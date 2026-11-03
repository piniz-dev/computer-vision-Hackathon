"""Demo dal vivo sul PC: Visione A ("Occhi") + Visione B ("Cervello") nella stessa finestra.

  [G]          -> Grok trova gli oggetti (anche quelli che YOLO non conosce, es. penne e matite)
                  e YOLO-World impara a seguirli dal vivo, anche se li muovi in mano
  [N]          -> Grok dà un nome breve a tutti gli oggetti numerati
  [S]          -> Scrivi un comando nella finestra (es. 'prendi la penna blu'), Invio per mandarlo
  [D] / SPAZIO -> Grok descrive la scena
  [Click Sx]   -> Seleziona un oggetto sotto il cursore (sagoma + pannello); la scelta va al
                  robot in logs/selection.json (anche da [S] e quando la selezione si riaggancia)
  [Z]          -> Zoom sull'oggetto attualmente selezionato (con S o Click Sx)
  [Click Dx]   -> Dezooma (riporta alla vista panoramica 1.0x, come [0] / [R])
  [C]          -> Dimentica tutto: oggetti di Grok e classi imparate (che altrimenti restano
                  in logs/vocab.json e tornano al prossimo avvio)
  [1-9]        -> Seleziona l'oggetto con quel numero e zooma
  [K]          -> Mostra/nascondi percentuale di confidenza
  [+] / [-]    -> Alza/abbassa la soglia minima di confidenza (+/- 5%)
  [T]          -> Imposta target per il cane robotico
  [H]          -> Aiuto
  [Q] / ESC    -> Esci

Senza XAI_API_KEY nel .env la demo parte in modalità finta (GROK_MOCK=1).

Utilizzo:
  python demo_live.py                 # Webcam 0 con YOLOv8 ONNX + YOLO-World + Grok (.env)
  python demo_live.py --cam 1         # Seconda webcam
  python demo_live.py --no-yolo       # Solo Grok (senza YOLO)
  python demo_live.py --no-world      # Senza YOLO-World (gli oggetti di Grok restano fermi)
  python demo_live.py --once          # Scatta una foto, YOLO + Grok, salva in logs/webcam_marked.jpg ed esce
  python demo_live.py --replay        # Da video/simulazione replay
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import threading
import time
import unicodedata
from pathlib import Path

import cv2

from vision.eyes.depth import CLASS_PRIORS_CM, estimate_distance_m, format_distance
from vision.eyes.marks import draw_marks
from vision.eyes.pipeline import EyePipeline
from vision.grok.geometry import refine_box
from vision.grok.handoff import send_selection
from vision.grok.local_resolver import display
from vision.grok.marks import object_at, reacquire
from vision.grok.open_vocab import OpenVocab, load_vocab, save_vocab
from vision.grok.schemas import TrackedObject
from vision.grok.tasks import describe_scene, detect_objects, name_objects, select
from vision.planner import PlanExecutor, RobotPlan, RobotPlanAgent

# vision.grok.client ha già caricato il .env. Senza chiave si va in modalità finta.
if not os.getenv("XAI_API_KEY") and not os.getenv("GROK_MOCK"):
    os.environ["GROK_MOCK"] = "1"

WIN = "Grok Hackathon - Occhi + Cervello Live"
LOGS_DIR = Path("logs")
VOCAB_PATH = LOGS_DIR / "vocab.json"          # classi imparate da Grok, ricaricate al prossimo avvio
SELECTION_PATH = LOGS_DIR / "selection.json"  # ultima scelta, letta dal braccio (formato in vision/grok/handoff.py)
FONT = cv2.FONT_HERSHEY_SIMPLEX
STATUS_COLOR = {"ok": (90, 220, 90), "wait": (0, 200, 255), "err": (70, 70, 255)}  # BGR
HELP = [
    "G       Grok trova gli oggetti (poi YOLO-World li segue)",
    "N       Grok da' i nomi",
    "S       scrivi un comando (es. prendi la penna blu)",
    "P       pianifica missione con Grok (Plan & Solve Markdown)",
    "D/Spazio  descrivi la scena (oppure esegui passo del piano)",
    "Click Sx  seleziona oggetto",
    "Z       zoom sull'oggetto selezionato",
    "Click Dx  dezooma (oppure 0 / R)",
    "C       dimentica tutto (oggetti di Grok e classi imparate)",
    "+/-     soglia confidenza    K  mostra %",
    "T       target cane",
    "H       chiudi aiuto    Q/Esc  esci",
]


def _ascii(s: str) -> str:
    """cv2.putText non sa scrivere le lettere accentate: "è" -> "e"."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


def display_obj(o: dict) -> str:
    return display(TrackedObject.model_validate(o))


def _learn_size(label: str, size_cm) -> None:
    """Aggiunge la misura reale stimata da Grok alla tabella usata da Visione A per la distanza.
    ponytail: scriviamo in memoria nel dict di A (setdefault: le misure che hanno già restano);
    se A aggiunge un modo ufficiale per registrare misure, usare quello."""
    if size_cm and len(size_cm) == 2 and all(v > 0 for v in size_cm):
        CLASS_PRIORS_CM.setdefault(label.strip().lower(), (float(size_cm[0]), float(size_cm[1])))


def _distance(o: dict) -> float | None:
    return o.get("distance_m") or estimate_distance_m(o.get("label", "object"), o["bbox"])


def _source(obj_id: int) -> str:
    """Chi ha trovato l'oggetto, dagli intervalli di id: YOLO < 1000 <= Grok < 2000 <= YOLO-World."""
    return "YOLO" if obj_id < 1000 else "Grok" if obj_id < 2000 else "YOLO-World"


def _box(frame, x1, y1, x2, y2, alpha=0.75):
    """Rettangolo scuro semitrasparente (sfondo dei pannelli)."""
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    roi[:] = (roi * (1 - alpha)).astype(roi.dtype) + int(18 * alpha)


def _text(frame, s, x, y, color=(255, 255, 255), scale=0.45, thick=1):
    cv2.putText(frame, _ascii(s), (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _draw_top(frame, fps, n_objs, zoom, live_status, has_plan=False):
    w = frame.shape[1]
    _box(frame, 0, 0, w, 44)
    hint_plan = "   [Piano attivo: SPAZIO passo, A tutto]" if has_plan else ""
    _text(frame, f"H aiuto   G trova   N nomi   S comando   P piano Grok   Z zoom   Click Dx dezooma{hint_plan}", 10, 18, (0, 230, 255))
    _text(frame, f"FPS {fps:.0f}  |  {n_objs} oggetti  |  zoom {zoom:.1f}x  |  live: {live_status}",
          10, 36, (190, 190, 190), 0.4)


def _draw_status(frame, msg, kind, latency):
    h, w = frame.shape[:2]
    lines = textwrap.wrap(_ascii(msg), max(30, w // 9))[:3] or [""]
    top = h - 16 - 20 * len(lines)
    _box(frame, 0, top, w, h)
    cv2.rectangle(frame, (0, top), (6, h), STATUS_COLOR[kind], -1)
    for i, line in enumerate(lines):
        _text(frame, line, 16, top + 20 + 20 * i, scale=0.5)
    if latency:
        _text(frame, latency, w - 150, top + 20, (150, 150, 150), 0.4)


def _draw_panel(frame, obj, sel_id, last_cmd):
    """Scheda dell'oggetto selezionato, in alto a destra."""
    w = frame.shape[1]
    if obj is None:
        lines = [(f"#{sel_id} non e' piu' visibile", (0, 200, 255))]
    else:
        lines = [(f"#{obj['id']}  {display_obj(obj)}", (0, 255, 255)),
                 (f"tipo: {obj['label']} ({_source(obj['id'])})", (230, 230, 230)),
                 (f"colore: {obj.get('color') or '-'}", (230, 230, 230)),
                 (f"distanza: {format_distance(_distance(obj))}", (230, 230, 230)),
                 (f"sicurezza: {int(round(obj.get('conf', 0) * 100))}%", (230, 230, 230))]
        if obj.get("tags"):
            lines.append((f"tag: {', '.join(obj['tags'][:4])}", (230, 230, 230)))
        if last_cmd and last_cmd.get("target_id") == obj["id"]:
            lines.append((f"azione: {last_cmd['action']} ({last_cmd['source']})", (90, 220, 90)))
    x1, y1 = w - 290, 54
    _box(frame, x1, y1, w - 8, y1 + 14 + 22 * len(lines))
    for i, (s, color) in enumerate(lines):
        _text(frame, s[:38], x1 + 10, y1 + 24 + 22 * i, color)


def _draw_plan_panel(frame, plan: RobotPlan, executor: PlanExecutor | None):
    """Visualizza il piano d'azione in formato Markdown con checkbox a schermo."""
    h, w = frame.shape[:2]
    pw = min(680, w - 30)
    steps = plan.steps
    n_show = min(8, len(steps))
    ph = 58 + 23 * n_show + 36
    x1 = (w - pw) // 2
    y1 = max(50, (h - ph) // 2)

    _box(frame, x1, y1, x1 + pw, y1 + ph, 0.90)
    cv2.rectangle(frame, (x1, y1), (x1 + pw, y1 + ph), (0, 230, 255), 2)

    # Titolo obiettivo e stato
    _text(frame, f"PIANO GROK: {plan.goal[:38]}", x1 + 14, y1 + 24, (0, 255, 255), 0.50, 2)
    _text(frame, f"{plan.progress_str()}  [{plan.status.value}]", x1 + pw - 240, y1 + 24, (120, 255, 120), 0.42)

    # Elenco passi atomici
    for i, step in enumerate(steps[:n_show]):
        sy = y1 + 50 + 23 * i
        chk = "[x]" if step.done else "[ ]"
        is_cur = (plan.current_step() == step)
        col = (90, 220, 90) if step.done else ((0, 230, 255) if is_cur else (210, 210, 210))
        arrow = "-> " if is_cur else "   "
        line_str = f"{arrow}{step.index}. {chk} {step.action.value}"
        if step.target_id is not None:
            line_str += f" #{step.target_id}"
        coords = []
        if step.x_mm is not None and step.y_mm is not None:
            coords.append(f"X={step.x_mm:.0f},Y={step.y_mm:.0f}")
        if step.z_mm is not None:
            coords.append(f"Z={step.z_mm:.0f}")
        if coords:
            line_str += f" ({', '.join(coords)})"
        if step.note:
            line_str += f" | {step.note[:24]}"
        _text(frame, line_str[:64], x1 + 14, sy, col, 0.44, 2 if is_cur else 1)

    fy = y1 + ph - 12
    _text(frame, "SPAZIO = esegui passo    A = esegui tutto    X / ESC = chiudi piano", x1 + 14, fy, (0, 230, 255), 0.40)


def _draw_input(frame, buf, prompt="Comando per il robot  (Invio = manda, Esc = annulla)"):
    h, w = frame.shape[:2]
    y = h // 2 - 30
    _box(frame, 20, y, w - 20, y + 64, 0.85)
    cv2.rectangle(frame, (20, y), (w - 20, y + 64), (0, 230, 255), 2)
    _text(frame, prompt, 34, y + 22, (0, 230, 255), 0.45)
    blink = "_" if int(time.time() * 2) % 2 else " "
    _text(frame, buf[-55:] + blink, 34, y + 50, scale=0.6)


def _draw_help(frame):
    h, w = frame.shape[:2]
    x1, y1 = max(10, w // 2 - 250), max(50, h // 2 - 130)
    _box(frame, x1, y1, x1 + 500, y1 + 40 + 22 * len(HELP), 0.85)
    _text(frame, "Comandi", x1 + 14, y1 + 26, (0, 230, 255), 0.55)
    for i, s in enumerate(HELP):
        _text(frame, s, x1 + 14, y1 + 52 + 22 * i)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam", type=int, default=0, help="Indice webcam (default 0)")
    parser.add_argument("--replay", action="store_true", help="Usa simulazione replay al posto della webcam")
    parser.add_argument("--no-yolo", action="store_true", help="Disabilita YOLO locale (solo rilevamento via Grok)")
    parser.add_argument("--no-world", action="store_true", help="Disabilita YOLO-World (oggetti di Grok fermi)")
    parser.add_argument("--once", action="store_true", help="Scatta una foto, rileva con YOLO + Grok, salva logs/webcam_marked.jpg ed esce")
    parser.add_argument("--model", type=str, default="data/models/yolov8n.onnx", help="Pesi YOLO (.onnx o .pt)")
    parser.add_argument("--min-conf", type=float, default=0.40, help="Soglia minima di confidenza per visualizzare riquadri (default 0.40)")
    args = parser.parse_args()

    min_conf = max(0.05, min(0.95, args.min_conf))
    show_conf = True  # K lo cambia

    # Video e tracker YOLO di Visione A
    pipeline = EyePipeline(fps=30.0)
    pipeline.add_source(
        source="arm",
        mode="replay" if args.replay else "webcam",
        target="data/replays/sample_arm.mp4" if args.replay else args.cam,
        use_mock_tracker=args.no_yolo,
        model_name=args.model,
        conf_threshold=0.25,
        ema_alpha=0.65,
    )
    arm_node = pipeline.nodes.get("arm")
    zoom_mgr = arm_node.zoom if arm_node else None

    # --no-yolo: tracker senza modello e senza oggetti finti, così gli oggetti arrivano solo da Grok
    if args.no_yolo and arm_node:
        arm_node.tracker.use_mock = False
        arm_node.tracker._yolo_model = None
        arm_node.tracker._onnx_net = None

    # --once: una foto, YOLO + Grok, immagine salvata e risultato nel terminale
    if args.once:
        print("[Demo Once] Acquisizione fotogramma...")
        pipeline.step("arm")
        raw_frame = pipeline.state.get_frame("arm")
        yolo_objs = pipeline.state.get_objects("arm")

        print(f"[Demo Once] YOLO ha rilevato {len(yolo_objs)} oggetti. Invio a Grok...")
        res = detect_objects(raw_frame, known=yolo_objs)
        grok_objs = []
        if res.ok:
            for o in res.objects:
                _learn_size(o.label, o.size_cm)
                d = o.model_dump()
                d["distance_m"] = estimate_distance_m(d["label"], d["bbox"])
                grok_objs.append(d)
        all_objs = yolo_objs + grok_objs

        marked = draw_marks(raw_frame, all_objs, show_conf=show_conf, show_dist=True)
        LOGS_DIR.mkdir(exist_ok=True)
        out_path = LOGS_DIR / "webcam_marked.jpg"
        cv2.imwrite(str(out_path), marked)

        print("\n" + "=" * 60)
        print(" RISULTATO SCATTO SINGOLO:")
        print(f" YOLO: {len(yolo_objs)} oggetti | Grok: {len(grok_objs)} oggetti")
        if res.meta:
            print(f" Latenza Grok: {res.meta.get('latency_ms', 0):.0f} ms")
            for why in res.meta.get("discard_reasons", []):
                print(f" Scartato: {why}")
        if not res.ok:
            print(f" Errore Grok: {res.error}")
        print("=" * 60)
        for o in all_objs:
            dist_txt = format_distance(_distance(o))
            print(f"  #{o['id']:<4} {o['label']:<15} {display_obj(o)} | dist={dist_txt:<6} | bbox={[round(v, 2) for v in o['bbox']]}")
        print(f"\nImmagine salvata in: {out_path.resolve()}\n")
        pipeline.stop()
        return

    ov = None if args.no_world else OpenVocab()  # carica in background, il video parte subito
    planner_agent = RobotPlanAgent()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 980, 620)

    # Stato condiviso tra ciclo video, click e thread delle chiamate a Grok.
    # ponytail: dict senza lock, basta per una demo (una sola chiamata Grok alla volta).
    st = {
        "grok_objects": [],   # riquadri fermi di Grok (finché YOLO-World non segue quella classe)
        "objects": [],        # tutti gli oggetti dell'ultimo frame
        "frame_size": (640, 480),
        "meta": {},           # id -> {name, color, tags} dati da Grok con N
        "label_meta": load_vocab(VOCAB_PATH),  # label -> {name, color, tags, size_cm}, anche dalle sessioni prima
        "selected_id": None,
        "sel_last": None,     # ultimo dict dell'oggetto selezionato, per riagganciarlo se cambia id
        "sel_seen": 0.0,      # quando l'abbiamo visto l'ultima volta
        "sel_cmd": {},        # command/action/destination_id della scelta, riscritti se si riaggancia
        "last_cmd": None,     # ultimo comando S andato a buon fine, per il pannello
        "typing": None,       # testo del comando mentre lo scrivi (None = non stai scrivendo)
        "typing_mode": "command", # 'command' per S, 'plan' per P
        "executor": None,     # esecutore attivo del piano robotico (Parte C)
        "help": False,
        "is_busy": False,     # una chiamata a Grok è in corso
        "status": ("Pronto. G = Grok trova, S = comando, P = piano robot, click = seleziona, H = aiuto.", "ok"),
        "latency_msg": "",
        "dog_info": "",
    }

    def set_status(msg: str, kind: str = "ok") -> None:
        st["status"] = (msg, kind)

    # Classi imparate nelle sessioni precedenti: YOLO-World le segue subito, senza premere G
    for label, m in st["label_meta"].items():
        _learn_size(label, m.get("size_cm"))
    if ov and st["label_meta"]:
        ov.set_classes(st["label_meta"])
        set_status(f"Ricordo {', '.join(st['label_meta'])}: le seguo dal vivo. C = dimentica tutto, H = aiuto.")

    def latency(res) -> str:
        return f"{res.meta.get('latency_ms', 0) / 1000:.1f} s" if res.meta else ""

    def reset_zoom() -> None:
        if zoom_mgr is not None:
            zoom_mgr.reset()
            set_status("Zoom resettato (1.0x)")

    def send(obj: dict) -> str:
        """Scrive la scelta in logs/selection.json per il braccio; restituisce il testo per la barra."""
        obj["distance_m"] = _distance(obj)
        msg = send_selection(SELECTION_PATH, obj, **st["sel_cmd"])
        if msg is None:
            return "NON inviato al robot (logs/selection.json non scrivibile)"
        print("\n[ROBOT] logs/selection.json:\n" + json.dumps(msg, ensure_ascii=False, indent=2, default=str))
        return "inviato al robot"

    def select_obj(obj: dict, command: str | None = None, action: str | None = None,
                   destination_id: int | None = None) -> str:
        """Seleziona l'oggetto e lo manda al braccio. command/action ci sono solo se arriva da S."""
        st["selected_id"] = obj["id"]
        st["sel_last"], st["sel_seen"] = obj, time.time()
        st["sel_cmd"] = {"command": command, "action": action, "destination_id": destination_id}
        sent = send(obj)
        set_status(f"Selezionato #{obj['id']}: {display_obj(obj)} - distanza {format_distance(obj['distance_m'])}, "
                   f"{sent}  (Premi [Z] per zoomare)")
        return sent

    # Click sinistro = seleziona l'oggetto sotto il cursore, click destro = dezooma
    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            return reset_zoom()
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        win_w, win_h = st["frame_size"]
        fx, fy = x / max(1, win_w), y / max(1, win_h)
        if zoom_mgr is not None and zoom_mgr.is_active():  # da coordinate dello zoom a quelle del frame intero
            fx, fy = zoom_mgr.zoomed_to_full_coords(fx, fy)
        clicked_obj = object_at(st["objects"], fx, fy)
        if clicked_obj is not None:
            select_obj(clicked_obj)
        else:
            set_status("Nessun oggetto qui. Clicca su un oggetto o premi [S] per selezionarlo.", "wait")

    cv2.setMouseCallback(WIN, on_mouse_click)

    # Chiamate a Grok: ognuna in un thread, così il video non si ferma
    def run_grok_detect(raw_frame, yolo_objs):
        res = detect_objects(raw_frame, known=yolo_objs)
        if not res.ok:
            return set_status(f"Errore Grok: {res.error}", "err")
        for o in res.objects:
            _learn_size(o.label, o.size_cm)
            st["label_meta"].setdefault(o.label, {"name": o.name, "color": o.color, "tags": o.tags,
                                                  "size_cm": o.size_cm})
        save_vocab(VOCAB_PATH, st["label_meta"])
        st["grok_objects"] = [dict(o.model_dump(), distance_m=estimate_distance_m(o.label, o.bbox))
                              for o in res.objects]
        names = ", ".join(o.name for o in res.objects) or "nessuno"
        if ov and res.objects:
            ov.set_classes(o.label for o in res.objects)
            names += ". YOLO-World ora li segue dal vivo"
        dropped = res.meta.get("discard_reasons", [])
        if dropped:
            names += f" (scartati {len(dropped)} riquadri dubbi)"
            print("[Grok] scartati: " + "; ".join(dropped))
        set_status(f"Grok ha trovato {len(res.objects)} oggetti: {names}")
        st["latency_msg"] = latency(res)

    def run_describe(raw_img, objs):
        res = describe_scene(raw_img, objs)
        set_status(res.text) if res.ok else set_status(f"Errore: {res.error}", "err")
        st["latency_msg"] = latency(res)

    def run_names(marked_img, objs):
        res = name_objects(marked_img, objs)
        if not res.ok:
            return set_status(f"Errore: {res.error}", "err")
        by_id = {o["id"]: o for o in objs}
        for n in res.objects:
            meta = {"name": n.name, "color": n.color, "tags": n.tags}
            st["meta"][n.id] = meta
            if n.id >= 2000:  # gli id di YOLO-World cambiano quando perde l'oggetto: il nome si ricorda per classe
                lbl = by_id[n.id]["label"]
                st["label_meta"][lbl] = {**st["label_meta"].get(lbl, {}), **meta}  # size_cm resta
            if arm_node and n.id < 1000:
                arm_node.tracker.set_object_metadata(n.id, name=n.name, color=n.color)
            for go in st["grok_objects"]:
                if go["id"] == n.id:
                    go.update(meta)
        save_vocab(VOCAB_PATH, st["label_meta"])
        set_status("Nomi: " + (", ".join(f"#{n.id} {n.name}" for n in res.objects) or "nessuno"))
        st["latency_msg"] = latency(res)

    def run_select(marked_img, objs, text):
        res = select(marked_img, objs, text)
        st["latency_msg"] = f"{latency(res)} ({res.source})"
        if not res.ok:
            return set_status(f"Errore: {res.error}", "err")
        if res.ambiguous:
            return set_status(f"Quale? {res.question}", "wait")
        hit = next((o for o in objs if o["id"] == res.target_id), None)
        if hit is None:
            return set_status(f"Non trovato: {res.reason}", "err")
        st["last_cmd"] = {"target_id": res.target_id, "action": res.action, "source": res.source}
        sent = select_obj(hit, text, res.action, res.destination_id)
        set_status(f"'{text}' -> #{hit['id']} {display_obj(hit)}, azione: {res.action}, {sent}")

    def run_robot_plan(objs, raw_img, goal: str):
        try:
            plan, plan_file = planner_agent.plan(goal=goal, objects=objs, raw_frame=raw_img)
            executor = PlanExecutor(
                plan=plan,
                plan_path=plan_file,
                on_step_done=lambda step, d, t: set_status(f"Eseguito passo {d}/{t}: {step.action.value}"),
                on_completed=lambda p: set_status(f"Piano completato con successo: '{p.goal}'!"),
            )
            st["executor"] = executor
            set_status(f"Piano generato ({len(plan.steps)} passi): {plan_file.name}. SPAZIO = passo, A = tutto, X/Esc = chiudi")
        except Exception as e:
            set_status(f"Errore piano: {e}", "err")

    def start(job, *job_args, msg):
        st["is_busy"] = True
        set_status(msg, "wait")

        def run():
            try:
                job(*job_args)
            finally:
                st["is_busy"] = False

        threading.Thread(target=run, daemon=True).start()

    print("\n" + "=" * 65)
    print("  DEMO LIVE: OCCHI (Parte A) + CERVELLO (Parte B)   -   H nella finestra = aiuto")
    print("=" * 65 + "\n")

    t_prev = time.perf_counter()
    fps_smooth = 30.0
    win_shown = False  # la finestra può risultare "non visibile" nei primi frame: non è la X

    try:
        while True:
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_smooth = 0.90 * fps_smooth + 0.10 * (1.0 / dt)

            pipeline.step("arm", min_display_conf=min_conf, show_conf=show_conf, apply_zoom=False)
            raw = pipeline.state.get_frame("arm")
            st["frame_size"] = (raw.shape[1], raw.shape[0])
            yolo_objs = pipeline.state.get_objects("arm")

            # YOLO-World: oggetti vivi delle classi trovate da Grok, con i nomi dati da Grok
            live = []
            if ov:
                ov.feed(raw)
                live = [dict(o) for o in ov.objects]
                seen = {o["label"] for o in live}
                if seen:  # quella classe ora si muove dal vivo: via il riquadro fermo di Grok
                    st["grok_objects"] = [o for o in st["grok_objects"] if o["label"] not in seen]
                for o in live:
                    o.update(st["label_meta"].get(o["label"], {}))
            all_objs = yolo_objs + live + st["grok_objects"]
            for o in all_objs:
                o.update(st["meta"].get(o["id"], {}))
            st["objects"] = all_objs

            frame_display = draw_marks(raw, all_objs, min_conf=min_conf, show_conf=show_conf)

            sel = next((o for o in all_objs if o["id"] == st["selected_id"]), None)
            # Selezionato sparito: forse il tracker gli ha dato un altro id. Lo cerchiamo per 3 s
            if sel is None and st["sel_last"] and time.time() - st["sel_seen"] < 3.0:
                sel = reacquire(all_objs, st["sel_last"])
                if sel is not None:
                    old = st["selected_id"]
                    st["selected_id"] = sel["id"]
                    if st["last_cmd"] and st["last_cmd"]["target_id"] == old:
                        st["last_cmd"]["target_id"] = sel["id"]
                    if zoom_mgr is not None and zoom_mgr.target_id == old:
                        zoom_mgr.set_target_object(sel["id"])
                    set_status(f"#{sel['id']} riagganciato (prima #{old}): {display_obj(sel)}, {send(sel)}")
            # Oggetto selezionato: sagoma, oppure riquadro bianco se la sagoma non si trova
            if sel is not None:
                st["sel_last"], st["sel_seen"] = sel, time.time()
                fit = refine_box(raw, sel["bbox"])
                if fit is not None:
                    cv2.drawContours(frame_display, [fit[1]], -1, (255, 255, 255), 4, cv2.LINE_AA)
                    cv2.drawContours(frame_display, [fit[1]], -1, (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    h_f, w_f = frame_display.shape[:2]
                    bx, by, bw, bh = sel["bbox"]
                    cv2.rectangle(frame_display, (int(bx * w_f) - 3, int(by * h_f) - 3),
                                  (int((bx + bw) * w_f) + 3, int((by + bh) * h_f) + 3), (255, 255, 255), 3)

            if zoom_mgr is not None and zoom_mgr.is_active():
                zoom_mgr.update_with_tracked_objects(all_objs)
                frame_display = zoom_mgr.apply(frame_display)

            # Interfaccia sopra il video
            has_plan = st.get("executor") is not None
            _draw_top(frame_display, fps_smooth, sum(1 for o in all_objs if o.get("conf", 0) >= min_conf),
                      zoom_mgr.current_factor if zoom_mgr else 1.0, ov.status if ov else "spento",
                      has_plan=has_plan)
            msg, kind = st["status"]
            _draw_status(frame_display, ("[Grok lavora...] " if st["is_busy"] else "") + msg,
                         "wait" if st["is_busy"] else kind, st["latency_msg"])
            if st["selected_id"] is not None:
                _draw_panel(frame_display, sel, st["selected_id"], st["last_cmd"])
            if st["dog_info"]:
                _text(frame_display, st["dog_info"], 10, 64, (0, 165, 255), 0.5, 2)
            if st["typing"] is not None:
                prompt = "Obiettivo missione robot (Grok Plan & Solve)  [Invio=pianifica, Esc=annulla]" if st.get("typing_mode") == "plan" else "Comando per il robot  (Invio = manda, Esc = annulla)"
                _draw_input(frame_display, st["typing"], prompt=prompt)
            elif st.get("executor") is not None:
                _draw_plan_panel(frame_display, st["executor"].plan, st["executor"])
            elif st["help"]:
                _draw_help(frame_display)

            cv2.imshow(WIN, frame_display)
            key = cv2.waitKey(15) & 0xFF
            visible = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) >= 1
            if win_shown and not visible:  # chiusa con la X
                break
            win_shown = win_shown or visible
            if key == 255:
                continue

            # Mentre scrivi un comando, i tasti vanno nel testo
            if st["typing"] is not None:
                if key == 27:
                    st["typing"] = None
                    set_status("Comando annullato")
                elif key in (13, 10):
                    text, st["typing"] = st["typing"].strip(), None
                    if text and not st["is_busy"]:
                        if st.get("typing_mode") == "plan":
                            start(run_robot_plan, list(all_objs), raw.copy(), text, msg=f"Pianifico con Grok: '{text}'...")
                        else:
                            marked = draw_marks(raw, all_objs)
                            start(run_select, marked, list(all_objs), text, msg=f"Cerco: '{text}'...")
                elif key == 8:
                    st["typing"] = st["typing"][:-1]
                elif 32 <= key < 127:
                    st["typing"] += chr(key)
                continue

            # Controllo piano robotico attivo
            if st.get("executor") is not None:
                if key in (32, 13):  # SPAZIO o INVIO: esegui un passo atomico
                    step = st["executor"].step()
                    if step:
                        set_status(f"Eseguito passo {step.index}: {step.action.value} ({st['executor'].plan.progress_str()})")
                    else:
                        set_status(f"Piano terminato [{st['executor'].plan.status.value}]")
                    continue
                elif key in (ord("a"), ord("A")):
                    if not st["executor"].is_running:
                        st["executor"].start_background(delay_sec=0.35)
                        set_status("Esecuzione automatica del piano avviata...")
                    continue
                elif key in (27, ord("x"), ord("X")):
                    st["executor"].stop()
                    st["executor"] = None
                    set_status("Piano chiuso")
                    continue

            if key in (27, ord("q"), ord("Q")):
                break

            elif key in (ord("h"), ord("H")):
                st["help"] = not st["help"]

            elif key in (ord("g"), ord("G")) and not st["is_busy"]:
                start(run_grok_detect, raw.copy(), list(yolo_objs), msg="Grok cerca gli oggetti nella scena...")

            elif key in (ord("d"), ord("D"), 32) and not st["is_busy"]:
                start(run_describe, raw.copy(), list(all_objs), msg="Grok descrive la scena...")

            # N e S mandano a Grok il frame con i soli numeri, senza HUD né zoom
            elif key in (ord("n"), ord("N")) and not st["is_busy"]:
                start(run_names, draw_marks(raw, all_objs), list(all_objs), msg="Grok da' i nomi agli oggetti...")

            elif key in (ord("s"), ord("S")):
                st["typing"], st["typing_mode"], st["help"] = "", "command", False

            elif key in (ord("p"), ord("P")):
                st["typing"], st["typing_mode"], st["help"] = "", "plan", False

            # C: dimentica gli oggetti di Grok, le classi di YOLO-World e vocab.json
            elif key in (ord("c"), ord("C")):
                st["grok_objects"], st["label_meta"] = [], {}
                if ov:
                    ov.forget()
                save_vocab(VOCAB_PATH, {})
                set_status("Dimenticato tutto: oggetti di Grok e classi imparate. G = ricomincia.")

            elif key in (ord("k"), ord("K")):
                show_conf = not show_conf
                set_status(f"Percentuale di confidenza: {'visibile' if show_conf else 'nascosta'}")

            # + e - anche con = _ ] [ (stesso tasto senza Shift su layout diversi)
            elif key in (ord("+"), ord("="), ord("]")):
                min_conf = min(0.95, round(min_conf + 0.05, 2))
                set_status(f"Soglia di confidenza: {int(min_conf * 100)}%")

            elif key in (ord("-"), ord("_"), ord("[")):
                min_conf = max(0.05, round(min_conf - 0.05, 2))
                set_status(f"Soglia di confidenza: {int(min_conf * 100)}%")

            elif key in (ord("z"), ord("Z")):
                target_obj = sel or st["sel_last"]
                if target_obj is not None and zoom_mgr is not None:
                    zoom_mgr.set_target_object(target_obj["id"])
                    zoom_mgr.zoom_to_bbox(target_obj["bbox"])
                    set_status(
                        f"Zoom su #{target_obj['id']} ({display_obj(target_obj)}). "
                        f"Click destro del mouse (o 0/R) per dezoomare."
                    )
                else:
                    set_status("Seleziona prima un oggetto (con S o Click sinistro) e poi premi Z per zoomare.", "wait")

            elif ord("1") <= key <= ord("9"):
                target_num = key - ord("0")
                if zoom_mgr is not None:
                    matched = [o for o in all_objs if o.get("id") == target_num]
                    if matched:
                        select_obj(matched[0])
                        zoom_mgr.set_target_object(target_num)
                        zoom_mgr.zoom_to_bbox(matched[0]["bbox"])
                        set_status(f"Zoom su #{target_num} ({display_obj(matched[0])}). Click destro = dezooma.")
                    else:
                        set_status(f"Oggetto #{target_num} non visibile", "wait")

            elif key in (ord("0"), ord("r"), ord("R")):
                reset_zoom()

            elif key in (ord("t"), ord("T")):
                target = sel or (all_objs[0] if all_objs else None)
                if target:
                    st["dog_info"] = f"Target cane: #{target['id']}"
                    set_status(f"Target cane impostato su #{target['id']} ({display_obj(target)}).")
                else:
                    set_status("Nessun oggetto da agganciare.", "err")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
