"""Demo interattiva Desktop unificata: Visione A ("Occhi") + Visione B ("Cervello").

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
  [K]          -> Mostra/nascondi percentuale di confidenza
  [+] / [-]    -> Alza/abbassa la soglia minima di confidenza (+/- 5%)
  [T]          -> Imposta target per il cane robotico
  [H]          -> Aiuto
  [Q] / ESC    -> Esci

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
from dotenv import find_dotenv, load_dotenv

# Carica .env dalla root del progetto con massima priorità
_env_path = find_dotenv(usecwd=True)
if not _env_path:
    _candidate = Path(__file__).resolve().parent / ".env"
    _env_path = str(_candidate) if _candidate.exists() else None
load_dotenv(_env_path)

# Se non è impostata alcuna chiave, fallback a GROK_MOCK=1 per consentire demo offline
if not os.getenv("XAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY") and not os.getenv("GROK_MOCK"):
    os.environ["GROK_MOCK"] = "1"

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

WIN = "Grok Hackathon - Occhi + Cervello Live"
LOGS_DIR = Path("logs")
FONT = cv2.FONT_HERSHEY_SIMPLEX
STATUS_COLOR = {"ok": (90, 220, 90), "wait": (0, 200, 255), "err": (70, 70, 255)}  # BGR
HELP = [
    "G       Grok trova gli oggetti (poi YOLO-World li segue)",
    "N       Grok da' i nomi",
    "S       scrivi un comando (es. prendi la penna blu)",
    "D/Spazio  descrivi la scena",
    "Click Sx  seleziona oggetto",
    "Z       zoom sull'oggetto selezionato",
    "Click Dx  dezooma (oppure 0 / R)",
    "C       dimentica tutto (oggetti di Grok e classi imparate)",
    "+/-     soglia confidenza    K  mostra %",
    "T       target cane",
    "H       chiudi aiuto    Q/Esc  esci",
]


def _ascii(s: str) -> str:
    """Rimuove caratteri accentati per compatibilità con cv2.putText."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


def display_obj(o: dict) -> str:
    return display(TrackedObject.model_validate(o))


VOCAB_PATH = LOGS_DIR / "vocab.json"  # classi imparate da Grok, ricaricate al prossimo avvio
SELECTION_PATH = LOGS_DIR / "selection.json"  # ultima scelta, letta dal braccio (formato in vision/grok/handoff.py)


def _learn_size(label: str, size_cm) -> None:
    """Misura reale stimata da Grok -> tabella delle misure della stima di distanza di Visione A.
    ponytail: scriviamo nel dict di A in memoria (setdefault: le loro misure restano); se A
    aggiunge un modo ufficiale per registrare misure, usare quello."""
    if size_cm and len(size_cm) == 2 and all(v > 0 for v in size_cm):
        CLASS_PRIORS_CM.setdefault(label.strip().lower(), (float(size_cm[0]), float(size_cm[1])))


def _distance(o: dict) -> float | None:
    return o.get("distance_m") or estimate_distance_m(o.get("label", "object"), o["bbox"])


def _source(obj_id: int) -> str:
    return "YOLO" if obj_id < 1000 else "Grok" if obj_id < 2000 else "YOLO-World"


def _box(frame, x1, y1, x2, y2, alpha=0.75):
    """Rettangolo scuro semitrasparente (sfondo dei pannelli)."""
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    roi[:] = (roi * (1 - alpha)).astype(roi.dtype) + int(18 * alpha)


def _text(frame, s, x, y, color=(255, 255, 255), scale=0.45, thick=1):
    cv2.putText(frame, _ascii(s), (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _draw_top(frame, fps, n_objs, zoom, live_status):
    w = frame.shape[1]
    _box(frame, 0, 0, w, 44)
    _text(frame, "H aiuto   G trova   N nomi   S comando   Z zoom sel   Click Dx dezooma   D descrivi   Q esci", 10, 18, (0, 230, 255))
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


def _draw_input(frame, buf):
    h, w = frame.shape[:2]
    y = h // 2 - 30
    _box(frame, 20, y, w - 20, y + 64, 0.85)
    cv2.rectangle(frame, (20, y), (w - 20, y + 64), (0, 230, 255), 2)
    _text(frame, "Comando per il robot  (Invio = manda, Esc = annulla)", 34, y + 22, (0, 230, 255), 0.45)
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
    parser.add_argument("--show-conf", action="store_true", default=True, help="Mostra percentuale confidenza nel badge")
    args = parser.parse_args()

    mode = "replay" if args.replay else "webcam"
    target = "data/replays/sample_arm.mp4" if args.replay else args.cam

    min_conf = max(0.05, min(0.95, args.min_conf))
    show_conf = args.show_conf

    # Inizializza pipeline di Visione A
    pipeline = EyePipeline(fps=30.0)
    pipeline.add_source(
        source="arm",
        mode=mode,
        target=target,
        use_mock_tracker=False if not args.no_yolo else True,
        model_name=args.model,
        conf_threshold=0.25,
        ema_alpha=0.65,
    )

    # Se --no-yolo è attivo, disabilitiamo il tracker locale impostando use_mock=False senza modelli
    if args.no_yolo:
        node = pipeline.nodes.get("arm")
        if node:
            node.tracker.use_mock = False
            node.tracker._yolo_model = None
            node.tracker._onnx_net = None

    # Modalità scatto singolo (--once)
    if args.once:
        print("[Demo Once] Acquisizione fotogramma...")
        payload = pipeline.step("arm")
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
        print(f" RISULTATO SCATTO SINGOLO:")
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

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 980, 620)

    # Stato condiviso tra ciclo video, click e thread delle chiamate a Grok.
    # ponytail: dict senza lock, basta per una demo (una sola chiamata Grok alla volta).
    st = {
        "grok_objects": [],   # riquadri fermi di Grok (finché YOLO-World non segue quella classe)
        "objects": [],        # tutti gli oggetti dell'ultimo frame
        "meta": {},           # id -> {name, color, tags} dati da Grok (N)
        "label_meta": load_vocab(VOCAB_PATH),  # label -> {name, color, tags, size_cm}, anche dalle sessioni prima
        "selected_id": None,
        "sel_last": None,     # ultimo dict dell'oggetto selezionato, per riagganciarlo se cambia id
        "sel_seen": 0.0,      # quando l'abbiamo visto l'ultima volta
        "last_cmd": None,
        "sel_cmd": {},  # comando/azione dell'ultima scelta, riusati se la selezione si riaggancia
        "typing": None,       # testo del comando mentre lo scrivi (None = non stai scrivendo)
        "help": False,
        "is_busy": False,
        "status": ("Pronto. G = Grok trova gli oggetti, click = seleziona, H = aiuto.", "ok"),
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

    zoom_node = pipeline.nodes.get("arm")
    zoom_mgr = zoom_node.zoom if zoom_node else None

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
        st["selected_id"] = obj["id"]
        st["sel_last"], st["sel_seen"] = obj, time.time()
        st["sel_cmd"] = {"command": command, "action": action, "destination_id": destination_id}
        sent = send(obj)
        set_status(f"Selezionato #{obj['id']}: {display_obj(obj)} - distanza {format_distance(obj['distance_m'])}, "
                   f"{sent}  (Premi [Z] per zoomare)")
        return sent

    # Handler click del mouse:
    # - Click Sinistro: seleziona l'oggetto sotto il cursore
    # - Click Destro:   dezooma (riporta alla vista panoramica 1.0x)
    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            if zoom_mgr is not None:
                zoom_mgr.reset()
                set_status("Zoom resettato (1.0x)")
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        win_w, win_h = st.get("frame_size", (640, 480))
        nx = float(x) / max(1, win_w)
        ny = float(y) / max(1, win_h)

        # Se lo zoom è attivo, converti da coordinate zoomate a full
        if zoom_mgr is not None and zoom_mgr.is_active():
            fx, fy = zoom_mgr.zoomed_to_full_coords(nx, ny)
        else:
            fx, fy = nx, ny

        clicked_obj = object_at(st["objects"], fx, fy)
        if clicked_obj is not None:
            select_obj(clicked_obj)
        else:
            set_status("Nessun oggetto qui. Clicca su un oggetto o premi [S] per selezionarlo.", "wait")

    cv2.setMouseCallback(WIN, on_mouse_click)

    # Thread runners per chiamate a Grok
    def run_grok_detect(raw_frame, yolo_objs):
        try:
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
        finally:
            st["is_busy"] = False

    def run_describe(raw_img, objs):
        try:
            res = describe_scene(raw_img, objs)
            set_status(res.text, "ok") if res.ok else set_status(f"Errore: {res.error}", "err")
            st["latency_msg"] = latency(res)
        finally:
            st["is_busy"] = False

    def run_names(marked_img, objs):
        try:
            res = name_objects(marked_img, objs)
            if not res.ok:
                return set_status(f"Errore: {res.error}", "err")
            by_id = {o["id"]: o for o in objs}
            node = pipeline.nodes.get("arm")
            for n in res.objects:
                meta = {"name": n.name, "color": n.color, "tags": n.tags}
                st["meta"][n.id] = meta
                if n.id >= 2000:  # gli id di YOLO-World cambiano se perde l'oggetto: ricorda per classe
                    lbl = by_id[n.id]["label"]
                    st["label_meta"][lbl] = {**st["label_meta"].get(lbl, {}), **meta}  # tiene size_cm
                if node and n.id < 1000:
                    node.tracker.set_object_metadata(n.id, name=n.name, color=n.color)
                for go in st["grok_objects"]:
                    if go["id"] == n.id:
                        go.update(meta)
            save_vocab(VOCAB_PATH, st["label_meta"])
            set_status("Nomi: " + (", ".join(f"#{n.id} {n.name}" for n in res.objects) or "nessuno"))
            st["latency_msg"] = latency(res)
        finally:
            st["is_busy"] = False

    def run_select(marked_img, objs, text):
        try:
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
        finally:
            st["is_busy"] = False

    def start(job, *job_args, msg):
        st["is_busy"] = True
        set_status(msg, "wait")
        threading.Thread(target=job, args=job_args, daemon=True).start()

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

            # Oggetto selezionato: sagoma (o riquadro bianco se la sagoma non si trova)
            sel = next((o for o in all_objs if o["id"] == st["selected_id"]), None)
            # Il tracker l'ha perso per un attimo e gli ha dato un altro id: lo riagganciamo (per 3 s)
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

            # Se lo zoom è attivo, applica il ritaglio
            if zoom_mgr is not None and zoom_mgr.is_active():
                zoom_mgr.update_with_tracked_objects(all_objs)
                frame_display = zoom_mgr.apply(frame_display)

            # Interfaccia sopra il video
            _draw_top(frame_display, fps_smooth, sum(1 for o in all_objs if o.get("conf", 0) >= min_conf),
                      zoom_mgr.current_factor if zoom_mgr else 1.0, ov.status if ov else "spento")
            msg, kind = st["status"]
            _draw_status(frame_display, ("[Grok lavora...] " if st["is_busy"] else "") + msg,
                         "wait" if st["is_busy"] else kind, st["latency_msg"])
            if st["selected_id"] is not None:
                _draw_panel(frame_display, sel, st["selected_id"], st["last_cmd"])
            if st["dog_info"]:
                _text(frame_display, st["dog_info"], 10, 64, (0, 165, 255), 0.5, 2)
            if st["typing"] is not None:
                _draw_input(frame_display, st["typing"])
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
                        marked = draw_marks(raw, all_objs)
                        start(run_select, marked, list(all_objs), text, msg=f"Cerco: '{text}'...")
                elif key == 8:
                    st["typing"] = st["typing"][:-1]
                elif 32 <= key < 127:
                    st["typing"] += chr(key)
                continue

            if key in (27, ord("q"), ord("Q")):
                break

            elif key in (ord("h"), ord("H")):
                st["help"] = not st["help"]

            # Grok Detect [G]
            elif key in (ord("g"), ord("G")) and not st["is_busy"]:
                start(run_grok_detect, raw.copy(), list(yolo_objs), msg="Grok cerca gli oggetti nella scena...")

            # Grok Describe [D] o SPAZIO
            elif key in (ord("d"), ord("D"), 32) and not st["is_busy"]:
                start(run_describe, raw.copy(), list(all_objs), msg="Grok descrive la scena...")

            # Grok Naming [N]: immagine con i soli numeri, senza HUD né zoom
            elif key in (ord("n"), ord("N")) and not st["is_busy"]:
                start(run_names, draw_marks(raw, all_objs), list(all_objs), msg="Grok da' i nomi agli oggetti...")

            # Comando scritto nella finestra [S]
            elif key in (ord("s"), ord("S")):
                st["typing"], st["help"] = "", False

            # Dimentica tutto [C]: oggetti di Grok, classi di YOLO-World e vocab.json
            elif key in (ord("c"), ord("C")):
                st["grok_objects"], st["label_meta"] = [], {}
                if ov:
                    ov.forget()
                save_vocab(VOCAB_PATH, {})
                set_status("Dimenticato tutto: oggetti di Grok e classi imparate. G = ricomincia.")

            # Toggle confidenza nel badge [K]
            elif key in (ord("k"), ord("K")):
                show_conf = not show_conf
                set_status(f"Percentuale di confidenza: {'visibile' if show_conf else 'nascosta'}")

            # Regola soglia confidenza [+] / [-]
            elif key in (ord("+"), ord("="), ord("]")):
                min_conf = min(0.95, round(min_conf + 0.05, 2))
                set_status(f"Soglia di confidenza: {int(min_conf * 100)}%")

            elif key in (ord("-"), ord("_"), ord("[")):
                min_conf = max(0.05, round(min_conf - 0.05, 2))
                set_status(f"Soglia di confidenza: {int(min_conf * 100)}%")

            # Zoom sull'oggetto selezionato [Z]
            elif key in (ord("z"), ord("Z")):
                target_obj = sel or st.get("sel_last")
                if target_obj is not None and zoom_mgr is not None:
                    zoom_mgr.set_target_object(target_obj["id"])
                    zoom_mgr.zoom_to_bbox(target_obj["bbox"])
                    set_status(
                        f"Zoom su #{target_obj['id']} ({display_obj(target_obj)}). "
                        f"Click destro del mouse (o 0/R) per dezoomare."
                    )
                else:
                    set_status("Seleziona prima un oggetto (con S o Click sinistro) e poi premi Z per zoomare.", "wait")

            # Zoom numerico su ID [1-9] (seleziona e zooma)
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

            # Reset Zoom [0] o [R] (oltre al Click Destro del mouse)
            elif key in (ord("0"), ord("r"), ord("R")):
                if zoom_mgr is not None:
                    zoom_mgr.reset()
                    set_status("Zoom resettato (1.0x)")

            # Target Cane [T]
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
