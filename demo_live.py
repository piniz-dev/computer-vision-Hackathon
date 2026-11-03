"""Demo interattiva Desktop unificata: Visione A ("Occhi") + Visione B ("Cervello").

  [G]          -> Grok trova gli oggetti (anche quelli che YOLO non conosce, es. penne e matite)
                  e YOLO-World impara a seguirli dal vivo, anche se li muovi in mano
  [N]          -> Grok dà un nome breve a tutti gli oggetti numerati
  [S]          -> Scrivi un comando nella finestra (es. 'prendi la penna blu'), Invio per mandarlo
  [D] / SPAZIO -> Grok descrive la scena
  [Click]      -> Su un oggetto: SELEZIONE (sagoma + pannello, JSON nel terminale)
                  Sullo sfondo:  ZOOM DIGITALE sul punto cliccato
  [1-9]        -> Zoom e tracking automatico sull'oggetto con quell'ID
  [0] / [R]    -> Reset zoom
  [C]          -> Cancella gli oggetti trovati da Grok
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

from vision.eyes.depth import estimate_distance_m, format_distance
from vision.eyes.marks import draw_marks
from vision.eyes.pipeline import EyePipeline
from vision.grok.geometry import refine_box
from vision.grok.local_resolver import display
from vision.grok.marks import object_at
from vision.grok.open_vocab import OpenVocab
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
    "Click   seleziona oggetto / zoom sullo sfondo",
    "1-9     zoom su oggetto    0/R  reset zoom",
    "C       cancella oggetti di Grok",
    "+/-     soglia confidenza    K  mostra %",
    "T       target cane",
    "H       chiudi aiuto    Q/Esc  esci",
]


def _ascii(s: str) -> str:
    """Rimuove caratteri accentati per compatibilità con cv2.putText."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


def display_obj(o: dict) -> str:
    return display(TrackedObject.model_validate(o))


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
    _text(frame, "H aiuto   G trova   N nomi   S comando   D descrivi   Q esci", 10, 18, (0, 230, 255))
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
        "label_meta": {},     # label -> {name, color, tags}: nomi per gli oggetti di YOLO-World
        "selected_id": None,
        "last_cmd": None,
        "typing": None,       # testo del comando mentre lo scrivi (None = non stai scrivendo)
        "help": False,
        "is_busy": False,
        "status": ("Pronto. G = Grok trova gli oggetti, click = seleziona, H = aiuto.", "ok"),
        "latency_msg": "",
        "dog_info": "",
    }

    def set_status(msg: str, kind: str = "ok") -> None:
        st["status"] = (msg, kind)

    def latency(res) -> str:
        return f"{res.meta.get('latency_ms', 0) / 1000:.1f} s" if res.meta else ""

    zoom_node = pipeline.nodes.get("arm")
    zoom_mgr = zoom_node.zoom if zoom_node else None

    def select_obj(obj: dict) -> None:
        st["selected_id"] = obj["id"]
        obj["distance_m"] = _distance(obj)
        set_status(f"Selezionato #{obj['id']}: {display_obj(obj)} - distanza {format_distance(obj['distance_m'])}")
        print("\n[ROBOT PICK] Oggetto selezionato da passare al robot:")
        print(json.dumps(obj, ensure_ascii=False, indent=2))

    # Handler click del mouse:
    # 1. Se clicchi su un oggetto -> SELEZIONE per il robot
    # 2. Se clicchi su spazio vuoto -> ZOOM DIGITALE sul punto cliccato
    def on_mouse_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        node_r = zoom_node.reader if zoom_node else None
        win_w = node_r.width if node_r and node_r.width else 640
        win_h = node_r.height if node_r and node_r.height else 480

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
        elif zoom_mgr is not None:
            # Click su sfondo vuoto: effettua zoom digitale centrato sul punto
            zoom_mgr.zoom_to_point(fx, fy, factor=2.5)
            set_status(f"Zoom 2.5x su ({fx:.2f}, {fy:.2f}). 0 = torna indietro")

    cv2.setMouseCallback(WIN, on_mouse_click)

    # Thread runners per chiamate a Grok
    def run_grok_detect(raw_frame, yolo_objs):
        try:
            res = detect_objects(raw_frame, known=yolo_objs)
            if not res.ok:
                return set_status(f"Errore Grok: {res.error}", "err")
            st["grok_objects"] = [dict(o.model_dump(), distance_m=estimate_distance_m(o.label, o.bbox))
                                  for o in res.objects]
            for o in res.objects:
                st["label_meta"].setdefault(o.label, {"name": o.name, "color": o.color, "tags": o.tags})
            names = ", ".join(o.name for o in res.objects) or "nessuno"
            if ov and res.objects:
                ov.set_classes(o.label for o in res.objects)
                names += ". YOLO-World ora li segue dal vivo"
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
                    st["label_meta"][by_id[n.id]["label"]] = meta
                if node and n.id < 1000:
                    node.tracker.set_object_metadata(n.id, name=n.name, color=n.color)
                for go in st["grok_objects"]:
                    if go["id"] == n.id:
                        go.update(meta)
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
            select_obj(hit)
            set_status(f"'{text}' -> #{hit['id']} {display_obj(hit)}, azione: {res.action}")
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

    try:
        while True:
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_smooth = 0.90 * fps_smooth + 0.10 * (1.0 / dt)

            pipeline.step("arm", min_display_conf=min_conf, show_conf=show_conf)
            raw = pipeline.state.get_frame("arm")
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
            if sel is not None:
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
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:  # chiusa con la X
                break
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

            # Cancella oggetti Grok [C]
            elif key in (ord("c"), ord("C")):
                st["grok_objects"] = []
                set_status("Oggetti di Grok rimossi dalla scena.")

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

            # Zoom numerico su ID [1-9]
            elif ord("1") <= key <= ord("9"):
                target_num = key - ord("0")
                if zoom_mgr is not None:
                    matched = [o for o in all_objs if o.get("id") == target_num]
                    zoom_mgr.set_target_object(target_num)
                    if matched:
                        zoom_mgr.zoom_to_bbox(matched[0]["bbox"])
                        set_status(f"Zoom su #{target_num} ({display_obj(matched[0])})")
                    else:
                        set_status(f"Zoom su #{target_num} appena compare", "wait")

            # Reset Zoom [0] o [R]
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
