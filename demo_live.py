"""Demo interattiva Desktop unificata: Visione A ("Occhi") + Visione B ("Cervello").

Unisce la precisione e le funzionalità di webcam_demo con le nuove feature di demo_live:
  [G]          -> Grok rileva oggetti (anche quelli sconosciuti a YOLO, riquadri precisi)
  [N]          -> Grok assegna nomi descrittivi a tutti gli oggetti numerati
  [D] / SPAZIO -> Grok descrive l'intera scena
  [S]          -> Seleziona oggetto (richiesta da tastiera es. 'prendi la bottiglia')
  [Click]      -> Click su oggetto: SELEZIONE (da passare al robot)
                  Click su sfondo:  ZOOM DIGITALE fluido sul punto cliccato
  [1-9]        -> Zoom e tracking automatico sull'oggetto con quell'ID
  [0] / [R]    -> Reset zoom alla panoramica (1.0x)
  [C]          -> Cancella gli oggetti trovati da Grok
  [K]          -> Mostra/nascondi percentuale di accuratezza nel badge
  [+] / [-]    -> Alza/abbassa la soglia minima di accuratezza (+/- 5%)
  [T]          -> Imposta target per il cane robotico
  [Q] / ESC    -> Esci

Utilizzo:
  python demo_live.py                 # Webcam 0 con YOLOv8 ONNX nativo + Grok (.env)
  python demo_live.py --cam 1         # Seconda webcam
  python demo_live.py --no-yolo       # Solo Grok (senza YOLO)
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

from vision.eyes.marks import draw_marks
from vision.eyes.pipeline import EyePipeline
from vision.grok.local_resolver import display
from vision.grok.marks import object_at
from vision.grok.schemas import TrackedObject
from vision.grok.tasks import describe_scene, detect_objects, name_objects, select

WIN = "Grok Hackathon - Occhi + Cervello Live"
LOGS_DIR = Path("logs")


def _ascii(s: str) -> str:
    """Rimuove caratteri accentati per compatibilità con cv2.putText."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


def display_obj(o: dict) -> str:
    return display(TrackedObject.model_validate(o))


def _draw_hud(
    frame,
    status_text: str,
    latency_text: str = "",
    dog_info: str = "",
    fps_val: float = 0.0,
    min_conf: float = 0.40,
    total_objs: int = 0,
    shown_objs: int = 0,
    show_conf: bool = True,
    zoom_factor: float = 1.0,
    zoom_target_id: int | None = None,
    selected_id: int | None = None,
):
    h, w = frame.shape[:2]

    # Barra superiore con scorciatoie e statistiche
    cv2.rectangle(frame, (0, 0), (w, 42), (20, 20, 20), -1)
    hud_keys = "[G] Grok Obj | [N] Nomi | [D/Spazio] Descrivi | [S] Seleziona | [Click] Pick/Zoom | [1-9] Zoom | [0] Reset"
    cv2.putText(frame, hud_keys, (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 230, 255), 1, cv2.LINE_AA)

    zoom_str = f"Zoom: {zoom_factor:.1f}x" + (f" [#{zoom_target_id}]" if zoom_target_id is not None else "")
    sel_str = f"Sel: #{selected_id}" if selected_id is not None else "Sel: Nessuno"
    stats_str = f"FPS: {fps_val:.1f} | {zoom_str} | {sel_str} | Soglia Acc: {int(min_conf*100)}% | Visti: {shown_objs}/{total_objs}"
    cv2.putText(frame, stats_str, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 120), 1, cv2.LINE_AA)

    # Box inferiore con risposta di Grok / Stato
    lines = textwrap.wrap(_ascii(status_text), 75)[:4]
    box_h = 32 + 22 * max(1, len(lines))
    cv2.rectangle(frame, (0, h - box_h), (w, h), (15, 15, 15), -1)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, h - box_h + 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    if latency_text:
        cv2.putText(frame, latency_text, (w - 210, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 100), 1, cv2.LINE_AA)
    if dog_info:
        cv2.putText(frame, dog_info, (w - 240, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 165, 255), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam", type=int, default=0, help="Indice webcam (default 0)")
    parser.add_argument("--replay", action="store_true", help="Usa simulazione replay al posto della webcam")
    parser.add_argument("--no-yolo", action="store_true", help="Disabilita YOLO locale (solo rilevamento via Grok)")
    parser.add_argument("--once", action="store_true", help="Scatta una foto, rileva con YOLO + Grok, salva logs/webcam_marked.jpg ed esce")
    parser.add_argument("--model", type=str, default="data/models/yolov8n.onnx", help="Pesi YOLO (.onnx o .pt)")
    parser.add_argument("--min-conf", type=float, default=0.40, help="Soglia minima di accuratezza per visualizzare riquadri (default 0.40)")
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
        grok_objs = [o.model_dump() for o in res.objects] if res.ok else []
        all_objs = yolo_objs + grok_objs

        marked = draw_marks(raw_frame, all_objs)
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
            print(f"  #{o['id']:<4} {o['label']:<15} {display_obj(o)} | bbox={[round(v, 2) for v in o['bbox']]}")
        print(f"\nImmagine salvata in: {out_path.resolve()}\n")
        pipeline.stop()
        return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 980, 620)

    # Stato locale condiviso per chiamate asincrone e selezioni
    st = {
        "grok_objects": [],
        "selected_id": None,
        "is_busy": False,
        "status_msg": "Pronto. Premi [G] per Grok detector, [N] per nomi, [D/Spazio] per descrivere, Click per selezionare.",
        "latency_msg": "",
        "dog_info": "",
    }

    zoom_node = pipeline.nodes.get("arm")
    zoom_mgr = zoom_node.zoom if zoom_node else None

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

        # Recupera tutti gli oggetti correnti (YOLO + Grok)
        yolo_objs = pipeline.state.get_objects("arm")
        current_all = yolo_objs + st["grok_objects"]
        clicked_obj = object_at(current_all, fx, fy)

        if clicked_obj is not None:
            st["selected_id"] = clicked_obj["id"]
            lbl = clicked_obj.get("name") or clicked_obj.get("label", "oggetto")
            st["status_msg"] = f"Selezionato ID #{clicked_obj['id']} ({lbl})"
            print("\n[ROBOT PICK] Oggetto selezionato da passare al robot:")
            print(json.dumps(clicked_obj, ensure_ascii=False, indent=2))
        elif zoom_mgr is not None:
            # Click su sfondo vuoto: effettua zoom digitale centrato sul punto
            zoom_mgr.zoom_to_point(fx, fy, factor=2.5)
            st["status_msg"] = f"Zoom 2.5x su ({fx:.2f}, {fy:.2f})"

    cv2.setMouseCallback(WIN, on_mouse_click)

    # Thread runners per chiamate a Grok
    def run_grok_detect(raw_frame, yolo_objs):
        try:
            res = detect_objects(raw_frame, known=yolo_objs)
            if res.ok:
                st["grok_objects"] = [o.model_dump() for o in res.objects]
                lat = res.meta.get("latency_ms", 0) if res.meta else 0
                st["status_msg"] = f"Grok ha trovato {len(res.objects)} oggetti"
                st["latency_msg"] = f"Lat: {lat:.0f} ms (grok)"
            else:
                st["status_msg"] = f"Errore Grok: {res.error}"
        finally:
            st["is_busy"] = False

    def run_describe(raw_img, objs):
        try:
            res = describe_scene(raw_img, objs)
            if res.ok:
                lat = res.meta.get("latency_ms", 0) if res.meta else 0
                st["status_msg"] = f"Grok: {res.text}"
                st["latency_msg"] = f"Lat: {lat:.0f} ms"
            else:
                st["status_msg"] = f"Errore: {res.error}"
        finally:
            st["is_busy"] = False

    def run_names(marked_img, objs):
        try:
            res = name_objects(marked_img, objs)
            if res.ok:
                assigned = [f"#{o.id}:{o.name}" for o in res.objects]
                st["status_msg"] = f"Nomi: {', '.join(assigned) if assigned else 'Nessuno'}"
                lat = res.meta.get("latency_ms", 0) if res.meta else 0
                st["latency_msg"] = f"Lat: {lat:.0f} ms"
                # Salva i nomi nel tracker di Visione A
                node = pipeline.nodes.get("arm")
                if node:
                    for o in res.objects:
                        node.tracker.set_object_metadata(o.id, name=o.name, color=o.color)
                # Aggiorna anche gli oggetti di Grok
                for n in res.objects:
                    for go in st["grok_objects"]:
                        if go["id"] == n.id:
                            go.update(name=n.name, color=n.color)
            else:
                st["status_msg"] = f"Errore: {res.error}"
        finally:
            st["is_busy"] = False

    def run_select(marked_img, objs, text):
        try:
            res = select(marked_img, objs, text)
            if res.ok:
                if res.ambiguous:
                    st["status_msg"] = f"Grok (Ambiguo): {res.question}"
                elif res.target_id is not None:
                    st["selected_id"] = res.target_id
                    st["status_msg"] = f"Selezionato ID #{res.target_id} ({res.action}). Motivo: {res.reason}"
                else:
                    st["status_msg"] = f"Non trovato. Motivo: {res.reason}"
                src = getattr(res, "source", "grok")
                lat = res.meta.get("latency_ms", 0) if res.meta else 0
                st["latency_msg"] = f"Lat: {lat:.0f} ms ({src})"
            else:
                st["status_msg"] = f"Errore: {res.error}"
        finally:
            st["is_busy"] = False

    print("\n" + "=" * 65)
    print("  AVVIO DEMO LIVE UNIFICATA: OCCHI (Parte A) + CERVELLO (Parte B)")
    print("=" * 65)
    print(" Comandi rapidi:")
    print("   [G]          -> Grok trova oggetti (riquadri stabili)")
    print("   [N]          -> Grok battezza oggetti con nomi ricchi")
    print("   [D] / SPAZIO -> Grok descrive la scena")
    print("   [S]          -> Seleziona oggetto (es. 'la tazza rossa')")
    print("   [Click]      -> Su oggetto: SELEZIONA | Su sfondo: ZOOM 2.5x")
    print("   [1-9]        -> Zoom e segui l'oggetto con quell'ID")
    print("   [0] o [R]    -> Reset Zoom panoramico (1.0x)")
    print("   [C]          -> Cancella oggetti trovati da Grok")
    print("   [+] / [-]    -> Alza / Abbassa soglia accuratezza (+/- 5%)")
    print("   [K]          -> Mostra / Nascondi percentuale accuratezza")
    print("   [Q] / ESC    -> Esci")
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

            payload = pipeline.step("arm", min_display_conf=min_conf, show_conf=show_conf)
            raw = pipeline.state.get_frame("arm")
            yolo_objs = pipeline.state.get_objects("arm")

            # Unisci oggetti YOLO con eventuali oggetti trovati da Grok
            all_objs = yolo_objs + st["grok_objects"]
            total_objs = len(all_objs)
            shown_objs = sum(1 for o in all_objs if o.get("conf", 0.0) >= min_conf)

            # Rendering con evidenziazione oggetto selezionato ed etichette
            frame_display = raw.copy()
            frame_display = draw_marks(
                frame_display,
                all_objs,
                min_conf=min_conf,
                show_conf=show_conf,
            )

            # Evidenziazione visiva dell'oggetto selezionato per il robot
            h_f, w_f = frame_display.shape[:2]
            for o in all_objs:
                if o["id"] == st["selected_id"]:
                    bx, by, bw, bh = o["bbox"]
                    px1 = max(0, int(bx * w_f) - 3)
                    py1 = max(0, int(by * h_f) - 3)
                    px2 = min(w_f, int((bx + bw) * w_f) + 3)
                    py2 = min(h_f, int((by + bh) * h_f) + 3)
                    cv2.rectangle(frame_display, (px1, py1), (px2, py2), (255, 255, 255), 3)

            # Se lo zoom è attivo, applica il ritaglio
            if zoom_mgr is not None and zoom_mgr.is_active():
                zoom_mgr.update_with_tracked_objects(all_objs)
                frame_display = zoom_mgr.apply(frame_display)

            cur_zoom = zoom_mgr.current_factor if zoom_mgr else 1.0
            cur_target_id = zoom_mgr.target_id if zoom_mgr else None

            busy_str = " [In elaborazione...] " if st["is_busy"] else ""
            _draw_hud(
                frame_display,
                busy_str + st["status_msg"],
                latency_text=st["latency_msg"],
                dog_info=st["dog_info"],
                fps_val=fps_smooth,
                min_conf=min_conf,
                total_objs=total_objs,
                shown_objs=shown_objs,
                show_conf=show_conf,
                zoom_factor=cur_zoom,
                zoom_target_id=cur_target_id,
                selected_id=st["selected_id"],
            )

            cv2.imshow(WIN, frame_display)
            key = cv2.waitKey(15) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            # Grok Detect [G]
            elif key in (ord("g"), ord("G")) and not st["is_busy"]:
                st["is_busy"] = True
                st["status_msg"] = "Grok sta analizzando la scena per trovare oggetti..."
                threading.Thread(target=run_grok_detect, args=(raw.copy(), list(yolo_objs)), daemon=True).start()

            # Grok Describe [D] o SPAZIO
            elif (key in (ord("d"), ord("D")) or key == 32) and not st["is_busy"]:
                st["is_busy"] = True
                st["status_msg"] = "Grok sta descrivendo la scena..."
                threading.Thread(target=run_describe, args=(raw.copy(), list(all_objs)), daemon=True).start()

            # Grok Naming [N]
            elif key in (ord("n"), ord("N")) and not st["is_busy"]:
                st["is_busy"] = True
                st["status_msg"] = "Grok sta assegnando nomi descrittivi..."
                threading.Thread(target=run_names, args=(frame_display.copy(), list(all_objs)), daemon=True).start()

            # Selezione Oggetto via Voce/Testo [S]
            elif key in (ord("s"), ord("S")) and not st["is_busy"]:
                print("\n[Comando Vocale/Testo] Cosa deve prendere o indicare il robot? (es. 'la tazza rossa', 'la bottiglia'):")
                query = input("> ").strip()
                if query:
                    st["is_busy"] = True
                    st["status_msg"] = f"Risoluzione per: '{query}'..."
                    threading.Thread(target=run_select, args=(frame_display.copy(), list(all_objs), query), daemon=True).start()

            # Cancella oggetti Grok [C]
            elif key in (ord("c"), ord("C")):
                st["grok_objects"] = []
                st["status_msg"] = "Oggetti rilevati da Grok rimossi dalla scena."

            # Toggle accuratezza nel badge [K]
            elif key in (ord("k"), ord("K")):
                show_conf = not show_conf
                st["status_msg"] = f"Percentuale accuratezza: {'ATTIVA' if show_conf else 'DISATTIVATA'}"

            # Regola soglia accuratezza [+] / [-]
            elif key in (ord("+"), ord("="), ord("]")):
                min_conf = min(0.95, round(min_conf + 0.05, 2))
                st["status_msg"] = f"Soglia accuratezza aumentata a {int(min_conf*100)}%"

            elif key in (ord("-"), ord("_"), ord("[")):
                min_conf = max(0.05, round(min_conf - 0.05, 2))
                st["status_msg"] = f"Soglia accuratezza abbassata a {int(min_conf*100)}%"

            # Zoom numerico su ID [1-9]
            elif ord("1") <= key <= ord("9"):
                target_num = key - ord("0")
                if zoom_mgr is not None:
                    matched = [o for o in all_objs if o.get("id") == target_num]
                    if matched:
                        zoom_mgr.set_target_object(target_num)
                        zoom_mgr.zoom_to_bbox(matched[0]["bbox"])
                        st["status_msg"] = f"Zoom agganciato su #{target_num} ({matched[0].get('label', '')})"
                    else:
                        zoom_mgr.set_target_object(target_num)
                        st["status_msg"] = f"Target Zoom impostato su #{target_num} (in attesa)"

            # Reset Zoom [0] o [R]
            elif key in (ord("0"), ord("r"), ord("R")):
                if zoom_mgr is not None:
                    zoom_mgr.reset()
                    st["status_msg"] = "Zoom resettato a visuale panoramica (1.0x)"

            # Target Cane [T]
            elif key in (ord("t"), ord("T")):
                if all_objs:
                    first_id = all_objs[0]["id"]
                    st["dog_info"] = f"Target Cane: #{first_id}"
                    st["status_msg"] = f"Target cane impostato su ID #{first_id}."
                else:
                    st["status_msg"] = "Nessun oggetto presente da agganciare."

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
