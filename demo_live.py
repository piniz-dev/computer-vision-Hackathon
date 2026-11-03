"""Demo interattiva Desktop: Visione A ("Occhi") + Visione B ("Cervello").

Mostra la finestra OpenCV in tempo reale con tracciamento Set-of-Mark e
permette di testare a caldo i compiti di Grok (o resolver locale):
  [D] -> Grok descrive la scena ("describe_scene")
  [S] -> Seleziona un oggetto digitando nel terminale ("select_object")
  [N] -> Battezza gli oggetti con nomi ricchi ("name_objects")
  [T] -> Imposta target cane sull'oggetto selezionato
  [Q] o [ESC] -> Esci

Utilizzo:
  python demo_live.py                 # Webcam 0 + Modalità finta (GROK_MOCK=1 se non c'è .env)
  python demo_live.py --cam 1         # Altra webcam
  python demo_live.py --replay        # Da video sintetico/replay
"""
from __future__ import annotations

import argparse
import os
import textwrap
import threading
import time
import unicodedata

import cv2
from dotenv import load_dotenv

load_dotenv()

# Se non è impostata alcuna chiave, default a GROK_MOCK=1 per non andare in errore
if not os.getenv("XAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY") and not os.getenv("GROK_MOCK"):
    os.environ["GROK_MOCK"] = "1"

from vision.eyes.pipeline import EyePipeline
from vision.grok.tasks import describe_scene, name_objects, select


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


def _draw_hud(
    frame,
    status_text: str,
    latency_text: str = "",
    dog_info: str = "",
    fps_val: float = 0.0,
    min_conf: float = 0.50,
    total_objs: int = 0,
    shown_objs: int = 0,
    show_conf: bool = True,
    zoom_factor: float = 1.0,
    zoom_target_id: int | None = None,
):
    h, w = frame.shape[:2]
    # Barra superiore con comandi e statistiche real-time
    cv2.rectangle(frame, (0, 0), (w, 38), (20, 20, 20), -1)
    hud_keys = "[D] Descrivi | [S] Seleziona | [1-9] Zoom Obj | [0] Reset Zoom | [+/-] Soglia | [C] Conf | [Q] Esci"
    cv2.putText(frame, hud_keys, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)

    zoom_str = f"Zoom: {zoom_factor:.1f}x" + (f" [#{zoom_target_id}]" if zoom_target_id is not None else "")
    stats_str = f"FPS: {fps_val:.1f} | {zoom_str} | Soglia Acc: {int(min_conf*100)}% | Visti: {shown_objs}/{total_objs}"
    cv2.putText(frame, stats_str, (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1, cv2.LINE_AA)

    # Box inferiore con risposta di Grok / Stato
    lines = textwrap.wrap(_ascii(status_text), 70)[:4]
    box_h = 30 + 24 * max(1, len(lines))
    cv2.rectangle(frame, (0, h - box_h), (w, h), (15, 15, 15), -1)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, h - box_h + 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    if latency_text:
        cv2.putText(frame, latency_text, (w - 200, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 100), 1, cv2.LINE_AA)
    if dog_info:
        cv2.putText(frame, dog_info, (w - 240, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 165, 255), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Demo Occhi + Cervello")
    parser.add_argument("--cam", type=int, default=0, help="Indice webcam (default 0)")
    parser.add_argument("--replay", action="store_true", help="Usa simulazione replay al posto della webcam")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pesi YOLO (default yolov8n.pt)")
    parser.add_argument("--min-conf", type=float, default=0.50, help="Soglia minima di accuratezza per mostrare bbox (default 0.50)")
    parser.add_argument("--show-conf", action="store_true", default=True, help="Mostra percentuale confidenza nel badge")
    args = parser.parse_args()

    mode = "replay" if args.replay else "webcam"
    target = "data/replays/sample_arm.mp4" if args.replay else args.cam

    min_conf = max(0.05, min(0.95, args.min_conf))
    show_conf = args.show_conf

    pipeline = EyePipeline(fps=30.0)
    pipeline.add_source(
        source="arm",
        mode=mode,
        target=target,
        use_mock_tracker=False,
        model_name=args.model,
        conf_threshold=0.25,  # tracking interno accetta da 0.25, display filtra con min_conf
        ema_alpha=0.65,       # smoothing anti-jitter
    )

    window_name = "Grok Hackathon - Occhi + Cervello Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 600)

    status_msg = "Pronto. Premi [D] per descrivere, [S] per selezionare, [N] per nomi."
    latency_msg = ""
    dog_info = ""
    is_busy = False

    def run_describe(raw_img, objs):
        nonlocal status_msg, latency_msg, is_busy
        res = describe_scene(raw_img, objs)
        if res.ok:
            lat = res.meta.get("latency_ms", 0) if res.meta else 0
            status_msg = f"Grok: {res.text}"
            latency_msg = f"Lat: {lat:.0f} ms"
        else:
            status_msg = f"Errore: {res.error}"
        is_busy = False

    def run_names(marked_img, objs):
        nonlocal status_msg, latency_msg, is_busy
        res = name_objects(marked_img, objs)
        if res.ok:
            assigned = [f"#{o.id}:{o.name}" for o in res.objects]
            status_msg = f"Nomi: {', '.join(assigned) if assigned else 'Nessuno'}"
            lat = res.meta.get("latency_ms", 0) if res.meta else 0
            latency_msg = f"Lat: {lat:.0f} ms"
            # Aggiorna il tracker con i nomi di Grok
            node = pipeline.nodes.get("arm")
            if node:
                for o in res.objects:
                    node.tracker.set_object_metadata(o.id, name=o.name, color=o.color)
        else:
            status_msg = f"Errore: {res.error}"
        is_busy = False

    def run_select(marked_img, objs, text):
        nonlocal status_msg, latency_msg, is_busy
        res = select(marked_img, objs, text)
        if res.ok:
            if res.ambiguous:
                status_msg = f"Grok (Ambiguo): {res.question}"
            elif res.target_id is not None:
                status_msg = f"Selezionato ID #{res.target_id} ({res.action}). Motivo: {res.reason}"
            else:
                status_msg = f"Non trovato. Motivo: {res.reason}"
            src = getattr(res, "source", "grok")
            lat = res.meta.get("latency_ms", 0) if res.meta else 0
            latency_msg = f"Lat: {lat:.0f} ms ({src})"
        else:
            status_msg = f"Errore: {res.error}"
        is_busy = False

    print("\n" + "=" * 60)
    print("  AVVIO DEMO LIVE: OCCHI (Parte A) + CERVELLO (Parte B)")
    print("=" * 60)
    print(" Comandi disponibili nella finestra:")
    print("   [D] -> Chiedi a Grok di descrivere la scena")
    print("   [S] -> Seleziona oggetto (es. 'prendi la bottiglia')")
    print("   [N] -> Battezza oggetti con name_objects")
    print("   [1-9] -> Zoom sull'oggetto con quell'ID")
    print("   [0] o [R] -> Reset Zoom panoramico")
    print("   [Click Mouse] -> Zoom sul punto cliccato")
    print("   [+] o [=] -> Aumenta soglia accuratezza (+5%)")
    print("   [-] o [_] -> Diminuisci soglia accuratezza (-5%)")
    print("   [C] -> Mostra/nascondi percentuale confidenza")
    print("   [Q] o ESC -> Esci")
    print("=" * 60 + "\n")

    zoom_node = pipeline.nodes.get("arm")
    zoom_mgr = zoom_node.zoom if zoom_node else None

    # Callback mouse per zoom interattivo
    def on_mouse_click(event, x, y, flags, param):
        nonlocal status_msg
        if event == cv2.EVENT_LBUTTONDOWN and zoom_mgr is not None:
            # Ricava dimensioni finestra
            win_h, win_w = 480, 640
            node_r = zoom_node.reader if zoom_node else None
            if node_r and node_r.width and node_r.height:
                win_w, win_h = node_r.width, node_r.height
            nx = float(x) / max(1, win_w)
            ny = float(y) / max(1, win_h)
            fx, fy = zoom_mgr.zoomed_to_full_coords(nx, ny)
            zoom_mgr.zoom_to_point(fx, fy, factor=2.5)
            status_msg = f"Zoom al 2.5x su coordinate ({fx:.2f}, {fy:.2f})"

    cv2.setMouseCallback(window_name, on_mouse_click)

    t_prev = time.perf_counter()
    fps_smooth = 30.0

    try:
        while True:
            # Calcolo FPS reale
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_smooth = 0.90 * fps_smooth + 0.10 * (1.0 / dt)

            payload = pipeline.step("arm", min_display_conf=min_conf, show_conf=show_conf)
            marked = pipeline.state.get_marked_frame("arm")
            raw = pipeline.state.get_frame("arm")
            objs = pipeline.state.get_objects("arm")

            total_objs = len(objs)
            shown_objs = sum(1 for o in objs if o.get("conf", 0.0) >= min_conf)

            cur_zoom = zoom_mgr.current_factor if zoom_mgr else 1.0
            cur_target_id = zoom_mgr.target_id if zoom_mgr else None

            display_frame = marked.copy()
            busy_txt = " [Chiamata in corso...] " if is_busy else ""
            _draw_hud(
                display_frame,
                busy_txt + status_msg,
                latency_text=latency_msg,
                dog_info=dog_info,
                fps_val=fps_smooth,
                min_conf=min_conf,
                total_objs=total_objs,
                shown_objs=shown_objs,
                show_conf=show_conf,
                zoom_factor=cur_zoom,
                zoom_target_id=cur_target_id,
            )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(15) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            elif ord("1") <= key <= ord("9"):
                target_num = key - ord("0")
                if zoom_mgr is not None:
                    # Cerca se l'oggetto esiste
                    matched = [o for o in objs if o.get("id") == target_num]
                    if matched:
                        zoom_mgr.set_target_object(target_num)
                        zoom_mgr.zoom_to_bbox(matched[0]["bbox"])
                        status_msg = f"Zoom agganciato su Oggetto #{target_num} ({matched[0].get('label', '')})"
                    else:
                        zoom_mgr.set_target_object(target_num)
                        status_msg = f"Target Zoom impostato su #{target_num} (in attesa di rilevamento)"

            elif key in (ord("0"), ord("r"), ord("R")):
                if zoom_mgr is not None:
                    zoom_mgr.reset()
                    status_msg = "Zoom resettato a visuale panoramica (1.0x)"

            elif key in (ord("+"), ord("="), ord("]")):
                min_conf = min(0.95, round(min_conf + 0.05, 2))
                status_msg = f"Soglia accuratezza aumentata a {int(min_conf*100)}%"

            elif key in (ord("-"), ord("_"), ord("[")):
                min_conf = max(0.05, round(min_conf - 0.05, 2))
                status_msg = f"Soglia accuratezza abbassata a {int(min_conf*100)}%"

            elif key in (ord("c"), ord("C")):
                show_conf = not show_conf
                status_msg = f"Display percentuale confidenza: {'ATTIVO' if show_conf else 'DISATTIVATO'}"

            elif key in (ord("d"), ord("D")) and not is_busy:
                is_busy = True
                status_msg = "Grok sta elaborando la descrizione..."
                threading.Thread(target=run_describe, args=(raw, objs), daemon=True).start()

            elif key in (ord("n"), ord("N")) and not is_busy:
                is_busy = True
                status_msg = "Grok sta assegnando i nomi agli oggetti..."
                threading.Thread(target=run_names, args=(marked, objs), daemon=True).start()

            elif key in (ord("s"), ord("S")) and not is_busy:
                # Chiedi input da terminale in modo pulito
                print("\n[Comando Vocale/Testo] Digita la richiesta (es. 'prendi la tazza rossa', 'la bottiglia'):")
                text = input("> ").strip()
                if text:
                    is_busy = True
                    status_msg = f"Risoluzione per: '{text}'..."
                    threading.Thread(target=run_select, args=(marked, objs, text), daemon=True).start()

            elif key in (ord("t"), ord("T")):
                if objs:
                    first_id = objs[0]["id"]
                    dog_info = f"Target Cane: ID #{first_id}"
                    status_msg = f"Impostato target inseguimento cane su ID #{first_id}."
                else:
                    status_msg = "Nessun oggetto presente da agganciare."

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
