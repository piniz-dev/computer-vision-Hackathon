"""Valutazione di select() sul dataset in eval/data: con e senza scorciatoia locale.

    python -m vision.grok.eval.run_eval                 (con GROK_MOCK=1: modalità finta)
    python -m vision.grok.eval.run_eval --data DIR --out FILE.md

Scrive eval/results.md con una tabella pronta per README e slide.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from datetime import datetime
from pathlib import Path

import cv2

from ..client import DEFAULT_MODEL, is_mock
from ..marks import draw_marks
from ..schemas import Detections
from ..tasks import select

HERE = Path(__file__).parent


def load(data: Path) -> tuple[list[dict], dict]:
    """Comandi del CSV + per ogni scena (frame numerato, oggetti)."""
    with (data / "commands.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scenes = {}
    for scene in {r["scene"] for r in rows}:
        img = cv2.imread(str(data / f"{scene}.jpg"))
        if img is None:
            raise FileNotFoundError(data / f"{scene}.jpg")
        msg = Detections.model_validate_json((data / f"{scene}.json").read_text(encoding="utf-8"))
        objs = [o.model_dump() for o in msg.objects]
        scenes[scene] = (draw_marks(img, objs), objs)
    return rows, scenes


def run(rows: list[dict], scenes: dict, use_local: bool) -> list[dict]:
    out = []
    for r in rows:
        marked, objs = scenes[r["scene"]]
        out.append({"scene": r["scene"], "text": r["text"],
                    "exp": int(r["expected_target_id"]) if r["expected_target_id"].strip() else None,
                    "exp_amb": r["expected_ambiguous"].strip().lower() == "true",
                    "res": select(marked, objs, r["text"], use_local=use_local)})
    return out


def _ratio(k: int, n: int) -> str:
    return f"{k / n:.0%} ({k}/{n})" if n else "n/d"


def metrics(out: list[dict]) -> dict:
    res = [o["res"] for o in out]
    tp = sum(r.ambiguous and o["exp_amb"] for r, o in zip(res, out))
    pred_amb = sum(r.ambiguous for r in res)
    real_amb = sum(o["exp_amb"] for o in out)
    absent = [o for o in out if o["exp"] is None and not o["exp_amb"]]
    lat = sorted(r.meta.get("latency_ms", 0.0) for r in res)
    p95 = statistics.quantiles(lat, n=20, method="inclusive")[18] if len(lat) > 1 else lat[0]
    return {
        "Accuratezza sull'id": _ratio(sum(o["res"].target_id == o["exp"] for o in out), len(out)),
        "Ambiguità: precisione": _ratio(tp, pred_amb),
        "Ambiguità: richiamo": _ratio(tp, real_amb),
        "Rifiuto oggetti assenti": _ratio(sum(o["res"].target_id is None for o in absent), len(absent)),
        "Latenza p50 (ms)": f"{statistics.median(lat):.0f}",
        "Latenza p95 (ms)": f"{p95:.0f}",
        "Risolti in locale": _ratio(sum(r.source == "local" for r in res), len(res)),
        "Errori (ok=False)": str(sum(not r.ok for r in res)),
    }


def wrong(out: list[dict]) -> list[dict]:
    """Comandi sbagliati: id diverso dall'atteso o ambiguità non riconosciuta."""
    return [o for o in out if o["res"].target_id != o["exp"] or o["res"].ambiguous != o["exp_amb"]]


def report(with_local: list[dict], grok_only: list[dict], n_scenes: int) -> str:
    mode = "finta (GROK_MOCK=1)" if is_mock() else "reale"
    m1, m2 = metrics(with_local), metrics(grok_only)
    lines = [
        "# Risultati della valutazione",
        "",
        f"- Data: {datetime.now():%Y-%m-%d %H:%M} · Modalità: {mode} · Modello: {os.getenv('XAI_MODEL', DEFAULT_MODEL)}",
        f"- Dataset: {n_scenes} scene, {len(with_local)} comandi",
    ]
    if is_mock():
        lines.append("- **Modalità finta: i numeri verificano solo la pipeline, non la qualità di Grok.**")
    lines += ["", "| Metrica | Con scorciatoia locale | Solo Grok |", "|---|---|---|"]
    lines += [f"| {k} | {m1[k]} | {m2[k]} |" for k in m1]
    errors = [("con locale", o) for o in wrong(with_local)] + [("solo Grok", o) for o in wrong(grok_only)]
    lines += ["", "## Comandi sbagliati", ""]
    if errors:
        lines += ["| Modalità | Scena | Comando | Atteso | Ottenuto | Strada | Motivo |", "|---|---|---|---|---|---|---|"]
        for mode_name, o in errors:
            r = o["res"]
            exp = "ambiguo" if o["exp_amb"] else (o["exp"] if o["exp"] is not None else "nessuno")
            got = "ambiguo" if r.ambiguous else (r.target_id if r.target_id is not None else "nessuno")
            why = (r.error or r.reason).replace("|", "/")
            lines.append(f"| {mode_name} | {o['scene']} | {o['text']} | {exp} | {got} | {r.source} | {why} |")
    else:
        lines.append("Nessuno.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--out", type=Path, default=HERE / "results.md")
    args = ap.parse_args()

    rows, scenes = load(args.data)
    text = report(run(rows, scenes, use_local=True), run(rows, scenes, use_local=False), len(scenes))
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Scritto {args.out}")


if __name__ == "__main__":
    main()
