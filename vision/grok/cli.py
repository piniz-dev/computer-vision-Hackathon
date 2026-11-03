"""Riga di comando per provare i compiti su un'immagine.

  python -m vision.grok.cli resolve  --image X.jpg --objects X.json --text "prendi la tazza rossa" [--no-local]
  python -m vision.grok.cli name     --image X.jpg --objects X.json
  python -m vision.grok.cli describe --image X.jpg --objects X.json
  python -m vision.grok.cli verify   --before A.jpg --after B.jpg --objects-before A.json --objects-after B.json \\
                                     --action pick --target 3 [--destination 5]

Le immagini sono frame originali: i numeri li disegna draw_marks. Il file oggetti
può essere il messaggio completo del gruppo ({"objects": [...]}) o solo la lista.
Con GROK_MOCK=1 non si chiama la rete. Esce con 0 se ok, 1 se ok=False.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .marks import draw_marks
from .tasks import describe_scene, name_objects, select, verify_action


def _img(path: str):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"immagine non leggibile: {path}")
    return img


def _objects(path: str) -> list:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["objects"] if isinstance(data, dict) else data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vision.grok.cli", description="Prova i compiti di Visione B.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("resolve", "name", "describe"):
        p = sub.add_parser(name)
        p.add_argument("--image", required=True)
        p.add_argument("--objects", required=True)
        if name == "resolve":
            p.add_argument("--text", required=True)
            p.add_argument("--no-local", action="store_true", help="salta la scorciatoia locale")
    p = sub.add_parser("verify")
    for arg in ("--before", "--after", "--objects-before", "--objects-after", "--action"):
        p.add_argument(arg, required=True)
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--destination", type=int)
    a = ap.parse_args(argv)

    if a.cmd == "verify":
        res = verify_action(_img(a.before), _img(a.after), _objects(a.objects_before), _objects(a.objects_after),
                            a.action, a.target, a.destination)
    else:
        img, objs = _img(a.image), _objects(a.objects)
        if a.cmd == "resolve":
            res = select(draw_marks(img, objs), objs, a.text, use_local=not a.no_local)
        elif a.cmd == "name":
            res = name_objects(draw_marks(img, objs), objs)
        else:
            res = describe_scene(img, objs)
    print(res.model_dump_json(indent=2))
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
