"""La scelta dell'utente passa al robot attraverso un file JSON (logs/selection.json).

Il team del braccio legge il file a intervalli e riparte quando cambia `ts`. La scrittura
è atomica (file temporaneo e poi os.replace), così chi legge non trova mai un file a metà.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

TARGET_KEYS = ("id", "label", "name", "color", "tags", "conf", "bbox", "distance_m", "coords_mm")


def send_selection(path: Path, obj: dict, command: str | None = None, action: str | None = None,
                   destination_id: int | None = None, source: str = "arm") -> dict | None:
    """Scrive la selezione e restituisce il messaggio scritto. Restituisce None se il disco rifiuta la scrittura."""
    msg = {"ts": round(time.time(), 3), "source": source, "command": command, "action": action or "none",
           "destination_id": destination_id, "target": {k: obj.get(k) for k in TARGET_KEYS}}
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(msg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)  # ponytail: su Windows fallisce se il lettore tiene il file aperto proprio ora
    except OSError:
        return None
    return msg
