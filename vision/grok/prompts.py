"""Prompt dei compiti, in italiano. Chiedono SOLO JSON: lo schema esatto arriva
all'API tramite response_format (o in coda al prompt se si ripiega su json_object)."""
from __future__ import annotations

import json

from .schemas import TrackedObject

_BASE = (
    "Sei il sistema di visione di un robot. Nell'immagine ogni oggetto riconosciuto ha un riquadro "
    "con un numero: quel numero è l'id dell'oggetto. "
    "bbox = [x, y, larghezza, altezza] normalizzati 0-1, origine in alto a sinistra.\n"
    "Rispondi SOLO con JSON, senza testo prima o dopo.\n\n"
)

_ACTIONS = ("pick = prendere, place = posare/mettere, point = indicare, go_to = andare verso, "
            "follow = seguire, look = guardare, none = nessuna azione chiara")


def objects_block(objs: list[TrackedObject]) -> str:
    """Lista oggetti compatta da inserire nel prompt."""
    return json.dumps([{"id": o.id, "label": o.label, "name": o.name, "color": o.color,
                        "bbox": [round(v, 3) for v in o.bbox]} for o in objs], ensure_ascii=False)


def name_objects(objs: list[TrackedObject]) -> str:
    return _BASE + f"""Oggetti: {objects_block(objs)}

Compito: dai a ogni oggetto della lista un nome breve in italiano (2-5 parole) che includa il colore
e gli attributi utili a distinguerlo dagli oggetti simili presenti (forma, materiale, dimensione).
In "color" metti il colore principale con una sola parola italiana (null se non si capisce).
Usa solo gli id della lista, uno per oggetto."""


def resolve_command(objs: list[TrackedObject], text: str, context: dict | None) -> str:
    ctx = json.dumps(context, ensure_ascii=False) if context else "nessuno"
    return _BASE + f"""Oggetti: {objects_block(objs)}
Contesto (ultimo oggetto selezionato e ultima domanda fatta all'utente): {ctx}
Comando dell'utente: {json.dumps(text, ensure_ascii=False)}

Compito: capisci a quale oggetto si riferisce il comando.
- target_id: id dell'oggetto su cui agire. destination_id: id dell'oggetto di destinazione
  (per esempio "mettilo sopra ..."), altrimenti null.
- action: {_ACTIONS}.
- Se più oggetti corrispondono alla descrizione: ambiguous=true, target_id=null e in "question"
  una domanda breve da leggere a voce che nomina le alternative con colore e tipo.
- Se l'oggetto richiesto non c'è: target_id=null e spiega il motivo in "reason".
  Non inventare mai: usa solo gli id della lista.
- Se il comando risponde alla domanda del contesto (per esempio "quello a sinistra"),
  usa il contesto e le posizioni (bbox) per risolverlo.
- confidence: da 0 a 1. reason: una frase breve in italiano."""


def describe_scene(objs: list[TrackedObject]) -> str:
    return _BASE + f"""Oggetti: {objects_block(objs)}

Compito: descrivi la scena in "text" con al massimo 2 frasi brevi in italiano, adatte a essere
lette a voce. Cita gli oggetti principali e, quando aiuta a sceglierli, il loro numero."""


def detect_objects(known: list[TrackedObject], max_objects: int) -> str:
    skip = (f"Questi oggetti sono già stati trovati, NON ripeterli: {objects_block(known)}\n"
            if known else "")
    return (
        "Sei il sistema di visione di un robot. Rispondi SOLO con JSON, senza testo prima o dopo.\n\n"
        + skip
        + f"""Compito: trova nell'immagine gli oggetti fisici distinti e ben visibili (al massimo {max_objects}),
quelli che si potrebbero indicare o afferrare. Ignora superfici e sfondo (pareti, pavimento, soffitto, piano del tavolo).
Per ogni oggetto:
- label: classe breve in inglese, minuscola (es. stile COCO)
- name: nome breve in italiano (2-5 parole) con colore e attributi utili a distinguerlo
- color: colore principale, una parola italiana (null se non si capisce)
- conf: quanto sei sicuro, da 0 a 1
- bbox: [x, y, larghezza, altezza] normalizzati 0-1 rispetto all'immagine intera, origine in alto
  a sinistra, riquadro stretto attorno all'oggetto."""
    )


def verify_action(objs_before: list[TrackedObject], objs_after: list[TrackedObject], action: str,
                  target_id: int, destination_id: int | None) -> str:
    dest = f", destinazione oggetto {destination_id}" if destination_id is not None else ""
    return _BASE + f"""Ti mostro due immagini: la prima PRIMA dell'azione del robot, la seconda DOPO.
Oggetti prima: {objects_block(objs_before)}
Oggetti dopo: {objects_block(objs_after)}
Azione richiesta: {action} sull'oggetto {target_id}{dest}. (Azioni: {_ACTIONS}.)

Compito: confronta le due immagini e decidi se l'azione è riuscita.
- success: true/false. reason: una frase breve in italiano.
- suggestion: se è fallita, cosa fare (una frase breve), altrimenti null.
- confidence: da 0 a 1."""
