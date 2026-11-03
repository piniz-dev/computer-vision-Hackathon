"""Scorciatoia locale: risolve i comandi semplici senza chiamare Grok.

Confronta la frase con `name`, `label` e `tags` degli oggetti (con sinonimi italiano ->
inglese, colori italiani e plurali) e con gli id detti a voce ("prendi il 3").
Se trova ESATTAMENTE un oggetto lo restituisce subito con source="local";
con zero o più oggetti restituisce None e si passa a Grok.
"""
from __future__ import annotations

import re
import unicodedata

from .schemas import CommandResult, TrackedObject, parse_objects

# Sinonimi di base italiano -> etichetta inglese (classi COCO più comuni).
# Le parole inglesi funzionano già da sole perché si confrontano con `label`.
SYNONYMS = {
    "persona": "person", "uomo": "person", "donna": "person", "bambino": "person",
    "bicicletta": "bicycle", "bici": "bicycle", "auto": "car", "macchina": "car", "moto": "motorcycle",
    "cane": "dog", "gatto": "cat", "uccello": "bird",
    "zaino": "backpack", "ombrello": "umbrella", "borsa": "handbag", "valigia": "suitcase",
    "palla": "sports ball", "pallone": "sports ball",
    "bottiglia": "bottle", "bicchiere": "wine glass", "calice": "wine glass", "tazza": "cup", "tazzina": "cup",
    "forchetta": "fork", "coltello": "knife", "cucchiaio": "spoon", "ciotola": "bowl", "scodella": "bowl",
    "banana": "banana", "mela": "apple", "arancia": "orange", "carota": "carrot", "panino": "sandwich",
    "sedia": "chair", "divano": "couch", "pianta": "potted plant", "tavolo": "dining table",
    "televisore": "tv", "portatile": "laptop", "computer": "laptop", "telecomando": "remote",
    "tastiera": "keyboard", "telefono": "cell phone", "cellulare": "cell phone", "smartphone": "cell phone",
    "libro": "book", "orologio": "clock", "vaso": "vase", "forbici": "scissors",
    "peluche": "teddy bear", "orsacchiotto": "teddy bear", "spazzolino": "toothbrush",
}

# Colore canonico (italiano) -> forme accettate (generi, plurali, inglese).
COLORS = {
    "rosso": ["rosso", "rossa", "rossi", "rosse", "red"],
    "verde": ["verde", "verdi", "green"],
    "blu": ["blu", "blue"],
    "azzurro": ["azzurro", "azzurra", "azzurri", "azzurre", "celeste", "celesti", "cyan"],
    "giallo": ["giallo", "gialla", "gialli", "gialle", "yellow"],
    "arancione": ["arancione", "arancioni", "arancio", "orange"],
    "viola": ["viola", "purple", "violet"],
    "rosa": ["rosa", "pink"],
    "nero": ["nero", "nera", "neri", "nere", "black"],
    "bianco": ["bianco", "bianca", "bianchi", "bianche", "white"],
    "grigio": ["grigio", "grigia", "grigi", "grigie", "gray", "grey"],
    "marrone": ["marrone", "marroni", "brown"],
}
_COLOR_OF = {form: canon for canon, forms in COLORS.items() for form in forms}

# Numeri detti a parole ("uno"/"una" esclusi: sono quasi sempre articoli).
NUMBERS = {"due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6, "sette": 7, "otto": 8, "nove": 9,
           "dieci": 10, "undici": 11, "dodici": 12, "tredici": 13, "quattordici": 14, "quindici": 15,
           "sedici": 16, "diciassette": 17, "diciotto": 18, "diciannove": 19, "venti": 20}

# Radice del verbo (anche con pronome attaccato: "prendila", "mettilo") -> azione.
VERBS = {"prend": "pick", "afferr": "pick", "raccogl": "pick", "mett": "place", "pos": "place",
         "appoggi": "place", "indic": "point", "vai": "go_to", "avvicinat": "go_to", "segu": "follow",
         "guard": "look", "inquadr": "look"}

_STOP = {"il", "lo", "la", "i", "gli", "le", "l", "un", "uno", "una", "di", "del", "dello", "della", "dei",
         "degli", "delle", "a", "al", "allo", "alla", "ai", "agli", "alle", "da", "in", "nel", "nella", "su",
         "sul", "sulla", "con", "per", "e", "o", "che", "quello", "quella", "questo", "questa", "mi", "me",
         "numero", "oggetto", "favore", "the", "an", "of"}


def _tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", s)


def _stem(t: str) -> str:
    # Toglie la vocale finale: tazza/tazze, penna/penne, bicchiere/bicchieri coincidono.
    # ponytail: radice grezza, confonde coppie come manico/manica; uno stemmer vero se dà fastidio.
    return t[:-1] if len(t) > 3 and t[-1] in "aeiou" else t


_SYN = {_stem(k): v for k, v in SYNONYMS.items()}


def _verb(t: str) -> str | None:
    return next((a for stem, a in VERBS.items() if t.startswith(stem)), None)


def action_of(text: str) -> str:
    return next((a for t in _tokens(text) if (a := _verb(t))), "none")


def display(o: TrackedObject) -> str:
    """Nome da dire a voce: name se c'è, altrimenti label + colore."""
    return o.name or " ".join(x for x in (o.label, o.color) if x)


def _words(o: TrackedObject) -> set[str]:
    return {_stem(t) for t in _tokens(" ".join([o.label, o.name or "", *o.tags]))}


def _colors(o: TrackedObject) -> set[str]:
    return {_COLOR_OF[t] for t in _tokens(f"{o.color or ''} {o.name or ''}") if t in _COLOR_OF}


def candidates(objects: list, text: str) -> list[TrackedObject] | None:
    """Oggetti compatibili con la frase; None se la frase non contiene nulla di riconoscibile."""
    objs = parse_objects(objects)
    toks = _tokens(text)
    known = set().union(*map(_words, objs))
    colors = {_COLOR_OF[t] for t in toks if t in _COLOR_OF}
    numbers = {int(t) if t.isdigit() else NUMBERS[t] for t in toks if t.isdigit() or t in NUMBERS}
    nouns = [_stem(t) for t in toks if t not in _STOP and t not in _COLOR_OF and not _verb(t)
             and (_stem(t) in _SYN or _stem(t) in known)]
    if not (colors or numbers or nouns):
        return None

    def match(o: TrackedObject) -> bool:
        words = _words(o)
        return ((not numbers or numbers == {o.id})        # più numeri diversi -> nessuno
                and colors <= _colors(o)                   # tutti i colori detti
                and all(t in words or _SYN.get(t) == o.label.lower() for t in nouns))

    # ponytail: niente negazioni ("non la rossa") né plurali; quei casi finiscono a Grok o sbagliano.
    return [o for o in objs if match(o)]


def resolve_local(objects: list, text: str) -> CommandResult | None:
    """CommandResult con source="local" se la frase indica esattamente un oggetto, altrimenti None."""
    try:
        hits = candidates(objects, text)
    except Exception:  # input non valido: lo gestirà (e segnalerà) la strada Grok
        return None
    if not hits or len(hits) != 1:
        return None
    o = hits[0]
    return CommandResult(target_id=o.id, action=action_of(text), confidence=0.9,
                         reason=f"Riconosciuto in locale: {display(o)}.", source="local")
