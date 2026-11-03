import pytest

from vision.grok.local_resolver import resolve_local


@pytest.mark.parametrize("text, expected", [
    ("prendi la tazza rossa", 3),
    ("prendila, la tazza con il manico", 3),
    ("prendi la bottiglia", 7),          # sinonimo italiano -> label inglese
    ("prendi la bottiglia blu", 7),      # colore inglese nel campo color
    ("take the green cup", 8),           # inglese
    ("afferra la tazza verde", 8),       # colore italiano, name assente
    ("prendi il 5", 5),                  # id detto a voce
    ("prendi il numero tre", 3),         # id a parole
    ("prendi quello rosso", None),       # due oggetti rossi -> Grok
    ("prendi la tazza", None),           # due tazze -> Grok
    ("prendi la banana", None),          # assente -> Grok
    ("quello a sinistra", None),         # posizione -> Grok
    ("metti la tazza rossa sulla bottiglia", None),  # due oggetti -> Grok
    ("prendi la tazza 7", None),         # id e tipo non coincidono -> Grok
    ("", None),
])
def test_resolve_local(objs, text, expected):
    r = resolve_local(objs, text)
    assert (r.target_id if r else None) == expected
    if r:
        assert r.source == "local" and r.ok


def test_action_from_verb(objs):
    assert resolve_local(objs, "prendila, la tazza con il manico").action == "pick"
    assert resolve_local(objs, "guarda la bottiglia").action == "look"
    assert resolve_local(objs, "la bottiglia").action == "none"


def test_short_names_tags_and_plurals():
    mug = {"id": 1, "label": "mug", "name": "tazza rossa", "tags": ["boccale", "manico", "ceramica"],
           "conf": 0.9, "bbox": [0.1, 0.1, 0.2, 0.2], "color": "rosso"}
    pen = {"id": 2, "label": "pen", "name": "penna blu", "conf": 0.9, "bbox": [0.5, 0.1, 0.1, 0.3], "color": "blu"}
    assert resolve_local([mug, pen], "prendi la tazza rossa con il manico").target_id == 1
    assert resolve_local([mug, pen], "prendi il boccale").target_id == 1      # dai tag
    assert resolve_local([mug, pen], "dammi le penne").target_id == 2         # plurale
    assert resolve_local([mug, pen], "la tazza col cucchiaino").target_id == 1  # parola sconosciuta ignorata
    twin = dict(mug, id=3, name="tazza rossa grande", tags=[])
    assert resolve_local([mug, twin, pen], "prendi la tazza rossa") is None   # due tazze rosse -> Grok
    assert resolve_local([mug, twin, pen], "prendi la tazza rossa grande").target_id == 3


def test_bad_input_returns_none():
    assert resolve_local([{"bad": 1}], "tazza") is None
