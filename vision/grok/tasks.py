"""I quattro compiti che usano Grok per capire le immagini.

Nessuno solleva eccezioni verso il chiamante: in caso di errore il risultato ha
ok=False e il motivo in `error`. Ogni id restituito da Grok viene controllato
contro gli id presenti: un id inventato viene scartato.

Le funzioni _mock_* sono le risposte della modalità finta (GROK_MOCK=1):
deterministiche, plausibili, costruite solo dagli oggetti in ingresso.
"""
from __future__ import annotations

import functools
import re

from . import prompts
from .client import ask_json
from .schemas import (CommandResult, NamesResult, SceneResult, TrackedObject, VerifyResult,
                      parse_objects)


def _never_raise(result_cls):
    """Qualsiasi eccezione (input non valido, bug) diventa result_cls(ok=False, error=...)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                return result_cls(ok=False, error=f"{type(e).__name__}: {e}")
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# 1. Nomi degli oggetti
# ---------------------------------------------------------------------------

@_never_raise(NamesResult)
def name_objects(marked_img, objects: list) -> NamesResult:
    objs = parse_objects(objects)
    if not objs:
        return NamesResult()
    res, meta = ask_json([marked_img], prompts.name_objects(objs), NamesResult, task="name_objects",
                         mock=lambda: _mock_names(objs))
    if res is None:
        return NamesResult(ok=False, error=meta["error"], meta=meta)
    ids = {o.id for o in objs}
    meta["discarded_ids"] = [n.id for n in res.objects if n.id not in ids]
    return res.model_copy(update={"objects": [n for n in res.objects if n.id in ids], "meta": meta})


# ---------------------------------------------------------------------------
# 2. Comando -> oggetto
# ---------------------------------------------------------------------------

@_never_raise(CommandResult)
def resolve_command(marked_img, objects: list, text: str, context: dict | None = None) -> CommandResult:
    """context: es. {"last_selected_id": 3, "last_question": "Il primo o il secondo?"}."""
    objs = parse_objects(objects)
    if not objs:
        return CommandResult(reason="Non vedo nessun oggetto.")
    res, meta = ask_json([marked_img], prompts.resolve_command(objs, text, context), CommandResult,
                         task="resolve_command", mock=lambda: _mock_resolve(objs, text))
    if res is None:
        return CommandResult(ok=False, error=meta["error"], meta=meta)

    # Regola di sicurezza: solo id presenti nella lista.
    ids = {o.id for o in objs}
    invented = [i for i in (res.target_id, res.destination_id) if i is not None and i not in ids]
    if invented:
        return CommandResult(ok=False, error=f"Grok ha restituito id non presenti: {invented}",
                             reason=res.reason, meta=meta)
    if res.ambiguous and not res.question:
        return CommandResult(ok=False, error="comando ambiguo ma senza domanda", reason=res.reason, meta=meta)
    return res.model_copy(update={"meta": meta})


# ---------------------------------------------------------------------------
# 3. Descrizione della scena
# ---------------------------------------------------------------------------

@_never_raise(SceneResult)
def describe_scene(img, objects: list) -> SceneResult:
    objs = parse_objects(objects)
    res, meta = ask_json([img], prompts.describe_scene(objs), SceneResult, task="describe_scene",
                         mock=lambda: _mock_describe(objs))
    if res is None:
        return SceneResult(ok=False, error=meta["error"], meta=meta)
    return res.model_copy(update={"meta": meta})


# ---------------------------------------------------------------------------
# 4. Verifica dell'azione
# ---------------------------------------------------------------------------

@_never_raise(VerifyResult)
def verify_action(img_before, img_after, objects_before: list, objects_after: list, action: str,
                  target_id: int, destination_id: int | None = None) -> VerifyResult:
    before, after = parse_objects(objects_before), parse_objects(objects_after)
    res, meta = ask_json([img_before, img_after],
                         prompts.verify_action(before, after, action, target_id, destination_id), VerifyResult,
                         task="verify_action",
                         mock=lambda: _mock_verify(before, after, action, target_id, destination_id))
    if res is None:
        return VerifyResult(ok=False, error=meta["error"], meta=meta)
    return res.model_copy(update={"meta": meta})


# ---------------------------------------------------------------------------
# Risposte finte (GROK_MOCK=1)
# ---------------------------------------------------------------------------

_VERBS = {"prendi": "pick", "afferra": "pick", "raccogli": "pick", "metti": "place", "posa": "place",
          "appoggia": "place", "indica": "point", "vai": "go_to", "avvicinati": "go_to", "segui": "follow",
          "guarda": "look", "inquadra": "look"}


def _words(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def _display(o: TrackedObject) -> str:
    return o.name or " ".join(x for x in (o.label, o.color) if x)


def _mock_names(objs: list[TrackedObject]) -> dict:
    return {"objects": [{"id": o.id, "name": _display(o), "color": o.color} for o in objs]}


def _mock_resolve(objs: list[TrackedObject], text: str) -> dict:
    # ponytail: confronto a parole intere, senza sinonimi; la versione seria è local_resolver (punto 4).
    words = _words(text)
    action = next((a for w, a in _VERBS.items() if w in words), "none")
    scores = {o.id: len(words & _words(f"{o.label} {o.name or ''} {o.color or ''}")) for o in objs}
    best = max(scores.values())
    hits = [o for o in objs if best > 0 and scores[o.id] == best]
    out = {"target_id": None, "destination_id": None, "action": action, "ambiguous": False,
           "question": None, "confidence": 0.0, "reason": ""}
    if len(hits) == 1:
        out.update(target_id=hits[0].id, confidence=0.9, reason=f"Corrisponde a {_display(hits[0])}.")
    elif hits:
        out.update(ambiguous=True, confidence=0.5, reason="Più oggetti corrispondono.",
                   question=" o ".join(_display(o) for o in hits).capitalize() + "?")
    else:
        out.update(reason="Nessun oggetto corrisponde alla richiesta.")
    return out


def _mock_describe(objs: list[TrackedObject]) -> dict:
    if not objs:
        return {"text": "Non vedo oggetti riconoscibili."}
    names = [f"{_display(o)} ({o.id})" for o in objs]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " e " + names[-1]
    return {"text": f"Vedo {len(objs)} oggetti: {listed}."}


def _center(o: TrackedObject) -> tuple[float, float]:
    x, y, w, h = o.bbox
    return x + w / 2, y + h / 2


def _mock_verify(before, after, action, target_id, destination_id) -> dict:
    b = {o.id: o for o in before}
    a = {o.id: o for o in after}
    if target_id not in b:
        return {"success": False, "reason": "L'oggetto non era presente prima dell'azione.",
                "suggestion": "Seleziona di nuovo l'oggetto.", "confidence": 0.8}
    if action in ("pick", "place"):
        if action == "place" and destination_id in a and target_id in a:
            # posato: il centro dell'oggetto cade dentro la destinazione
            cx, cy = _center(a[target_id])
            x, y, w, h = a[destination_id].bbox
            done = x <= cx <= x + w and y <= cy <= y + h
        else:
            # preso/spostato: sparito dalla scena o centro spostato di oltre il 5%
            done = target_id not in a or max(
                abs(p - q) for p, q in zip(_center(b[target_id]), _center(a[target_id]))) > 0.05
        if not done:
            return {"success": False, "reason": "L'oggetto è ancora nella stessa posizione.",
                    "suggestion": "Riprova avvicinando il braccio.", "confidence": 0.7}
    return {"success": True, "reason": "L'azione sembra riuscita.", "suggestion": None, "confidence": 0.7}
