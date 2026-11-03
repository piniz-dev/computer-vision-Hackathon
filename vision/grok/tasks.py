"""I compiti che usano Grok per capire le immagini: nomi, comando, descrizione, verifica, ricerca oggetti.

Nessuno solleva eccezioni verso il chiamante: in caso di errore il risultato ha
ok=False e il motivo in `error`. Ogni id restituito da Grok viene controllato
contro gli id presenti: un id inventato viene scartato.

Le funzioni _mock_* sono le risposte della modalità finta (GROK_MOCK=1):
deterministiche, plausibili, costruite solo dagli oggetti in ingresso.
"""
from __future__ import annotations

import functools
import time

from . import prompts
from .client import ask_json
from .geometry import center, iou, refine_box
from .local_resolver import action_of, candidates, display, resolve_local
from .schemas import (CommandResult, DetectResult, NamesResult, SceneResult, TrackedObject, VerifyResult,
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


def select(marked_img, objects: list, text: str, context: dict | None = None, use_local: bool = True) -> CommandResult:
    """Funzione pubblica: prima la scorciatoia locale, poi Grok.

    Il risultato dice sempre la strada (res.source = "local" | "grok") e la
    latenza totale (res.meta["latency_ms"]). use_local=False serve alla valutazione.
    """
    t0 = time.perf_counter()
    res = resolve_local(objects, text) if use_local else None
    if res is None:
        res = resolve_command(marked_img, objects, text, context)
    res.meta["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return res


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
# 5. Trova oggetti (quando il rilevatore locale non li conosce)
# ---------------------------------------------------------------------------

@_never_raise(DetectResult)
def detect_objects(img, known: list | None = None, start_id: int = 1000, max_objects: int = 15,
                   min_conf: float = 0.5) -> DetectResult:
    """Grok trova gli oggetti nell'immagine con il loro riquadro, anche quelli fuori dalle classi di YOLO.

    known:    oggetti già trovati (es. da YOLO) che Grok non deve ripetere.
    start_id: gli id li assegniamo noi, consecutivi da qui (lontani da quelli del tracker).
    Gli oggetti hanno il formato del gruppo più `tags` e `size_cm` (misura reale stimata).
    I riquadri di Grok sono approssimativi e fermi: valgono per il frame inviato.
    La conf di un LLM non è calibrata, quindi i riquadri sospetti li scartiamo noi;
    il motivo di ogni scarto finisce in meta["discard_reasons"].
    """
    known = parse_objects(known or [])
    res, meta = ask_json([img], prompts.detect_objects(known, max_objects), DetectResult,
                         task="detect_objects", mock=_mock_detect)
    if res is None:
        return DetectResult(ok=False, error=meta["error"], meta=meta)
    check_crop = not meta.get("mock")  # la scena finta è uniforme: GrabCut scarterebbe tutto
    kept, reasons = [], []
    for o in res.objects[:max_objects]:
        x, y, w, h = (min(max(v, 0.0), 1.0) for v in o.bbox)
        w, h = min(w, 1 - x), min(h, 1 - y)  # il riquadro resta dentro l'immagine
        box = [x, y, w, h]
        why = ("confidenza bassa" if o.conf < min_conf
               else "troppo piccolo" if w <= 0.01 or h <= 0.01 or w * h < 0.002
               else "troppo grande" if w * h > 0.5
               else "doppione di YOLO" if any(iou(box, k.bbox) > 0.5 for k in known)
               else "riquadro vuoto" if check_crop and refine_box(img, box) is None
               else None)
        if why:
            reasons.append(f"{o.name or o.label}: {why}")
        else:
            kept.append(o.model_copy(update={"id": start_id + len(kept), "bbox": box}))
    meta["discarded"] = len(res.objects) - len(kept)
    meta["discard_reasons"] = reasons
    return res.model_copy(update={"objects": kept, "meta": meta})


# ---------------------------------------------------------------------------
# Risposte finte (GROK_MOCK=1)
# ---------------------------------------------------------------------------

def _mock_names(objs: list[TrackedObject]) -> dict:
    return {"objects": [{"id": o.id, "name": display(o), "color": o.color} for o in objs]}


def _mock_resolve(objs: list[TrackedObject], text: str) -> dict:
    # Stesso confronto della scorciatoia locale, ma qui più candidati diventano una domanda.
    hits = candidates(objs, text) or []
    out = {"target_id": None, "destination_id": None, "action": action_of(text), "ambiguous": False,
           "question": None, "confidence": 0.0, "reason": ""}
    if len(hits) == 1:
        out.update(target_id=hits[0].id, confidence=0.9, reason=f"Corrisponde a {display(hits[0])}.")
    elif hits:
        out.update(ambiguous=True, confidence=0.5, reason="Più oggetti corrispondono.",
                   question=" o ".join(display(o) for o in hits).capitalize() + "?")
    else:
        out.update(reason="Nessun oggetto corrisponde alla richiesta.")
    return out


def _mock_detect() -> dict:
    return {"objects": [{"label": "object", "name": "oggetto al centro", "color": None, "conf": 0.5,
                         "bbox": [0.4, 0.4, 0.2, 0.2]}]}


def _mock_describe(objs: list[TrackedObject]) -> dict:
    if not objs:
        return {"text": "Non vedo oggetti riconoscibili."}
    names = [f"{display(o)} ({o.id})" for o in objs]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " e " + names[-1]
    return {"text": f"Vedo {len(objs)} oggetti: {listed}."}


def _mock_verify(before, after, action, target_id, destination_id) -> dict:
    b = {o.id: o for o in before}
    a = {o.id: o for o in after}
    if target_id not in b:
        return {"success": False, "reason": "L'oggetto non era presente prima dell'azione.",
                "suggestion": "Seleziona di nuovo l'oggetto.", "confidence": 0.8}
    if action in ("pick", "place"):
        if action == "place" and destination_id in a and target_id in a:
            # posato: il centro dell'oggetto cade dentro la destinazione
            cx, cy = center(a[target_id].bbox)
            x, y, w, h = a[destination_id].bbox
            done = x <= cx <= x + w and y <= cy <= y + h
        else:
            # preso/spostato: sparito dalla scena o centro spostato di oltre il 5%
            done = target_id not in a or max(
                abs(p - q) for p, q in zip(center(b[target_id].bbox), center(a[target_id].bbox))) > 0.05
        if not done:
            return {"success": False, "reason": "L'oggetto è ancora nella stessa posizione.",
                    "suggestion": "Riprova avvicinando il braccio.", "confidence": 0.7}
    return {"success": True, "reason": "L'azione sembra riuscita.", "suggestion": None, "confidence": 0.7}
