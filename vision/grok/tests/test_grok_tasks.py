"""Compiti: regola di sicurezza sugli id, nessuna eccezione, risposte finte, select."""
import pytest

from vision.grok import tasks
from vision.grok.schemas import CommandResult, DetectedObject, DetectResult, NamesResult, ObjectName
from vision.grok.marks import object_at


def _grok_answers(monkeypatch, result):
    monkeypatch.setattr(tasks, "ask_json", lambda *a, **k: (result, {"error": None}))


@pytest.mark.parametrize("answer", [
    CommandResult(target_id=99, action="pick", confidence=0.9, reason="x"),
    CommandResult(target_id=3, destination_id=42, action="place", confidence=0.9, reason="x"),
])
def test_invented_ids_are_rejected(monkeypatch, img, objs, answer):
    _grok_answers(monkeypatch, answer)
    r = tasks.resolve_command(img, objs, "prendi la tazza")
    assert not r.ok and r.target_id is None and "non presenti" in r.error


def test_names_drop_invented_ids(monkeypatch, img, objs):
    _grok_answers(monkeypatch, NamesResult(objects=[ObjectName(id=3, name="tazza", color="rosso"),
                                                     ObjectName(id=99, name="fantasma", color=None)]))
    r = tasks.name_objects(img, objs)
    assert r.ok and [n.id for n in r.objects] == [3] and r.meta["discarded_ids"] == [99]


def test_ambiguous_without_question_is_rejected(monkeypatch, img, objs):
    _grok_answers(monkeypatch, CommandResult(ambiguous=True, confidence=0.5, reason="x"))
    assert not tasks.resolve_command(img, objs, "quello").ok


def test_grok_failure_becomes_ok_false(monkeypatch, img, objs):
    monkeypatch.setattr(tasks, "ask_json", lambda *a, **k: (None, {"error": "timeout"}))
    r = tasks.resolve_command(img, objs, "prendi la tazza")
    assert not r.ok and r.error == "timeout"


@pytest.mark.parametrize("call", [
    lambda img: tasks.name_objects(img, [{"id": "x"}]),
    lambda img: tasks.resolve_command(img, None, "ciao"),
    lambda img: tasks.describe_scene(img, [{"bbox": 1}]),
    lambda img: tasks.verify_action(img, img, "boh", [], "pick", 1),
])
def test_bad_input_never_raises(img, call):
    r = call(img)
    assert not r.ok and r.error


def test_mock_answers(img, objs):
    assert [n.id for n in tasks.name_objects(img, objs).objects] == [3, 5, 7, 8]
    r = tasks.resolve_command(img, objs, "prendi la tazza rossa")
    assert r.ok and r.target_id == 3 and r.action == "pick"
    r = tasks.resolve_command(img, objs, "prendi quello rosso")
    assert r.ambiguous and r.target_id is None and r.question
    r = tasks.resolve_command(img, objs, "prendi la banana")
    assert r.ok and r.target_id is None and r.reason
    assert tasks.describe_scene(img, objs).text


def test_mock_verify(img, objs):
    moved = [dict(objs[0], bbox=[0.6, 0.6, 0.2, 0.2])] + objs[1:]
    assert tasks.verify_action(img, img, objs, moved, "pick", 3).success
    assert not tasks.verify_action(img, img, objs, objs, "pick", 3).success


def test_no_objects_no_call(monkeypatch, img):
    monkeypatch.setattr(tasks, "ask_json", lambda *a, **k: pytest.fail("non doveva chiamare Grok"))
    assert tasks.resolve_command(img, [], "prendi la tazza").target_id is None


def test_detect_objects_assigns_ids_and_fixes_boxes(monkeypatch, img):
    det = lambda bbox: DetectedObject(label="pen", name="penna blu", color="blu", conf=0.8, bbox=bbox)
    _grok_answers(monkeypatch, DetectResult(objects=[det([0.1, 0.1, 0.2, 0.2]),
                                                      det([0.9, 0.9, 0.5, 0.5]),     # esce dall'immagine: tagliato
                                                      det([1.2, 0.5, 0.1, 0.1])]))  # fuori: scartato
    r = tasks.detect_objects(img, start_id=100)
    assert r.ok and [o.id for o in r.objects] == [100, 101] and r.meta["discarded"] == 1
    assert r.objects[1].bbox == pytest.approx([0.9, 0.9, 0.1, 0.1])
    assert set(r.objects[0].model_dump()) == {"id", "label", "name", "color", "tags", "conf", "bbox"}  # formato del gruppo


def test_detect_objects_mock_and_failure(monkeypatch, img):
    assert tasks.detect_objects(img).objects[0].id == 1000
    monkeypatch.setattr(tasks, "ask_json", lambda *a, **k: (None, {"error": "timeout"}))
    assert not tasks.detect_objects(img).ok


def test_click_picks_smallest_object(objs):
    big = dict(objs[0], id=1, bbox=[0.0, 0.0, 1.0, 1.0])
    assert object_at([big] + objs, 0.15, 0.15)["id"] == 3   # dentro la tazza e dentro il riquadro grande
    assert object_at([big], 0.5, 0.5)["id"] == 1
    assert object_at(objs, 0.99, 0.01) is None


def test_select_reports_path_and_latency(img, objs):
    r = tasks.select(img, objs, "prendi la tazza rossa")
    assert r.source == "local" and r.target_id == 3 and "latency_ms" in r.meta
    r = tasks.select(img, objs, "prendi la tazza rossa", use_local=False)
    assert r.source == "grok" and r.target_id == 3 and "latency_ms" in r.meta
    r = tasks.select(img, objs, "prendi quello rosso")
    assert r.source == "grok" and r.ambiguous
