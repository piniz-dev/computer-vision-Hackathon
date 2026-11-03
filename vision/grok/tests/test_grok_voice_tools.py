"""Strumenti Grok Voice: definizioni, dispatch con FakeVisionState, contesto, errori."""
import json

import cv2
import pytest

from vision.grok import voice_tools
from vision.grok.voice_tools import TOOLS, FakeVisionState, dispatch


@pytest.fixture
def state(tmp_path, img, objs):
    msg = lambda o: json.dumps({"source": "arm", "frame_id": 1, "ts": 0.0, "objects": o})
    cv2.imwrite(str(tmp_path / "arm.jpg"), img)
    cv2.imwrite(str(tmp_path / "arm_before.jpg"), img)
    (tmp_path / "arm_before.json").write_text(msg(objs))
    (tmp_path / "arm.json").write_text(msg([dict(objs[0], bbox=[0.6, 0.6, 0.2, 0.2])] + objs[1:]))
    (tmp_path / "arm_action.json").write_text(json.dumps({"action": "pick", "target_id": 3, "destination_id": None}))
    return FakeVisionState(tmp_path)


def test_tool_definitions():
    assert [t["name"] for t in TOOLS] == ["list_objects", "select_object", "describe_scene", "verify_last_action"]
    for t in TOOLS:  # formato Grok Voice: tool "piatti"
        assert t["type"] == "function" and t["parameters"]["type"] == "object" and "source" in t["parameters"]["required"]
    json.dumps(TOOLS)


def test_list_objects(state):
    r = dispatch("list_objects", {"source": "arm"}, state)
    assert r["ok"] and [o["id"] for o in r["objects"]] == [3, 5, 7, 8] and r["say"]


def test_select_object_and_context(state):
    r = dispatch("select_object", '{"source": "arm", "description": "quello rosso"}', state)  # argomenti come stringa
    assert not r["ok"] and r["ambiguous"] and r["say"].endswith("?")
    assert voice_tools._context["arm"]["last_question"] == r["say"]

    r = dispatch("select_object", {"source": "arm", "description": "la tazza rossa"}, state)
    assert r["ok"] and r["id"] == 3 and r["source"] == "local" and "selezionato" in r["say"]
    assert voice_tools._context["arm"] == {"last_selected_id": 3, "last_question": None}


def test_select_absent(state):
    r = dispatch("select_object", {"source": "arm", "description": "la banana"}, state)
    assert not r["ok"] and r["id"] is None and r["say"]


def test_describe_and_verify(state):
    assert dispatch("describe_scene", {"source": "arm"}, state)["ok"]
    r = dispatch("verify_last_action", {"source": "arm"}, state)
    assert r["ok"] and r["success"]


def test_verify_without_action(state):
    r = dispatch("verify_last_action", {"source": "dog"}, state)
    assert not r["ok"] and r["say"]


@pytest.mark.parametrize("name, args", [("boh", {}), ("list_objects", "non json"), ("list_objects", {"source": "dog"})])
def test_errors_never_raise(state, name, args):
    r = dispatch(name, args, state)
    assert not r["ok"] and r["say"] and r["error"]
