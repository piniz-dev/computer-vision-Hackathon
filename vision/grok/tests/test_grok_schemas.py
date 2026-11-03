import pytest
from pydantic import ValidationError

from vision.grok.schemas import CommandResult, Detections, NamesResult, TrackedObject, VerifyResult

GOOD = ('{"target_id": 3, "destination_id": null, "action": "pick", "ambiguous": false, "question": null, '
        '"confidence": 0.9, "reason": "ok"}')


def test_command_valid():
    r = CommandResult.model_validate_json(GOOD)
    assert r.ok and r.target_id == 3 and r.action == "pick" and r.source == "grok"


@pytest.mark.parametrize("bad", [
    GOOD.replace('"pick"', '"vola"'),        # azione non prevista
    GOOD.replace("0.9", "1.5"),              # confidence fuori 0-1
    GOOD.replace('"target_id": 3', '"target_id": "tre"'),
    "non json",
])
def test_command_invalid(bad):
    with pytest.raises(ValidationError):
        CommandResult.model_validate_json(bad)


def test_schema_sent_to_grok_hides_our_fields_and_requires_all():
    for model in (CommandResult, NamesResult, VerifyResult):
        schema = model.model_json_schema()
        props = set(schema["properties"])
        assert not props & {"ok", "error", "meta", "source"}
        assert set(schema["required"]) == props
        assert all("default" not in p for p in schema["properties"].values())


def test_input_format_of_the_group():
    msg = Detections.model_validate_json(
        '{"source":"arm","frame_id":1842,"ts":1730700000.12,"objects":[{"id":3,"label":"cup","name":null,'
        '"conf":0.87,"bbox":[0.41,0.30,0.12,0.15],"color":null}]}')
    assert msg.objects[0].bbox == (0.41, 0.30, 0.12, 0.15)


@pytest.mark.parametrize("bad", [{"bbox": [0.1, 0.2, 0.3]}, {"conf": 1.2}, {"id": "x"}])
def test_tracked_object_invalid(bad):
    base = {"id": 1, "label": "cup", "conf": 0.5, "bbox": [0.1, 0.1, 0.1, 0.1]}
    with pytest.raises(ValidationError):
        TrackedObject.model_validate(base | bad)
