import json
from pathlib import Path

from vision.grok.cli import main

DATA = Path(__file__).parents[1] / "eval" / "data"


def test_cli_commands(capsys):
    img, objs = str(DATA / "scene_01.jpg"), str(DATA / "scene_01.json")
    assert main(["resolve", "--image", img, "--objects", objs, "--text", "prendi il cerchio verde"]) == 0
    assert json.loads(capsys.readouterr().out)["target_id"] == 4
    assert main(["name", "--image", img, "--objects", objs]) == 0
    assert main(["describe", "--image", img, "--objects", objs]) == 0
    assert main(["verify", "--before", img, "--after", img, "--objects-before", objs, "--objects-after", objs,
                 "--action", "look", "--target", "1"]) == 0
