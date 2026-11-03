"""Test suite per la Parte C: Agente di Pianificazione Robotica e Piano Markdown."""
from __future__ import annotations

from pathlib import Path

import pytest

from vision.planner import (
    ActionStep,
    ActionType,
    ArmPose,
    PlanExecutor,
    PlanStatus,
    RobotPlan,
    RobotPlanAgent,
    Z_SAFE_MM,
    Z_TABLE_LIMIT_MM,
)


@pytest.fixture(autouse=True)
def force_mock_grok(monkeypatch):
    """Garantisce test offline ultra-rapidi e deterministici per il planner."""
    monkeypatch.setenv("GROK_MOCK", "1")


def test_action_step_and_markdown_serialization():
    step = ActionStep(
        index=1,
        action=ActionType.MOVE_ABOVE,
        target_id=2,
        x_mm=120.0,
        y_mm=80.0,
        z_mm=150.0,
        note="Posizionamento sopra l'oggetto",
        done=False,
    )
    line = step.to_markdown_line()
    assert "1. [ ] **MOVE_ABOVE**" in line
    assert "Target: #2" in line
    assert "Coord: X=120mm, Y=80mm, Z=150mm" in line
    assert "Nota: Posizionamento sopra l'oggetto" in line

    # Test checkbox completato
    step.done = True
    assert "1. [x] **MOVE_ABOVE**" in step.to_markdown_line()


def test_robot_plan_markdown_roundtrip():
    plan = RobotPlan(
        goal="Prendi la penna blu e mettila nel secchio",
        target_id=3,
        target_label="penna blu",
        destination_id=7,
        destination_label="secchio",
        preconditions_ok=True,
        preconditions_notes="Area sgombra",
        steps=[
            ActionStep(index=1, action=ActionType.MOVE_ABOVE, target_id=3, x_mm=100.0, y_mm=140.0, z_mm=150.0, note="Approccio"),
            ActionStep(index=2, action=ActionType.RELEASE, note="Apri pinza"),
            ActionStep(index=3, action=ActionType.DESCEND, z_mm=25.0, note="Scendi"),
            ActionStep(index=4, action=ActionType.GRIP, target_id=3, grip_force_pct=70.0, note="Afferra"),
            ActionStep(index=5, action=ActionType.LIFT, z_mm=150.0, note="Risali"),
            ActionStep(index=6, action=ActionType.MOVE_ABOVE, target_id=7, x_mm=220.0, y_mm=140.0, z_mm=160.0, note="Vai al secchio"),
            ActionStep(index=7, action=ActionType.RELEASE, note="Rilascia"),
            ActionStep(index=8, action=ActionType.HOME, note="Riposo"),
        ],
    )

    md = plan.to_markdown()
    assert "# 🤖 Piano di Esecuzione Robotica" in md
    assert "**Obiettivo:** Prendi la penna blu e mettila nel secchio" in md
    assert "**Target ID:** #3 (penna blu)" in md
    assert "**Destinazione ID:** #7 (secchio)" in md
    assert "8. [ ] **HOME**" in md

    # Deserializza da Markdown e verifica corrispondenza
    parsed = RobotPlan.from_markdown(md)
    assert parsed.goal == plan.goal
    assert parsed.target_id == 3
    assert parsed.destination_id == 7
    assert len(parsed.steps) == 8
    assert parsed.steps[0].action == ActionType.MOVE_ABOVE
    assert parsed.steps[0].x_mm == 100.0
    assert parsed.steps[3].action == ActionType.GRIP
    assert parsed.steps[3].grip_force_pct == 70.0
    assert parsed.steps[7].action == ActionType.HOME


def test_agent_plan_generation_and_reflection(tmp_path: Path):
    agent = RobotPlanAgent(plans_dir=tmp_path)

    sample_objects = [
        {"id": 2, "label": "pen", "name": "penna blu", "color": "blu", "coords_mm": {"x": 110.0, "y": 140.0}, "distance_m": 0.45},
        {"id": 5, "label": "cup", "name": "tazza rossa", "color": "rosso", "coords_mm": {"x": 210.0, "y": 160.0}, "distance_m": 0.50},
    ]

    plan, plan_file = agent.plan(
        goal="Prendi la penna blu e mettila nella tazza",
        objects=sample_objects,
    )

    assert plan_file.exists()
    assert plan.target_id == 2
    assert plan.destination_id == 5
    assert len(plan.steps) >= 6
    assert plan.reviewed is True

    # Verifica che il file sul disco sia un markdown valido
    saved_md = plan_file.read_text(encoding="utf-8")
    assert "Piano di Esecuzione Robotica" in saved_md
    assert "MOVE_ABOVE" in saved_md
    assert "HOME" in saved_md


def test_agent_safety_reflection_corrections(tmp_path: Path):
    agent = RobotPlanAgent(plans_dir=tmp_path)

    # Piano non sicuro: quota Z troppo bassa (< 10mm), chiusura pinza prima della discesa, assenza di HOME
    unverified_plan = RobotPlan(
        goal="Test collisione",
        steps=[
            ActionStep(index=1, action=ActionType.GRIP, grip_force_pct=80.0),  # Pinza chiusa
            ActionStep(index=2, action=ActionType.DESCEND, z_mm=5.0),          # Quota vietata (5mm < 10mm)
            ActionStep(index=3, action=ActionType.GRIP, target_id=1),          # Presa con pinza già chiusa
        ],
    )

    reviewed = agent.review_plan(unverified_plan, objects=[{"id": 1, "label": "box"}])

    # 1. Quota Z limitata a Z_TABLE_LIMIT_MM (10mm)
    descend_steps = [s for s in reviewed.steps if s.action == ActionType.DESCEND]
    assert len(descend_steps) > 0
    assert descend_steps[0].z_mm >= Z_TABLE_LIMIT_MM

    # 2. Inserito RELEASE prima di scendere ad afferrare
    actions = [s.action for s in reviewed.steps]
    descend_idx = actions.index(ActionType.DESCEND)
    assert ActionType.RELEASE in actions[:descend_idx]

    # 3. Termina con HOME
    assert actions[-1] == ActionType.HOME


def test_plan_executor_step_by_step_and_sync(tmp_path: Path):
    plan_file = tmp_path / "test_exec_plan.md"

    plan = RobotPlan(
        goal="Sposta oggetto",
        target_id=1,
        steps=[
            ActionStep(index=1, action=ActionType.MOVE_ABOVE, target_id=1, x_mm=100.0, y_mm=120.0, z_mm=150.0),
            ActionStep(index=2, action=ActionType.RELEASE),
            ActionStep(index=3, action=ActionType.DESCEND, z_mm=25.0),
            ActionStep(index=4, action=ActionType.HOME),
        ],
    )

    steps_done_log = []
    def on_done(step, done_cnt, total):
        steps_done_log.append((step.action, done_cnt, total))

    executor = PlanExecutor(
        plan=plan,
        plan_path=plan_file,
        on_step_done=on_done,
    )

    assert plan_file.exists()
    assert "1. [ ] **MOVE_ABOVE**" in plan_file.read_text(encoding="utf-8")

    # Esegui passo 1
    s1 = executor.step()
    assert s1 is not None and s1.action == ActionType.MOVE_ABOVE
    assert s1.done is True
    assert executor.arm_pose.x == 100.0
    assert executor.arm_pose.z == 150.0

    # Verifica sincronizzazione su file markdown ([ ] -> [x])
    md_content = plan_file.read_text(encoding="utf-8")
    assert "1. [x] **MOVE_ABOVE**" in md_content
    assert "2. [ ] **RELEASE**" in md_content

    # Esegui tutti i passi rimanenti
    success = executor.run_all(delay_sec=0.01)
    assert success is True
    assert executor.plan.status == PlanStatus.COMPLETED
    assert len(steps_done_log) == 4

    # File markdown deve avere tutti i checkbox [x]
    final_md = plan_file.read_text(encoding="utf-8")
    assert "[ ]" not in final_md
    assert "4. [x] **HOME**" in final_md

    # Test reset
    executor.reset()
    assert executor.plan.status == PlanStatus.READY
    reset_md = plan_file.read_text(encoding="utf-8")
    assert "1. [ ] **MOVE_ABOVE**" in reset_md
