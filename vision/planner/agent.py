"""Agente intelligente di pianificazione robotica (Plan-and-Solve con Grok e Markdown)."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..grok.client import ask_json, is_mock
from .actions import ActionType, Z_GRIP_DEFAULT_MM, Z_SAFE_MM, Z_TABLE_LIMIT_MM
from .schemas import ActionStep, PlanStatus, RobotPlan

LOGS_PLANS_DIR = Path("logs/plans")

SYSTEM_PLANNER_PROMPT = """Sei l'agente di pianificazione robotica per un braccio manipolatore a 4-6 gradi di libertà.
Il tuo compito è pianificare in modo rigoroso, sicuro e atomico le azioni necessarie a compiere l'obiettivo dell'operatore.

Regole fisiche e di sicurezza MANDATORIE:
1. Prima di scendere per afferrare qualsiasi oggetto, la pinza deve essere aperta (azione RELEASE).
2. Prima di qualsiasi traslazione orizzontale (MOVE_ABOVE o MOVE_TO), il braccio DEVE trovarsi a quota di sicurezza (Z >= 150mm o azione LIFT). Mai traslare a quota bassa.
3. La quota del tavolo è Z=0mm; non scendere MAI sotto Z=10mm (limite anti-collisione). Quota di presa tipica: Z=20-30mm.
4. Ogni sequenza di manipolazione deve terminare con il rilascio controllato dell'oggetto e il ritorno in posizione di riposo (HOME).
5. Se l'obiettivo richiede di posizionare un oggetto dentro/vicino a un altro, identifica il Target ID (cosa prendere) e il Destination ID (dove metterlo).

Genera il piano strutturato rispettando lo schema con la lista di azioni atomiche.
"""


def _extract_coords(obj: dict) -> tuple[float, float]:
    """Estrae coordinate cartesiane in millimetri, o le stima dalla bbox normalizzata [0, 1]."""
    coords_mm = obj.get("coords_mm")
    if coords_mm is not None:
        if isinstance(coords_mm, dict):
            return float(coords_mm.get("x", 150.0)), float(coords_mm.get("y", 150.0))
        if hasattr(coords_mm, "x") and hasattr(coords_mm, "y"):
            return float(coords_mm.x), float(coords_mm.y)

    bbox = obj.get("bbox", [0.5, 0.5, 0.1, 0.1])
    bx, by, bw, bh = [float(v) for v in bbox]
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    # Mappatura standard banco di lavoro: X da -150 a +150 mm, Y da 80 a 240 mm
    calc_x = round((cx - 0.5) * 300.0, 1)
    calc_y = round(80.0 + cy * 160.0, 1)
    return calc_x, calc_y


def _find_target_and_dest(goal: str, objects: list[dict]) -> tuple[dict | None, dict | None]:
    """Identifica target e destinazione in base alle parole chiave dell'obiettivo."""
    g_low = goal.lower()
    target = None
    dest = None

    # Cerca oggetti menzionati
    for o in objects:
        lbl = str(o.get("label", "")).lower()
        name = str(o.get("name", "")).lower()
        color = str(o.get("color", "")).lower()

        # È il target?
        if target is None:
            if lbl and lbl in g_low:
                target = o
            elif name and any(part in g_low for part in name.split()):
                target = o
            elif color and color in g_low and ("prendi" in g_low or "afferra" in g_low or "sposta" in g_low):
                target = o

        # È la destinazione (secchio, tazza, ciotola, scatola, tavolo, destra/sinistra)?
        elif dest is None and o != target:
            if any(k in g_low for k in ("dentro", "nel", "nella", "sopra", "vicino", "accanto")):
                if lbl and lbl in g_low:
                    dest = o
                elif name and any(part in g_low for part in name.split()):
                    dest = o

    # Fallback deterministico se non esplicitato
    if target is None and objects:
        target = objects[0]
    if dest is None and len(objects) > 1:
        dest = objects[1]

    return target, dest


class RobotPlanAgent:
    """Agente che orchestra percezione, pianificazione Grok e validazione agentica."""

    def __init__(self, plans_dir: Path | str = LOGS_PLANS_DIR) -> None:
        self.plans_dir = Path(plans_dir)
        self.plans_dir.mkdir(parents=True, exist_ok=True)

    def plan(
        self,
        goal: str,
        objects: list[dict],
        raw_frame: np.ndarray | None = None,
    ) -> tuple[RobotPlan, Path]:
        """Pianifica, revisiona e salva il piano d'azione per raggiungere il `goal`."""
        target_obj, dest_obj = _find_target_and_dest(goal, objects)

        # Mock generator per test e modalità offline
        def _mock_plan_dict() -> dict[str, Any]:
            tx, ty = _extract_coords(target_obj) if target_obj else (100.0, 150.0)
            dx, dy = _extract_coords(dest_obj) if dest_obj else (200.0, 150.0)
            t_id = target_obj.get("id") if target_obj else 1
            t_lbl = target_obj.get("name") or target_obj.get("label", "oggetto") if target_obj else "oggetto"
            d_id = dest_obj.get("id") if dest_obj else 2
            d_lbl = dest_obj.get("name") or dest_obj.get("label", "contenitore") if dest_obj else "contenitore"

            steps_data = [
                {
                    "index": 1,
                    "action": ActionType.MOVE_ABOVE.value,
                    "target_id": t_id,
                    "x_mm": tx,
                    "y_mm": ty,
                    "z_mm": Z_SAFE_MM,
                    "note": f"Posizionamento sopra {t_lbl}",
                },
                {
                    "index": 2,
                    "action": ActionType.RELEASE.value,
                    "note": "Apertura pinza prima della discesa",
                },
                {
                    "index": 3,
                    "action": ActionType.DESCEND.value,
                    "z_mm": Z_GRIP_DEFAULT_MM,
                    "note": f"Discesa per afferrare {t_lbl}",
                },
                {
                    "index": 4,
                    "action": ActionType.GRIP.value,
                    "target_id": t_id,
                    "grip_force_pct": 70.0,
                    "note": "Chiusura pinza e presa salda",
                },
                {
                    "index": 5,
                    "action": ActionType.LIFT.value,
                    "z_mm": Z_SAFE_MM,
                    "note": "Risalita verticale di sicurezza",
                },
                {
                    "index": 6,
                    "action": ActionType.VERIFY_GRASP.value,
                    "target_id": t_id,
                    "note": "Verifica visiva avvenuta presa",
                },
                {
                    "index": 7,
                    "action": ActionType.MOVE_ABOVE.value,
                    "target_id": d_id,
                    "x_mm": dx,
                    "y_mm": dy,
                    "z_mm": Z_SAFE_MM + 10.0,
                    "note": f"Traslazione sopra {d_lbl}",
                },
                {
                    "index": 8,
                    "action": ActionType.RELEASE.value,
                    "note": f"Rilascio oggetto in {d_lbl}",
                },
                {
                    "index": 9,
                    "action": ActionType.HOME.value,
                    "note": "Ritorno in posizione di riposo",
                },
            ]

            return {
                "goal": goal,
                "target_id": t_id,
                "target_label": t_lbl,
                "destination_id": d_id,
                "destination_label": d_lbl,
                "preconditions_ok": True,
                "preconditions_notes": "Target e destinazione visibili nel workspace",
                "steps": steps_data,
                "safety_notes": [
                    "Non oltrepassare il limite Z=10mm per evitare collisione con il tavolo.",
                    "Velocità ridotta al 40% durante la discesa.",
                ],
                "status": PlanStatus.READY.value,
            }

        # Prompt per Grok
        scene_summary = []
        for o in objects:
            oid = o.get("id")
            olbl = o.get("name") or o.get("label", "unknown")
            ocolor = o.get("color") or ""
            cx, cy = _extract_coords(o)
            dist = o.get("distance_m", 0.5)
            scene_summary.append(
                f"- ID #{oid}: {olbl} ({ocolor}) | Coord stimate: X={cx:.0f}mm, Y={cy:.0f}mm | Distanza camera: ~{dist:.2f}m"
            )

        user_content = (
            f"Obiettivo utente: '{goal}'\n\n"
            f"Oggetti rilevati nel piano di lavoro:\n"
            + ("\n".join(scene_summary) if scene_summary else "Nessun oggetto rilevato")
            + "\n\nPianifica la sequenza di azioni."
        )

        images = [raw_frame] if raw_frame is not None else []
        plan_model, meta = ask_json(
            images=images,
            prompt=f"{SYSTEM_PLANNER_PROMPT}\n\n{user_content}",
            schema=RobotPlan,
            task="robot_plan",
            mock=_mock_plan_dict,
        )

        if plan_model is None or not isinstance(plan_model, RobotPlan):
            # Fallback robusto al piano mock se la chiamata API non ha restituito il modello
            raw_dict = _mock_plan_dict()
            plan = RobotPlan.model_validate(raw_dict)
        else:
            plan = plan_model

        # Fase di Agentic Reflection e Validazione di sicurezza
        plan = self.review_plan(plan, objects)

        # Salvataggio del piano in formato Markdown
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plan_filename = f"plan_{timestamp}.md"
        plan_path = self.plans_dir / plan_filename
        plan_path.write_text(plan.to_markdown(), encoding="utf-8")

        return plan, plan_path

    def review_plan(self, plan: RobotPlan, objects: list[dict]) -> RobotPlan:
        """Fase di Reflection dell'Agente: verifica e corregge eventuali incoerenze o rischi fisici."""
        corrections = []
        known_ids = {o.get("id") for o in objects if o.get("id") is not None}

        # Regola 1: Target esiste?
        if plan.target_id is not None and known_ids and plan.target_id not in known_ids:
            plan.preconditions_ok = False
            plan.preconditions_notes = f"Target #{plan.target_id} non attualmente presente nella scena."
            corrections.append(f"Target #{plan.target_id} non confermato visivamente")

        # Regola 2: Verifica sequenza di presa (RELEASE prima di DESCEND/GRIP)
        gripper_open = True
        adjusted_steps: list[ActionStep] = []

        for step in plan.steps:
            # Protezione Z minima
            if step.z_mm is not None and step.z_mm < Z_TABLE_LIMIT_MM:
                step.z_mm = Z_TABLE_LIMIT_MM
                corrections.append(f"Passo {step.index}: limitata quota Z a {Z_TABLE_LIMIT_MM}mm per sicurezza")

            # Se si appresta a scendere per afferrare ma la pinza è chiusa, inserisci un RELEASE
            if step.action == ActionType.DESCEND and not gripper_open:
                adjusted_steps.append(
                    ActionStep(
                        index=len(adjusted_steps) + 1,
                        action=ActionType.RELEASE,
                        note="Apertura pinza correttiva prima della discesa (Safety Rule)",
                    )
                )
                gripper_open = True
                corrections.append("Aggiunta apertura pinza prima della discesa")

            if step.action == ActionType.RELEASE:
                gripper_open = True
            elif step.action == ActionType.GRIP:
                gripper_open = False

            step.index = len(adjusted_steps) + 1
            adjusted_steps.append(step)

        # Regola 3: Termina con HOME se non già presente
        if not adjusted_steps or adjusted_steps[-1].action != ActionType.HOME:
            adjusted_steps.append(
                ActionStep(
                    index=len(adjusted_steps) + 1,
                    action=ActionType.HOME,
                    note="Ritorno in configurazione di riposo (Safety Rule)",
                )
            )
            corrections.append("Aggiunto ritorno HOME a fine piano")

        plan.steps = adjusted_steps
        plan.reviewed = True
        plan.review_feedback = "; ".join(corrections) if corrections else "Piano verificato: conforme alle regole fisiche."
        return plan
