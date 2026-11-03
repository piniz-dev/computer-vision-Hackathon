"""Azioni atomiche e stato del braccio robotico per la Parte C (Planner).

Definisce i comandi elementari che l'esecutore invia al controller del robot
(o simula) per compiere compiti di manipolazione sicura.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """Tipi di azione atomica eseguibili dal braccio."""
    MOVE_ABOVE = "MOVE_ABOVE"      # Spostamento orizzontale in sicurezza sopra il bersaglio (Z alta)
    DESCEND = "DESCEND"            # Discesa verticale controllata verso l'oggetto
    GRIP = "GRIP"                  # Chiusura della pinza/gripper
    LIFT = "LIFT"                  # Risalita verticale a quota sicura prima di traslare
    MOVE_TO = "MOVE_TO"            # Spostamento 3D cartesiano
    RELEASE = "RELEASE"            # Apertura completa della pinza
    HOME = "HOME"                  # Ritorno alla posizione di riposo sicura
    WAIT = "WAIT"                  # Pausa di assestamento / stabilizzazione
    VERIFY_GRASP = "VERIFY_GRASP"  # Controllo visivo per verificare che l'oggetto sia nella pinza


# Quote Z standard di sicurezza (in millimetri)
Z_SAFE_MM: float = 150.0       # Quota di traslazione sicura sopra gli ostacoli
Z_APPROACH_MM: float = 75.0    # Quota di pre-approccio
Z_GRIP_DEFAULT_MM: float = 25.0  # Quota standard di presa sul tavolo
Z_TABLE_LIMIT_MM: float = 10.0   # Quota minima assoluta (limite di sicurezza anti-collisione)


@dataclass
class ArmPose:
    """Stato fisico/cinematico corrente del braccio."""
    x: float = 150.0
    y: float = 150.0
    z: float = Z_SAFE_MM
    gripper_open: bool = True
    grip_force_pct: float = 0.0
    holding_object_id: int | None = None

    def clamp_z(self) -> None:
        """Impedisce la collisione con il piano del tavolo."""
        if self.z < Z_TABLE_LIMIT_MM:
            self.z = Z_TABLE_LIMIT_MM


def simulate_action(action_type: ActionType, params: dict[str, Any], current_pose: ArmPose) -> ArmPose:
    """Aggiorna lo stato simulato del braccio in base all'azione eseguita."""
    pose = ArmPose(
        x=current_pose.x,
        y=current_pose.y,
        z=current_pose.z,
        gripper_open=current_pose.gripper_open,
        grip_force_pct=current_pose.grip_force_pct,
        holding_object_id=current_pose.holding_object_id,
    )

    if action_type == ActionType.MOVE_ABOVE:
        pose.x = float(params.get("x_mm", pose.x))
        pose.y = float(params.get("y_mm", pose.y))
        pose.z = max(float(params.get("z_mm", Z_SAFE_MM)), Z_APPROACH_MM)

    elif action_type == ActionType.DESCEND:
        pose.z = max(float(params.get("z_mm", Z_GRIP_DEFAULT_MM)), Z_TABLE_LIMIT_MM)

    elif action_type == ActionType.GRIP:
        pose.gripper_open = False
        pose.grip_force_pct = float(params.get("grip_force_pct", 70.0))
        target_id = params.get("target_id")
        if target_id is not None:
            pose.holding_object_id = int(target_id)

    elif action_type == ActionType.LIFT:
        pose.z = max(float(params.get("z_mm", Z_SAFE_MM)), Z_APPROACH_MM)

    elif action_type == ActionType.MOVE_TO:
        pose.x = float(params.get("x_mm", pose.x))
        pose.y = float(params.get("y_mm", pose.y))
        pose.z = float(params.get("z_mm", pose.z))
        pose.clamp_z()

    elif action_type == ActionType.RELEASE:
        pose.gripper_open = True
        pose.grip_force_pct = 0.0
        pose.holding_object_id = None

    elif action_type == ActionType.HOME:
        pose.x = 150.0
        pose.y = 150.0
        pose.z = Z_SAFE_MM

    elif action_type in (ActionType.WAIT, ActionType.VERIFY_GRASP):
        pass

    return pose
