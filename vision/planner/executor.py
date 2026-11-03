"""Esecutore sequenziale dei piani Markdown per il braccio robotico."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from .actions import ActionType, ArmPose, simulate_action
from .schemas import ActionStep, PlanStatus, RobotPlan


class PlanExecutor:
    """Interpreta ed esegue un `RobotPlan` passo dopo passo, aggiornando i checkbox nel Markdown."""

    def __init__(
        self,
        plan: RobotPlan,
        plan_path: Path | str | None = None,
        hardware_handler: Callable[[ActionStep, ArmPose], bool] | None = None,
        on_step_start: Callable[[ActionStep], None] | None = None,
        on_step_done: Callable[[ActionStep, int, int], None] | None = None,
        on_completed: Callable[[RobotPlan], None] | None = None,
    ) -> None:
        self.plan = plan
        self.plan_path = Path(plan_path) if plan_path else None
        self.hardware_handler = hardware_handler
        self.on_step_start = on_step_start
        self.on_step_done = on_step_done
        self.on_completed = on_completed

        self.arm_pose = ArmPose()
        self.last_error: str | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        # Salva subito il file Markdown se indicato
        self._sync_markdown_file()

    def _sync_markdown_file(self) -> None:
        if self.plan_path is not None:
            try:
                self.plan_path.parent.mkdir(parents=True, exist_ok=True)
                self.plan_path.write_text(self.plan.to_markdown(), encoding="utf-8")
            except OSError:
                pass

    def step(self) -> ActionStep | None:
        """Esegue il singolo passo successivo del piano."""
        next_step = self.plan.current_step()
        if next_step is None:
            self.plan.status = PlanStatus.COMPLETED
            if self.on_completed:
                self.on_completed(self.plan)
            return None

        self.plan.status = PlanStatus.RUNNING
        if self.on_step_start:
            self.on_step_start(next_step)

        params = {
            "x_mm": next_step.x_mm,
            "y_mm": next_step.y_mm,
            "z_mm": next_step.z_mm,
            "grip_force_pct": next_step.grip_force_pct,
            "target_id": next_step.target_id,
        }

        # 1. Se è configurato un driver hardware reale, invoca il comando
        success = True
        if self.hardware_handler is not None:
            try:
                success = bool(self.hardware_handler(next_step, self.arm_pose))
            except Exception as e:
                self.last_error = str(e)
                success = False

        if not success:
            self.plan.status = PlanStatus.FAILED
            return None

        # 2. Aggiorna lo stato simulato della cinematica
        self.arm_pose = simulate_action(next_step.action, params, self.arm_pose)
        next_step.done = True

        # 3. Sincronizza il file Markdown sul disco ([ ] -> [x])
        self._sync_markdown_file()

        total = len(self.plan.steps)
        done_cnt = sum(1 for s in self.plan.steps if s.done)

        if self.on_step_done:
            self.on_step_done(next_step, done_cnt, total)

        if done_cnt == total:
            self.plan.status = PlanStatus.COMPLETED
            if self.on_completed:
                self.on_completed(self.plan)

        return next_step

    def run_all(self, delay_sec: float = 0.35) -> bool:
        """Esegue tutti i passi rimanenti in sequenza sincrona."""
        self._running = True
        try:
            while self._running:
                step = self.step()
                if step is None:
                    break
                if delay_sec > 0:
                    time.sleep(delay_sec)
            return self.plan.status == PlanStatus.COMPLETED
        finally:
            self._running = False

    def start_background(self, delay_sec: float = 0.35) -> None:
        """Avvia l'esecuzione continua in un thread in background."""
        if self._running:
            return
        self._thread = threading.Thread(
            target=self.run_all,
            args=(delay_sec,),
            name="PlanExecutorThread",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Ferma l'esecuzione in background."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.plan.status == PlanStatus.RUNNING:
            self.plan.status = PlanStatus.CANCELLED

    def reset(self) -> None:
        """Riporta tutti i passi allo stato iniziale [ ]."""
        self.stop()
        for step in self.plan.steps:
            step.done = False
        self.plan.status = PlanStatus.READY
        self.arm_pose = ArmPose()
        self._sync_markdown_file()
