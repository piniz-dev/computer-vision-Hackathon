"""Schemi Pydantic e serializzazione/deserializzazione Markdown per il Planner Robotico."""
from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .actions import ActionType, Z_SAFE_MM


class PlanStatus(str, Enum):
    """Stato del ciclo di vita del piano robotico."""
    DRAFT = "DRAFT"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ActionStep(BaseModel):
    """Singolo passo atomico del piano d'azione del braccio."""
    index: int = Field(ge=1, description="Numero progressivo del passo (1-based)")
    action: ActionType = Field(description="Comando atomico del braccio")
    target_id: int | None = Field(default=None, description="ID oggetto coinvolto (se applicabile)")
    x_mm: float | None = Field(default=None, description="Coordinata X cartesiana in mm")
    y_mm: float | None = Field(default=None, description="Coordinata Y cartesiana in mm")
    z_mm: float | None = Field(default=None, description="Quota Z in mm")
    grip_force_pct: float | None = Field(default=None, description="Percentuale forza pinza [0-100]")
    note: str = Field(default="", description="Spiegazione operativa o motivo dell'azione")
    done: bool = Field(default=False, description="True se il passo è stato eseguito con successo")

    def to_markdown_line(self) -> str:
        """Formatta il passo nel formato standard markdown con checkbox [ ] o [x]."""
        chk = "[x]" if self.done else "[ ]"
        parts = [f"{self.index}. {chk} **{self.action.value}**"]

        if self.target_id is not None:
            parts.append(f"Target: #{self.target_id}")

        coords = []
        if self.x_mm is not None:
            coords.append(f"X={self.x_mm:.0f}mm")
        if self.y_mm is not None:
            coords.append(f"Y={self.y_mm:.0f}mm")
        if self.z_mm is not None:
            coords.append(f"Z={self.z_mm:.0f}mm")
        if coords:
            parts.append(f"Coord: {', '.join(coords)}")

        if self.grip_force_pct is not None:
            parts.append(f"Forza: {int(self.grip_force_pct)}%")

        if self.note:
            parts.append(f"Nota: {self.note}")

        return " | ".join(parts)


class RobotPlan(BaseModel):
    """Piano completo di manipolazione robotica generato e verificato da Grok."""
    goal: str = Field(description="Obiettivo formulato dall'utente in linguaggio naturale")
    target_id: int | None = Field(default=None, description="ID dell'oggetto principale da manipolare")
    target_label: str | None = Field(default=None, description="Nome o classe dell'oggetto target")
    destination_id: int | None = Field(default=None, description="ID del contenitore o zona di destinazione")
    destination_label: str | None = Field(default=None, description="Nome o classe della destinazione")
    preconditions_ok: bool = Field(default=True, description="Esito verifica precondizioni")
    preconditions_notes: str = Field(default="OK (Percorso libero, target raggiungibile)", description="Dettaglio precondizioni")
    steps: list[ActionStep] = Field(default_factory=list, description="Sequenza ordinata di azioni")
    safety_notes: list[str] = Field(
        default_factory=lambda: [
            "Non oltrepassare il limite Z=10mm per evitare collisione con il tavolo.",
            "Velocità ridotta al 40% durante la discesa.",
        ],
        description="Regole di sicurezza per l'esecutore",
    )
    status: PlanStatus = Field(default=PlanStatus.READY, description="Stato di avanzamento del piano")
    reviewed: bool = Field(default=False, description="True se il piano è passato dalla fase di revisione Grok")
    review_feedback: str | None = Field(default=None, description="Esito o correzioni applicate dalla revisione")
    created_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_markdown(self) -> str:
        """Serializza il piano completo in formato Markdown leggibile e modificabile."""
        target_str = f"#{self.target_id} ({self.target_label})" if self.target_id is not None else "Nessuno"
        dest_str = f"#{self.destination_id} ({self.destination_label})" if self.destination_id is not None else "Nessuna"

        lines = [
            "# 🤖 Piano di Esecuzione Robotica",
            f"**Obiettivo:** {self.goal}",
            f"**Target ID:** {target_str}",
            f"**Destinazione ID:** {dest_str}",
            f"**Stato Pre-condizioni:** {'OK' if self.preconditions_ok else 'ATTENZIONE'} ({self.preconditions_notes})",
            "",
            "---",
            "",
            "### 📝 Azioni Sequenziali",
        ]

        for step in self.steps:
            lines.append(step.to_markdown_line())

        lines.extend([
            "",
            "---",
            "### 🛡️ Note di Sicurezza",
        ])
        for note in self.safety_notes:
            lines.append(f"- {note}")

        return "\n".join(lines) + "\n"

    @classmethod
    def from_markdown(cls, md_text: str) -> "RobotPlan":
        """Ricostruisce un oggetto RobotPlan partendo dal testo Markdown."""
        goal_match = re.search(r"\*\*Obiettivo:\*\*\s*(.+)", md_text)
        goal = goal_match.group(1).strip() if goal_match else "Azione robotica"

        target_id = None
        target_label = None
        target_m = re.search(r"\*\*Target ID:\*\*\s*#(\d+)(?:\s*\((.*?)\))?", md_text)
        if target_m:
            target_id = int(target_m.group(1))
            target_label = target_m.group(2) if target_m.group(2) else None

        dest_id = None
        dest_label = None
        dest_m = re.search(r"\*\*Destinazione ID:\*\*\s*#(\d+)(?:\s*\((.*?)\))?", md_text)
        if dest_m:
            dest_id = int(dest_m.group(1))
            dest_label = dest_m.group(2) if dest_m.group(2) else None

        precond_ok = True
        precond_notes = "OK"
        prec_m = re.search(r"\*\*Stato Pre-condizioni:\*\*\s*(OK|ATTENZIONE)?(?:\s*\((.*?)\))?", md_text)
        if prec_m:
            if prec_m.group(1) == "ATTENZIONE":
                precond_ok = False
            if prec_m.group(2):
                precond_notes = prec_m.group(2).strip()

        steps: list[ActionStep] = []
        step_pattern = re.compile(
            r"^\s*(\d+)\.\s*\[([ xX])\]\s*\*\*([A-Z_]+)\*\*(.*?)$",
            re.MULTILINE,
        )

        for match in step_pattern.finditer(md_text):
            idx = int(match.group(1))
            is_done = match.group(2).lower() == "x"
            act_str = match.group(3).strip()
            rest = match.group(4).strip()

            try:
                action_type = ActionType(act_str)
            except ValueError:
                action_type = ActionType.WAIT

            # Parsing segmenti delimitati da pipe '|'
            segments = [s.strip() for s in rest.split("|") if s.strip()]
            t_id = None
            x_val = None
            y_val = None
            z_val = None
            force_val = None
            note_val = ""

            for seg in segments:
                # Target: #2
                tm = re.search(r"Target:\s*#(\d+)", seg)
                if tm:
                    t_id = int(tm.group(1))

                # Coord: X=120mm, Y=80mm, Z=150mm
                xm = re.search(r"X=([-\d.]+)mm", seg)
                if xm:
                    x_val = float(xm.group(1))
                ym = re.search(r"Y=([-\d.]+)mm", seg)
                if ym:
                    y_val = float(ym.group(1))
                zm = re.search(r"Z=([-\d.]+)mm", seg)
                if zm:
                    z_val = float(zm.group(1))

                # Forza: 70%
                fm = re.search(r"Forza:\s*(\d+)%", seg)
                if fm:
                    force_val = float(fm.group(1))

                # Nota: ...
                nm = re.search(r"Nota:\s*(.+)", seg)
                if nm:
                    note_val = nm.group(1).strip()
                elif not (tm or xm or ym or zm or fm):
                    note_val = seg

            steps.append(
                ActionStep(
                    index=idx,
                    action=action_type,
                    target_id=t_id,
                    x_mm=x_val,
                    y_mm=y_val,
                    z_mm=z_val,
                    grip_force_pct=force_val,
                    note=note_val,
                    done=is_done,
                )
            )

        # Note di sicurezza
        safety_notes: list[str] = []
        safety_sec = re.search(r"### 🛡️ Note di Sicurezza\s*\n((?:- .*\n?)*)", md_text)
        if safety_sec:
            for sline in safety_sec.group(1).splitlines():
                clean_s = sline.lstrip("- ").strip()
                if clean_s:
                    safety_notes.append(clean_s)

        if not safety_notes:
            safety_notes = [
                "Non oltrepassare il limite Z=10mm per evitare collisione con il tavolo.",
                "Velocità ridotta al 40% durante la discesa.",
            ]

        # Determina stato da checkbox
        status = PlanStatus.READY
        if steps:
            done_count = sum(1 for s in steps if s.done)
            if done_count == len(steps):
                status = PlanStatus.COMPLETED
            elif done_count > 0:
                status = PlanStatus.RUNNING

        return cls(
            goal=goal,
            target_id=target_id,
            target_label=target_label,
            destination_id=dest_id,
            destination_label=dest_label,
            preconditions_ok=precond_ok,
            preconditions_notes=precond_notes,
            steps=steps,
            safety_notes=safety_notes,
            status=status,
            reviewed=True,
        )

    def progress_str(self) -> str:
        """Restituisce una stringa breve di avanzamento (es. 'Passo 3/8 (38%)')."""
        total = len(self.steps)
        if total == 0:
            return "Nessun passo"
        done_count = sum(1 for s in self.steps if s.done)
        pct = int(round(done_count / total * 100))
        return f"Passo {done_count}/{total} ({pct}%)"

    def current_step(self) -> ActionStep | None:
        """Restituisce il prossimo passo da eseguire (primo con done=False)."""
        for step in self.steps:
            if not step.done:
                return step
        return None
