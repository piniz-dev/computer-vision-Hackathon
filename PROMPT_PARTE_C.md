# PROMPT OPERATIVO - PARTE C: AGENTE DI PIANIFICAZIONE ROBOTICA (PLAN-AND-SOLVE)

## 🎯 Obiettivo del Modulo
Implementare la **Parte C** dell'architettura di Computer Vision & Robotica per l'Hackathon Grok.
Attualmente il sistema dispone di:
- **Parte A (`vision/eyes/`)**: Percezione in tempo reale (webcam, YOLO, ArUco per millimetri reali, tracking, zoom, depth monoculare).
- **Parte B (`vision/grok/`)**: Cervello multimodale (client Grok, open_vocab, descrizioni, visual task).

La **Parte C** implementa un **Agente Intelligente di Pianificazione Robotica** (*Agentic Plan-and-Solve Architecture*) per il **Braccio Robotico**.
L'AI non esegue comandi alla cieca o d'impulso:
1. Riceve un obiettivo naturale in italiano (es. *"Prendi la chiave inglese e mettila nel secchio"* oppure *"Ordina gli oggetti sul tavolo per colore"*).
2. Analizza lo stato visivo della scena fornito da Visione A/B (oggetti con ID, label, coordinate $(X, Y)$ in mm, colore, distanza).
3. **Genera un Piano Strutturato in formato Markdown** (`plan.md`) con analisi pre-condizioni, controlli di sicurezza, e una sequenza di passi atomici formattati chiaramente.
4. **Agentic Reflection / Rilettura**: rilegge e convalida il Markdown generato verificando che i passaggi rispettino le regole fisiche del braccio (es. aprire il gripper prima di scendere, salire prima di muoversi orizzontalmente per evitare collisioni).
5. **Esecuzione Sequenziale**: esegue il piano passo dopo passo, notificando lo stato di avanzamento (`STEP 1/6: IN CORSO...`) ed esponendo le azioni al controller del braccio (callback Python o JSON WebSocket).
6. **Integrazione in `demo_live.py`**: shortcut da tastiera (tasto `[P]`) per digitare un obiettivo, visualizzare il Markdown generato e vedere l'avanzamento visivo.

---

## 🏗️ Architettura dei File da Creare

Creare il package `vision/planner/`:

```
vision/planner/
├── __init__.py          # Export delle classi principali (RobotPlanAgent, ActionStep, PlanStatus)
├── actions.py          # Primitivi d'azione del braccio (MOVE_XY, DESCEND, ASCEND, GRIP, RELEASE, HOME, WAIT)
├── schemas.py          # Modelli Pydantic per validazione del piano, precondizioni e step
├── agent.py            # Orchestratore Grok: Goal + VisionState -> Markdown -> Verifica -> Execution
└── executor.py         # Interprete/Runner del piano Markdown (simulato o collegato al braccio reale)
```

---

## 📋 Specifiche di Implementazione

### 1. Azioni Atomiche del Braccio (`vision/planner/actions.py`)
Definire un'enumerazione `ActionType` e i parametri corrispondenti:
- `MOVE_ABOVE(x_mm, y_mm, z_safe_mm=150)`: spostamento in sicurezza sopra l'oggetto.
- `DESCEND(z_grip_mm=25)`: discesa verticale controllata verso l'oggetto.
- `GRIP(force_pct=70)`: chiusura pinza/gripper.
- `LIFT(z_safe_mm=150)`: risalita verticale prima di qualsiasi traslazione.
- `MOVE_TO(x_mm, y_mm, z_mm)`: spostamento verso destinazione.
- `RELEASE()`: apertura completa pinza.
- `HOME()`: ritorno alla posizione di riposo sicura.
- `WAIT(seconds)`: pausa di assestamento.
- `VERIFY_GRASP(target_id)`: verifica visiva post-presa (con Visione A/B).

### 2. Struttura del Piano in Markdown (`vision/planner/schemas.py` & Prompt Grok)
Il piano Markdown generato da Grok deve seguire questo schema rigoroso:

```markdown
# 🤖 Piano di Esecuzione Robotica
**Obiettivo:** {Descrizione obiettivo}
**Target ID:** #{target_id} ({label_target})
**Destinazione ID:** #{dest_id} ({label_dest})
**Stato Pre-condizioni:** OK (Percorso libero, target raggiungibile)

---

### 📝 Azioni Sequenziali
1. [ ] **MOVE_ABOVE** | Target: #{target_id} | Coord: X={x}mm, Y={y}mm, Z=150mm | Nota: Posizionamento sopra l'oggetto
2. [ ] **RELEASE** | Apertura pinza prima della discesa
3. [ ] **DESCEND** | Z=25mm | Nota: Discesa per presa
4. [ ] **GRIP** | Forza: 70% | Nota: Presa salda dell'oggetto
5. [ ] **LIFT** | Z=150mm | Nota: Sollevamento di sicurezza
6. [ ] **MOVE_ABOVE** | Dest: #{dest_id} | Coord: X={x_dest}mm, Y={y_dest}mm, Z=160mm | Nota: Spostamento verso il contenitore
7. [ ] **RELEASE** | Nota: Rilascio nel secchio
8. [ ] **HOME** | Nota: Ritorno in configurazione sicura

---
### 🛡️ Note di Sicurezza
- Non oltrepassare il limite Z=10mm per evitare collisione con il tavolo.
- Velocità ridotta al 40% durante la discesa.
```

### 3. Logica dell'Agente (`vision/planner/agent.py`)
- Funzione `generate_markdown_plan(goal: str, objects: list[dict], client: GrokClient) -> str`:
  Invia un prompt di sistema a Grok con la descrizione dei primitivi del braccio e il dump JSON della scena reale vista da Visione A (`objects` con `coords_mm`, `distance_m`, `color`, `label`).
- Funzione `review_plan(markdown_plan: str, objects: list[dict]) -> tuple[bool, str]`:
  Fase agentica di reflection: Grok rilegge il markdown generato e verifica che nessun oggetto sia sollevato senza aver prima chiuso la pinza, e che non ci siano traslazioni a Z bassa. Se rileva incoerenze, corregge il markdown prima di restituirlo.
- Salvataggio automatico: ogni piano viene salvato in `logs/plans/plan_<timestamp>.md`.

### 4. Esecutore del Piano (`vision/planner/executor.py`)
- Interpreta il markdown riga per riga trasformando i checkbox `[ ]` in `[x]` a mano a mano che ogni azione viene completata.
- Supporta callback ad ogni step: `on_step_start(step_idx, action_name, details)`, `on_step_done(step_idx)`.
- Espone un metodo `step()` o `run_all()` non bloccante (tramite thread dedicato).

### 5. Integrazione nella Demo Live (`demo_live.py`)
- Aggiungere il tasto rapido **`[P]`**:
  - Apre un input su schermo per digitare l'obiettivo (es. *"Prendi la matita e mettila nella tazza"*).
  - Mostra a video lo stato: `[Grok sta pianificando il piano d'azione...]`.
  - Appena generato, mostra un riquadro/pannello con il piano Markdown e lo stato di avanzamento dei passi.
  - Permette di avviare l'esecuzione con `[INVIO]` o annullare con `[ESC]`.

### 6. Suite di Test (`tests/test_planner.py`)
- Test di generazione del piano Markdown (con mock Grok deterministico offline).
- Test del parser e validatore delle azioni robot.
- Test dell'esecutore sequenziale con completamento dei checkbox `[x]`.
- Assicurarsi che `pytest` continui a passare al 100% per tutti i test del repository.

---

## 🚀 Istruzioni di Consegna
1. Implementare tutti i file indicati in `vision/planner/`.
2. Aggiungere i test in `tests/test_planner.py`.
3. Integrare il planner in `demo_live.py` con shortcut `[P]`.
4. Verificare che l'intera suite `python -m pytest` passi senza alcun errore.
5. Effettuare commit e push pulito sul branch `main`.
