# vision/grok — Visione B ("Cervello")

Tutto ciò che usa Grok per capire le immagini: dare un nome agli oggetti, capire a quale oggetto si
riferisce un comando (tocco o voce), descrivere la scena, verificare se un'azione del robot è riuscita.

Toolkit generico: nessun oggetto o demo scritto nel codice, funziona con qualsiasi lista di oggetti.
Lavora **a eventi** (un comando, una chiamata), mai a ogni frame.

## Installazione e `.env`

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r vision/grok/requirements.txt
copy .env.example .env            # Linux/macOS: cp .env.example .env
```

`.env` (nella radice del repo, **mai** su git):

| Variabile | Default | Significato |
|---|---|---|
| `XAI_API_KEY` | — | chiave xAI (obbligatoria se non in modalità finta) |
| `XAI_MODEL` | `grok-4.7` | modello con input di immagini |
| `GROK_MOCK` | `0` | `1` = modalità finta: risposte deterministiche, nessuna chiamata di rete |
| `GROK_TIMEOUT` | `8` | secondi massimi per chiamata (con il nuovo tentativo, al massimo il doppio) |

## Formato degli oggetti (deciso dal gruppo)

```json
{"source":"arm","frame_id":1842,"ts":1730700000.12,
 "objects":[{"id":3,"label":"cup","name":null,"conf":0.87,"bbox":[0.41,0.30,0.12,0.15],"color":null}]}
```

`bbox` = x, y, larghezza, altezza normalizzati 0–1 (x, y = angolo in alto a sinistra).
Le funzioni accettano la lista `objects` (dict o `TrackedObject`).

## Cosa fa ogni file

| File | Funzioni | Cosa fa |
|---|---|---|
| `client.py` | `encode_image(img_bgr, max_side=1024, quality=85)` | ridimensiona mantenendo le proporzioni, JPEG, base64 → data URL |
| | `ask_json(images, prompt, schema, timeout, task, mock)` | chiama Grok e restituisce `(modello pydantic \| None, meta)`. Prova `json_schema`, se rifiutato ripiega su `json_object` (e lo ricorda). 1 nuovo tentativo su JSON non valido o timeout. Mai eccezioni. `meta` = `latency_ms`, `tokens`, `attempts`, `error`, `mode`, `mock`. Ogni chiamata finisce in `logs/grok_calls.jsonl` (senza immagini) |
| `schemas.py` | `TrackedObject`, `Detections`, `parse_objects` | formato in ingresso, validato |
| | `NamesResult`, `CommandResult`, `SceneResult`, `VerifyResult` | risposte dei compiti. Tutte hanno `ok`, `error`, `meta` (aggiunti da noi, Grok non li vede) |
| `marks.py` | `draw_marks(frame, objects)` | riquadro + id su ogni oggetto (Set-of-Mark). Versione semplice per test e dataset; A ha la sua |
| `prompts.py` | un prompt per compito | prompt in italiano che chiedono solo JSON |
| `tasks.py` | `name_objects(marked_img, objects)` | nomi brevi (2–5 parole) con colore e attributi distintivi |
| | `resolve_command(marked_img, objects, text, context=None)` | comando → `target_id`, `destination_id`, `action` (`pick/place/point/go_to/follow/look/none`), `ambiguous` + `question` da dire a voce, `confidence`, `reason`. `context` = `{"last_selected_id", "last_question"}` |
| | `describe_scene(img, objects)` | massimo 2 frasi, da leggere a voce |
| | `verify_action(img_before, img_after, objects_before, objects_after, action, target_id, destination_id=None)` | `success`, `reason`, `suggestion`, `confidence` |
| | **`select(marked_img, objects, text, context=None, use_local=True)`** | **funzione pubblica**: prima la scorciatoia locale, poi Grok. `res.source` = `"local"` o `"grok"`, `res.meta["latency_ms"]` = latenza totale |
| `local_resolver.py` | `resolve_local(objects, text)` | confronta la frase con `name`/`label` (sinonimi italiano→inglese delle classi COCO comuni, colori italiani in tutte le forme, id detti a voce "prendi il 3"). Esattamente un oggetto → risultato con `source="local"`, altrimenti `None` |
| `voice_tools.py` | `TOOLS`, `dispatch(tool_name, args, state)`, `VisionState`, `FakeVisionState(folder)` | strumenti per Grok Voice (vedi sotto) |
| `cli.py` | `main()` | riga di comando |
| `eval/` | `make_dataset.py`, `run_eval.py` | dataset sintetico e valutazione |

**Regole valide per tutti i compiti**
- Non sollevano mai eccezioni: in caso di errore `ok=False` e il motivo in `error`.
- **Sicurezza:** ogni `target_id` / `destination_id` restituito da Grok è controllato contro gli id presenti;
  un id inventato dà `ok=False`. In `name_objects` gli id inventati vengono scartati (`meta["discarded_ids"]`).
- Comando ambiguo senza domanda → `ok=False`. Oggetto assente → `ok=True`, `target_id=null` e motivo in `reason`.
- Lista oggetti vuota in `resolve_command` → nessuna chiamata a Grok.

## Grok Voice (per il gruppo app)

1. Manda `voice_tools.TOOLS` in `session.update` → `session.tools`.
2. A ogni evento `response.function_call_arguments.done`:
   ```python
   result = dispatch(event["name"], event["arguments"], state)   # arguments è una stringa JSON: va bene così
   ```
   e rispondi con `conversation.item.create` → `{"type": "function_call_output", "call_id": event["call_id"], "output": json.dumps(result)}`.
3. Dopo aver risposto a **tutte** le chiamate, manda `response.create`.

Strumenti: `list_objects(source)`, `select_object(source, description)`, `describe_scene(source)`,
`verify_last_action(source)`. `source` = `"arm"` o `"dog"`. Ogni risultato ha `ok` e `say`
(la frase breve da dire), es. `{"ok": true, "id": 3, "say": "Ho selezionato la tazza rossa."}`.

`state` implementa `VisionState` (lo fa Visione A): `get_frame`, `get_marked_frame`, `get_objects`,
`get_last_action` (formato nella docstring). Per lavorare senza robot c'è `FakeVisionState(cartella)`, che legge
`<source>.jpg` + `<source>.json` e, per la verifica, `<source>_before.jpg/.json` + `<source>_action.json`.

## Riga di comando

```bash
python -m vision.grok.cli resolve  --image X.jpg --objects X.json --text "prendi la tazza rossa" [--no-local]
python -m vision.grok.cli name     --image X.jpg --objects X.json
python -m vision.grok.cli describe --image X.jpg --objects X.json
python -m vision.grok.cli verify   --before A.jpg --after B.jpg --objects-before A.json --objects-after B.json --action pick --target 3 [--destination 5]
```

Le immagini sono frame originali (i numeri li disegna `draw_marks`). Esce con 0 se `ok`, 1 altrimenti.
Prova senza chiave: `GROK_MOCK=1` (PowerShell: `$env:GROK_MOCK=1`).

## Test

```bash
python -m pytest vision/grok/tests
```

Girano sempre in modalità finta (forzata da `conftest.py`), il log va in una cartella temporanea.
Coprono: validazione degli schemi, scarto degli id inventati, scorciatoia locale, dispatch degli strumenti,
timeout e JSON non valido (simulati anche con le eccezioni vere dell'SDK), ripiego su `json_object`, CLI.
Usare `python -m pytest` dalla radice del repo (serve per importare `vision.grok`).

## Valutazione

```bash
python -m vision.grok.eval.make_dataset    # rigenera il dataset sintetico (già presente in eval/data)
python -m vision.grok.eval.run_eval        # scrive eval/results.md
```

Dataset in `eval/data/`: `scene_XX.jpg` + `scene_XX.json` (formato del gruppo) e `commands.csv`
(`scene, text, expected_target_id, expected_ambiguous`; `expected_target_id` vuoto = nessun oggetto).
Quello presente è sintetico (4 scene di forme colorate, 20 comandi con casi ambigui e oggetti assenti):
al museo si sostituisce con foto vere nello stesso formato.

`run_eval.py` esegue `select` su tutti i comandi con e senza scorciatoia locale e misura accuratezza sull'id,
precisione/richiamo sulle ambiguità, rifiuto degli oggetti assenti, latenza p50/p95, percentuale risolta
in locale, errori. In modalità finta i numeri verificano solo la pipeline: quelli da presentare vanno
rigenerati con la chiave vera.

## Cosa è stato verificato sulla documentazione xAI (3 novembre 2026)

| Cosa | Risultato | Fonte |
|---|---|---|
| Nome del modello | `grok-4.7` esiste ("the most capable model we've built", contesto 500k). La pagina non dice quali modelli accettano immagini; tutti gli esempi di immagini usano `grok-4.7` | [Models](https://docs.x.ai/docs/models), [Image Understanding](https://docs.x.ai/docs/guides/image-understanding) |
| Formato immagini | JPG/JPEG o PNG, massimo 20 MiB. In Chat Completions: `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "auto\|low\|high"}}` (data URL base64 o URL pubblico) | [Chat Completions](https://docs.x.ai/developers/model-capabilities/legacy/chat-completions) |
| Endpoint | Chat Completions è "legacy" (le novità arrivano prima sulla Responses API) ma è ancora supportato e documentato. Usiamo Chat Completions con l'SDK OpenAI e `base_url=https://api.x.ai/v1` | [Chat Completions](https://docs.x.ai/developers/model-capabilities/legacy/chat-completions) |
| `response_format` | Supporta `json_schema` (schema in `response_format.json_schema`) e `json_object`. Limiti: `additionalProperties` è `false` di default; rifiutati `enum`/`anyOf` vuoti, `items` come array, `minContains`/`maxContains`; `$ref` solo non circolari; `pattern` solo un sottoinsieme delle regex | [Structured Outputs](https://docs.x.ai/docs/guides/structured-outputs) |
| Strumenti Grok Voice | Tool "piatti" `{"type": "function", "name", "description", "parameters"}` in `session.update`; chiamata in `response.function_call_arguments.done` con `arguments` stringa JSON; risposta con `function_call_output` + `response.create` | [Speech to Speech](https://docs.x.ai/developers/model-capabilities/audio/voice-agent) |

## Da verificare con la chiave vera

1. **Modello:** che `grok-4.7` accetti davvero immagini in Chat Completions (altrimenti cambiare `XAI_MODEL`).
2. **`json_schema`:** che venga accettato con `"strict": true` sui nostri schemi. Se no, il client ripiega da
   solo su `json_object`: nel log compare `"mode": "json_object"`.
3. **Latenza reale:** p50/p95 con `run_eval` (e se 8 s di `GROK_TIMEOUT` bastano). Stessa cosa per `verify_action`,
   che manda due immagini.
4. Che `usage` (token) sia presente nella risposta.
5. Rigenerare `eval/results.md` con la chiave vera e con le foto vere del museo.

## Limiti noti

- Timeout per chiamata = `GROK_TIMEOUT`, ma con il nuovo tentativo il caso peggiore è il doppio (16 s di default).
- La scorciatoia locale non gestisce negazioni ("non la rossa"), plurali e verbi inglesi; i sinonimi sono solo
  le classi COCO più comuni (le altre parole funzionano se compaiono in `name` o `label`).
- Il contesto della conversazione vocale (`voice_tools._context`) è uno solo per processo: un utente alla volta.
- Le risposte finte sono deterministiche ma semplici: non ragionano sulle posizioni ("quello più a destra").
- `list_objects` usa `name` se c'è, altrimenti label + colore: non chiama Grok. Per nomi migliori chiamare
  `name_objects` una volta quando cambia la scena.
- Chat Completions è un endpoint legacy: se xAI lo dismette, va cambiato solo `client._call_api`.
