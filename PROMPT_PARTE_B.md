# Persona B: “Cervello”, Grok (xAI) LLM Multimodale

Sei un ingegnere Python esperto di computer vision e LLM multimodali. Devi scrivere un modulo per una squadra che parteciperà a un hackathon. Leggi tutto prima di iniziare.

# Contesto

- Evento: "Grok Hackathon at the Museum" (SpaceXAI × Ctrl+Alt Museum, Pavia), 4–5 novembre 2026. Bisogna collegare un hardware (braccio robotico e cane robot) alle API di Grok (xAI).
- Il progetto: un'app web per il telefono che mostra il video della telecamera del braccio o del cane. Gli oggetti riconosciuti hanno un riquadro con un numero. L'utente può toccare un oggetto, oppure dirlo a voce con Grok Voice, e il robot ci interagisce.
- La squadra (6 persone) è divisa così:
  - **Visione A ("Occhi")**: acquisizione video, YOLO + tracciamento, numeri sugli oggetti, calibrazione, stream MJPEG.
  - **Visione B ("Cervello")**: tutto ciò che usa Grok per capire le immagini. **È il tuo compito.**
  - Robotica: comandi a braccio e cane.
  - App: interfaccia web per il telefono e Grok Voice.
- IMPORTANTE: oggi prepariamo un **toolkit generico e riutilizzabile**, non il progetto finale. Niente idee di demo specifiche, niente oggetti specifici scritti nel codice. Deve funzionare con qualsiasi lista di oggetti e qualsiasi immagine.

# Cosa devi costruire: il pacchetto `vision/grok/`

Python 3.11. Dipendenze minime: `openai`, `pydantic>=2`, `opencv-python`, `numpy`, `python-dotenv`, `pytest`.

## 1. Client Grok: `client.py`
- SDK OpenAI con `base_url="https://api.x.ai/v1"`. Chiave da `XAI_API_KEY` (file `.env`, MAI nel codice). Modello da `XAI_MODEL`.
- **Non dare per scontati i dettagli dell'API.** Prima di scrivere il client consulta la documentazione ufficiale (docs.x.ai) e verifica:
  - il nome del modello con input di immagini (sul sito dell'evento è indicato "grok-4.7")
  - il formato delle immagini in input
  - il supporto a `response_format` con `json_schema`

  Scrivi nel README cosa hai verificato e da dove.
- Funzione `encode_image(img_bgr, max_side=1024, quality=85) -> data_url`: ridimensiona mantenendo le proporzioni, JPEG, base64.
- Funzione `ask_json(images: list, prompt: str, schema: type[BaseModel], timeout=8.0) -> (BaseModel | None, meta)`:
  - prova prima l'output strutturato con lo schema JSON, se non è supportato passa a `json_object`
  - valida con pydantic
  - **1 nuovo tentativo** in caso di JSON non valido o timeout
  - `meta` contiene latenza in ms, token se disponibili, numero di tentativi, errore
- **Modalità finta** `GROK_MOCK=1`: risposte plausibili e deterministiche senza chiamare la rete, per lavorare senza chiave e nei test.
- Ogni chiamata viene scritta in `logs/grok_calls.jsonl`: timestamp, compito, latenza, esito, risposta (senza immagini).

## 2. Formato degli oggetti in ingresso (deciso dal gruppo, non cambiarlo)

```json
{"source":"arm","frame_id":1842,"ts":1730700000.12,
 "objects":[{"id":3,"label":"cup","name":null,"conf":0.87,"bbox":[0.41,0.30,0.12,0.15],"color":null}]}
```

- `bbox` è normalizzato 0–1 (x, y, w, h).
- Riceverai due immagini: il frame originale e il **frame numerato**, con riquadri e id disegnati da A (tecnica Set-of-Mark).
- Per lavorare in autonomia scrivi anche `marks.py` con `draw_marks(frame, objects)`. È una versione semplice: A ha la sua, la tua serve per test e dataset.

## 3. I quattro compiti: `tasks.py` + `prompts.py` + `schemas.py`
I prompt sono in **italiano** e chiedono **solo JSON**. Ogni compito restituisce un modello pydantic e non solleva mai eccezioni verso il chiamante: in caso di errore restituisce `ok=False` con il motivo.

1. `name_objects(marked_img, objects)` → `{"objects":[{"id":3,"name":"tazza rossa con manico","color":"rosso"}]}`. Nomi brevi (2–5 parole), con colore e attributi utili a distinguere oggetti simili.
2. `resolve_command(marked_img, objects, text, context=None)` → `{"target_id":3|null,"destination_id":5|null,"action":"pick|place|point|go_to|follow|look|none","ambiguous":false,"question":null,"confidence":0.0-1.0,"reason":"..."}`
   - Se più oggetti corrispondono → `ambiguous=true` e una **domanda breve** da far pronunciare alla voce ("La tazza rossa o il pennarello rosso?").
   - Se l'oggetto non c'è → `target_id=null` con il motivo. Non inventare mai.
   - `context` contiene l'ultimo oggetto selezionato e l'ultima domanda fatta, così una risposta come "quella a sinistra" si risolve.
3. `describe_scene(img, objects)` → `{"text":"..."}`: massimo 2 frasi, adatte a essere lette a voce.
4. `verify_action(img_before, img_after, objects_before, objects_after, action, target_id, destination_id=None)` → `{"success":bool,"reason":"...","suggestion":"..."|null,"confidence":0.0-1.0}`

**Regola di sicurezza obbligatoria:** ogni `target_id` o `destination_id` restituito da Grok va **controllato contro gli id presenti** nella lista. Se non esiste, lo scarti e restituisci `ok=False`.

## 4. Scorciatoia locale: `local_resolver.py`
Prima di chiamare Grok, prova a risolvere il comando in locale:
- confronta la frase con `name` e `label`, con sinonimi italiano↔inglese di base (tazza/cup, bottiglia/bottle...) e i colori italiani
- se trova **esattamente un** oggetto, lo restituisci subito con `source="local"`
- se ne trova zero o più di uno, restituisci `None` e si passa a Grok

La funzione pubblica `select(marked_img, objects, text, context)` usa prima la scorciatoia locale e poi Grok. Restituisce sempre da quale strada è arrivata la risposta e la latenza.

## 5. Strumenti per Grok Voice: `voice_tools.py`
- Definizioni JSON degli strumenti (function tools), da dare al gruppo app che gestisce la sessione realtime di Grok Voice:
  - `list_objects(source)`
  - `select_object(source, description)`
  - `describe_scene(source)`
  - `verify_last_action(source)`
- Un `dispatch(tool_name, args, state) -> dict` che esegue lo strumento giusto. `state` è un'interfaccia astratta `VisionState` con:
  - `get_frame(source)`
  - `get_marked_frame(source)`
  - `get_objects(source)`
  - `get_last_action(source)`

  Implementala poi A. Tu scrivi anche una `FakeVisionState` che legge da cartella.
- Le risposte degli strumenti sono brevi e pensate per la voce. Esempio: `{"ok":true,"id":3,"say":"Ho selezionato la tazza rossa."}`

## 6. Valutazione: `eval/`
Questi sono i numeri che presenteremo ai giudici.
- Formato del dataset in `eval/data/`:
  - `scene_XX.jpg` + `scene_XX.json` (oggetti)
  - `commands.csv` con le colonne: `scene, text, expected_target_id, expected_ambiguous`
- Crea **un piccolo dataset sintetico di esempio** per provare la pipeline: immagini generate con OpenCV (forme colorate su sfondo uniforme) e 20 comandi, inclusi casi ambigui e oggetti assenti. Al museo lo sostituiremo con foto vere.
- `run_eval.py`:
  - esegue `select` su tutti i comandi, con e senza scorciatoia locale
  - calcola: accuratezza sull'id, rilevamento delle ambiguità (precisione e richiamo), rifiuto degli oggetti assenti, latenza p50/p95, percentuale risolta in locale
  - scrive `eval/results.md` con una tabella pronta per README e slide

## 7. Riga di comando e test
- CLI: `python -m vision.grok.cli resolve --image X.jpg --objects X.json --text "prendi la tazza rossa"`, e lo stesso per `name`, `describe` e `verify`.
- Test `pytest` in modalità finta per:
  - validazione degli schemi
  - scarto degli id inventati
  - scorciatoia locale
  - dispatch degli strumenti
  - gestione di timeout e JSON non valido (simulati)

# Vincoli
- Codice semplice, leggibile e commentato quanto basta. Niente framework pesanti (no LangChain).
- Nessuna chiamata a Grok dentro cicli per ogni frame. Questo modulo lavora **a eventi**.
- Tempo massimo di una chiamata configurabile. Se Grok non risponde, il sistema non si blocca.
- Nessun riferimento a demo o oggetti specifici dell'evento nel codice.
- Non toccare le cartelle di A, della robotica o dell'app.

# Cosa voglio alla fine
1. L'albero dei file creati.
2. `vision/grok/README.md` con: cosa fa ogni funzione, come configurare `.env`, come lanciare CLI, test e valutazione, cosa hai verificato sulla documentazione xAI, limiti noti.
3. L'output di `pytest` e di `run_eval.py` in modalità finta.
4. Un elenco delle cose da verificare con la chiave vera: nome del modello, supporto a `json_schema`, latenza reale.
