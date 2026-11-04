# Visione → App: come collegarsi

Tutto quello che vede la visione esce da un server HTTP sul PC con la camera.
CORS è aperto, quindi l'app può chiamarlo da qualsiasi origine.

## 1. Avvio (lo facciamo noi, sul PC)

```bat
.venv\Scripts\python.exe -m vision.eyes.cli run --mode webcam --device 2 --port 8000
```

- `--device 2` = RealSense, `--device 0` = webcam del portatile. `--mode synthetic` = scena finta, senza camera.
- Indirizzo base: `http://<IP-del-PC>:8000` (l'IP lo trovate con `ipconfig`, stessa Wi-Fi). Se dal telefono
  non risponde, è il firewall di Windows: va permesso Python sulla rete privata.
- Prova veloce: `GET /health` → `{"ok": true, "sources": ["arm", "dog"]}`.
- Documentazione interattiva generata dal server: `http://<IP-del-PC>:8000/docs`.
- Il server e `run_demo.bat` non possono usare la stessa camera contemporaneamente.

## 2. Endpoint

`source` è sempre `"arm"` (camera del braccio) o `"dog"` (cane robot).

| Metodo | Percorso | Cosa fa |
|---|---|---|
| `GET` | `/health` | Server vivo e sorgenti attive |
| `GET` | `/stream/{source}` | Video MJPEG con riquadri e numeri. Basta `<img src>` |
| `GET` | `/stream/{source}/raw` | Video MJPEG pulito |
| `GET` | `/frame/{source}.jpg?raw=false` | Un solo fotogramma JPEG |
| `WS` | `/ws/objects/{source}?hz=15` | `FramePayload` continuo (default 15 al secondo) |
| `GET` | `/state/{source}` | Un solo `FramePayload` |
| `POST` | `/dog/target` | `{target_id?, target_label?}` → il cane insegue quell'oggetto |
| `POST` | `/action/{source}/start` | `{action, target_id, destination_id?}`: segna lo stato "prima" dell'azione del robot |
| `POST` | `/action/{source}/verify_local` | Confronta prima/dopo: l'oggetto è sparito / arrivato? |
| `GET` | `/voice/tools` | Strumenti da passare a Grok Voice (`session.tools`) |
| `POST` | `/voice/dispatch` | `{name, arguments}` → esegue lo strumento chiesto da Grok Voice |

## 3. Tipi TypeScript

```ts
export type Source = "arm" | "dog";

/** [x, y, w, h] normalizzati 0..1, (x, y) = angolo in alto a sinistra. Moltiplicate per la larghezza/altezza del video. */
export type BBox = [number, number, number, number];

export interface VisionObject {
  id: number;               // numero mostrato sul video; può cambiare se il tracker perde l'oggetto
  label: string;            // classe, es. "cup"
  name: string | null;      // nome breve dato da Grok, se c'è
  conf: number;             // 0..1
  bbox: BBox;
  color: string | null;     // in italiano, es. "rosso"
  coords_mm: { x: number; y: number; angle_deg: number } | null; // null finché il braccio non è calibrato
  distance_m: number | null; // stima, non misura
}

export interface DogTelemetry {
  target_id: number | null;
  target_visible: boolean;
  error_x: number;     // -1 sinistra .. 0 centro .. +1 destra
  error_y: number;     // -1 alto .. +1 basso
  bbox_height: number; // 0..1, più grande = più vicino
}

export interface FramePayload {
  source: Source;
  frame_id: number;
  ts: number;          // secondi Unix
  objects: VisionObject[];
  dog_telemetry: DogTelemetry | null; // solo per "dog" con un target impostato
  safety_stop: boolean;
  safety_reason: string | null;
}

export interface VerifyResult {
  ok: boolean;
  success: boolean;
  disappeared_from_source: boolean;
  arrived_at_destination: boolean | null;
  source_iou: number;
  destination_iou: number;
  reason: string;
}

/** Risposta di /voice/dispatch: c'è sempre `say`, la frase che la voce deve dire. */
export interface VoiceResult {
  ok: boolean;
  say: string;
  error?: string;
  // list_objects:
  objects?: { id: number; name: string }[];
  // select_object:
  id?: number | null;
  action?: string;               // "pick" | "place" | "none" ...
  destination_id?: number | null;
  ambiguous?: boolean;           // true = `say` è una domanda da fare all'utente
  // verify_last_action:
  success?: boolean;
}
```

## 4. Esempi

```ts
const BASE = "http://192.168.1.50:8000"; // IP del PC

// Video in groky-app: nessun codice, basta il .env (cameras.ts legge un JPEG alla volta)
//   EXPO_PUBLIC_CAMERAS=Visione=http://192.168.1.50:8000/frame/arm.jpg
// (?raw=true per il frame senza riquadri. /stream/arm è MJPEG: va bene solo in un browser, non in <Image> di React Native)

// Oggetti dal vivo
const ws = new WebSocket(`${BASE.replace("http", "ws")}/ws/objects/arm?hz=10`);
ws.onmessage = (e) => {
  const frame: FramePayload = JSON.parse(e.data);
  // frame.objects -> lista / overlay cliccabile (bbox * dimensioni dell'<img>)
};

// Il cane insegue l'oggetto 3
await fetch(`${BASE}/dog/target`, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ target_id: 3 }),
});
```

### Grok Voice (realtime)

L'app apre la sessione vocale con xAI; la visione risponde agli strumenti.

1. `const tools = await (await fetch(`${BASE}/voice/tools`)).json()` → mandateli in `session.update` → `session.tools`.
2. A ogni evento `response.function_call_arguments.done`:
   ```ts
   const result: VoiceResult = await (await fetch(`${BASE}/voice/dispatch`, {
     method: "POST", headers: { "Content-Type": "application/json" },
     body: JSON.stringify({ name: ev.name, arguments: ev.arguments }), // arguments può restare la stringa JSON
   })).json();
   send({ type: "conversation.item.create",
          item: { type: "function_call_output", call_id: ev.call_id, output: JSON.stringify(result) } });
   ```
3. Dopo aver risposto a **tutte** le chiamate di quel turno: `send({ type: "response.create" })`.

Strumenti: `list_objects`, `select_object` (con `description` = le parole dell'utente), `describe_scene`,
`verify_last_action`. Se `ambiguous` è `true`, la voce fa la domanda in `say` e la risposta dell'utente
va in un nuovo `select_object`: il contesto lo teniamo noi.

In groky-app: aggiungete i nostri strumenti alla lista di `grokTools()` (per esempio con il prefisso `vision_`)
e, in `runTool`, mandate quei nomi a `POST /voice/dispatch` invece che a `{bridge}/tools/{name}`.

`XAI_API_KEY` non va nel codice dell'app (finirebbe nel bundle): ditecelo se vi serve un modo per ottenerla dal server.

## 5. Esempi reali (scena di prova)

```json
// GET /state/arm
{"source": "arm", "frame_id": 3, "ts": 1793787986.08,
 "objects": [{"id": 1, "label": "box", "name": null, "conf": 0.87, "bbox": [0.7188, 0.5208, 0.1266, 0.1479],
              "color": "verde", "coords_mm": null, "distance_m": 1.2}],
 "dog_telemetry": null, "safety_stop": false, "safety_reason": null}

// POST /voice/dispatch {"name": "select_object", "arguments": {"source": "arm", "description": "il numero 1"}}
{"source": "local", "latency_ms": 0.6, "ok": true, "id": 1, "destination_id": null, "action": "none",
 "say": "Ho selezionato box verde."}

// GET /state/dog (dopo POST /dog/target {"target_id": 1})
{"dog_telemetry": {"target_id": 1, "target_visible": true, "error_x": 0.5642, "error_y": 0.1895, "bbox_height": 0.1479}}
```

## 6. Cosa ancora non c'è

- `select_object` restituisce l'`id` ma **non muove il braccio**: oggi la scelta arriva al braccio
  (`logs/selection.json`) solo dalla demo sul PC. Da decidere con il gruppo robotica chi la manda.
  Il vostro `arm_pick_place` passa il nome dell'oggetto ("red cube"), noi diamo un `id`: da allineare.
- Il tasto G della demo (Grok trova oggetti che YOLO non conosce) non è esposto dal server.
- `coords_mm` resta `null` finché non si calibra il braccio con i marker ArUco.
