# Risultati della valutazione

- Data: 2026-11-03 10:05 · Modalità: finta (GROK_MOCK=1) · Modello: grok-4.7
- Dataset: 4 scene, 20 comandi
- **Modalità finta: i numeri verificano solo la pipeline, non la qualità di Grok.**

| Metrica | Con scorciatoia locale | Solo Grok |
|---|---|---|
| Accuratezza sull'id | 95% (19/20) | 95% (19/20) |
| Ambiguità: precisione | 100% (6/6) | 100% (6/6) |
| Ambiguità: richiamo | 100% (6/6) | 100% (6/6) |
| Rifiuto oggetti assenti | 100% (4/4) | 100% (4/4) |
| Latenza p50 (ms) | 3 | 4 |
| Latenza p95 (ms) | 5 | 5 |
| Risolti in locale | 45% (9/20) | 0% (0/20) |
| Errori (ok=False) | 0 | 0 |

## Comandi sbagliati

| Modalità | Scena | Comando | Atteso | Ottenuto | Strada | Motivo |
|---|---|---|---|---|---|---|
| con locale | scene_04 | prendi quello più a destra | 13 | nessuno | grok | Nessun oggetto corrisponde alla richiesta. |
| solo Grok | scene_04 | prendi quello più a destra | 13 | nessuno | grok | Nessun oggetto corrisponde alla richiesta. |
