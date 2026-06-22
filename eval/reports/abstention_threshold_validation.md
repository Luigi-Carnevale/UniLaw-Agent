# Validazione held-out della soglia di astensione (Ciclo 2 — FASE 6 + 13)

- Generato: 2026-06-22T11:29:27
- Set di calibrazione: ['q17', 'q18', 'q19']
- Set held-out (mai visto in calibrazione): ['q35', 'q36', 'q37']

## Forza LESSICALE (overlap di token) — FASE 6

- Soglia in config (`ABSTENTION_OOD_MAX_STRENGTH`): **0.37**
- Soglia calibrata sui soli negativi storici: **0.3667**

| Insieme | Soglia | Accuratezza |
|---|---|---|
| calibrazione | 0.3667 | 1.0 |
| held-out | 0.37 (config) | 1.0 |
| held-out | 0.3667 (calibrata) | 1.0 |

### Calibrazione

| id | strength | causa attesa | causa predetta | ok |
|---|---|---|---|:--:|
| q17 | 0.4 | evidenza_insufficiente | evidenza_insufficiente | ✓ |
| q18 | 0.6667 | evidenza_insufficiente | evidenza_insufficiente | ✓ |
| q19 | 0.3333 | fuori_dominio | fuori_dominio | ✓ |

### Held-out (soglia di config)

| id | strength | causa attesa | causa predetta | ok |
|---|---|---|---|:--:|
| q35 | 0.3333 | fuori_dominio | fuori_dominio | ✓ |
| q36 | 0.25 | fuori_dominio | fuori_dominio | ✓ |
| q37 | 0.5 | evidenza_insufficiente | evidenza_insufficiente | ✓ |

## Forza SEMANTICA (similarità di embedding) — FASE 13

- Soglia in config (`ABSTENTION_OOD_SEMANTIC_MAX_STRENGTH`): **0.53**
- Soglia calibrata sui soli negativi storici: **0.5286**

| Insieme | Soglia | Accuratezza |
|---|---|---|
| calibrazione | 0.5286 | 1.0 |
| held-out | 0.53 (config) | 1.0 |
| held-out | 0.5286 (calibrata) | 1.0 |

### Calibrazione

| id | strength | causa attesa | causa predetta | ok |
|---|---|---|---|:--:|
| q17 | 0.597 | evidenza_insufficiente | evidenza_insufficiente | ✓ |
| q18 | 0.6741 | evidenza_insufficiente | evidenza_insufficiente | ✓ |
| q19 | 0.4601 | fuori_dominio | fuori_dominio | ✓ |

### Held-out (soglia di config)

| id | strength | causa attesa | causa predetta | ok |
|---|---|---|---|:--:|
| q35 | 0.3242 | fuori_dominio | fuori_dominio | ✓ |
| q36 | 0.4036 | fuori_dominio | fuori_dominio | ✓ |
| q37 | 0.6634 | evidenza_insufficiente | evidenza_insufficiente | ✓ |

