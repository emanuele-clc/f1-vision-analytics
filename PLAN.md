# F1 Strategy & ML — Piano di progetto

> Progetto di **analisi dati, machine learning e ottimizzazione strategica**
> in Formula 1, a partire dalla telemetria ufficiale (FastF1). Nessuna computer vision.

---

## 1. Obiettivo del progetto

Il progetto risponde a una domanda concreta da muretto box: **qual è la strategia gomme
ottimale per questa gara, e quanto sono prevedibili i tempi sul giro?**

Il progetto integra tre aree tecniche:

- **Data engineering** — pipeline riproducibile da API reale (FastF1) a dataset puliti.
- **Machine learning** — modello predittivo dei tempi sul giro con validazione seria (cross-validation, MAE/R²).
- **Planning / ottimizzazione** — simulatore di gara e ricerca della strategia ottimale, con analisi undercut/overcut.

Il tutto è testabile, documentato e mostrabile in una dashboard. Ispirato a progetti come
`f1-race-replay`, ma spostato dal "guardare la gara" al "decidere la strategia".

---

## 2. Architettura

```
FastF1 (dati reali)
      │
      ▼
┌──────────────┐   ┌───────────────┐   ┌────────────────────┐
│  data.py     │──▶│  features.py  │──▶│  models.py (ML)    │
│ load + clean │   │ dataset + deg │   │ tempi sul giro     │
└──────────────┘   └──────┬────────┘   └────────────────────┘
                          │
                          ▼
                 ┌────────────────────┐    ┌──────────────┐
                 │  strategy.py       │    │  replay.py   │
                 │ simulatore + opt.  │    │ posizioni    │
                 └─────────┬──────────┘    └──────┬───────┘
                           ▼                      ▼
                        ┌──────────────────────────────┐
                        │  dashboard.py (Streamlit)     │
                        └──────────────────────────────┘
```

---

## 3. Stack

Python 3.11 · FastF1 · pandas/numpy · scikit-learn (HistGradientBoosting) · Streamlit · Plotly · pytest · ruff.

---

## 4. Componenti in dettaglio

**Data (`data.py`)** — `load_session()` scarica una sessione (cache locale); `laps_dataframe()`
la trasforma in un DataFrame ordinato; `quicklaps()` filtra out/in-lap e giri anomali.

**Feature (`features.py`)** — `build_laptime_dataset()` produce (X, y) con età gomma, giro,
stint e mescola one-hot; `degradation_table()` stima base pace e degrado per mescola via regressione.

**ML (`models.py`)** — `LapTimeModel` (gradient boosting) con `fit/predict/evaluate/save/load` e
`cross_validate()` per una stima onesta dell'errore (MAE in secondi).

**Strategia (`strategy.py`)** — `TyreModel` (base + degrado·età), `simulate_strategy()` calcola il
tempo gara di un piano, `optimize_strategy()` esplora le combinazioni plausibili e restituisce
l'ottimo; `undercut_gain()` quantifica la convenienza di un undercut.

**Replay (`replay.py`)** — `build_frames()` ricostruisce le posizioni (X, Y) dei piloti nel tempo
per la visualizzazione; `track_outline()` disegna il tracciato.

---

## 5. Roadmap a milestone

**M0 — Setup** ✅ Repo, ambiente, cache FastF1, moduli base e test.

**M1 — Dati & feature** ✅ Caricamento, pulizia, dataset ML, tabella degrado.

**M2 — Modello ML** Allenare `LapTimeModel` su più gare, tuning, report CV. Confrontare con baseline
(media per mescola). *Deliverable*: MAE < ~0.3s e grafici predetto-vs-reale.

**M3 — Strategy optimizer** ✅ Simulatore + ottimizzatore + undercut. Estensione: pit-loss per
circuito, Safety Car, meteo variabile.

**M4 — Validazione** Confrontare la strategia consigliata con quella realmente adottata in gara e
misurare il delta di tempo, come misura oggettiva della qualità del modello.

**M5 — Dashboard & storytelling** Rifinire la UI, aggiungere il replay animato, README con GIF e numeri.

---

## 6. Idee di estensione (ML avanzato)

- Previsione della **posizione finale** o del **risultato qualifica** (classificazione/regressione).
- Modello di **probabilità di Safety Car** per gara/circuito.
- **Ottimizzazione stocastica**: strategia robusta rispetto all'incertezza sul degrado.
- Confronto **pilota-vs-pilota** sul passo gara depurato dal carburante.

---

## 7. Note conclusive

Il valore del progetto risiede nella pipeline completa: dai dati grezzi ufficiali a una decisione
di strategia motivata e misurabile, con validazione contro la strategia realmente adottata in gara.
