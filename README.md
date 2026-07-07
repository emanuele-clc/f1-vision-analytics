# 🏎️ F1 Vision Analytics

Pipeline end-to-end che trasforma video di Formula 1 in analitica quantitativa:
**detection → tracking → omografia → cinematica → strategia**, con validazione contro la
telemetria ufficiale (FastF1).

> Progetto di portfolio. Vedi [`PLAN.md`](./PLAN.md) per roadmap, milestone e scelte di design.

## Demo (placeholder)

_Inserisci qui una GIF: onboard con bounding box + mini-mappa bird's-eye + curva velocità ricostruita._

## Funzionalità

- 🎯 **Detection** delle monoposto con YOLO11 (fine-tunabile su frame F1).
- 🔗 **Multi-object tracking** con ByteTrack (ID persistenti).
- 🗺️ **Omografia** frame → mappa 2D del circuito (bird's-eye).
- 📈 **Cinematica**: velocità, gap e racing line dai movimenti proiettati (smoothing/Kalman).
- 🧠 **Strategia**: stint gomme, degrado, finestre di pit, undercut/overcut (dati FastF1).
- ✅ **Validazione**: velocità stimata dal video vs telemetria reale (errore %).
- 📊 **Dashboard** Streamlit interattiva.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Scarica una sessione reale (cache FastF1)
python scripts/download_session.py --year 2024 --gp "Monza" --session R

# 2) Baseline: pipeline su un onboard con YOLO pre-addestrato
python scripts/run_pipeline.py --video data/samples/onboard.mp4 --out out/

# 3) Dashboard
streamlit run apps/dashboard.py
```

## Struttura

```
f1-vision-analytics/
├── PLAN.md                 # piano di progetto (leggi prima questo)
├── README.md
├── requirements.txt
├── configs/default.yaml    # parametri pipeline
├── data/                   # dataset e campioni (non versionare i video)
├── src/f1va/
│   ├── config.py           # caricamento config
│   ├── detection.py        # wrapper YOLO
│   ├── tracking.py         # ByteTrack
│   ├── homography.py       # calibrazione + proiezione 2D
│   ├── speed.py            # stima velocità/gap + Kalman
│   ├── strategy.py         # analisi strategica + FastF1
│   ├── pipeline.py         # orchestrazione end-to-end
│   └── viz.py              # overlay e mini-mappa
├── apps/dashboard.py       # UI Streamlit
├── scripts/                # entrypoint CLI
├── notebooks/              # esplorazione e validazione
└── tests/                  # unit test
```

## Note

- I diritti dei video di F1 sono di FOM: usa clip solo per dimostrazione tecnica, cita la fonte,
  non ridistribuire. Valuta onboard sintetici (sim/game) per materiale riutilizzabile.
- FastF1 usa una cache locale: la prima esecuzione scarica i dati, le successive sono istantanee.

## Licenza

MIT (codice). I dati e i video restano dei rispettivi proprietari.
