# 🏎️ F1 Strategy & ML

Analisi dati, machine learning e ottimizzazione della strategia di gara in Formula 1,
a partire dalla telemetria ufficiale (FastF1). Nessun video: tutto guidato dai dati.

> 🔴 **Demo live (GitHub Pages):** https://emanuele-clc.github.io/f1-vision-analytics/ — l'ottimizzatore di strategia gira nel browser.


![Replay delle posizioni in pista](images/replay.gif)

> Vedi [`PLAN.md`](./PLAN.md) per l'architettura e la roadmap del progetto.

## Cosa fa

- 📥 **Data layer** — scarica sessioni reali via FastF1 e le pulisce in DataFrame pronti all'uso.
- 🧪 **Feature engineering** — costruisce dataset per il ML e stima il degrado gomma per mescola.
- 🤖 **Machine learning** — modello di previsione del tempo sul giro (gradient boosting) con cross-validation.
- 🧠 **Strategy optimizer (planning)** — simulatore di gara che trova la strategia gomme/pit ottimale
  (numero di soste, mescole, lunghezza degli stint), con analisi di undercut/overcut.
- 🖥️ **Dashboard** — interfaccia Streamlit: degrado, strategia consigliata, replay.

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r requirements.txt

# Analisi + strategia ottimale su una gara reale
python scripts/optimize_strategy.py --year 2024 --gp "Monza" --laps 53

# Allena il modello di previsione tempi sul giro
python scripts/train_models.py --year 2024 --gp "Monza"

# Dashboard
streamlit run apps/dashboard.py
```

## Struttura

```
src/f1va/
├── data.py         # caricamento sessioni FastF1 + pulizia giri
├── features.py     # dataset ML + tabella degrado gomme
├── models.py       # modello tempi sul giro (scikit-learn) + CV + save/load
├── strategy.py     # simulatore e ottimizzatore di strategia (planning)
├── replay.py       # ricostruzione posizioni in pista per il replay
└── config.py       # config YAML
apps/dashboard.py   # dashboard Streamlit
scripts/            # CLI: download, train, optimize
tests/              # unit test (strategia + feature)
```

## Come funziona la strategia

Il degrado di ogni mescola è modellato come `tempo_giro = base + degrado * età_gomma`, stimato
per regressione dai dati reali. Il simulatore somma i tempi degli stint più il tempo perso ai box,
e l'ottimizzatore esplora tutte le combinazioni plausibili di soste, mescole e lunghezze
per trovare quella col tempo gara minimo (rispettando la regola delle due mescole).

## Dati

FastF1 fornisce timing e telemetria ufficiali, gratuiti, con cache locale (`.fastf1_cache/`).
Nessun dato proprietario nel repo.

## Licenza

MIT. Dati F1 di proprietà dei rispettivi titolari, usati a fini educativi e non commerciali.
