# F1 Vision Analytics — Piano di progetto

> **Progetto di portfolio**: sistema end-to-end che trasforma video di Formula 1 (broadcast / onboard)
> in analitica quantitativa, combinando **computer vision** e **analisi strategica** su dati reali.

---

## 1. L'angolo scelto (e perché)

Hai detto di essere avanzato e di non aver ancora scelto l'angolo. Ti propongo quello con il
**miglior rapporto impatto/fattibilità** per un portfolio:

**"Dal video alla strategia": una pipeline che rileva e traccia le monoposto in video, le proietta
su una mappa 2D del circuito, stima velocità / gap / traiettoria, e incrocia il tutto con la
telemetria reale (FastF1) per fare analisi strategica (gomme, stint, pit-stop).**

Perché funziona come portfolio:

- **È visivamente impressionante** — overlay con bounding box, mini-mappa bird's-eye, tracciato della
  racing line. Le prime 5 righe del README con una GIF fanno colpo su un recruiter in 10 secondi.
- **Copre l'intero stack ML** — detection, multi-object tracking, geometria (omografia), stima cinematica,
  validazione con ground truth, e un layer di *planning*/strategia. Non è "solo un modello".
- **Usa dati pubblici e gratuiti** — nessun dataset a pagamento. FastF1 dà timing e telemetria ufficiali;
  i video di gara/onboard si prendono da clip pubbliche per uso dimostrativo.
- **Ha una storia** — "ho ricostruito la telemetria dal solo video e l'ho validata contro i dati ufficiali"
  è una frase che si racconta bene in un colloquio.

### Alternative (se vuoi restringere lo scope)

| Angolo | Difficoltà | Pro | Contro |
|---|---|---|---|
| **A. Pipeline video→strategia** (consigliato) | Alta | Completo, d'impatto, storia forte | Molte parti mobili |
| **B. Analisi pit-stop** | Media | Self-contained, camera fissa = più facile | Meno "wow" visivo |
| **C. Solo strategia/telemetria (FastF1)** | Bassa | Veloce, dati puliti | Poca CV, si vede tanto in giro |

Suggerimento: **parti dal MVP dell'angolo A limitato a un onboard** (camera meno mobile del broadcast),
e tieni B come modulo aggiuntivo. Se il tempo stringe, il progetto resta completo anche solo con l'onboard.

---

## 2. Architettura

```
Video (broadcast/onboard)
        │
        ▼
┌───────────────┐   ┌────────────────┐   ┌─────────────────┐
│  Detection    │──▶│    Tracking    │──▶│   Homography    │
│ (YOLO fine-   │   │ (ByteTrack /   │   │ frame → mappa   │
│  tuned su F1) │   │  BoT-SORT)     │   │ 2D del circuito │
└───────────────┘   └────────────────┘   └────────┬────────┘
                                                   ▼
                                          ┌─────────────────┐
                                          │  Cinematica     │
                                          │ velocità, gap,  │
                                          │ racing line     │
                                          └────────┬────────┘
                                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │  Layer strategico + validazione (FastF1)             │
        │  stint gomme, degrado, undercut/overcut, pit window  │
        └──────────────────────────┬───────────────────────────┘
                                    ▼
                          ┌───────────────────┐
                          │  Dashboard        │
                          │  (Streamlit)      │
                          └───────────────────┘
```

---

## 3. Stack tecnico

- **Linguaggio**: Python 3.11
- **CV / Detection**: [Ultralytics YOLO](https://docs.ultralytics.com) (YOLO11), fine-tuning su F1
- **Tracking**: `supervision` (ByteTrack), in alternativa BoT-SORT integrato in Ultralytics
- **Geometria**: OpenCV (omografia, `cv2.findHomography`, `perspectiveTransform`)
- **Dati F1 reali**: [FastF1](https://docs.fastf1.dev) (timing, telemetria, meteo, stint)
- **Analisi**: NumPy, pandas, SciPy (smoothing, filtro di Kalman per la velocità)
- **Dashboard**: Streamlit + Plotly
- **Annotazione dataset**: CVAT o Roboflow (per il fine-tuning del detector)
- **MLOps leggero**: `uv`/`pip` + `ruff` + `pytest` + GitHub Actions

---

## 4. Dataset

1. **Video dimostrativi** — clip pubbliche di onboard/gara *solo per demo tecnica*. Documenta la fonte
   e non ridistribuire i file nel repo (metti solo un piccolo campione o istruzioni per scaricarlo).
2. **Annotazioni detection** — 300–800 frame etichettati (classe `car`, opzionale `car_number`,
   `marshal`, `pit_crew`). Bastano poche centinaia di frame ben scelti + augmentation.
3. **Telemetria** — via FastF1: gratuita, ufficiale, cache locale. Nessuna annotazione manuale.
4. **Calibrazione circuito** — coordinate della mappa 2D del tracciato (SVG/GeoJSON dei circuiti,
   oppure punti noti curva/rettilineo per stimare l'omografia).

> ⚠️ **Nota legale**: i diritti dei video di F1 sono di FOM. Usa le clip solo per dimostrazione tecnica,
> cita la fonte, non ridistribuire, e considera anche footage di sim/game (es. onboard sintetici) per
> avere materiale riutilizzabile senza vincoli.

---

## 5. Roadmap a milestone

**M0 — Setup (2–3 giorni)**
Repo, ambiente, `requirements`, CI, cache FastF1 funzionante, un notebook che scarica una sessione.

**M1 — Detection & Tracking (1–2 settimane)**
YOLO pre-addestrato → baseline su un onboard. Poi fine-tuning su frame annotati. Integrazione ByteTrack,
ID persistenti. *Deliverable*: video con box + track ID stabili.

**M2 — Omografia & mappa 2D (1 settimana)**
Selezione punti di calibrazione, proiezione dei centroidi sulla mappa del circuito, mini-mappa live.
*Deliverable*: bird's-eye view delle auto in movimento.

**M3 — Cinematica (1 settimana)**
Velocità e gap dai movimenti proiettati (con smoothing/Kalman), estrazione racing line.
*Deliverable*: curve velocità-vs-tempo ricostruite dal video.

**M4 — Validazione + Strategia (1–2 settimane)**
Confronto velocità stimata vs FastF1 (errore %, grafici). Layer strategico: stint gomme, degrado,
finestre di pit, simulazione undercut/overcut. *Deliverable*: report di validazione + insight strategici.

**M5 — Dashboard & storytelling (1 settimana)**
Streamlit: carica un video → mostra overlay, mini-mappa, telemetria ricostruita, confronto coi dati reali.
README con GIF, architettura, risultati. *Deliverable*: demo cliccabile + README da portfolio.

Totale realistico: **6–9 settimane** part-time. MVP mostrabile già a fine M3.

---

## 6. Come presentarlo nel portfolio

- **README con hook visivo** nelle prime righe (GIF del tracking + mini-mappa).
- **Sezione "Risultati"** con numeri: mAP del detector, MOTA/IDF1 del tracker, errore % sulla velocità vs FastF1.
- **Sezione "Cosa ho imparato / limiti"** — onestà tecnica (camera mobile del broadcast, scale ambigue).
  I recruiter tecnici apprezzano molto la discussione dei limiti.
- **Demo live** (Streamlit su HuggingFace Spaces) + notebook riproducibili.
- **Blog post / thread** che racconta la ricostruzione della telemetria dal video: ottimo per LinkedIn.

---

## 7. Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Camera broadcast troppo mobile | Parti da onboard o da inquadrature stabili; omografia per-shot |
| Scala/velocità ambigua senza riferimenti | Calibra su elementi noti (lunghezza rettilineo, passo vettura) |
| Annotare frame è noioso | Usa un modello pre-addestrato per pre-annotare, poi correggi |
| FastF1 rate limit / cache | Abilita la cache locale sin da subito |
| Scope che esplode | Congela l'MVP a M3; M4/M5 sono "nice to have" |

---

## 8. Prossimi passi immediati

1. `pip install -r requirements.txt`
2. Esegui `scripts/download_session.py` per scaricare una sessione FastF1 (verifica cache).
3. Esegui `scripts/run_pipeline.py --video path/al/onboard.mp4` con YOLO pre-addestrato (baseline).
4. Annota ~300 frame e lancia il fine-tuning.
5. Itera per milestone.
