"""Dashboard Streamlit: replay gara, previsioni ML e ottimizzatore di strategia."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va import data as f1data      # noqa: E402
from f1va import features, strategy  # noqa: E402

st.set_page_config(page_title="F1 Strategy & ML", layout="wide")
st.title("F1 Strategy & ML")
st.caption("Dati reali FastF1 -> analisi degrado -> previsioni -> strategia ottimale")

with st.sidebar:
    st.header("Sessione")
    year = st.number_input("Anno", 2018, 2025, 2024)
    gp = st.text_input("Gran Premio", "Monza")
    load = st.button("Carica sessione", type="primary")

tab_deg, tab_strat, tab_replay = st.tabs(["Degrado gomme", "Strategia ottimale", "Replay"])

if load:
    with st.spinner("Scarico i dati (la prima volta puo richiedere un minuto)..."):
        ses = f1data.load_session(int(year), gp, "R")
        laps = f1data.quicklaps(f1data.laps_dataframe(ses))
        deg = features.degradation_table(laps)
    st.session_state["laps"] = laps
    st.session_state["deg"] = deg

if "deg" in st.session_state:
    deg = st.session_state["deg"]
    with tab_deg:
        st.subheader("Degrado stimato per mescola")
        st.dataframe(deg)
        st.bar_chart(deg.set_index("compound")["deg_s_per_lap"])

    with tab_strat:
        st.subheader("Strategia ottimale")
        total_laps = st.slider("Giri gara", 30, 78, 53)
        max_stops = st.slider("Soste massime", 1, 3, 2)
        models = strategy.fit_tyre_models(deg)
        if len(models) >= 2:
            best = strategy.optimize_strategy(
                total_laps, list(models.keys()), models, max_stops=max_stops)
            plan = "  ->  ".join(f"{c} ({n} giri)" for c, n in best.stints)
            st.metric("Piano consigliato", plan)
            st.metric("Tempo gara stimato", f"{best.total_time_s/60:.2f} min",
                      help=f"{best.n_stops} sosta/e")
        else:
            st.warning("Servono almeno due mescole con dati sufficienti.")

    with tab_replay:
        st.info("Il replay usa f1va.replay.build_frames(session). "
                "Collega qui la visualizzazione delle posizioni (X, Y) nel tempo.")
else:
    with tab_deg:
        st.info("Scegli anno e GP nella barra laterale e premi «Carica sessione».")
