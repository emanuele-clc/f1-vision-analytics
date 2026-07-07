"""Dashboard Streamlit: carica un video, mostra overlay e telemetria ricostruita.

Uso:
    streamlit run apps/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

st.set_page_config(page_title="F1 Vision Analytics", layout="wide")
st.title("🏎️ F1 Vision Analytics")
st.caption("Dal video alla telemetria: detection → tracking → mappa → velocità → strategia")

with st.sidebar:
    st.header("Input")
    video = st.file_uploader("Video onboard/broadcast", type=["mp4", "mov", "mkv"])
    config = st.text_input("Config", "configs/default.yaml")
    run_btn = st.button("Esegui pipeline", type="primary")

tab_video, tab_tel, tab_strat = st.tabs(["Overlay", "Telemetria", "Strategia"])

if run_btn and video is not None:
    tmp = Path("out") / "input_video.mp4"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_bytes(video.read())

    from f1va.config import load_config
    from f1va.pipeline import run

    with st.spinner("Analisi in corso..."):
        df = run(str(tmp), load_config(config), out_dir="out")

    with tab_video:
        overlay = Path("out") / "overlay.mp4"
        if overlay.exists():
            st.video(str(overlay))
    with tab_tel:
        st.dataframe(df.head(500))
        if "speed_kmh" in df:
            pivot = df.pivot_table(index="time_s", columns="track_id", values="speed_kmh")
            st.line_chart(pivot)
    with tab_strat:
        st.info("Collega una sessione FastF1 per il confronto con i dati reali "
                "(vedi f1va.strategy.validate_speed).")
else:
    with tab_video:
        st.info("Carica un video e premi «Esegui pipeline».")
