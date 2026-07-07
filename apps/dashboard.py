"""Dashboard Streamlit — F1 Strategy & ML.

Tema scuro in stile paddock: KPI, analisi degrado, strategia ottimale e replay,
tutto da dati reali FastF1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va import data as f1data      # noqa: E402
from f1va import features, strategy  # noqa: E402

# ----------------------------------------------------------------------------- #
#  Stile
# ----------------------------------------------------------------------------- #
st.set_page_config(page_title="F1 Strategy & ML", page_icon="🏎️", layout="wide")

COMPOUND_COLORS = {
    "SOFT": "#E10600", "MEDIUM": "#FFD12E", "HARD": "#EFEFEF",
    "INTERMEDIATE": "#43B02A", "WET": "#0067AD",
}

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.2rem; max-width: 1250px;}
.hero {
  background: linear-gradient(100deg, #E10600 0%, #8B0000 45%, #17171B 100%);
  border-radius: 16px; padding: 26px 30px; margin-bottom: 22px;
  box-shadow: 0 8px 30px rgba(225,6,0,.18);
}
.hero h1 {margin: 0; font-size: 2.05rem; font-weight: 800; letter-spacing: -.5px; color: #fff;}
.hero p {margin: 6px 0 0; color: rgba(255,255,255,.82); font-size: .98rem;}
.kpi {
  background: #17171B; border: 1px solid #262630; border-radius: 14px;
  padding: 16px 18px; height: 100%;
}
.kpi .label {color: #9A9AA5; font-size: .78rem; text-transform: uppercase; letter-spacing: .6px;}
.kpi .value {color: #fff; font-size: 1.7rem; font-weight: 800; margin-top: 2px;}
.pill {
  display:inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 700;
  font-size: .82rem; margin-right: 6px; color:#111;
}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {
  background:#17171B; border-radius: 10px 10px 0 0; padding: 8px 18px; color:#B9B9C3;
}
.stTabs [aria-selected="true"] {background:#E10600 !important; color:#fff !important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>🏎️ F1 Strategy &amp; ML</h1>'
    '<p>Telemetria reale FastF1 → analisi del degrado → strategia gomme ottimale</p></div>',
    unsafe_allow_html=True,
)


def kpi(col, label: str, value: str) -> None:
    col.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div></div>', unsafe_allow_html=True)


def plot_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111116",
        margin=dict(l=10, r=10, t=40, b=10), font=dict(color="#EDEDED"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def stint_chart(stints) -> go.Figure:
    """Barra orizzontale degli stint (Gantt gomme)."""
    fig = go.Figure()
    start = 0
    for compound, length in stints:
        fig.add_trace(go.Bar(
            y=["Strategia"], x=[length], base=start, orientation="h",
            marker=dict(color=COMPOUND_COLORS.get(compound, "#888"),
                        line=dict(color="#0E0E10", width=2)),
            text=f"{compound} · {length}", textposition="inside",
            insidetextanchor="middle", showlegend=False,
            textfont=dict(color="#111" if compound != "SOFT" else "#fff", size=13),
        ))
        start += length
    fig.update_layout(barmode="stack", xaxis_title="Giro")
    return plot_layout(fig, height=170)


# ----------------------------------------------------------------------------- #
#  Sidebar
# ----------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Sessione")
    year = st.number_input("Anno", 2018, 2025, 2024)
    gp = st.text_input("Gran Premio", "Monza")
    session_type = st.selectbox("Sessione", ["R", "Q", "FP1", "FP2", "FP3"], index=0)
    load = st.button("Carica sessione", type="primary", use_container_width=True)
    st.caption("La prima volta il download può richiedere ~1 minuto (poi va in cache).")

if load:
    with st.spinner("Scarico e preparo i dati…"):
        ses = f1data.load_session(int(year), gp, session_type)
        raw = f1data.laps_dataframe(ses)
        laps = f1data.quicklaps(raw)
        st.session_state.update(
            raw=raw, laps=laps, deg=features.degradation_table(laps),
            meta=f"{int(year)} · {gp} · {session_type}",
        )

# ----------------------------------------------------------------------------- #
#  Corpo
# ----------------------------------------------------------------------------- #
if "deg" not in st.session_state:
    st.info("Scegli anno e Gran Premio nella barra laterale e premi **Carica sessione**.")
    st.stop()

laps = st.session_state["laps"]
deg = st.session_state["deg"]
st.caption(f"Sessione caricata: **{st.session_state['meta']}**")

c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Giri validi", f"{len(laps)}")
kpi(c2, "Piloti", f"{laps['Driver'].nunique()}")
kpi(c3, "Mescole", f"{laps['Compound'].nunique()}")
kpi(c4, "Miglior giro", f"{laps['laptime_s'].min():.3f}s")
st.write("")

tab_deg, tab_strat, tab_replay = st.tabs(["📉 Degrado gomme", "🧠 Strategia", "🗺️ Replay"])

with tab_deg:
    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("Degrado per mescola")
        st.dataframe(deg, use_container_width=True, hide_index=True)
        fig = go.Figure(go.Bar(
            x=deg["compound"], y=deg["deg_s_per_lap"],
            marker_color=[COMPOUND_COLORS.get(c, "#888") for c in deg["compound"]],
            text=deg["deg_s_per_lap"], textposition="outside",
        ))
        fig.update_layout(yaxis_title="Degrado (s/giro)")
        st.plotly_chart(plot_layout(fig, 300), use_container_width=True)
    with right:
        st.subheader("Tempo sul giro vs età gomma")
        fig = go.Figure()
        for comp, g in laps.groupby(laps["Compound"].astype(str).str.upper()):
            fig.add_trace(go.Scatter(
                x=g["TyreLife"], y=g["laptime_s"], mode="markers",
                name=comp, marker=dict(color=COMPOUND_COLORS.get(comp, "#888"), size=7,
                                       line=dict(color="#0E0E10", width=.5)),
            ))
        fig.update_layout(xaxis_title="Età gomma (giri)", yaxis_title="Tempo (s)")
        st.plotly_chart(plot_layout(fig, 430), use_container_width=True)

with tab_strat:
    st.subheader("Strategia ottimale")
    a, b, c = st.columns(3)
    total_laps = a.slider("Giri gara", 30, 78, 53)
    max_stops = b.slider("Soste massime", 1, 3, 2)
    pit_loss = c.slider("Pit-loss (s)", 15.0, 30.0, 22.0, 0.5)

    models = strategy.fit_tyre_models(deg)
    if len(models) < 2:
        st.warning("Servono almeno due mescole con dati sufficienti in questa sessione.")
    else:
        best = strategy.optimize_strategy(
            total_laps, list(models.keys()), models,
            max_stops=max_stops, pit_loss=pit_loss)
        pills = "".join(
            f'<span class="pill" style="background:{COMPOUND_COLORS.get(c, "#888")}">'
            f'{c} · {n}</span>' for c, n in best.stints)
        st.markdown(f"**Piano consigliato:** {pills}", unsafe_allow_html=True)
        st.plotly_chart(stint_chart(best.stints), use_container_width=True)
        m1, m2 = st.columns(2)
        kpi(m1, "Soste", f"{best.n_stops}")
        kpi(m2, "Tempo gara stimato", f"{best.total_time_s/60:.2f} min")

with tab_replay:
    st.subheader("Replay posizioni")
    st.info("Collega qui la visualizzazione: `f1va.replay.build_frames(session)` fornisce "
            "le posizioni (X, Y) dei piloti nel tempo, e `track_outline(session)` il tracciato.")
