import streamlit as st
import pandas as pd
import numpy as np

# --- 1. GLOBAL UI & BRANDING ---
st.set_page_config(page_title="HighStakes Pro", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: white; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; }
    .stButton>button { background: #d32f2f; color: white; width: 100%; height: 4em; font-weight: bold; border-radius: 10px; }
    .section-header { color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 10px; margin-top: 30px; font-family: 'Helvetica'; }
    .stTextInput>div>div>input { font-size: 20px; font-weight: bold; color: #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ HighStakes | Elite Match Command")
st.caption("Strategic Simulation Engine | v5.0 Final Build")

# --- SECTION 1: THE TEAMS & MATCH DYNAMICS ---
st.markdown("<h2 class='section-header'>🏟️ Match Setup & Dynamics</h2>", unsafe_allow_html=True)
col_team1, col_team2, col_meta = st.columns([1, 1, 1])

with col_team1:
    h_name = st.text_input("Home Team Name", "Arsenal").upper()
    h_lineup = st.text_area(f"{h_name} Starting XI", height=150, placeholder="List the 11 players...")

with col_team2:
    a_name = st.text_input("Away Team Name", "Man City").upper()
    a_lineup = st.text_area(f"{a_name} Starting XI", height=150, placeholder="List the 11 players...")

with col_meta:
    match_type = st.selectbox("Competition Type", ["League Match", "Local Derby", "Cup Final", "Relegation Battle", "Friendly"])
    seriousness = st.select_slider("Match Seriousness / Intensity", options=["Relaxed", "Normal", "High", "Maximum"])
    
    # Internal Math for Intensity
    seriousness_map = {"Relaxed": 0.85, "Normal": 1.0, "High": 1.15, "Maximum": 1.30}
    intensity_mult = seriousness_map[seriousness]

# --- SECTION 2: HISTORY & INJURIES ---
st.markdown("<h2 class='section-header'>📊 History & Medical Room</h2>", unsafe_allow_html=True)
c_hist, c_inj_h, c_inj_a = st.columns([1, 1, 1])

with c_hist:
    st.subheader("H2H & Last 5")
    h2h_notes = st.text_area("H2H / Form History", "If H2H is low, use recent xG form...", height=150)

with c_inj_h:
    st.subheader(f"🏥 {h_name} Injuries")
    h_inj_name = st.text_input("Player Name (H)", key="hn")
    h_inj_pos = st.selectbox("Role (H)", ["Attacker", "Midfielder", "Defender/GK"], key="hp")
    h_inj_rank = st.selectbox("Impact (H)", ["Key Player", "Starter", "Rotation"], key="hr")

with c_inj_a:
    st.subheader(f"🏥 {a_name} Injuries")
    a_inj_name = st.text_input("Player Name (A)", key="an")
    a_inj_pos = st.selectbox("Role (A)", ["Attacker", "Midfielder", "Defender/GK"], key="ap")
    a_inj_rank = st.selectbox("Impact (A)", ["Key Player", "Starter", "Rotation"], key="ar")

# --- SECTION 3: TACTICAL RATIOS ---
st.markdown("<h2 class='section-header'>📈 Tactical Performance Ratios</h2>", unsafe_allow_html=True)
st.caption("1.0 = League Average. Adjust based on recent form and injuries.")
r_col1, r_col2, r_col3, r_col4 = st.columns(4)

h_xg = r_col1.number_input(f"{h_name} xG Ratio", 1.8)
a_xg = r_col1.number_input(f"{a_name} xG Ratio", 1.2)
h_sot = r_col2.number_input(f"{h_name} SoT Ratio", 1.5)
a_sot = r_col2.number_input(f"{a_name} SoT Ratio", 0.9)
h_bc = r_col3.number_input(f"{h_name} Big Chance Ratio", 2.0)
a_bc = r_col3.number_input(f"{a_name} Big Chance Ratio", 1.1)
h_def = r_col4.number_input(f"{h_name} Defense Ratio", 1.0)
a_def = r_col4.number_input(f"{a_name} Defense Ratio", 1.4)

# --- SECTION 4: BOOKIE ODDS & SIMULATION ---
st.markdown("<h2 class='section-header'>💰 Market Odds & Execution</h2>", unsafe_allow_html=True)
o1, o2, o3, o4, o5 = st.columns(5)
bk_1 = o1.number_input("Bookie: 1", 2.10)
bk_x = o2.number_input("Bookie: X", 3.40)
bk_2 = o3.number_input("Bookie: 2", 3.80)
bk_ov = o4.number_input("Bookie: Ov 2.5", 1.90)
bk_gg = o5.number_input("Bookie: GG", 1.85)

if st.button("🚀 EXECUTE 10,000 RUN HIGHSTAKES SIMULATION"):
    # --- INJURY PENALTY CALCULATION ---
    h_pen = 1.0
    if len(h_inj_name) > 0:
        h_pen = 0.82 if h_inj_rank == "Key Player" else 0.92
        
    a_pen = 1.0
    if len(a_inj_name) > 0:
        a_pen = 0.82 if a_inj_rank == "Key Player" else 0.92

    # --- SIMULATION ENGINE ---
    # Clashing xG with Defense Ratios and Seriousness
    h_final = (h_xg * h_pen * (1/a_def)) * intensity_mult
    a_final = (a_xg * a_pen * (1/h_def)) * intensity_mult
    
    h_sim = np.random.poisson(h_final, 10000)
    a_sim = np.random.poisson(a_final, 10000)
    
    # PROBABILITIES
    p_1, p_x, p_2 = np.mean(h_sim > a_sim), np.mean(h_sim == a_sim), np.mean(h_sim < a_sim)
    p_gg = np.mean((h_sim > 0) & (a_sim > 0))
    p_ov25 = np.mean((h_sim + a_sim) > 2.5)
    p_un25 = 1 - p_ov25
    p_dc1x = p_1 + p_x
    p_fh_ov = np.mean((h_sim + a_sim)*0.44 > 0.5)

    # --- TRUE ODDS vs BOOKIE ---
    def get_val(p, bk): return (p * bk) - 1
    def to_true_odds(p): return round(1/p, 2) if p > 0.1 else "High"

    data = [
        {"Market": "Match Result (1)", "Prob": p_1, "True Odds": to_true_odds(p_1), "Bookie": bk_1},
        {"Market": "Match Result (X)", "Prob": p_x, "True Odds": to_true_odds(p_x), "Bookie": bk_x},
        {"Market": "Match Result (2)", "Prob": p_2, "True Odds": to_true_odds(p_2), "Bookie": bk_2},
        {"Market": "GG (BTTS)", "Prob": p_gg, "True Odds": to_true_odds(p_gg), "Bookie": bk_gg},
        {"Market": "Over 2.5", "Prob": p_ov25, "True Odds": to_true_odds(p_ov25), "Bookie": bk_ov},
        {"Market": "Double Chance (1X)", "Prob": p_dc1x, "True Odds": to_true_odds(p_dc1x), "Bookie": 1.35},
        {"Market": "1st Half Over 0.5", "Prob": p_fh_ov, "True Odds": to_true_odds(p_fh_ov), "Bookie": 1.45}
    ]

    for d in data: d["Value"] = get_val(d["Prob"], d["Bookie"])
    
    st.divider()
    res_df = pd.DataFrame(data)
    st.table(res_df.style.format({"Prob": "{:.1%}", "Value": "{:.1%}"}))
    
    best = res_df.loc[res_df['Value'].idxmax()]
    if best['Value'] > 0.05:
        st.success(f"💎 **HIGHSTAKES SUMMARY:** Highest Edge: **{best['Market']}**. Prob: {best['Prob']:.1%}. Edge: {best['Value']:.1%}")
    else:
        st.warning("⚠️ **SUMMARY:** Play carefully. No significant value found.")
