import streamlit as st
import pandas as pd
import numpy as np

# --- 1. BRANDING & UI CONFIG ---
st.set_page_config(page_title="HighStakes: Elite Analytics", layout="wide")

# Custom CSS for Professional Dark Mode
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; }
    .stButton>button { background: linear-gradient(90deg, #d32f2f, #b71c1c); color: white; font-weight: bold; border: none; height: 3.5em; border-radius: 8px; width: 100%; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #161b22; border-radius: 4px; padding: 10px 20px; color: white; }
    .value-high { color: #238636; font-weight: bold; }
    .value-low { color: #da3633; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔥 HighStakes | Strategic Match Engine")
st.caption("v4.0 Final Build | Monte Carlo 10k | Position-Weighted xG")

# --- 2. SIDEBAR: DYNAMIC CONTEXT & INJURIES ---
with st.sidebar:
    st.header("🏟️ Match Environment")
    game_type = st.selectbox("Competition Type", ["Standard League Match", "Local Derby", "Cup Final", "Relegation Battle", "Friendly"])
    
    # Internal Weighting for Match Seriousness
    seriousness_map = {"Standard League Match": 1.0, "Local Derby": 1.18, "Cup Final": 1.35, "Relegation Battle": 1.12, "Friendly": 0.8}
    intensity = seriousness_map[game_type]
    
    st.divider()
    st.header("🏥 HighStakes Injury Matrix")
    
    def injury_module(team_label):
        st.subheader(f"{team_label} Absence")
        name = st.text_input(f"Player Name", key=f"{team_label}_name", placeholder="e.g. Haaland")
        pos = st.selectbox("Role", ["Striker/Winger", "Midfielder", "Defender/GK"], key=f"{team_label}_pos")
        importance = st.select_slider("Impact", options=["Rotation", "Starter", "Key Player"], key=f"{team_label}_imp")
        
        # Math Penalty Logic
        penalty = 1.0
        if importance == "Key Player": penalty = 0.82  # 18% nerf
        elif importance == "Starter": penalty = 0.91   # 9% nerf
        return {"pos": pos, "penalty": penalty, "active": len(name) > 0}

    h_inj = injury_module("Home")
    a_inj = injury_module("Away")

# --- 3. COMMAND DASHBOARD ---
tab_prep, tab_data, tab_results = st.tabs(["📋 Prep & Lineups", "📈 Tactical Ratios", "🎯 Simulated Value"])

with tab_prep:
    c1, c2 = st.columns(2)
    with c1:
        h_name = st.text_input("Home Team", "Arsenal").upper()
        st.text_area(f"{h_name} Lineup / H2H Notes", height=150, placeholder="Paste lineup or last 5 results here...")
    with c2:
        a_name = st.text_input("Away Team", "Man City").upper()
        st.text_area(f"{a_name} Lineup / H2H Notes", height=150, placeholder="Paste lineup or last 5 results here...")
    
    st.info("💡 **H2H Tip:** Check if the Home team has won at least 3 of the last 5 meetings before finalizing ratios.")

with tab_data:
    st.subheader("Tactical Ratios (League Average = 1.0)")
    r_data = {
        "Metric": ["Goals/Game", "xG Ratio", "SoT/Game", "Big Chances", "BC Missed", "Possession", "Fouls", "Offsides", "Goal Kicks", "Yellow Cards", "Saves", "Corners", "Clean Sheet Ratio"],
        "Home Ratio": [1.8, 1.95, 1.6, 2.1, 0.7, 1.2, 1.0, 1.1, 0.8, 1.0, 1.0, 1.4, 1.2],
        "Away Ratio": [1.2, 1.1, 0.9, 1.1, 1.3, 0.9, 1.3, 0.7, 1.1, 1.5, 1.4, 0.8, 0.7]
    }
    df_ratios = st.data_editor(pd.DataFrame(r_data), use_container_width=True)

with tab_results:
    st.subheader("💰 Market Entry (Bookie Odds)")
    o1, o2, o3, o4 = st.columns(4)
    odd_1 = o1.number_input("Home (1)", value=2.0)
    odd_x = o2.number_input("Draw (X)", value=3.4)
    odd_2 = o3.number_input("Away (2)", value=3.8)
    odd_gg = o4.number_input("GG/BTTS", value=1.85)

    if st.button("🚀 EXECUTE 10,000 RUN SIMULATION"):
        # --- POSITION-WEIGHTED MATH ---
        h_atk_mod = h_inj["penalty"] if h_inj["pos"] == "Striker/Winger" else 1.0
        h_def_mod = h_inj["penalty"] if h_inj["pos"] == "Defender/GK" else 1.0
        a_atk_mod = a_inj["penalty"] if a_inj["pos"] == "Striker/Winger" else 1.0
        a_def_mod = a_inj["penalty"] if a_inj["pos"] == "Defender/GK" else 1.0

        # Base xG Calculation (Ratio Clash)
        h_exp = (df_ratios.iloc[1,1] * h_atk_mod * (1/a_def_mod)) * intensity
        a_exp = (df_ratios.iloc[1,2] * a_atk_mod * (1/h_def_mod)) * intensity
        
        # Monte Carlo Engine
        h_sim = np.random.poisson(h_exp, 10000)
        a_sim = np.random.poisson(a_exp, 10000)
        
        # Outcome Calculation
        p_h, p_x, p_a = np.mean(h_sim > a_sim), np.mean(h_sim == a_sim), np.mean(h_sim < a_sim)
        p_gg = np.mean((h_sim > 0) & (a_sim > 0))
        p_ov25 = np.mean((h_sim + a_sim) > 2.5)
        p_fh_ov = np.mean((h_sim + a_sim)*0.44 > 0.5)

        # Market List
        markets = [
            {"Market": f"{h_name} Win", "Prob": p_h, "Odds": odd_1},
            {"Market": "Draw", "Prob": p_x, "Odds": odd_x},
            {"Market": f"{a_name} Win", "Prob": p_a, "Odds": odd_2},
            {"Market": "GG (BTTS)", "Prob": p_gg, "Odds": odd_gg},
            {"Market": "Over 2.5 Goals", "Prob": p_ov25, "Odds": 1.95},
            {"Market": "1st Half Over 0.5", "Prob": p_fh_ov, "Odds": 1.45}
        ]

        for m in markets: m["Value"] = (m["Prob"] * m["Odds"]) - 1
        
        # Visual Results
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{h_name} Win %", f"{p_h:.1%}")
        m2.metric("Draw %", f"{p_x:.1%}")
        m3.metric(f"{a_name} Win %", f"{p_a:.1%}")
        
        final_df = pd.DataFrame(markets)
        
        # Display table with value identification
        st.table(final_df.style.format({"Prob": "{:.1%}", "Value": "{:.1%}"}))
        
        # HIGHEST VALUE SUMMARY
        best = final_df.loc[final_df['Value'].idxmax()]
        
        st.markdown("---")
        if best['Value'] > 0.05:
            st.success(f"💎 **HIGHSTAKES SUMMARY:** Highest Value detected in **{best['Market']}**. Edge: **{best['Value']:.1%}**.")
        else:
            st.warning("⚠️ **SUMMARY:** No high-value edges detected. Market is efficient (Risky).")
