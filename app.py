import streamlit as st
import pandas as pd
import numpy as np

# --- 1. SETTINGS & THEME ---
st.set_page_config(page_title="Truthful Analytics Pro", layout="wide")

# Theme Toggle
mode = st.sidebar.radio("UI Theme", ["Dark", "Light"])
if mode == "Dark":
    st.markdown("<style>.main {background-color: #0e1117; color: white; padding: 20px;} .stDataEditor {border: 1px solid #444;}</style>", unsafe_allow_html=True)

# --- 2. THE ENGINE (MONTE CARLO) ---
def run_simulation(h_exp, a_exp, rounds=10000):
    return np.random.poisson(h_exp, rounds), np.random.poisson(a_exp, rounds)

# --- 3. UI LAYOUT ---
st.title("⚽ Truthful Football Analytics v3.5 (Manual Elite)")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Data & Ratios", "📋 Lineup & Injuries", "🎯 Simulated Outcomes"])

with tab1:
    # VOLUME STATS (NON-RATIO RAW TOTALS)
    st.subheader("📁 Season Volume Totals")
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.write("**Home Team**")
        h_vol = {
            "GS": st.number_input("Total Goals Scored (H)", 50),
            "GC": st.number_input("Total Goals Conceded (H)", 25),
            "CS": st.number_input("Total Clean Sheets (H)", 12),
            "AS": st.number_input("Total Assists (H)", 40),
            "SV": st.number_input("Saves per Game (H)", 3.5)
        }
    with v_col2:
        st.write("**Away Team**")
        a_vol = {
            "GS": st.number_input("Total Goals Scored (A)", 35),
            "GC": st.number_input("Total Goals Conceded (A)", 45),
            "CS": st.number_input("Total Clean Sheets (A)", 6),
            "AS": st.number_input("Total Assists (A)", 25),
            "SV": st.number_input("Saves per Game (A)", 4.8)
        }

    st.divider()
    
    # THE 13 PERFORMANCE RATIOS
    st.subheader("📉 Tactical Ratio Editor")
    st.caption("Enter ratios relative to league average (e.g., 1.2 = 20% above average)")
    ratio_data = {
        "Metric": ["Goals/Game", "xG Ratio", "Shots on Target", "Big Chances", "BC Missed", "Possession", "Fouls", "Offsides", "Goal Kicks", "Yellow Cards", "Saves", "Corners", "Clean Sheet Ratio"],
        "Home Ratio": [1.8, 1.9, 1.6, 2.1, 0.7, 1.2, 1.0, 1.1, 0.8, 1.0, 1.0, 1.4, 1.2],
        "Away Ratio": [1.1, 1.0, 0.9, 1.1, 1.3, 0.8, 1.4, 0.7, 1.1, 1.5, 1.4, 0.8, 0.7]
    }
    df_ratios = st.data_editor(pd.DataFrame(ratio_data), use_container_width=True)

with tab2:
    st.subheader("🏟️ Match Context")
    st.text_area("Predicted Lineups", "Enter the 11 players here for reference...")
    st.text_area("Injury List / Missing Players", "List injuries here (Manual adjustment of ratios recommended)...")

with tab3:
    st.subheader("🎲 Market Odds & Value Detection")
    # GRID FOR ODDS
    o1, o2, o3, o4 = st.columns(4)
    odd_1 = o1.number_input("Home Win (1)", 2.10)
    odd_x = o2.number_input("Draw (X)", 3.40)
    odd_2 = o3.number_input("Away Win (2)", 3.80)
    odd_gg = o4.number_input("GG/BTTS", 1.85)
    
    o5, o6, o7, o8 = st.columns(4)
    odd_ov25 = o5.number_input("Over 2.5 Goals", 1.90)
    odd_un25 = o6.number_input("Under 2.5 Goals", 1.90)
    odd_dc = o7.number_input("Double Chance (1X)", 1.30)
    odd_fh = o8.number_input("1st Half Over 0.5", 1.45)

    if st.button("🚀 EXECUTE 10,000 TRUTHFUL SIMULATIONS"):
        # MATH: Calculate Expected Goals (xG)
        h_exp = df_ratios.iloc[0,1] * df_ratios.iloc[1,1] # Goals/Game * xG Ratio
        a_exp = df_ratios.iloc[0,2] * df_ratios.iloc[1,2]
        
        h_sim, a_sim = run_simulation(h_exp, a_exp)
        
        # PROBABILITIES
        res = [
            {"Market": "Home Win", "Prob": np.mean(h_sim > a_sim), "Odds": odd_1},
            {"Market": "Draw", "Prob": np.mean(h_sim == a_sim), "Odds": odd_x},
            {"Market": "Away Win", "Prob": np.mean(h_sim < a_sim), "Odds": odd_2},
            {"Market": "GG (Both Score)", "Prob": np.mean((h_sim > 0) & (a_sim > 0)), "Odds": odd_gg},
            {"Market": "NG (No Goal)", "Prob": 1 - np.mean((h_sim > 0) & (a_sim > 0)), "Odds": 1.95},
            {"Market": "Over 2.5", "Prob": np.mean((h_sim + a_sim) > 2.5), "Odds": odd_ov25},
            {"Market": "Under 2.5", "Prob": np.mean((h_sim + a_sim) < 2.5), "Odds": odd_un25},
            {"Market": "1st Half Over 0.5", "Prob": np.mean((h_sim + a_sim)*0.44 > 0.5), "Odds": odd_fh},
            {"Market": "Double Chance (1X)", "Prob": np.mean(h_sim >= a_sim), "Odds": odd_dc},
            {"Market": "3+ Goal Streak (Home)", "Prob": np.mean(h_sim >= 3), "Odds": 4.0},
        ]

        # Calculate Value
        for r in res:
            r["Value"] = (r["Prob"] * r["Odds"]) - 1
            
        final_df = pd.DataFrame(res)
        st.table(final_df.style.format({"Prob": "{:.1%}", "Value": "{:.1%}"}))
        
        # SUMMARY TEXT
        best = final_df.loc[final_df['Value'].idxmax()]
        st.markdown("---")
        st.success(f"🔥 **TRUTHFUL SUMMARY:** The highest value bet is **{best['Market']}** with a mathematical advantage of **{best['Value']:.1%}**.")

