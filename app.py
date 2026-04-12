import streamlit as st
import pandas as pd
import numpy as np

# --- 1. STUDIO-GRADE UI CONFIG ---
st.set_page_config(page_title="HighStakes: Elite Command", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e1e1e1; font-family: 'Inter', sans-serif; }
    .stTextInput>div>div>input, .stTextArea>div>textarea, .stSelectbox>div>div, .stNumberInput>div>div>input { 
        background-color: #161b22 !important; border: 1px solid #30363d !important; color: white !important; 
    }
    .stButton>button { 
        background: linear-gradient(135deg, #e63946, #a8222e); color: white; border: none; 
        font-weight: 700; height: 3.5em; border-radius: 8px; width: 100%; cursor: pointer;
    }
    h2, h3 { color: #58a6ff; font-weight: 800; border-bottom: 2px solid #30363d; padding-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ HighStakes | Elite Command Center")

# --- DATA INPUTS ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 🏠 Home Team Setup")
    h_name = st.text_input("Home Team Name", "ARSENAL").upper()
    h_sot = st.number_input(f"{h_name} Shots on Target (Avg)", value=5.0)
    h_bc = st.number_input(f"{h_name} Big Chances (Avg)", value=2.0)
    h_pos = st.slider(f"{h_name} Possession %", 30, 70, 55)
    h_def = st.number_input(f"{h_name} Defensive Rating (Goals Conceded Avg)", value=0.8, step=0.1)
    h_form = st.text_input(f"{h_name} Form (e.g., WWDLW)", "WWDLW")

with col2:
    st.markdown("## ✈️ Away Team Setup")
    a_name = st.text_input("Away Team Name", "NEWCASTLE").upper()
    a_sot = st.number_input(f"{a_name} Shots on Target (Avg)", value=4.5)
    a_bc = st.number_input(f"{a_name} Big Chances (Avg)", value=1.5)
    a_pos = 100 - h_pos
    st.info(f"{a_name} Possession: {a_pos}%")
    a_def = st.number_input(f"{a_name} Defensive Rating (Goals Conceded Avg)", value=1.4, step=0.1)
    a_form = st.text_input(f"{a_name} Form (e.g., LDWLL)", "LDWLL")

st.markdown("---")
st.markdown("## 💰 Bookie Odds (Enter 0.0 if unknown)")
o1, o2, o3, o4, o5 = st.columns(5)
bk_1 = o1.number_input("Home (1)", value=1.45)
bk_x = o2.number_input("Draw (X)", value=4.50)
bk_2 = o3.number_input("Away (2)", value=6.50)
bk_ov25 = o4.number_input("Over 2.5", value=1.65)
bk_gg = o5.number_input("GG (BTTS)", value=1.75)

# --- HELPER FUNCTIONS ---
def get_mult(form):
    if not form: return 1.0
    f = form.upper().replace(" ", "")
    pts = sum({'W':1.0, 'D':0.5, 'L':0.0}.get(c, 0.5) for c in f)
    return 0.9 + ((pts/len(f)) * 0.2) if len(f)>0 else 1.0

# --- SIMULATION ENGINE ---
if st.button("🚀 EXECUTE 10,000 RUN SIMULATION"):
    h_mult, a_mult = get_mult(h_form), get_mult(a_form)
    
    # Base xG Logic (Safe against Zero Division)
    h_xg = ((h_sot * 0.15) + (h_bc * 0.45) + (h_pos * 0.005)) * h_mult * (1/max(0.4, a_def))
    a_xg = ((a_sot * 0.15) + (a_bc * 0.45) + (a_pos * 0.005)) * a_mult * (1/max(0.4, h_def))
    
    h_sim = np.random.poisson(h_xg, 10000)
    a_sim = np.random.poisson(a_xg, 10000)
    tot = h_sim + a_sim
    half_tot = np.random.poisson((h_xg + a_xg) * 0.45, 10000)

    # Markets Dictionary
    results = [
        {"Market": "1x2: Home (1)", "Prob": np.mean(h_sim > a_sim), "Odds": bk_1},
        {"Market": "1x2: Draw (X)", "Prob": np.mean(h_sim == a_sim), "Odds": bk_x},
        {"Market": "1x2: Away (2)", "Prob": np.mean(h_sim < a_sim), "Odds": bk_2},
        {"Market": "GG (BTTS - Yes)", "Prob": np.mean((h_sim > 0) & (a_sim > 0)), "Odds": bk_gg},
        {"Market": "NG (BTTS - No)", "Prob": 1 - np.mean((h_sim > 0) & (a_sim > 0)), "Odds": 0.0},
        {"Market": "3+ Goal Streak (Yes)", "Prob": np.mean((h_sim >= 3) | (a_sim >= 3)), "Odds": 0.0},
        {"Market": "Double Chance: 1X", "Prob": np.mean(h_sim >= a_sim), "Odds": 0.0},
        {"Market": "Double Chance: X2", "Prob": np.mean(a_sim >= h_sim), "Odds": 0.0},
        {"Market": "Double Chance: 12", "Prob": np.mean(h_sim != a_sim), "Odds": 0.0},
        {"Market": "Corners Over 9.5", "Prob": 0.54 if (h_sot + a_sot) > 9 else 0.45, "Odds": 1.85},
        {"Market": "Over 0.5 Goals", "Prob": np.mean(tot > 0.5), "Odds": 0.0},
        {"Market": "Over 1.5 Goals", "Prob": np.mean(tot > 1.5), "Odds": 0.0},
        {"Market": "Over 2.5 Goals", "Prob": np.mean(tot > 2.5), "Odds": bk_ov25},
        {"Market": "Over 3.5 Goals", "Prob": np.mean(tot > 3.5), "Odds": 0.0},
        {"Market": "Over 4.5 Goals", "Prob": np.mean(tot > 4.5), "Odds": 0.0},
        {"Market": "Handicap: Home (-1.5)", "Prob": np.mean(h_sim - a_sim > 1.5), "Odds": 0.0},
        {"Market": "Handicap: Away (+1.5)", "Prob": np.mean(a_sim - h_sim > -1.5), "Odds": 0.0},
        {"Market": "1st Half: Over 0.5", "Prob": np.mean(half_tot > 0.5), "Odds": 0.0},
        {"Market": "1st Half: Over 1.5", "Prob": np.mean(half_tot > 1.5), "Odds": 0.0},
        {"Market": "1st Half: Over 2.5", "Prob": np.mean(half_tot > 2.5), "Odds": 0.0},
    ]

    df = pd.DataFrame(results)
    df["True Odds"] = df["Prob"].apply(lambda x: round(1/x, 2) if x > 0 else "N/A")
    df["Edge %"] = df.apply(lambda r: round(((r['Prob'] * r['Odds']) - 1) * 100, 1) if r['Odds'] > 0 else 0.0, axis=1)

    st.table(df[['Market', 'Prob', 'True Odds', 'Edge %']].style.format({"Prob": "{:.1%}"}))
    
    best_value = df[df["Edge %"] > 0]
    if not best_value.empty:
        top = best_value.loc[best_value["Edge %"].idxmax()]
        st.success(f"💎 **BEST VALUE:** {top['Market']} with a {top['Edge %']}% statistical edge.")
    else:
        st.warning("⚠️ No value found. The bookie odds are too efficient.")
