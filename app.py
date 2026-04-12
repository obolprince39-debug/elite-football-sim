import streamlit as st
import pandas as pd
import numpy as np

# --- 1. SETUP & STYLING ---
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
    h2, h3 { color: #58a6ff; font-weight: 800; border-bottom: 2px solid #30363d; padding-bottom: 5px; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ HighStakes | Elite Command Center")

# --- SECTION 1: MATCH ARCHITECTURE ---
st.markdown("## 🏟️ Match Architecture")
col1, col2, col3 = st.columns([1, 1, 1.2])

with col1:
    h_name = st.text_input("HOME TEAM", "ARSENAL").upper()
    h_xi = st.text_area(f"{h_name} Starting XI", "Enter starters...", height=100)
    h_form = st.text_input(f"{h_name} Form (e.g. WWDLW)", "WWDLW")

with col2:
    a_name = st.text_input("AWAY TEAM", "NEWCASTLE").upper()
    a_xi = st.text_area(f"{a_name} Starting XI", "Enter starters...", height=100)
    a_form = st.text_input(f"{a_name} Form (e.g. LDWLL)", "LDWLL")

with col3:
    m_type = st.selectbox("Match Category", ["Standard League", "Local Derby", "Cup Final", "Friendly"])
    h_adv = st.number_input("Home Field Advantage", value=1.10, step=0.05)
    h2h_data = st.text_area("H2H Context", "History/Trends...", height=68)

# --- SECTION 2: THE INJURY CLINIC ---
st.markdown("## 🏥 Medical & Injury Matrix")
inj_sev = {"None": 0.0, "Knock": 0.05, "Hamstring": 0.15, "ACL/Broken Bone": 0.35}
imp_val = {"Key Player (Irreplaceable)": 1.3, "Regular Starter": 1.0, "Squad Rotation": 0.5}

def get_injury_penalty(team, t_key):
    penalty = 0.0
    with st.expander(f"🚑 {team} Injuries"):
        for i in range(1, 4):
            c1, c2, c3 = st.columns([2, 1, 1])
            name = c1.text_input(f"Player {i}", key=f"{t_key}_n{i}")
            sev = c2.selectbox("Severity", list(inj_sev.keys()), key=f"{t_key}_s{i}")
            imp = c3.selectbox("Importance", list(imp_val.keys()), key=f"{t_key}_i{i}")
            if name:
                penalty += inj_sev[sev] * imp_val[imp]
    return penalty

h_penalty = get_injury_penalty(h_name, "H")
a_penalty = get_injury_penalty(a_name, "A")

# --- SECTION 3: TACTICAL DATA ENGINE ---
st.markdown("## 📈 Tactical Data Engine")
r1, r2, r3, r4 = st.columns(4)
h_sot = r1.number_input(f"{h_name} SoT", value=5.1)
a_sot = r1.number_input(f"{a_name} SoT", value=4.5)
h_bc = r2.number_input(f"{h_name} Big Chances", value=2.4)
a_bc = r2.number_input(f"{a_name} Big Chances", value=1.8)
h_pos = r3.number_input(f"{h_name} Possession %", value=55.0)
a_pos = r3.number_input(f"{a_name} Possession %", value=45.0)
h_def = r4.number_input(f"{h_name} Def. Rating", value=0.8)
a_def = r4.number_input(f"{a_name} Def. Rating", value=1.4)

st.markdown("## 💰 Market Analysis")
o1, o2, o3, o4, o5 = st.columns(5)
bk_1 = o1.number_input("Odds: 1", value=1.45)
bk_x = o2.number_input("Odds: X", value=4.50)
bk_2 = o3.number_input("Odds: 2", value=6.50)
bk_ov25 = o4.number_input("Odds: Ov 2.5", value=1.65)
bk_gg = o5.number_input("Odds: GG", value=1.75)

# --- CACHED SIMULATION ENGINE ---
@st.cache_data
def run_simulation(h_xg, a_xg):
    h_sim = np.random.poisson(h_xg, 10000)
    a_sim = np.random.poisson(a_xg, 10000)
    half_tot = np.random.poisson((h_xg + a_xg) * 0.44, 10000)
    return h_sim, a_sim, half_tot

# --- EXECUTION ---
if st.button("🚀 EXECUTE 10,000 RUN SIMULATION"):
    def f_score(f): 
        if not f: return 1.0
        clean = f.upper().replace(" ", "")
        pts = sum({'W':1,'D':0.5,'L':0}.get(c, 0.5) for c in clean)
        # Clamped form score to prevent extreme outliers
        return np.clip(0.9 + ((pts/max(1,len(clean))) * 0.2), 0.8, 1.2)
    
    m_int = {"Standard League": 1.0, "Local Derby": 1.15, "Cup Final": 1.3, "Friendly": 0.8}[m_type]
    
    # PROTECTIVE LOGIC: max(0.01, ...) prevents Poisson from crashing on negative values
    h_xg_raw = ((h_sot * 0.16) + (h_bc * 0.44) + (h_pos * 0.005)) * f_score(h_form) * (1 - h_penalty) * h_adv * m_int * (1/max(0.5, a_def))
    a_xg_raw = ((a_sot * 0.16) + (a_bc * 0.44) + (a_pos * 0.005)) * f_score(a_form) * (1 - a_penalty) * m_int * (1/max(0.5, h_def))
    
    h_xg = max(0.01, h_xg_raw)
    a_xg = max(0.01, a_xg_raw)
    
    h_sim, a_sim, half_tot = run_simulation(h_xg, a_xg)
    tot = h_sim + a_sim

    markets = [
        {"Market": "1x2: Home (1)", "Prob": np.mean(h_sim > a_sim), "Book": bk_1},
        {"Market": "1x2: Draw (X)", "Prob": np.mean(h_sim == a_sim), "Book": bk_x},
        {"Market": "1x2: Away (2)", "Prob": np.mean(h_sim < a_sim), "Book": bk_2},
        {"Market": "GG (BTTS - Yes)", "Prob": np.mean((h_sim > 0) & (a_sim > 0)), "Book": bk_gg},
        {"Market": "NG (BTTS - No)", "Prob": 1 - np.mean((h_sim > 0) & (a_sim > 0)), "Book": 1.95},
        {"Market": "3+ Goal Streak (Yes)", "Prob": np.mean((h_sim >= 3) | (a_sim >= 3)), "Book": 2.10},
        {"Market": "Double Chance (1X)", "Prob": np.mean(h_sim >= a_sim), "Book": 1.25},
        {"Market": "Double Chance (X2)", "Prob": np.mean(a_sim >= h_sim), "Book": 2.40},
        {"Market": "Double Chance (12)", "Prob": np.mean(h_sim != a_sim), "Book": 1.22},
        {"Market": "Corners Over 9.5", "Prob": 0.52 if (h_sot+a_sot) > 9 else 0.45, "Book": 1.85},
        {"Market": "Over 0.5 Goals", "Prob": np.mean(tot > 0.5), "Book": 1.05},
        {"Market": "Over 1.5 Goals", "Prob": np.mean(tot > 1.5), "Book": 1.25},
        {"Market": "Over 2.5 Goals", "Prob": np.mean(tot > 2.5), "Book": bk_ov25},
        {"Market": "Over 3.5 Goals", "Prob": np.mean(tot > 3.5), "Book": 2.80},
        {"Market": "Over 4.5 Goals", "Prob": np.mean(tot > 4.5), "Book": 5.50},
        {"Market": "Handicap: Home (-1.5)", "Prob": np.mean(h_sim - a_sim > 1.5), "Book": 2.10},
        {"Market": "1st Half: Over 0.5", "Prob": np.mean(half_tot > 0.5), "Book": 1.40},
        {"Market": "1st Half: Over 1.5", "Prob": np.mean(half_tot > 1.5), "Book": 2.90},
        {"Market": "1st Half: Over 2.5", "Prob": np.mean(half_tot > 2.5), "Book": 7.00}
    ]

    df = pd.DataFrame(markets)
    # Safer True Odds calculation
    df["True Odds"] = df["Prob"].apply(lambda x: round(1/x, 2) if x > 0.001 else "INF")
    df["Edge %"] = df.apply(lambda r: round(((r['Prob'] * r['Book']) - 1) * 100, 1) if (r['Book'] > 0 and isinstance(r['True Odds'], float)) else 0, axis=1)

    st.table(df[['Market', 'Prob', 'True Odds', 'Edge %']].style.format({"Prob": "{:.1%}"}))
    
    val = df[df["Edge %"] > 0]
    if not val.empty:
        best = val.loc[val["Edge %"].idxmax()]
        st.success(f"💎 **BEST VALUE:** {best['Market']} ({best['Edge %']}% Edge)")
    else:
        st.warning("⚠️ No positive edge found in current markets.")
        {"Market": "Over 0.5 Goals", "Prob": np.mean(tot > 0.5), "Book": 1.05},
        {"Market": "Over 1.5 Goals", "Prob": np.mean(tot > 1.5), "Book": 1.25},
        {"Market": "Over 2.5 Goals", "Prob": np.mean(tot > 2.5), "Book": bk_ov25},
        {"Market": "Over 3.5 Goals", "Prob": np.mean(tot > 3.5), "Book": 2.80},
        {"Market": "Over 4.5 Goals", "Prob": np.mean(tot > 4.5), "Book": 5.50},
        {"Market": "Handicap: Home (-1.5)", "Prob": np.mean(h_sim - a_sim > 1.5), "Book": 2.10},
        {"Market": "1st Half: Over 0.5", "Prob": np.mean(half_tot > 0.5), "Book": 1.40},
        {"Market": "1st Half: Over 1.5", "Prob": np.mean(half_tot > 1.5), "Book": 2.90},
        {"Market": "1st Half: Over 2.5", "Prob": np.mean(half_tot > 2.5), "Book": 7.00}
    ]

    df = pd.DataFrame(markets)
    df["True Odds"] = df["Prob"].apply(lambda x: round(1/x, 2) if x > 0.01 else "High")
    df["Edge %"] = df.apply(lambda r: round(((r['Prob'] * r['Book']) - 1) * 100, 1) if r['Book'] > 0 else 0, axis=1)

    st.table(df[['Market', 'Prob', 'True Odds', 'Edge %']].style.format({"Prob": "{:.1%}"}))
    
    val = df[df["Edge %"] > 0]
    if not val.empty:
        best = val.loc[val["Edge %"].idxmax()]
        st.success(f"💎 **BEST VALUE:** {best['Market']} ({best['Edge %']}% Edge)")
