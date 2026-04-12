import streamlit as st
import pandas as pd
import numpy as np

# --- 1. STUDIO-GRADE UI CONFIG ---
st.set_page_config(page_title="HighStakes: Elite Command", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e1e1e1; font-family: 'Inter', sans-serif; }
    .stTextInput>div>div>input, .stTextArea>div>textarea, .stSelectbox>div>div { 
        background-color: #161b22 !important; border: 1px solid #30363d !important; color: white !important; 
    }
    .stButton>button { 
        background: linear-gradient(135deg, #e63946, #a8222e); color: white; border: none; 
        font-weight: 700; height: 4.5em; border-radius: 12px; width: 100%; letter-spacing: 1.5px;
    }
    h2, h3 { color: #58a6ff; font-weight: 800; border-bottom: 2px solid #30363d; padding-bottom: 10px; margin-top: 35px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ HighStakes | Elite Simulation Command")

# --- SECTION 1: MATCH ARCHITECTURE ---
st.markdown("## 🏟️ Match Architecture")
col1, col2, col3 = st.columns([1, 1, 1.2])

with col1:
    h_name = st.text_input("HOME TEAM", placeholder="e.g. ARSENAL").upper()
    h_lineup = st.text_area("Starting XI", placeholder="List starters...", height=150)

with col2:
    a_name = st.text_input("AWAY TEAM", placeholder="e.g. MAN CITY").upper()
    a_lineup = st.text_area("Starting XI ", placeholder="List starters...", height=150)

with col3:
    league_type = st.selectbox("League / Competition", 
        ["EPL", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "NPFL", "UCL", "UEL", "International"])
    m_seriousness = st.selectbox("Match Category", 
        ["Friendly", "League Match", "Local Derby", "Relegation Battle", "Title Decider", "Cup Final"])
    h2h_data = st.text_area("H2H History", placeholder="Previous meetings...", height=65)
    form_data = st.text_area("Form History", placeholder="Recent form...", height=65)

# --- SECTION 2: THE INJURY CLINIC ---
st.markdown("## 🏥 Medical & Injury Matrix")
inj_list = ["None", "Knock", "Hamstring", "Quadriceps", "Ankle Sprain", "ACL Rupture", "Broken Bone", "Concussion"]
role_list = ["Main Goalscorer", "Creative Playmaker", "Defensive Mid", "Center Back", "Goalkeeper"]
imp_list = ["Key Player (Irreplaceable)", "Regular Starter", "Squad Rotation"]

def injury_row(team, idx):
    c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1.2, 1, 0.5])
    name = c1.text_input(f"Name", placeholder=f"Player {idx}", key=f"{team}_n{idx}")
    role = c2.selectbox("Role", role_list, key=f"{team}_p{idx}")
    inj = c3.selectbox("Injury", inj_list, key=f"{team}_t{idx}")
    imp = c4.selectbox("Impact", imp_list, key=f"{team}_i{idx}")
    is_starter = c5.checkbox("S11?", key=f"{team}_s{idx}")
    
    sev = {"None": 0.0, "Knock": 0.1, "Hamstring": 0.4, "ACL Rupture": 1.0, "Broken Bone": 0.8, "Quadriceps": 0.5, "Ankle Sprain": 0.3, "Concussion": 0.5}
    weight = {"Key Player (Irreplaceable)": 1.2, "Regular Starter": 1.0, "Squad Rotation": 0.6}
    return sev[inj] * weight[imp] if name and is_starter else 0

h_pen, a_pen = 0, 0
with st.expander(f"🔴 {h_name or 'HOME'} INJURIES"):
    for i in range(1, 6): h_pen += injury_row("H", i)
with st.expander(f"🔵 {a_name or 'AWAY'} INJURIES"):
    for i in range(1, 6): a_pen += injury_row("A", i)

# --- SECTION 3: TACTICAL DATA ENGINE ---
st.markdown("## 📈 Tactical Data Engine")
r1, r2, r3, r4 = st.columns(4)
h_sot = r1.number_input(f"{h_name or 'H'} SoT Ratio", value=0.0)
a_sot = r1.number_input(f"{a_name or 'A'} SoT Ratio", value=0.0)
h_bc = r2.number_input(f"{h_name or 'H'} Big Chances", value=0.0)
a_bc = r2.number_input(f"{a_name or 'A'} Big Chances", value=0.0)
h_pos = r3.number_input(f"{h_name or 'H'} Possession %", value=50.0)
a_pos = r3.number_input(f"{a_name or 'A'} Possession %", value=50.0)
h_def = r4.number_input(f"{h_name or 'H'} Def. Rating", value=1.0)
a_def = r4.number_input(f"{a_name or 'A'} Def. Rating", value=1.0)

# --- SECTION 4: ODDS & SIMULATION ---
st.markdown("## 💰 Market Analysis")
o1, o2, o3, o4, o5 = st.columns(5)
bk_1 = o1.number_input("Bookie: 1", value=0.0)
bk_x = o2.number_input("Bookie: X", value=0.0)
bk_2 = o3.number_input("Bookie: 2", value=0.0)
bk_ov25 = o4.number_input("Bookie: Ov 2.5", value=0.0)
bk_gg = o5.number_input("Bookie: GG", value=0.0)

if st.button("🚀 EXECUTE 10,000 RUN SIMULATION"):
    h_base = ((h_sot * 0.18) + (h_bc * 0.40) + (h_pos * 0.005)) * (1 - (h_pen * 0.12)) * (1/a_def)
    a_base = ((a_sot * 0.18) + (a_bc * 0.40) + (a_pos * 0.005)) * (1 - (a_pen * 0.12)) * (1/h_def)
    
    m_int = {"Friendly": 0.8, "League Match": 1.0, "Local Derby": 1.2, "Relegation Battle": 1.15, "Title Decider": 1.3, "Cup Final": 1.4}[m_seriousness]
    h_final, a_final = h_base * m_int, a_base * m_int
    
    h_sim, a_sim = np.random.poisson(max(0.1, h_final), 10000), np.random.poisson(max(0.1, a_final), 10000)
    total_goals = h_sim + a_sim
    
    # CALCULATIONS
    data = [
        {"Market": "HOME WIN (1)", "Prob": np.mean(h_sim > a_sim), "Bookie": bk_1},
        {"Market": "DRAW (X)", "Prob": np.mean(h_sim == a_sim), "Bookie": bk_x},
        {"Market": "AWAY WIN (2)", "Prob": np.mean(h_sim < a_sim), "Bookie": bk_2},
        {"Market": "GG (BTTS)", "Prob": np.mean((h_sim > 0) & (a_sim > 0)), "Bookie": bk_gg},
        {"Market": "NG (No Goal)", "Prob": 1 - np.mean((h_sim > 0) & (a_sim > 0)), "Bookie": 1.90},
        {"Market": "Double Chance (1X)", "Prob": np.mean(h_sim >= a_sim), "Bookie": 1.30},
        {"Market": "Double Chance (X2)", "Prob": np.mean(a_sim >= h_sim), "Bookie": 1.40},
        {"Market": "Over 0.5 Goals", "Prob": np.mean(total_goals > 0.5), "Bookie": 1.05},
        {"Market": "Over 1.5 Goals", "Prob": np.mean(total_goals > 1.5), "Bookie": 1.25},
        {"Market": "Over 2.5 Goals", "Prob": np.mean(total_goals > 2.5), "Bookie": bk_ov25},
        {"Market": "Over 3.5 Goals", "Prob": np.mean(total_goals > 3.5), "Bookie": 3.20},
        {"Market": "Over 4.5 Goals", "Prob": np.mean(total_goals > 4.5), "Bookie": 5.50},
        {"Market": "3+ Goals (Home)", "Prob": np.mean(h_sim >= 3), "Bookie": 4.50},
        {"Market": "Handicap (-1.5 Home)", "Prob": np.mean((h_sim - 1.5) > a_sim), "Bookie": 3.80},
        {"Market": "Corners Over 9.5", "Prob": 1 if (h_sot + a_sot) *
        {"Market": f"{h_name} 3+ GOALS", "Prob": p_h3, "Bookie": 4.5},
        {"Market": "CORNERS OV 9.5", "Prob": 1 if p_corn > 9.5 else 0.4, "Bookie": 1.8}
    ]

    for d in data: 
        d["True Odds"] = 1/d["Prob"] if d["Prob"] > 0 else 0
        d["Value"] = (d["Prob"] * d["Bookie"]) - 1 if d["Bookie"] > 0 else -1
    
    st.table(pd.DataFrame(data).style.format({"Prob": "{:.1%}", "True Odds": "{:.2f}", "Value": "{:.1%}"}))
    best = pd.DataFrame(data).loc[pd.DataFrame(data)['Value'].idxmax()]
    st.success(f"💎 **HIGHSTAKES SUMMARY:** Edge in **{best['Market']}** ({best['Value']:.1%})")
