import streamlit as st
import pandas as pd
import numpy as np

# --- 1. STUDIO-GRADE UI CONFIG ---
st.set_page_config(page_title="HighStakes: Command", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e1e1e1; font-family: 'Inter', sans-serif; }
    .stTextInput>div>div>input, .stTextArea>div>textarea, .stSelectbox>div>div { 
        background-color: #161b22 !important; border: 1px solid #30363d !important; color: white !important; 
    }
    .stMetric { background-color: #161b22; border-radius: 12px; padding: 20px; border: 1px solid #30363d; }
    .stButton>button { 
        background: linear-gradient(135deg, #e63946, #a8222e); color: white; border: none; 
        font-weight: 700; height: 4em; border-radius: 10px; width: 100%; letter-spacing: 1px;
    }
    h2, h3 { color: #58a6ff; font-weight: 800; border-bottom: 2px solid #30363d; padding-bottom: 10px; margin-top: 40px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ HighStakes | Elite Simulation Command")

# --- SECTION 1: MATCH ARCHITECTURE ---
st.markdown("## 🏟️ Match Architecture")
col1, col2, col3 = st.columns([1, 1, 1.2])

with col1:
    h_name = st.text_input("HOME TEAM", placeholder="e.g. ARSENAL").upper()
    h_lineup = st.text_area("Starting XI", placeholder="Input players here...", height=150)

with col2:
    a_name = st.text_input("AWAY TEAM", placeholder="e.g. MAN CITY").upper()
    a_lineup = st.text_area("Starting XI ", placeholder="Input players here...", height=150)

with col3:
    league_type = st.selectbox("League / Competition", 
        ["English Premier League", "Spanish La Liga", "Italian Serie A", "German Bundesliga", "Ligue 1", "NPFL", "Champions League", "Europa League", "International"])
    
    m_seriousness = st.selectbox("Match Intensity / Seriousness", 
        ["Friendly / Pre-season", "Normal League Game", "Local Derby", "Relegation Battle", "Title Decider", "Cup Final / Knockout"])
    
    h2h_data = st.text_area("H2H History", placeholder="Last 3 meetings...", height=65)
    form_data = st.text_area("Form History", placeholder="Last 5 results (W-W-D-L-W)...", height=65)

# --- SECTION 2: THE INJURY CLINIC (FULL DROPDOWNS) ---
st.markdown("## 🏥 Medical & Injury Matrix")

# Pre-defined Lists
injury_list = [
    "None", "Knock / Bruise", "Hamstring Strain", "Quadriceps Tear", "Ankle Sprain", 
    "Knee Ligament (MCL/LCL)", "ACL Rupture", "Groin / Adductor", "Calf Strain", 
    "Broken Bone (Foot/Leg)", "Metatarsal", "Concussion", "Shoulder Dislocation"
]
role_list = ["Main Goalscorer", "Creative Playmaker", "Box-to-Box Mid", "Defensive Mid", "Fullback / Wingback", "Center Back", "Goalkeeper"]
impact_list = ["Key Player (Irreplaceable)", "Regular Starter", "Squad Rotation", "Bench / Impact Sub"]

def injury_row(team, idx):
    c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1.2, 1, 0.5])
    name = c1.text_input(f"Player {idx}", placeholder="Name", key=f"{team}_n{idx}")
    role = c2.selectbox("Role", role_list, key=f"{team}_p{idx}")
    inj = c3.selectbox("Injury Type", injury_list, key=f"{team}_t{idx}")
    imp = c4.selectbox("Impact", impact_list, key=f"{team}_i{idx}")
    is_starter = c5.checkbox("S11?", key=f"{team}_s{idx}")
    
    # Severity Math
    sev_map = {
        "None": 0.0, "Knock / Bruise": 0.1, "Hamstring Strain": 0.4, "Quadriceps Tear": 0.5, 
        "Ankle Sprain": 0.3, "Knee Ligament (MCL/LCL)": 0.6, "ACL Rupture": 1.0, 
        "Groin / Adductor": 0.4, "Calf Strain": 0.3, "Broken Bone (Foot/Leg)": 0.8, 
        "Metatarsal": 0.7, "Concussion": 0.5, "Shoulder Dislocation": 0.6
    }
    imp_map = {"Key Player (Irreplaceable)": 1.2, "Regular Starter": 1.0, "Squad Rotation": 0.7, "Bench / Impact Sub": 0.4}
    
    if name and is_starter:
        return sev_map[inj] * imp_map[imp]
    return 0

h_penalty, a_penalty = 0, 0
with st.expander(f"🔴 {h_name or 'HOME'} INJURY SQUAD (MAX 5)"):
    for i in range(1, 6): h_penalty += injury_row("H", i)

with st.expander(f"🔵 {a_name or 'AWAY'} INJURY SQUAD (MAX 5)"):
    for i in range(1, 6): a_penalty += injury_row("A", i)

# --- SECTION 3: TACTICAL DATA ENGINE ---
st.markdown("## 📈 Tactical Data Engine")
r1, r2, r3, r4 = st.columns(4)
h_sot = r1.number_input(f"{h_name or 'H'} SoT Ratio", value=0.0)
a_sot = r1.number_input(f"{a_name or 'A'} SoT Ratio", value=0.0)
h_bc = r2.number_input(f"{h_name or 'H'} Big Chances", value=0.0)
a_bc = r2.number_input(f"{a_name or 'A'} Big Chances", value=0.0)
h_pos = r3.number_input(f"{h_name or 'H'} Possession %", value=50.0)
a_pos = r3.number_input(f"{a_name or 'A'} Possession %", value=50.0)
h_def = r4.number_input(f"{h_name or 'H'} Defensive Rating", value=1.0)
a_def = r4.number_input(f"{a_name or 'A'} Defensive Rating", value=1.0)

# --- SECTION 4: ODDS & SIMULATION ---
st.markdown("## 💰 Market Analysis")
o1, o2, o3, o4 = st.columns(4)
bk_1 = o1.number_input("Bookie: 1", value=0.0)
bk_x = o2.number_input("Bookie: X", value=0.0)
bk_2 = o3.number_input("Bookie: 2", value=0.0)
bk_ov = o4.number_input("Bookie: Ov 2.5", value=0.0)

if st.button("🚀 EXECUTE 10,000 RUN SIMULATION"):
    # AUTO-CALCULATE xG BASE
    h_base_xg = (h_sot * 0.18) + (h_bc * 0.40) + (h_pos * 0.005)
    a_base_xg = (a_sot * 0.18) + (a_bc * 0.40) + (a_pos * 0.005)
    
    # APPLY INTENSITY & INJURY NERFS
    int_map = {"Friendly / Pre-season": 0.8, "Normal League Game": 1.0, "Local Derby": 1.2, "Relegation Battle": 1.15, "Title Decider": 1.3, "Cup Final / Knockout": 1.4}
    intensity = int_map[m_seriousness]
    
    # Final Penalty is a compound of severity and impact
    h_final = (h_base_xg * (1 - (h_penalty * 0.12)) * (1/a_def)) * intensity
    a_final = (a_base_xg * (1 - (a_penalty * 0.12)) * (1/h_def)) * intensity
    
    h_sim, a_sim = np.random.poisson(max(0.1, h_final), 10000), np.random.poisson(max(0.1, a_final), 10000)
    
    # PROBABILITIES & TRUE ODDS
    p1, px, p2 = np.mean(h_sim > a_sim), np.mean(h_sim == a_sim), np.mean(h_sim < a_sim)
    p_ov25, p_gg = np.mean((h_sim + a_sim) > 2.5), np.mean((h_sim > 0) & (a_sim > 0))

    data = [
        {"Market": "HOME WIN", "Prob": p1, "True Odds": 1/p1 if p1 > 0 else 0, "Bookie": bk_1},
        {"Market": "DRAW", "Prob": px, "True Odds": 1/px if px > 0 else 0, "Bookie": bk_x},
        {"Market": "AWAY WIN", "Prob": p2, "True Odds": 1/p2 if p2 > 0 else 0, "Bookie": bk_2},
        {"Market": "OVER 2.5 GOALS", "Prob": p_ov25, "True Odds": 1/p_ov25 if p_ov25 > 0 else 0, "Bookie": bk_ov},
        {"Market": "GG (BTTS)", "Prob": p_gg, "True Odds": 1/p_gg if p_gg > 0 else 0, "Bookie": 1.85}
    ]

    for d in data: d["Value"] = (d["Prob"] * d["Bookie"]) - 1 if d["Bookie"] > 0 else -1
    
    # DISPLAY
    st.markdown("### 🎯 Predictive Summary")
    st.table(pd.DataFrame(data).style.format({"Prob": "{:.1%}", "True Odds": "{:.2f}", "Value": "{:.1%}"}))
    
    best = pd.DataFrame(data).loc[pd.DataFrame(data)['Value'].idxmax()]
    if best['Value'] > 0.05:
        st.success(f"💎 **HIGHSTAKES EDGE:** Found in **{best['Market']}**. Prob: {best['Prob']:.1%} | Value: {best['Value']:.1%}")
    else:
        st.warning("⚠️ **MARKET ALERT:** No significant mathematical edge detected.")
