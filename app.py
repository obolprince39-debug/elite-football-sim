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

# --- SECTION 1: MATCH ARCHITECTURE ---
st.markdown("## 🏟️ Match Architecture")
col_a, col_b, col_c = st.columns([1, 1, 1.2])

with col_a:
    h_name = st.text_input("HOME TEAM", "ARSENAL").upper()
    h_form = st.text_input(f"{h_name} Form (e.g. WWDLW)", "WWDLW")

with col_b:
    a_name = st.text_input("AWAY TEAM", "NEWCASTLE").upper()
    a_form = st.text_input(f"{a_name} Form (e.g. LDWLL)", "LDWLL")

with col_c:
    m_seriousness = st.selectbox("Match Category / Intensity", 
        ["Friendly", "Standard League Match", "Local Derby", "Relegation Battle", "Title Decider", "Cup Final / Knockout"])
    h2h_bias = st.slider("H2H Advantage (1.0 = Neutral)", 0.8, 1.2, 1.0, 0.05)

# --- SECTION 2: THE INJURY CLINIC (RESTORED) ---
st.markdown("## 🏥 Medical & Injury Matrix")
inj_list = ["None", "Knock", "Hamstring", "ACL/Broken Bone"]
role_list = ["Main Goalscorer", "Creative Playmaker", "Defensive Mid", "Center Back", "Goalkeeper"]
imp_list = ["Key Player (Irreplaceable)", "Regular Starter", "Squad Rotation"]

def injury_row(team, idx):
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
    name = c1.text_input(f"Player {idx}", placeholder="Name", key=f"{team}_n{idx}")
    role = c2.selectbox("Role", role_list, key=f"{team}_p{idx}")
    inj = c3.selectbox("Injury", inj_list, key=f"{team}_t{idx}")
    imp = c4.selectbox("Importance", imp_list, key=f"{team}_i{idx}")
    
    # Logic: Severity * Importance
    sev = {"None": 0.0, "Knock": 0.05, "Hamstring": 0.15, "ACL/Broken Bone": 0.3}
    weight = {"Key Player (Irreplaceable)": 1.2, "Regular Starter": 1.0, "Squad Rotation": 0.5}
    return sev[inj] * weight[imp] if name else 0

h_pen, a_pen = 0, 0
with st.expander(f"🔴 {h_name} INJURIES"):
    for i in range(1, 4): h_pen += injury_row("H", i)
with st.expander(f"🔵 {a_name} INJURIES"):
    for i in range(1, 4): a_pen += injury_row("A", i)

# --- SECTION 3: TACTICAL ENGINE ---
st.markdown("## 📈 Tactical Data Engine")
r1, r2, r3, r4 = st.columns(4)
h_sot = r1.number_input(f"{h_name} SoT", value=5.0)
a_sot = r1.number_input(f"{a_name} SoT", value=4.5)
h_bc = r2.number_input(f"{h_name} Big Chances", value=2.0)
a_bc = r2.number_input(f"{a_name} Big Chances", value=1.5)
h_pos = r3.number_input(f"{h_name} Possession %", value=55.0)
a_pos = r3.number_input(f"{a_name} Possession %", value=45.0)
h_def = r4.number_input(f"{h_name} Def. Rating", value=0.8)
a_def = r4.number_input(f"{a_name} Def. Rating", value=1.4)

st.markdown("## 💰 Bookie Odds")
o1, o2, o3, o4, o5 = st.columns(5)
bk_1 = o1.number_input("Home (1)", value=1.45)
bk_x = o2.number_input("Draw (X)", value=4.50)
bk_2 = o3.number_input("Away (2)", value=6.50)
bk_ov25 = o4.number_input("Over 2.5", value=1.65)
bk_gg = o5.number_input("GG (BTTS)", value=1.75)

# --- EXECUTION ---
if st.button("🚀 EXECUTE 10,000 RUN SIMULATION"):
    # Form Multipliers
    def get_f(f): return 0.9 + ((sum({'W':1,'D':0.5,'L':0}.get(c, 0.5) for c in f.upper())/max(1,len(f)))*0.2)
    m_int = {"Friendly": 0.8, "Standard League Match": 1.0, "Local Derby": 1.2, "Relegation Battle": 1.15, "Title Decider": 1.3, "Cup Final / Knockout": 1.4}[m_seriousness]
    
    # The Core xG Formula (Restored with Penalty & Intensity)
    h_xg = ((h_sot * 0.16) + (h_bc * 0.42) + (h_pos * 0.005)) * get_f(h_form) * (1 - h_pen) * m_int * h2h_bias * (1/max(0.4, a_def))
    a_xg = ((a_sot * 0.16) + (a_bc * 0.42) + (a_pos * 0.005)) * get_f(a_form) * (1 - a_pen) * m_int * (1/max(0.4, h_def))
    
    h_sim = np.random.poisson(h_xg, 10000)
    a_sim = np.random.poisson(a_xg, 10000)
    tot = h_sim + a_sim
    half_tot = np.random.poisson((h_xg + a_xg) * 0.44, 10000)

    data = [
        {"Market": "Match Result (1)", "Prob": np.mean(h_sim > a_sim), "Bookie": bk_1},
        {"Market": "Match Result (X)", "Prob": np.mean(h_sim == a_sim), "Bookie": bk_x},
        {"Market": "Match Result (2)", "Prob": np.mean(h_sim < a_sim), "Bookie": bk_2},
        {"Market": "GG (BTTS)", "Prob": np.mean((h_sim > 0) & (a_sim > 0)), "Bookie": bk_gg},
        {"Market": "NG (No Goal)", "Prob": 1 - np.mean((h_sim > 0) & (a_sim > 0)), "Bookie": 1.90},
        {"Market": "3+ Goal Streak (Yes)", "Prob": np.mean((h_sim >= 3) | (a_sim >= 3)), "Bookie": 2.20},
        {"Market": "Double Chance (1X)", "Prob": np.mean(h_sim >= a_sim), "Bookie": 1.25},
        {"Market": "Double Chance (X2)", "Prob": np.mean(a_sim >= h_sim), "Bookie": 2.50},
        {"Market": "Double Chance (12)", "Prob": np.mean(h_sim != a_sim), "Bookie": 1.20},
        {"Market": "Corners Over 9.5", "Prob": 0.52 if (h_sot + a_sot) > 9 else 0.44, "Bookie": 1.85},
        {"Market": "Over 0.5 Goals", "Prob": np.mean(tot > 0.5), "Bookie": 1.05},
        {"Market": "Over 1.5 Goals", "Prob": np.mean(tot > 1.5), "Bookie": 1.25},
        {"Market": "Over 2.5 Goals", "Prob": np.mean(tot > 2.5), "Bookie": bk_ov25},
        {"Market": "Over 3.5 Goals", "Prob": np.mean(tot > 3.5), "Bookie": 2.80},
        {"Market": "Over 4.5 Goals", "Prob": np.mean(tot > 4.5), "Bookie": 5.50},
        {"Market": "Handicap (-1.5 Home)", "Prob": np.mean(h_sim - a_sim > 1.5), "Bookie": 2.20},
        {"Market": "1st Half Over 0.5", "Prob": np.mean(half_tot > 0.5), "Bookie": 1.40},
        {"Market": "1st Half Over 1.5", "Prob": np.mean(half_tot > 1.5), "Bookie": 2.80},
        {"Market": "1st Half Over 2.5", "Prob": np.mean(half_tot > 2.5), "Bookie": 7.00}
    ]

    df = pd.DataFrame(data)
    df["True Odds"] = df["Prob"].apply(lambda x: round(1/x, 2) if x > 0 else "High")
    df["Value %"] = df.apply(lambda r: round(((r['Prob'] * r['Bookie']) - 1) * 100, 1) if r['Bookie'] > 0 else 0, axis=1)
    
    st.table(df.style.format({"Prob": "{:.1%}"}))
    
    top_v = df.loc[df["Value %"].idxmax()]
    st.success(f"💎 **BEST VALUE:** {top_v['Market']} ({top_v['Value %']}% Edge)")
