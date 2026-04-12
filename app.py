import streamlit as st
import pandas as pd
import numpy as np

# ==============================
# 🔧 CONFIG + STYLE
# ==============================
st.set_page_config(page_title="HighStakes: Elite Command", layout="wide")

st.markdown("""
<style>
.main { background-color: #0b0e14; color: #e1e1e1; }
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background-color: #161b22 !important;
    color: white !important;
}
.stButton button {
    background: linear-gradient(135deg, #e63946, #a8222e);
    color: white;
    font-weight: bold;
    height: 3em;
    border-radius: 8px;
}
h2, h3 { color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ HighStakes | Elite Command Center")

# ==============================
# 🧠 FORM SCORE FUNCTION
# ==============================
def f_score(f):
    if not f:
        return 1.0
    clean = "".join([c for c in f.upper() if c in 'WDL'])
    if not clean:
        return 1.0

    pts = sum({'W':1,'D':0.5,'L':0}.get(c, 0.5) for c in clean)
    return np.clip(0.9 + ((pts/len(clean)) * 0.2), 0.8, 1.2)

# ==============================
# 🎯 xG MODEL (UPGRADED)
# ==============================
def compute_xg(sot, bc, pos, form, penalty, adv, intensity, opp_def):

    base = (sot * 0.16) + (bc * 0.44)
    interaction = (sot * bc) * 0.015
    pos_adj = pos * 0.003

    xg = base + interaction + pos_adj

    xg *= f_score(form)

    attack_penalty = penalty * 0.7
    defense_leak = penalty * 0.3

    xg *= (1 - attack_penalty)
    xg *= adv
    xg *= intensity

    xg *= (1 / (opp_def * (1 - defense_leak + 0.01)))

    return max(0.01, xg)

# ==============================
# 🎲 SIMULATION ENGINE
# ==============================
@st.cache_data
def run_simulation(h_xg, a_xg):
    game_factor = np.random.normal(1, 0.12, 10000)

    h_sim = np.random.poisson(h_xg * game_factor)
    a_sim = np.random.poisson(a_xg * game_factor)

    half_tot = np.random.poisson((h_xg + a_xg) * 0.44, 10000)

    return h_sim, a_sim, half_tot

# ==============================
# 🏟️ INPUT FORM
# ==============================
with st.form("match_setup"):

    st.markdown("## 🏟️ Match Setup")

    col1, col2 = st.columns(2)

    with col1:
        h_name = st.text_input("Home Team", "ARSENAL")
        h_form = st.text_input("Form (W/D/L)", "WWDLW")
        h_sot = st.number_input("Shots on Target", 0.0, value=5.1)
        h_bc = st.number_input("Big Chances", 0.0, value=2.4)
        h_pos = st.number_input("Possession %", 0.0, 100.0, value=55.0)
        h_def = st.number_input("Def Rating", 0.1, value=0.8)

    with col2:
        a_name = st.text_input("Away Team", "NEWCASTLE")
        a_form = st.text_input("Form (W/D/L)", "LDWLL")
        a_sot = st.number_input("Shots on Target ", 0.0, value=4.5)
        a_bc = st.number_input("Big Chances ", 0.0, value=1.8)
        a_pos = st.number_input("Possession % ", 0.0, 100.0, value=45.0)
        a_def = st.number_input("Def Rating ", 0.1, value=1.4)

    st.markdown("## 🏥 Injury Matrix")

    inj_sev = {"None": 0.0, "Knock": 0.05, "Hamstring": 0.15, "ACL": 0.35}
    imp_val = {"Key": 1.3, "Starter": 1.0, "Rotation": 0.5}

    def injury_input(team, key):
        penalty = 0.0
        st.write(f"{team} Injuries")
        for i in range(2):
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Player", key=f"{key}{i}")
            sev = c2.selectbox("Severity", list(inj_sev.keys()), key=f"{key}s{i}")
            imp = c3.selectbox("Impact", list(imp_val.keys()), key=f"{key}i{i}")
            if name:
                penalty += inj_sev[sev] * imp_val[imp]
        return min(penalty, 0.6)

    h_penalty = injury_input(h_name, "H")
    a_penalty = injury_input(a_name, "A")

    st.markdown("## ⚔️ Match Context")

    m_type = st.selectbox(
        "Match Type",
        ["Standard League", "Local Derby", "Cup Final", "Friendly"]
    )

    h_adv = st.number_input("Home Advantage", value=1.1)

    st.markdown("## 💰 Market Odds")

    col_odds = st.columns(5)
    bk_1 = col_odds[0].number_input("Home", value=1.45)
    bk_x = col_odds[1].number_input("Draw", value=4.5)
    bk_2 = col_odds[2].number_input("Away", value=6.5)
    bk_ov = col_odds[3].number_input("Over 2.5", value=1.65)
    bk_gg = col_odds[4].number_input("BTTS", value=1.75)

    submit = st.form_submit_button("🚀 RUN SIMULATION")

# ==============================
# 🚀 EXECUTION
# ==============================
if submit:

    m_int_map = {
        "Standard League": 1.0,
        "Local Derby": 1.15,
        "Cup Final": 1.3,
        "Friendly": 0.8
    }

    m_int = m_int_map[m_type]

    h_xg = compute_xg(h_sot, h_bc, h_pos, h_form, h_penalty, h_adv, m_int, a_def)
    a_xg = compute_xg(a_sot, a_bc, a_pos, a_form, a_penalty, 1.0, m_int, h_def)

    h_sim, a_sim, half_tot = run_simulation(h_xg, a_xg)
    total = h_sim + a_sim

    st.markdown("## 📊 Goal Distribution")

    h_dist = np.bincount(h_sim, minlength=6)[:6] / len(h_sim)
    a_dist = np.bincount(a_sim, minlength=6)[:6] / len(a_sim)

    st.bar_chart(pd.DataFrame({h_name: h_dist, a_name: a_dist}))

    st.markdown("## 💰 Market Analysis")

    markets = [
        {"Market": "Home Win", "Prob": np.mean(h_sim > a_sim), "Book": bk_1},
        {"Market": "Draw", "Prob": np.mean(h_sim == a_sim), "Book": bk_x},
        {"Market": "Away Win", "Prob": np.mean(h_sim < a_sim), "Book": bk_2},
        {"Market": "BTTS", "Prob": np.mean((h_sim > 0) & (a_sim > 0)), "Book": bk_gg},
        {"Market": "Over 2.5", "Prob": np.mean(total > 2.5), "Book": bk_ov},
    ]

    df = pd.DataFrame(markets)

    df["True Odds"] = df["Prob"].apply(lambda x: round(1/x,2) if x > 0 else None)
    df["Edge %"] = ((df["Prob"] * df["Book"]) - 1) * 100

    # ==============================
    # 🧠 CONFIDENCE + VALUE FILTER
    # ==============================
    df["Confidence"] = df["Prob"].apply(
        lambda x: "High" if x > 0.6 else "Medium" if x > 0.45 else "Low"
    )

    st.dataframe(df.style.format({"Prob": "{:.2%}", "Edge %": "{:.1f}"}))

    st.markdown("## 🎯 Value Detection Engine")

    val = df[(df["Edge %"] > 3) & (df["Prob"] > 0.5)]

    if not val.empty:
        best = val.loc[val["Edge %"].idxmax()]

        st.success(
            f"💎 BEST VALUE: {best['Market']} | Edge: {best['Edge %']:.1f}% | Confidence: {best['Confidence']}"
        )

        st.dataframe(
            val.sort_values(by="Edge %", ascending=False)
            .style.format({"Prob": "{:.2%}"})
        )

    else:
        st.warning("⚠️ No strong value detected")

    st.markdown("## 🧾 Model Insight")

    st.write(f"""
    Expected Goals:
    {h_name}: {round(h_xg,2)}  
    {a_name}: {round(a_xg,2)}
    """)
