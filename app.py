import streamlit as st
import pandas as pd
import numpy as np

# ==============================
# 🔧 CONFIG
# ==============================
st.set_page_config(
    page_title="HighStakes: Elite Command",
    layout="wide"
)

# ==============================
# 🎨 UI STYLING
# ==============================
st.markdown("""
<style>
.main {
    background-color: #0b0e14;
    color: #e1e1e1;
}
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
</style>
""", unsafe_allow_html=True)

st.title("🛡️ HighStakes | Elite Command Center")

# ==============================
# 🧠 HELPER FUNCTIONS
# ==============================

def form_score(form):
    if not form:
        return 1.0
    clean = "".join([c for c in form.upper() if c in "WDL"])
    if not clean:
        return 1.0

    pts = sum({"W": 1, "D": 0.5, "L": 0}[c] for c in clean)
    return np.clip(0.9 + (pts / len(clean)) * 0.2, 0.8, 1.2)


def calculate_xg(sot, bc, pos, form, penalty, defense, adv=1.0, intensity=1.0):
    base = (sot * 0.14) + (bc * 0.5)
    interaction = (sot * bc) * 0.02
    pos_adj = np.log1p(pos) * 0.02

    xg = (base + interaction + pos_adj)
    xg *= form_score(form)
    xg *= (1 - penalty)
    xg *= adv
    xg *= intensity
    xg *= (1 / defense)

    return max(0.01, xg)


@st.cache_data
def run_simulation(h_xg, a_xg):
    game_factor = np.random.normal(1, 0.1, 10000)

    h_sim = np.random.poisson(h_xg * game_factor)
    a_sim = np.random.poisson(a_xg * game_factor)

    return h_sim, a_sim


# ==============================
# 🏟️ INPUT FORM
# ==============================
with st.form("match_form"):

    st.subheader("🏟️ Match Setup")

    col1, col2 = st.columns(2)

    with col1:
        h_name = st.text_input("Home Team", "ARSENAL")
        h_form = st.text_input("Form (W/D/L)", "WWDLW")
        h_sot = st.number_input("Shots on Target", 0.0, value=5.0)
        h_bc = st.number_input("Big Chances", 0.0, value=2.0)
        h_pos = st.number_input("Possession %", 0.0, 100.0, value=55.0)
        h_def = st.number_input("Def Rating", 0.1, value=0.8)

    with col2:
        a_name = st.text_input("Away Team", "NEWCASTLE")
        a_form = st.text_input("Form (W/D/L)", "LDWLL")
        a_sot = st.number_input("Shots on Target ", 0.0, value=4.0)
        a_bc = st.number_input("Big Chances ", 0.0, value=1.5)
        a_pos = st.number_input("Possession % ", 0.0, 100.0, value=45.0)
        a_def = st.number_input("Def Rating ", 0.1, value=1.4)

    st.subheader("⚔️ Match Context")

    m_type = st.selectbox(
        "Match Type",
        ["Standard", "Derby", "Final", "Friendly"]
    )

    h_adv = st.number_input("Home Advantage", value=1.1)

    submit = st.form_submit_button("🚀 Run Simulation")


# ==============================
# ⚙️ EXECUTION
# ==============================
if submit:

    intensity_map = {
        "Standard": 1.0,
        "Derby": 1.15,
        "Final": 1.3,
        "Friendly": 0.8
    }

    intensity = intensity_map[m_type]

    h_xg = calculate_xg(
        h_sot, h_bc, h_pos, h_form, 0, a_def, h_adv, intensity
    )

    a_xg = calculate_xg(
        a_sot, a_bc, a_pos, a_form, 0, h_def, 1.0, intensity
    )

    h_sim, a_sim = run_simulation(h_xg, a_xg)

    total = h_sim + a_sim

    # ==============================
    # 📊 RESULTS
    # ==============================
    st.subheader("📊 Probability Distribution")

    h_dist = np.bincount(h_sim, minlength=6)[:6] / len(h_sim)
    a_dist = np.bincount(a_sim, minlength=6)[:6] / len(a_sim)

    df_chart = pd.DataFrame({
        h_name: h_dist,
        a_name: a_dist
    })

    st.bar_chart(df_chart)

    # ==============================
    # 📈 MARKETS
    # ==============================
    st.subheader("💰 Market Probabilities")

    markets = [
        ["Home Win", np.mean(h_sim > a_sim)],
        ["Draw", np.mean(h_sim == a_sim)],
        ["Away Win", np.mean(h_sim < a_sim)],
        ["BTTS", np.mean((h_sim > 0) & (a_sim > 0))],
        ["Over 2.5", np.mean(total > 2.5)]
    ]

    df = pd.DataFrame(markets, columns=["Market", "Probability"])

    df["True Odds"] = df["Probability"].apply(
        lambda x: round(1 / x, 2) if x > 0 else None
    )

    df["Confidence"] = df["Probability"].apply(
        lambda x: "High" if x > 0.6 else "Medium" if x > 0.4 else "Low"
    )

    st.dataframe(df.style.format({"Probability": "{:.2%}"}))

    # ==============================
    # 🧾 SUMMARY
    # ==============================
    st.subheader("🧾 Model Insight")

    st.write(f"""
    **Expected Goals**
    - {h_name}: {round(h_xg,2)}
    - {a_name}: {round(a_xg,2)}

    This simulation reflects tactical input, form momentum, and match intensity.
    """)
