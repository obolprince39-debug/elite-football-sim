import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# 🔧 CONFIG
# ==============================
st.set_page_config(layout="wide")
st.title("🛡️ HighStakes | Intelligent Match Engine")

# ==============================
# 🧠 UTIL FUNCTIONS
# ==============================
def normalize(v, max_v):
    return min(v / max_v, 1)

def f_score(f):
    clean = "".join([c for c in f.upper() if c in "WDL"])
    if not clean:
        return 1.0
    pts = sum({"W":1,"D":0.5,"L":0}.get(c,0.5) for c in clean)
    return np.clip(0.9 + (pts/len(clean))*0.2, 0.8, 1.2)

# ==============================
# 🧍 PLAYER IMPACT (LIGHT)
# ==============================
def player_attack_boost(xi_text):
    if not xi_text:
        return 0.0
    players = [p.strip() for p in xi_text.split(",") if p.strip()]
    return min(len(players) * 0.01, 0.1)

# ==============================
# 🧠 FEATURE ENGINE
# ==============================
def build_features(team, opp, context, is_home):

    attack = (
        normalize(team["sot"],10)*0.6 +
        normalize(team["bc"],5)*1.2 -
        normalize(team["bcm"],5)*0.4 +
        normalize(team["gpg"],4)*0.8
    )

    defense = (
        normalize(opp["con_pg"],3)*1.0 +
        (1 - normalize(opp["cs"],20))*0.8
    )

    tempo = (
        (team["pos"]/100)*0.6 +
        normalize(team["offsides"],5)*0.1 -
        normalize(team["fouls"],20)*0.05
    )

    context_factor = (
        context["form"]*0.5 +
        (1-context["inj"])*0.3 +
        (1+context["h2h"])*0.2
    )

    home = 1 if is_home else 0

    return np.array([attack, defense, tempo, context_factor, home])

# ==============================
# 🎯 MODEL PREDICTION
# ==============================
def predict(model, feat):
    return max(0.01, model.predict([feat])[0])

# ==============================
# 🎲 SIMULATION
# ==============================
@st.cache_data
def simulate_goals(h_xg, a_xg):
    factor = np.random.normal(1,0.12,10000)
    h = np.random.poisson(h_xg * factor)
    a = np.random.poisson(a_xg * factor)
    fh = np.random.poisson((h_xg + a_xg)*0.44,10000)
    return h, a, fh

@st.cache_data
def simulate_corners(h_c, a_c):
    factor = np.random.normal(1,0.1,10000)
    h = np.random.poisson(h_c * factor)
    a = np.random.poisson(a_c * factor)
    return h, a

# ==============================
# 📦 LOAD MODELS
# ==============================
try:
    goal_model = joblib.load("model.pkl")
except:
    goal_model = None

try:
    corner_model = joblib.load("corner_model.pkl")
except:
    corner_model = None

# ==============================
# 🏟️ INPUT FORM
# ==============================
with st.form("match_input"):

    col1, col2 = st.columns(2)

    def team_inputs(name):
        return {
            "sot": st.number_input(f"{name} SoT",0.0),
            "bc": st.number_input(f"{name} Big Chances",0.0),
            "bcm": st.number_input(f"{name} Big Chances Missed",0.0),
            "gpg": st.number_input(f"{name} Goals/Game",0.0),
            "pos": st.number_input(f"{name} Possession %",0.0,100.0),
            "offsides": st.number_input(f"{name} Offsides/Game",0.0),
            "fouls": st.number_input(f"{name} Fouls/Game",0.0),
            "con_pg": st.number_input(f"{name} Conceded/Game",0.0),
            "cs": st.number_input(f"{name} Clean Sheets",0.0)
        }

    with col1:
        h_name = st.text_input("Home Team", "ARSENAL")
        h_form = st.text_input("Form", "WWDLW")
        h_xi = st.text_area("Starting XI (comma separated)")
        h_stats = team_inputs(h_name)

    with col2:
        a_name = st.text_input("Away Team", "NEWCASTLE")
        a_form = st.text_input("Form ", "LDWLL")
        a_xi = st.text_area("Starting XI ")
        a_stats = team_inputs(a_name)

    st.subheader("Context")
    h_inj = st.slider("Home Injury Impact",0.0,0.6,0.1)
    a_inj = st.slider("Away Injury Impact",0.0,0.6,0.1)
    h2h = st.slider("H2H Edge",-0.5,0.5,0.0)

    submit = st.form_submit_button("🚀 Run Simulation")

# ==============================
# 🚀 EXECUTION
# ==============================
if submit:

    if goal_model is None:
        st.error("⚠️ Goal model not found. Upload model.pkl")
        st.stop()

    # Context
    h_ctx = {"form": f_score(h_form), "inj": h_inj, "h2h": h2h}
    a_ctx = {"form": f_score(a_form), "inj": a_inj, "h2h": -h2h}

    # Features
    h_feat = build_features(h_stats, a_stats, h_ctx, True)
    a_feat = build_features(a_stats, h_stats, a_ctx, False)

    # Predictions
    h_xg = predict(goal_model, h_feat)
    a_xg = predict(goal_model, a_feat)

    # Player boost
    h_xg *= (1 + player_attack_boost(h_xi))
    a_xg *= (1 + player_attack_boost(a_xi))

    # Simulate
    h_sim, a_sim, fh = simulate_goals(h_xg, a_xg)
    total = h_sim + a_sim

    # Display xG
    st.subheader("⚽ Expected Goals (xG)")
    st.write(f"{h_name}: {round(h_xg,2)} | {a_name}: {round(a_xg,2)}")

    # ==============================
    # 📊 MARKETS
    # ==============================
    markets = {
        "Home Win": np.mean(h_sim > a_sim),
        "Draw": np.mean(h_sim == a_sim),
        "Away Win": np.mean(h_sim < a_sim),
        "BTTS": np.mean((h_sim > 0) & (a_sim > 0)),
        "Over 0.5": np.mean(total > 0.5),
        "Over 1.5": np.mean(total > 1.5),
        "Over 2.5": np.mean(total > 2.5),
        "Over 3.5": np.mean(total > 3.5),
        "Over 4.5": np.mean(total > 4.5),
        "First Half Over 0.5": np.mean(fh > 0.5),
        "First Half Over 1.5": np.mean(fh > 1.5),
        "First Half Over 2.5": np.mean(fh > 2.5),
        "Double Chance 1X": np.mean(h_sim >= a_sim),
        "Double Chance X2": np.mean(a_sim >= h_sim),
    }

    df = pd.DataFrame(markets.items(), columns=["Market","Probability"])
    st.subheader("📊 Market Probabilities")
    st.dataframe(df.style.format({"Probability": "{:.2%}"}))

    # ==============================
    # 🚩 CORNERS
    # ==============================
    if corner_model:
        h_c = predict(corner_model, h_feat)
        a_c = predict(corner_model, a_feat)

        hc_sim, ac_sim = simulate_corners(h_c, a_c)
        corners = hc_sim + ac_sim

        st.subheader("🚩 Corners Market")
        st.write({
            "Over 9.5": float(np.mean(corners > 9.5)),
            "Under 9.5": float(np.mean(corners < 9.5))
        })
    else:
        st.warning("⚠️ Corner model not loaded")
