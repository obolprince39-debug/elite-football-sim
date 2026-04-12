import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(layout="wide")
st.title("🛡️ HighStakes | Intelligent Match Engine")

# ==============================
# 🧠 UTIL
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
    players = xi_text.split(",")
    return min(len(players) * 0.01, 0.1)  # simple scaling

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
# 🎯 PREDICT
# ==============================
def predict(model, feat):
    return max(0.01, model.predict([feat])[0])

# ==============================
# 🎲 SIM
# ==============================
@st.cache_data
def simulate(h_xg, a_xg):
    factor = np.random.normal(1,0.12,10000)
    h = np.random.poisson(h_xg*factor)
    a = np.random.poisson(a_xg*factor)
    fh = np.random.poisson((h_xg+a_xg)*0.44,10000)
    return h,a,fh

@st.cache_data
def simulate_corners(h,a):
    f = np.random.normal(1,0.1,10000)
    return np.random.poisson(h*f), np.random.poisson(a*f)

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
# 🏟️ INPUT
# ==============================
with st.form("input"):

    col1,col2 = st.columns(2)

    def inputs(name):
        return {
            "sot": st.number_input(f"{name} SoT",0.0),
            "bc": st.number_input(f"{name} Big Chances",0.0),
            "bcm": st.number_input(f"{name} Missed Chances",0.0),
            "gpg": st.number_input(f"{name} Goals/Game",0.0),
            "pos": st.number_input(f"{name} Possession",0.0,100.0),
            "offsides": st.number_input(f"{name} Offsides",0.0),
            "fouls": st.number_input(f"{name} Fouls",0.0),
            "con_pg": st.number_input(f"{name} Conceded/Game",0.0),
            "cs": st.number_input(f"{name} Clean Sheets",0.0)
        }

    with col1:
        h_name = st.text_input("Home","ARSENAL")
        h_form = st.text_input("Form","WWDLW")
        h_xi = st.text_area("Starting XI (comma separated)")
        h_stats = inputs(h_name)

    with col2:
        a_name = st.text_input("Away","NEWCASTLE")
        a_form = st.text_input("Form ","LDWLL")
        a_xi = st.text_area("Starting XI ")
        a_stats = inputs(a_name)

    st.subheader("Context")
    h_inj = st.slider("Home Injury",0.0,0.6,0.1)
    a_inj = st.slider("Away Injury",0.0,0.6,0.1)
    h2h = st.slider("H2H",-0.5,0.5,0.0)

    submit = st.form_submit_button("Run")

# ==============================
# 🚀 EXECUTE
# ==============================
if submit and goal_model:

    h_ctx = {"form":f_score(h_form),"inj":h_inj,"h2h":h2h}
    a_ctx = {"form":f_score(a_form),"inj":a_inj,"h2h":-h2h}

    h_feat = build_features(h_stats,a_stats,h_ctx,True)
    a_feat = build_features(a_stats,h_stats,a_ctx,False)

    h_xg = predict(goal_model,h_feat)
    a_xg = predict(goal_model,a_feat)

    # player boost
    h_xg *= (1 + player_attack_boost(h_xi))
    a_xg *= (1 + player_attack_boost(a_xi))

    h_sim,a_sim,fh = simulate(h_xg,a_xg)
    total = h_sim + a_sim

    st.subheader("xG")
    st.write(f"{h_name}: {round(h_xg,2)} | {a_name}: {round(a_xg,2)}")

    # ==============================
    # 📊 MARKETS
    # ==============================
    markets = {
        "Home Win": np.mean(h_sim>a_sim),
        "Draw": np.mean(h_sim==a_sim),
        "Away Win": np.mean(h_sim<a_sim),
        "BTTS": np.mean((h_sim>0)&(a_sim>0)),
        "Over 0.5": np.mean(total>0.5),
        "Over 1.5": np.mean(total>1.5),
        "Over 2.5": np.mean(total>2.5),
        "Over 3.5": np.mean(total>3.5),
        "Over 4.5": np.mean(total>4.5),
        "First Half Over 0.5": np.mean(fh>0.5),
        "First Half Over 1.5": np.mean(fh>1.5),
        "First Half Over 2.5": np.mean(fh>2.5),
        "Double Chance 1X": np.mean(h_sim>=a_sim),
        "Double Chance X2": np.mean(a_sim>=h_sim),
    }

    df = pd.DataFrame(markets.items(),columns=["Market","Prob"])
    st.dataframe(df.style.format({"Prob":"{:.2%}"}))

    # ==============================
    # 🚩 CORNERS
    # ==============================
    if corner_model:
        hc = predict(corner_model,h_feat)
        ac = predict(corner_model,a_feat)

        hc_sim,ac_sim = simulate_corners(hc,ac)
        corners = hc_sim + ac_sim

        st.subheader("Corners")
        st.write({
            "Over 9.5": np.mean(corners>9.5),
            "Under 9.5": np.mean(corners<9.5)
        })        penalty = 0.0
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
