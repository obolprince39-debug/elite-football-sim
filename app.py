import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ==============================
# 🔌 API ENGINE
# ==============================
def get_api_data(team_name):
    """Fetches real-time stats from football-data.org"""
    try:
        api_key = st.secrets["FOOTBALL_API_KEY"]
        headers = {'X-Auth-Token': api_key}
        url = "https://api.football-data.org/v4/competitions/PL/standings"
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        for team in data['standings'][0]['table']:
            if team_name.lower() in team['team']['name'].lower():
                return {
                    "gpg": team['goalsFor'] / team['playedGames'],
                    "con_pg": team['goalsAgainst'] / team['playedGames'],
                    "form": team['form'].replace(',', ''),
                    "full_name": team['team']['name']
                }
    except Exception as e:
        st.error(f"API Connection Error: {e}")
    return None

# ==============================
# 🧠 CONFIGURATION
# ==============================
@dataclass
class Config:
    KEY_PLAYERS: List[str] = None
    MAX_ATTACK_BOOST: float = 0.15
    SIMULATION_COUNT: int = 10000
    GOAL_NOISE_STD: float = 0.12
    CORNER_NOISE_STD: float = 0.10
    FH_GOAL_RATIO: float = 0.44
    
    def __post_init__(self):
        if self.KEY_PLAYERS is None:
            self.KEY_PLAYERS = [
                "SALAH", "HAALAND", "SANE", "OSIMHEN", "MBAPPE", "BELLINGHAM",
                "DE BRUYNE", "FODEN", "RICE", "ODEGAARD", "SAKA", "MARTINELLI"
            ]

CONFIG = Config()

# ==============================
# 🧠 VALIDATION & UTILS
# ==============================
def validate_team_stats(stats: Dict, team_name: str) -> List[str]:
    errors = []
    if not 0 <= stats.get("pos", 0) <= 100: errors.append(f"{team_name}: Possession must be 0-100%")
    if stats.get("gpg", 0) > 5: errors.append(f"{team_name}: Goals/game seems unrealistic")
    return errors

def normalize(v: float, max_v: float) -> float:
    if max_v <= 0: return 0.0
    return float(np.clip(v / max_v, 0, 1))

def f_score(form_text: str) -> Tuple[float, List[str]]:
    if not form_text: return 1.0, ["No form provided"]
    clean = "".join([c for c in form_text.upper() if c in "WDL"])
    if not clean: return 1.0, ["No valid WDL characters found"]
    points = sum({"W": 3, "D": 1, "L": 0}.get(c, 0) for c in clean)
    form_ratio = points / (len(clean) * 3)
    score = 0.5 + (form_ratio * 0.8)
    return float(np.clip(score, 0.5, 1.3)), []

def player_attack_boost(xi_text: str, key_players: List[str] = None) -> Tuple[float, Dict]:
    metadata = {"total_players": 0, "key_players_found": 0, "found_names": []}
    if not xi_text or not xi_text.strip(): return 0.0, metadata
    key_players = key_players or CONFIG.KEY_PLAYERS
    players = [p.strip().upper() for p in xi_text.split(",") if p.strip()]
    found = []
    for p in players:
        for k in key_players:
            if k in p:
                found.append(p.title())
                break
    boost = min((len(players) * 0.005) + (len(found) * 0.02), CONFIG.MAX_ATTACK_BOOST)
    return float(boost), {"key_players_found": len(found), "found_names": found}

# ==============================
# 🧠 FEATURE ENGINE
# ==============================
def build_goal_features(team: Dict, opp: Dict, context: Dict, is_home: bool) -> np.ndarray:
    attack = (normalize(team["sot"], 10) * 0.6 + normalize(team["bc"], 5) * 1.2 - normalize(team["bcm"], 5) * 0.4 + normalize(team["gpg"], 4) * 0.8)
    defense = (normalize(opp["con_pg"], 3) * 1.0 + (1 - normalize(opp["cs"], 20)) * 0.8)
    tempo = ((team["pos"] / 100) * 0.6 + normalize(team["offsides"], 5) * 0.1 - normalize(team["fouls"], 20) * 0.05)
    context_factor = (context["form"] * 0.5 + (1 - context["inj"]) * 0.3 + (1 + context["h2h"]) * 0.2)
    return np.array([attack, defense, tempo, context_factor, 1.0 if is_home else 0.0], dtype=np.float32)

def build_corner_features(team: Dict, opp: Dict, is_home: bool) -> np.ndarray:
    volume = (normalize(team["sot"], 10) * 0.4 + normalize(team["bc"], 5) * 0.3 + normalize(team["pos"], 100) * 0.3)
    opp_defense = (normalize(opp["con_pg"], 3) * 0.4 + normalize(opp["fouls"], 20) * 0.3 + (1 - normalize(opp["cs"], 20)) * 0.3)
    wide_play = normalize(team["offsides"], 5) * 0.5 + normalize(team["fouls"], 20) * 0.2
    return np.array([volume, opp_defense, wide_play, 1.0 if is_home else 0.0], dtype=np.float32)

def predict(model, features: np.ndarray, model_name: str = "model") -> Tuple[float, Optional[str]]:
    if model is None: 
        return 0.0, f"{model_name} not loaded"
    try:
        pred = max(0.01, float(model.predict([features])[0]))
        return pred, None
    except Exception as e: 
        return 0.0, str(e)

# ==============================
# 🎲 SIMULATION ENGINE
# ==============================
@st.cache_data(show_spinner=False)
def simulate_goals(h_xg: float, a_xg: float, n_sims: int = 10000):
    h_goals = np.random.poisson(h_xg * np.random.normal(1, CONFIG.GOAL_NOISE_STD, n_sims))
    a_goals = np.random.poisson(a_xg * np.random.normal(1, CONFIG.GOAL_NOISE_STD, n_sims))
    fh_goals = np.random.poisson((h_xg + a_xg) * CONFIG.FH_GOAL_RATIO, n_sims)
    return h_goals, a_goals, fh_goals, {"h_ci": np.percentile(h_goals, [5, 95]), "a_ci": np.percentile(a_goals, [5, 95]), "h_mean": np.mean(h_goals), "a_mean": np.mean(a_goals)}

@st.cache_resource
def load_models():
    models = {}
    try: 
        models["goal"] = joblib.load("model.pkl")
    except: 
        models["goal"] = None
    try: 
        models["corner"] = joblib.load("corner_model.pkl")
    except: 
        models["corner"] = None
    return models

models = load_models()

# ==============================
# 🏟️ INTERFACE
# ==============================
st.set_page_config(layout="wide", page_title="HighStakes | Match Engine")
st.title("🛡️ HighStakes | Intelligent Match Engine")

st.subheader("📡 API Data Sync")
target_team = st.text_input("Enter Team Name (e.g., Arsenal, Chelsea)")
if st.button("Fetch Team Stats"):
    with st.spinner("Fetching..."):
        api_stats = get_api_data(target_team)
        if api_stats:
            st.success(f"Found {target_team}!")
            st.session_state['api_gpg'] = api_stats['gpg']
            st.session_state['api_con'] = api_stats['con_pg']
            st.session_state['api_form'] = api_stats['form']

def team_inputs(name: str, is_home: bool):
    color = "🔴" if is_home else "🔵"
    st.markdown(f"**{color} {name}**")
    return {
        "sot": st.number_input(f"Shots on Target", 0.0, 20.0, 4.5, key=f"{name}_sot"),
        "bc": st.number_input(f"Big Chances", 0.0, 10.0, 1.5, key=f"{name}_bc"),
        "bcm": st.number_input(f"Big Chances Missed", 0.0, 10.0, 0.8, key=f"{name}_bcm"),
        "gpg": st.number_input(f"Goals/Game", 0.0, 5.0, st.session_state.get("api_gpg", 1.2), key=f"{name}_gpg"),
        "pos": st.number_input(f"Possession %", 0.0, 100.0, 50.0, key=f"{name}_pos"),
        "offsides": st.number_input(f"Offsides/Game", 0.0, 10.0, 2.0, key=f"{name}_off"),
        "fouls": st.number_input(f"Fouls/Game", 0.0, 25.0, 10.0, key=f"{name}_fouls"),
        "con_pg": st.number_input(f"Conceded/Game", 0.0, 5.0, st.session_state.get("api_con", 1.0), key=f"{name}_con"),
        "cs": st.number_input(f"Clean Sheets", 0.0, 20.0, 5.0, key=f"{name}_cs")
    }

with st.form("match_input"):
    col1, col2 = st.columns(2)
    with col1:
        h_name = st.text_input("Home Team", "ARSENAL")
        h_form = st.text_input("Home Form", st.session_state.get("api_form", "WWDLW"))
        h_xi = st.text_area("Starting XI", "Raya, Saliba, Gabriel, Rice, Odegaard, Saka")
        h_stats = team_inputs(h_name, True)
    with col2:
        a_name = st.text_input("Away Team", "NEWCASTLE")
        a_form = st.text_input("Away Form", st.session_state.get("api_form", "LDWLL"))
        a_xi = st.text_area("Starting XI", "Pope, Schar, Guimaraes, Isak, Gordon")
        a_stats = team_inputs(a_name, False)
    
    st.markdown("---")
    h_inj = st.slider("Home Injury Impact", 0.0, 0.6, 0.1)
    a_inj = st.slider("Away Injury Impact", 0.0, 0.6, 0.1)
    h2h = st.slider("H2H Edge", -0.5, 0.5, 0.0)
    submit = st.form_submit_button("🚀 Run Simulation", use_container_width=True)

if submit:
    all_errors = validate_team_stats(h_stats, h_name) + validate_team_stats(a_stats, a_name)
    if all_errors or models["goal"] is None:
        st.error("Error: Check inputs or ensure model.pkl is in the root folder.")
        st.stop()

    h_f = f_score(h_form)[0]
    a_f = f_score(a_form)[0]
    h_b = player_attack_boost(h_xi)[0]
    a_b = player_attack_boost(a_xi)[0]
    
    h_xg, _ = predict(models["goal"], build_goal_features(h_stats, a_stats, {"form": h_f, "inj": h_inj, "h2h": h2h}, True))
    a_xg, _ = predict(models["goal"], build_goal_features(a_stats, h_stats, {"form": a_f, "inj": a_inj, "h2h": -h2h}, False))
    
    h_xg *= (1 + h_b)
    a_xg *= (1 + a_b)
    
    h_sim, a_sim, fh_sim, meta = simulate_goals(h_xg, a_xg)
    
    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric(f"{h_name} xG", f"{h_xg:.2f}")
    res_col2.metric(f"{a_name} xG", f"{a_xg:.2f}")
    res_col3.metric("Total xG", f"{h_xg + a_xg:.2f}")

    st.subheader("📊 Market Probabilities")
    m1, m2 = st.columns(2)
    m1.write(f"**Home Win:** {np.mean(h_sim > a_sim):.1%}")
    m1.write(f"**Draw:** {np.mean(h_sim == a_sim):.1%}")
    m1.write(f"**Away Win:** {np.mean(h_sim < a_sim):.1%}")
    m2.write(f"**Over 2.5:** {np.mean((h_sim + a_sim) > 2.5):.1%}")
    m2.write(f"**BTTS:** {np.mean((h_sim > 0) & (a_sim > 0)):.1%}")
