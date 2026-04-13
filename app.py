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
def get_api_data(team_name: str) -> Optional[Dict]:
    """Fetches real-time stats with safety check for missing form data"""
    if not team_name or not team_name.strip():
        return None
    
    try:
        api_key = st.secrets["FOOTBALL_API_KEY"]
        headers = {'X-Auth-Token': api_key}
        # Allow competition selection instead of hardcoded PL
        competition = st.session_state.get('competition', 'PL')
        url = f"https://api.football-data.org/v4/competitions/{competition}/standings"
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Check HTTP errors
        data = response.json()
        
        # Safer team matching - prefer exact match, fallback to contains
        matches = []
        search_name = team_name.lower().strip()
        
        for team in data['standings'][0]['table']:
            api_name = team['team']['name'].lower()
            if search_name == api_name:
                return extract_team_data(team)  # Exact match
            if search_name in api_name:
                matches.append((team, len(api_name) - len(search_name)))  # Track match quality
        
        # Return best partial match if no exact match
        if matches:
            matches.sort(key=lambda x: x[1])  # Sort by match quality
            return extract_team_data(matches[0][0])
            
    except requests.exceptions.RequestException:
        st.error("API Connection Error: Unable to fetch data. Check your connection or API key.")
    except (KeyError, IndexError) as e:
        st.error(f"API Data Error: Unexpected response format")
    except Exception:
        st.error("An unexpected error occurred while fetching data")
    
    return None

def extract_team_data(team: Dict) -> Dict:
    """Safely extract team data with division by zero protection"""
    played = team.get('playedGames', 0)
    goals_for = team.get('goalsFor', 0)
    goals_against = team.get('goalsAgainst', 0)
    
    raw_form = team.get('form', '')
    # Handle various form formats: "W,W,D,L,W" or "W W D L W" or "WWDLW"
    clean_form = ''.join([c for c in str(raw_form).upper() if c in "WDL"])
    if not clean_form:
        clean_form = "WWDLW"  # Default neutral form
    
    return {
        "gpg": goals_for / played if played > 0 else 0,
        "con_pg": goals_against / played if played > 0 else 0,
        "form": clean_form,
        "full_name": team['team']['name']
    }

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
    MIN_XG: float = 0.01  # Prevent zero/negative xG
    
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
    if not 0 <= stats.get("pos", 0) <= 100: 
        errors.append(f"{team_name}: Possession must be 0-100%")
    if stats.get("gpg", 0) > 5: 
        errors.append(f"{team_name}: Goals/game seems unrealistic (>5)")
    if stats.get("gpg", 0) < 0:
        errors.append(f"{team_name}: Goals/game cannot be negative")
    if stats.get("sot", 0) > 50:
        errors.append(f"{team_name}: Shots on target seems unrealistic (>50)")
    return errors

def normalize(v: float, max_v: float) -> float:
    if max_v <= 0: 
        return 0.0
    return float(np.clip(v / max_v, 0, 1))

def f_score(form_text: str) -> Tuple[float, List[str]]:
    if not form_text: 
        return 1.0, ["No form provided"]
    clean = "".join([c for c in str(form_text).upper() if c in "WDL"])
    if not clean: 
        return 1.0, ["No valid WDL characters found in form"]
    points = sum({"W": 3, "D": 1, "L": 0}.get(c, 0) for c in clean)
    form_ratio = points / (len(clean) * 3)
    score = 0.5 + (form_ratio * 0.8)
    return float(np.clip(score, 0.5, 1.3)), []

def player_attack_boost(xi_text: str, key_players: List[str] = None) -> Tuple[float, Dict]:
    metadata = {"total_players": 0, "key_players_found": 0, "found_names": []}
    if not xi_text or not xi_text.strip(): 
        return 0.0, metadata
    
    key_players = key_players or CONFIG.KEY_PLAYERS
    players = [p.strip().upper() for p in str(xi_text).split(",") if p.strip()]
    
    if not players:
        return 0.0, metadata
    
    found = []
    for p in players:
        for k in key_players:
            if k in p:
                found.append(p.title())
                break
    
    metadata["total_players"] = len(players)
    metadata["key_players_found"] = len(found)
    metadata["found_names"] = found
    
    # Base boost from squad depth + star player bonus
    depth_boost = min(len(players) * 0.005, 0.05)  # Cap depth contribution
    star_boost = len(found) * 0.02
    boost = min(depth_boost + star_boost, CONFIG.MAX_ATTACK_BOOST)
    
    return float(boost), metadata

# ==============================
# 🧠 FEATURE ENGINE
# ==============================
def build_goal_features(team: Dict, opp: Dict, context: Dict, is_home: bool) -> np.ndarray:
    attack = (
        normalize(team["sot"], 10) * 0.6 + 
        normalize(team["bc"], 5) * 1.2 - 
        normalize(team["bcm"], 5) * 0.4 + 
        normalize(team["gpg"], 4) * 0.8
    )
    defense = (
        normalize(opp["con_pg"], 3) * 1.0 + 
        (1 - normalize(opp["cs"], 20)) * 0.8
    )
    tempo = (
        (team["pos"] / 100) * 0.6 + 
        normalize(team["offsides"], 5) * 0.1 - 
        normalize(team["fouls"], 20) * 0.05
    )
    context_factor = (
        context["form"] * 0.5 + 
        (1 - context["inj"]) * 0.3 + 
        (1 + context["h2h"]) * 0.2
    )
    return np.array([
        attack, defense, tempo, context_factor, 
        1.0 if is_home else 0.0
    ], dtype=np.float32)

def build_corner_features(team: Dict, opp: Dict, is_home: bool) -> np.ndarray:
    volume = (
        normalize(team["sot"], 10) * 0.4 + 
        normalize(team["bc"], 5) * 0.3 + 
        normalize(team["pos"], 100) * 0.3
    )
    opp_defense = (
        normalize(opp["con_pg"], 3) * 0.4 + 
        normalize(opp["fouls"], 20) * 0.3 + 
        (1 - normalize(opp["cs"], 20)) * 0.3
    )
    wide_play = (
        normalize(team["offsides"], 5) * 0.5 + 
        normalize(team["fouls"], 20) * 0.2
    )
    return np.array([
        volume, opp_defense, wide_play, 
        1.0 if is_home else 0.0
    ], dtype=np.float32)

def predict(model, features: np.ndarray, model_name: str = "model") -> Tuple[float, Optional[str]]:
    if model is None: 
        return 0.0, f"{model_name} not loaded"
    try:
        pred = max(CONFIG.MIN_XG, float(model.predict([features])[0]))
        return pred, None
    except Exception as e: 
        return CONFIG.MIN_XG, str(e)

# ==============================
# 🎲 SIMULATION ENGINE
# ==============================
@st.cache_data(show_spinner=True)
def simulate_goals(h_xg: float, a_xg: float, n_sims: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Monte Carlo simulation of match outcomes
    Returns: (home_goals, away_goals, first_half_total_goals, metadata)
    """
    # Apply noise to xG values, ensuring they stay positive
    h_xg_noisy = h_xg * np.random.normal(1, CONFIG.GOAL_NOISE_STD, n_sims)
    a_xg_noisy = a_xg * np.random.normal(1, CONFIG.GOAL_NOISE_STD, n_sims)
    
    # Clip to prevent negative or zero lambdas for Poisson
    h_xg_noisy = np.clip(h_xg_noisy, CONFIG.MIN_XG, None)
    a_xg_noisy = np.clip(a_xg_noisy, CONFIG.MIN_XG, None)
    
    # Generate goal distributions
    h_goals = np.random.poisson(h_xg_noisy)
    a_goals = np.random.poisson(a_xg_noisy)
    
    # First half simulation (roughly 44% of total goals on average)
    total_xg = h_xg_noisy + a_xg_noisy
    fh_goals = np.random.poisson(total_xg * CONFIG.FH_GOAL_RATIO)
    
    metadata = {
        "h_ci": np.percentile(h_goals, [5, 95]),
        "a_ci": np.percentile(a_goals, [5, 95]),
        "h_mean": np.mean(h_goals),
        "a_mean": np.mean(a_goals),
        "fh_mean": np.mean(fh_goals)
    }
    
    return h_goals, a_goals, fh_goals, metadata

@st.cache_resource
def load_models():
    models = {}
    errors = []
    
    try: 
        models["goal"] = joblib.load("model.pkl")
        st.success("✅ Goal prediction model loaded")
    except Exception as e: 
        models["goal"] = None
        errors.append(f"Goal model: {e}")
    
    try: 
        models["corner"] = joblib.load("corner_model.pkl")
        st.success("✅ Corner prediction model loaded")
    except Exception as e: 
        models["corner"] = None
        errors.append(f"Corner model: {e}")
    
    if errors:
        st.warning(f"Some models failed to load. Running in fallback mode.")
    
    return models

# Initialize models
models = load_models()

# ==============================
# 🏟️ INTERFACE
# ==============================
st.set_page_config(layout="wide", page_title="HighStakes | Match Engine")
st.title("🛡️ HighStakes | Intelligent Match Engine")

# Competition selector
with st.expander("⚙️ Competition Settings"):
    competition = st.selectbox(
        "Select League",
        ["PL", "BL1", "SA", "PD", "FL1", "CL"],  # Premier League, Bundesliga, Serie A, La Liga, Ligue 1, Champions League
        format_func=lambda x: {
            "PL": "Premier League", "BL1": "Bundesliga", "SA": "Serie A",
            "PD": "La Liga", "FL1": "Ligue 1", "CL": "Champions League"
        }.get(x, x)
    )
    st.session_state['competition'] = competition

# API Data Sync Section
st.subheader("📡 API Data Sync")
api_col1, api_col2 = st.columns(2)

with api_col1:
    home_api_team = st.text_input("Home Team (API Search)", key="home_api_search")
    if st.button("Fetch Home Stats", key="fetch_home"):
        with st.spinner("Fetching..."):
            api_stats = get_api_data(home_api_team)
            if api_stats:
                st.success(f"Found: {api_stats['full_name']}")
                st.session_state['home_api_gpg'] = api_stats['gpg']
                st.session_state['home_api_con'] = api_stats['con_pg']
                st.session_state['home_api_form'] = api_stats['form']

with api_col2:
    away_api_team = st.text_input("Away Team (API Search)", key="away_api_search")
    if st.button("Fetch Away Stats", key="fetch_away"):
        with st.spinner("Fetching..."):
            api_stats = get_api_data(away_api_team)
            if api_stats:
                st.success(f"Found: {api_stats['full_name']}")
                st.session_state['away_api_gpg'] = api_stats['gpg']
                st.session_state['away_api_con'] = api_stats['con_pg']
                st.session_state['away_api_form'] = api_stats['form']

def team_inputs(name: str, is_home: bool, prefix: str):
    color = "🔴" if is_home else "🔵"
    st.markdown(f"**{color} {name}**")
    
    # Use team-specific session state keys
    gpg_key = f"{prefix}_api_gpg"
    con_key = f"{prefix}_api_con"
    form_key = f"{prefix}_api_form"
    
    return {
        "sot": st.number_input(f"Shots on Target", 0.0, 50.0, 4.5, key=f"{prefix}_sot"),
        "bc": st.number_input(f"Big Chances", 0.0, 20.0, 1.5, key=f"{prefix}_bc"),
        "bcm": st.number_input(f"Big Chances Missed", 0.0, 20.0, 0.8, key=f"{prefix}_bcm"),
        "gpg": st.number_input(
            f"Goals/Game", 0.0, 10.0, 
            st.session_state.get(gpg_key, 1.2), 
            key=f"{prefix}_gpg"
        ),
        "pos": st.number_input(f"Possession %", 0.0, 100.0, 50.0, key=f"{prefix}_pos"),
        "offsides": st.number_input(f"Offsides/Game", 0.0, 15.0, 2.0, key=f"{prefix}_off"),
        "fouls": st.number_input(f"Fouls/Game", 0.0, 35.0, 10.0, key=f"{prefix}_fouls"),
        "con_pg": st.number_input(
            f"Conceded/Game", 0.0, 10.0, 
            st.session_state.get(con_key, 1.0), 
            key=f"{prefix}_con"
        ),
        "cs": st.number_input(f"Clean Sheets", 0.0, 50.0, 5.0, key=f"{prefix}_cs")
    }

with st.form("match_input"):
    col1, col2 = st.columns(2)
    
    with col1:
        h_name = st.text_input("Home Team", "ARSENAL")
        h_form = st.text_input(
            "Home Form (W=Win, D=Draw, L=Loss)", 
            st.session_state.get("home_api_form", "WWDLW"),
            key="home_form_input"
        )
        h_xi = st.text_area("Starting XI (comma separated)", "Raya, Saliba, Gabriel, Rice, Odegaard, Saka", key="home_xi")
        h_stats = team_inputs(h_name, True, "home")
    
    with col2:
        a_name = st.text_input("Away Team", "NEWCASTLE")
        a_form = st.text_input(
            "Away Form (W=Win, D=Draw, L=Loss)", 
            st.session_state.get("away_api_form", "LDWLL"),
            key="away_form_input"
        )
        a_xi = st.text_area("Starting XI (comma separated)", "Pope, Schar, Guimaraes, Isak, Gordon", key="away_xi")
        a_stats = team_inputs(a_name, False, "away")
    
    st.markdown("---")
    
    # Context factors
    context_col1, context_col2, context_col3 = st.columns(3)
    with context_col1:
        h_inj = st.slider("Home Injury Impact", 0.0, 0.6, 0.1, help="0=No injuries, 0.6=Severe impact")
    with context_col2:
        a_inj = st.slider("Away Injury Impact", 0.0, 0.6, 0.1, help="0=No injuries, 0.6=Severe impact")
    with context_col3:
        h2h = st.slider("H2H Edge", -0.5, 0.5, 0.0, help="Positive favors home team")
    
    submit = st.form_submit_button("🚀 Run Simulation", use_container_width=True)

if submit:
    # Validation
    all_errors = validate_team_stats(h_stats, h_name) + validate_team_stats(a_stats, a_name)
    
    if all_errors:
        for error in all_errors:
            st.error(error)
        st.stop()
    
    # Check for model with fallback option
    if models["goal"] is None:
        st.error("❌ Goal prediction model not available. Please ensure 'model.pkl' exists in the app directory.")
        use_fallback = st.checkbox("Use fallback xG estimation (less accurate)", value=False)
        if not use_fallback:
            st.stop()
    
    # Calculate form scores
    h_f, h_f_warnings = f_score(h_form)
    a_f, a_f_warnings = f_score(a_form)
    
    for w in h_f_warnings + a_f_warnings:
        st.info(f"Form note: {w}")
    
    # Calculate player boosts
    h_b, h_b_meta = player_attack_boost(h_xi)
    a_b, a_b_meta = player_attack_boost(a_xi)
    
    with st.expander("📋 Squad Analysis"):
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.write(f"**{h_name}:** {h_b_meta['key_players_found']} key players")
            if h_b_meta['found_names']:
                st.write(", ".join(h_b_meta['found_names']))
        with col_meta2:
            st.write(f"**{a_name}:** {a_b_meta['key_players_found']} key players")
            if a_b_meta['found_names']:
                st.write(", ".join(a_b_meta['found_names']))
    
    # Predict xG
    if models["goal"] is not None:
        h_xg, h_err = predict(models["goal"], build_goal_features(h_stats, a_stats, {"form": h_f, "inj": h_inj, "h2h": h2h}, True), "home")
        a_xg, a_err = predict(models["goal"], build_goal_features(a_stats, h_stats, {"form": a_f, "inj": a_inj, "h2h": -h2h}, False), "away")
        
        if h_err or a_err:
            st.warning(f"Prediction warnings: {h_err or ''} {a_err or ''}")
    else:
        # Fallback xG estimation based on stats
        h_xg = (h_stats["gpg"] * 0.7 + h_stats["bc"] * 0.3) * (1 + h2h)
        a_xg = (a_stats["gpg"] * 0.7 + a_stats["bc"] * 0.3) * (1 - h2h)
        st.info("Using fallback xG estimation")
    
    # Apply attack boosts
    h_xg_final = max(CONFIG.MIN_XG, h_xg * (1 + h_b))
    a_xg_final = max(CONFIG.MIN_XG, a_xg * (1 + a_b))
    
    # Run simulation
    h_sim, a_sim, fh_sim, meta = simulate_goals(h_xg_final, a_xg_final)
    
    # Results display
    st.markdown("---")
    st.subheader("📊 Expected Goals (xG)")
    
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    res_col1.metric(f"{h_name} xG", f"{h_xg_final:.2f}", f"+{h_b:.0%} boost" if h_b > 0 else None)
    res_col2.metric(f"{a_name} xG", f"{a_xg_final:.2f}", f"+{a_b:.0%} boost" if a_b > 0 else None)
    res_col3.metric("Total xG", f"{h_xg_final + a_xg_final:.2f}")
    res_col4.metric("First Half xG", f"{meta['fh_mean']:.2f}")
    
    # Market probabilities
    st.subheader("🎯 Market Probabilities")
    
    # Calculate probabilities
    h_win_prob = np.mean(h_sim > a_sim)
    draw_prob = np.mean(h_sim == a_sim)
    a_win_prob = np.mean(h_sim < a_sim)
    over_2_5_prob = np.mean((h_sim + a_sim) > 2.5)
    btts_prob = np.mean((h_sim > 0) & (a_sim > 0))
    over_1_5_fh = np.mean(fh_sim > 1.5)
    
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown("**Match Result**")
        st.write(f"Home Win: **{h_win_prob:.1%}**")
        st.write(f"Draw: **{draw_prob:.1%}**")
        st.write(f"Away Win: **{a_win_prob:.1%}**")
        st.progress(h_win_prob)
    
    with m2:
        st.markdown("**Goals**")
        st.write(f"Over 2.5: **{over_2_5_prob:.1%}**")
        st.write(f"Under 2.5: **{1-over_2_5_prob:.1%}**")
        st.write(f"BTTS Yes: **{btts_prob:.1%}**")
        st.progress(over_2_5_prob)
    
    with m3:
        st.markdown("**First Half**")
        st.write(f"FH Over 1.5: **{over_1_5_fh:.1%}**")
        st.write(f"FH Under 1.5: **{1-over_1_5_fh:.1%}**")
        st.write(f"Expected FH Goals: **{meta['fh_mean']:.1f}**")
        st.progress(over_1_5_fh)
    
    # Score distribution
    with st.expander("📈 Detailed Score Distribution"):
        score_df = pd.DataFrame({
            f"{h_name} Goals": h_sim,
            f"{a_name} Goals": a_sim,
            "Total": h_sim + a_sim,
            "FH Total": fh_sim
        })
        
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            st.write("**Most Likely Scores**")
            score_counts = pd.crosstab(h_sim, a_sim)
            # Get top 5 most common scorelines
            flat_scores = score_counts.stack().reset_index()
            flat_scores.columns = [h_name, a_name, 'Probability']
            flat_scores['Probability'] = flat_scores['Probability'] / CONFIG.SIMULATION_COUNT
            top_scores = flat_scores.nlargest(5, 'Probability')
            st.dataframe(top_scores.style.format({'Probability': '{:.1%}'}))
        
        with dist_col2:
            st.write("**Goal Distribution Stats**")
            stats_summary = pd.DataFrame({
                'Metric': ['Mean', 'Median', '90th Percentile', '95% CI Lower', '95% CI Upper'],
                h_name: [
                    f"{meta['h_mean']:.2f}",
                    f"{np.median(h_sim):.0f}",
                    f"{np.percentile(h_sim, 90):.0f}",
                    f"{meta['h_ci'][0]:.0f}",
                    f"{meta['h_ci'][1]:.0f}"
                ],
                a_name: [
                    f"{meta['a_mean']:.2f}",
                    f"{np.median(a_sim):.0f}",
                    f"{np.percentile(a_sim, 90):.0f}",
                    f"{meta['a_ci'][0]:.0f}",
                    f"{meta['a_ci'][1]:.0f}"
                ]
            })
            st.dataframe(stats_summary.set_index('Metric'))
