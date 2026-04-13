import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ==============================
# 🔧 CONFIG
# ==============================
st.set_page_config(layout="wide", page_title="HighStakes | Match Engine")
st.title("🛡️ HighStakes | Intelligent Match Engine")

# ==============================
# 🧠 CONFIGURATION
# ==============================
@dataclass
class Config:
    """Centralized configuration"""
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
# 🧠 VALIDATION
# ==============================
def validate_team_stats(stats: Dict, team_name: str) -> List[str]:
    """Validate input statistics"""
    errors = []
    
    if not 0 <= stats.get("pos", 0) <= 100:
        errors.append(f"{team_name}: Possession must be 0-100%")
    
    if stats.get("gpg", 0) > 5:
        errors.append(f"{team_name}: Goals/game ({stats['gpg']}) seems unrealistic")
    
    if stats.get("con_pg", 0) > 5:
        errors.append(f"{team_name}: Conceded/game ({stats['con_pg']}) seems unrealistic")
    
    if stats.get("sot", 0) > 20:
        errors.append(f"{team_name}: Shots on target ({stats['sot']}) seems unrealistic")
    
    if stats.get("bc", 0) > 10:
        errors.append(f"{team_name}: Big chances ({stats['bc']}) seems unrealistic")
    
    negative_fields = ["sot", "bc", "bcm", "gpg", "offsides", "fouls", "con_pg", "cs"]
    for field in negative_fields:
        if stats.get(field, 0) < 0:
            errors.append(f"{team_name}: {field} cannot be negative")
    
    return errors

# ==============================
# 🧠 UTIL FUNCTIONS
# ==============================
def normalize(v: float, max_v: float) -> float:
    """Safe normalization with bounds checking"""
    if max_v <= 0:
        return 0.0
    return float(np.clip(v / max_v, 0, 1))

def f_score(form_text: str) -> Tuple[float, List[str]]:
    """
    Calculate form score from WDL string.
    Returns: (score, warnings)
    """
    warnings = []
    
    if not form_text:
        return 1.0, ["No form provided, using neutral 1.0"]
    
    clean = "".join([c for c in form_text.upper() if c in "WDL"])
    invalid_chars = len(form_text) - len(clean)
    
    if invalid_chars > 0:
        warnings.append(f"Ignored {invalid_chars} invalid characters in form")
    
    if not clean:
        return 1.0, ["No valid WDL characters found, using neutral 1.0"]
    
    # Standard football points: 3 for win, 1 for draw, 0 for loss
    points = sum({"W": 3, "D": 1, "L": 0}.get(c, 0) for c in clean)
    max_points = len(clean) * 3
    form_ratio = points / max_points
    
    # Map 0-1 to 0.5-1.3 range (wider impact for form)
    score = 0.5 + (form_ratio * 0.8)
    
    return float(np.clip(score, 0.5, 1.3)), warnings

def player_attack_boost(xi_text: str, key_players: List[str] = None) -> Tuple[float, Dict]:
    """
    Calculate attack boost from starting XI with quality weighting.
    Returns: (boost, metadata)
    """
    metadata = {"total_players": 0, "key_players_found": 0, "boost_breakdown": ""}
    
    if not xi_text or not xi_text.strip():
        return 0.0, metadata
    
    key_players = key_players or CONFIG.KEY_PLAYERS
    players = [p.strip().upper() for p in xi_text.split(",") if p.strip()]
    
    metadata["total_players"] = len(players)
    
    if not players:
        return 0.0, metadata
    
    # Count key players
    key_count = 0
    found_names = []
    for player in players:
        for key in key_players:
            if key in player or player in key:
                key_count += 1
                found_names.append(player.title())
                break
    
    metadata["key_players_found"] = key_count
    metadata["found_names"] = found_names
    
    # Base boost: 0.5% per player + 2% per key player
    base_boost = len(players) * 0.005
    key_boost = key_count * 0.02
    
    total_boost = min(base_boost + key_boost, CONFIG.MAX_ATTACK_BOOST)
    
    metadata["boost_breakdown"] = f"Base: {base_boost:.1%} + Key: {key_boost:.1%}"
    
    return float(total_boost), metadata

# ==============================
# 🧠 FEATURE ENGINE
# ==============================
def build_goal_features(team: Dict, opp: Dict, context: Dict, is_home: bool) -> np.ndarray:
    """
    Build feature vector for goal prediction model.
    Features: [attack, defense, tempo, context_factor, home]
    """
    # Attack metrics
    attack = (
        normalize(team["sot"], 10) * 0.6 +
        normalize(team["bc"], 5) * 1.2 -
        normalize(team["bcm"], 5) * 0.4 +
        normalize(team["gpg"], 4) * 0.8
    )
    
    # Defense metrics (from opponent's perspective)
    defense = (
        normalize(opp["con_pg"], 3) * 1.0 +
        (1 - normalize(opp["cs"], 20)) * 0.8
    )
    
    # Tempo/control metrics
    tempo = (
        (team["pos"] / 100) * 0.6 +
        normalize(team["offsides"], 5) * 0.1 -
        normalize(team["fouls"], 20) * 0.05
    )
    
    # Context factor
    context_factor = (
        context["form"] * 0.5 +
        (1 - context["inj"]) * 0.3 +
        (1 + context["h2h"]) * 0.2
    )
    
    home = 1.0 if is_home else 0.0
    
    return np.array([attack, defense, tempo, context_factor, home], dtype=np.float32)

def build_corner_features(team: Dict, opp: Dict, is_home: bool) -> np.ndarray:
    """
    Build feature vector for corner prediction model.
    Corners have different dynamics than goals.
    Features: [volume, opp_defense, wide_play, home]
    """
    # Attacking volume (crosses, shots from wide)
    volume = (
        normalize(team["sot"], 10) * 0.4 +      # Shots create deflections/corners
        normalize(team["bc"], 5) * 0.3 +        # Big chances often from wide crosses
        normalize(team["pos"], 100) * 0.3      # Possession pressure
    )
    
    # Opponent defensive style
    opp_defense = (
        normalize(opp["con_pg"], 3) * 0.4 +      # Poor defense = clearances for corners
        normalize(opp["fouls"], 20) * 0.3 +      # Desperation defending
        (1 - normalize(opp["cs"], 20)) * 0.3   # Lack of clean sheets
    )
    
    # Wide play indicator (offsides suggest aggressive wide runs)
    wide_play = normalize(team["offsides"], 5) * 0.5 + normalize(team["fouls"], 20) * 0.2
    
    home = 1.0 if is_home else 0.0
    
    return np.array([volume, opp_defense, wide_play, home], dtype=np.float32)

# ==============================
# 🎯 MODEL PREDICTION
# ==============================
def predict(model, features: np.ndarray, model_name: str = "model") -> Tuple[float, Optional[str]]:
    """
    Safe prediction with validation.
    Returns: (prediction, error_message)
    """
    if model is None:
        return 0.0, f"{model_name} not loaded"
    
    # Validate feature count if model exposes feature names
    if hasattr(model, 'feature_names_in_'):
        expected = len(model.feature_names_in_)
        actual = len(features)
        if actual != expected:
            return 0.0, f"{model_name} expects {expected} features, got {actual}"
    
    try:
        pred = model.predict([features])[0]
        # Ensure positive prediction
        pred = max(0.01, float(pred))
        return pred, None
    except Exception as e:
        return 0.0, f"Prediction error: {str(e)}"

# ==============================
# 🎲 SIMULATION
# ==============================
@st.cache_data(show_spinner=False)
def simulate_goals(h_xg: float, a_xg: float, n_sims: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Monte Carlo simulation for goal outcomes.
    Returns: (home_goals, away_goals, first_half_goals, metadata)
    """
    # Random factors for each team
    h_factor = np.random.normal(1, CONFIG.GOAL_NOISE_STD, n_sims)
    a_factor = np.random.normal(1, CONFIG.GOAL_NOISE_STD, n_sims)
    
    # Poisson simulation
    h_goals = np.random.poisson(h_xg * h_factor)
    a_goals = np.random.poisson(a_xg * a_factor)
    
    # First half goals (typically ~44% of total)
    total_xg = h_xg + a_xg
    fh_goals = np.random.poisson(total_xg * CONFIG.FH_GOAL_RATIO, n_sims)
    
    metadata = {
        "h_ci": np.percentile(h_goals, [5, 95]),
        "a_ci": np.percentile(a_goals, [5, 95]),
        "h_mean": np.mean(h_goals),
        "a_mean": np.mean(a_goals),
        "fh_mean": np.mean(fh_goals)
    }
    
    return h_goals, a_goals, fh_goals, metadata

@st.cache_data(show_spinner=False)
def simulate_corners(h_c: float, a_c: float, n_sims: int = 10000) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Monte Carlo simulation for corner outcomes"""
    h_factor = np.random.normal(1, CONFIG.CORNER_NOISE_STD, n_sims)
    a_factor = np.random.normal(1, CONFIG.CORNER_NOISE_STD, n_sims)
    
    h_corners = np.random.poisson(h_c * h_factor)
    a_corners = np.random.poisson(a_c * a_factor)
    
    metadata = {
        "h_ci": np.percentile(h_corners, [5, 95]),
        "a_ci": np.percentile(a_corners, [5, 95]),
        "h_mean": np.mean(h_corners),
        "a_mean": np.mean(a_corners)
    }
    
    return h_corners, a_corners, metadata

# ==============================
# 📦 LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    """Load models with error handling"""
    models = {}
    
    try:
        models["goal"] = joblib.load("model.pkl")
        st.sidebar.success("✅ Goal model loaded")
    except Exception as e:
        st.sidebar.error(f"⚠️ Goal model error: {e}")
        models["goal"] = None
    
    try:
        models["corner"] = joblib.load("corner_model.pkl")
        st.sidebar.success("✅ Corner model loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Corner model not found: {e}")
        models["corner"] = None
    
    return models

models = load_models()

# ==============================
# 🏟️ INPUT FORM
# ==============================
with st.form("match_input"):
    st.markdown("### Match Setup")
    
    col1, col2 = st.columns(2)

    def team_inputs(name: str, is_home: bool):
        color = "🔴" if is_home else "🔵"
        st.markdown(f"**{color} {name}**")
        
        return {
            "sot": st.number_input(f"Shots on Target", 0.0, 20.0, 4.5, key=f"{name}_sot"),
            "bc": st.number_input(f"Big Chances", 0.0, 10.0, 1.5, key=f"{name}_bc"),
            "bcm": st.number_input(f"Big Chances Missed", 0.0, 10.0, 0.8, key=f"{name}_bcm"),
            "gpg": st.number_input(f"Goals/Game", 0.0, 5.0, 1.2, key=f"{name}_gpg"),
            "pos": st.number_input(f"Possession %", 0.0, 100.0, 50.0, key=f"{name}_pos"),
            "offsides": st.number_input(f"Offsides/Game", 0.0, 10.0, 2.0, key=f"{name}_off"),
            "fouls": st.number_input(f"Fouls/Game", 0.0, 25.0, 10.0, key=f"{name}_fouls"),
            "con_pg": st.number_input(f"Conceded/Game", 0.0, 5.0, 1.0, key=f"{name}_con"),
            "cs": st.number_input(f"Clean Sheets", 0.0, 20.0, 5.0, key=f"{name}_cs")
        }

    with col1:
        h_name = st.text_input("Home Team", "ARSENAL", key="home_name")
        h_form = st.text_input("Recent Form (W=Win, D=Draw, L=Loss)", "WWDLW", key="home_form")
        h_xi = st.text_area("Starting XI (comma separated)", "Raya, White, Saliba, Gabriel, Zinchenko, Rice, Odegaard, Havertz, Saka, Martinelli, Trossard", key="home_xi")
        h_stats = team_inputs(h_name, True)

    with col2:
        a_name = st.text_input("Away Team", "NEWCASTLE", key="away_name")
        a_form = st.text_input("Recent Form", "LDWLL", key="away_form")
        a_xi = st.text_area("Starting XI", "Pope, Trippier, Schar, Botman, Burn, Guimaraes, Tonali, Joelinton, Almiron, Isak, Gordon", key="away_xi")
        a_stats = team_inputs(a_name, False)

    st.markdown("---")
    st.subheader("Context Factors")
    
    ctx_col1, ctx_col2, ctx_col3 = st.columns(3)
    
    with ctx_col1:
        h_inj = st.slider("Home Injury Impact", 0.0, 0.6, 0.1, help="0=none, 0.6=severe")
        a_inj = st.slider("Away Injury Impact", 0.0, 0.6, 0.1)
    
    with ctx_col2:
        h2h = st.slider("H2H Edge (Home advantage)", -0.5, 0.5, 0.0, help="Positive favors home")
    
    with ctx_col3:
        weather = st.selectbox("Weather", ["Clear", "Rain", "Wind"], index=0)
        importance = st.selectbox("Match Importance", ["Regular", "Derby", "Title Decider", "Relegation"], index=0)

    submit = st.form_submit_button("🚀 Run Simulation", use_container_width=True)

# ==============================
# 🚀 EXECUTION
# ==============================
if submit:
    # Validation
    all_errors = validate_team_stats(h_stats, h_name) + validate_team_stats(a_stats, a_name)
    
    if all_errors:
        st.error("### ❌ Validation Errors")
        for error in all_errors:
            st.error(error)
        st.stop()
    
    if models["goal"] is None:
        st.error("⚠️ Goal prediction model required. Please upload model.pkl")
        st.stop()
    
    # Calculate form scores
    h_form_score, h_warnings = f_score(h_form)
    a_form_score, a_warnings = f_score(a_form)
    
    # Display warnings
    for w in h_warnings + a_warnings:
        st.warning(w)
    
    # Calculate player boosts
    h_boost, h_boost_meta = player_attack_boost(h_xi, CONFIG.KEY_PLAYERS)
    a_boost, a_boost_meta = player_attack_boost(a_xi, CONFIG.KEY_PLAYERS)
    
    # Build contexts
    h_ctx = {"form": h_form_score, "inj": h_inj, "h2h": h2h}
    a_ctx = {"form": a_form_score, "inj": a_inj, "h2h": -h2h}
    
    # Build features
    h_goal_feat = build_goal_features(h_stats, a_stats, h_ctx, True)
    a_goal_feat = build_goal_features(a_stats, h_stats, a_ctx, False)
    
    # Predict goals
    h_xg, h_err = predict(models["goal"], h_goal_feat, "Goal Model (Home)")
    a_xg, a_err = predict(models["goal"], a_goal_feat, "Goal Model (Away)")
    
    if h_err or a_err:
        st.error(f"Prediction error: {h_err or a_err}")
        st.stop()
    
    # Apply player boosts
    h_xg_raw, a_xg_raw = h_xg, a_xg
    h_xg *= (1 + h_boost)
    a_xg *= (1 + a_boost)
    
    # Run simulations
    h_sim, a_sim, fh_sim, sim_meta = simulate_goals(h_xg, a_xg, CONFIG.SIMULATION_COUNT)
    total_goals = h_sim + a_sim
    
    # ==============================
    # 📊 DISPLAY RESULTS
    # ==============================
    
    st.markdown("---")
    
    # xG Display with confidence intervals
    col_xg1, col_xg2, col_xg3 = st.columns([2, 2, 1])
    
    with col_xg1:
        st.metric(
            f"{h_name} xG", 
            f"{h_xg:.2f}",
            delta=f"+{h_boost:.1%} boost" if h_boost > 0 else None
        )
        st.caption(f"Raw: {h_xg_raw:.2f} | 90% CI: {sim_meta['h_ci'][0]:.0f}-{sim_meta['h_ci'][1]:.0f} goals")
        if h_boost_meta['key_players_found'] > 0:
            st.caption(f"⭐ Key players: {', '.join(h_boost_meta['found_names'][:3])}")
    
    with col_xg2:
        st.metric(
            f"{a_name} xG",
            f"{a_xg:.2f}",
            delta=f"+{a_boost:.1%} boost" if a_boost > 0 else None
        )
        st.caption(f"Raw: {a_xg_raw:.2f} | 90% CI: {sim_meta['a_ci'][0]:.0f}-{sim_meta['a_ci'][1]:.0f} goals")
        if a_boost_meta['key_players_found'] > 0:
            st.caption(f"⭐ Key players: {', '.join(a_boost_meta['found_names'][:3])}")
    
    with col_xg3:
        st.metric("Total xG", f"{h_xg + a_xg:.2f}")
        st.caption(f"Simulated mean: {sim_meta['h_mean'] + sim_meta['a_mean']:.1f}")
    
    # ==============================
    # 📊 MARKET PROBABILITIES
    # ==============================
    st.subheader("📊 Market Probabilities")
    
    markets = {
        "Match Result": {
            "Home Win": np.mean(h_sim > a_sim),
            "Draw": np.mean(h_sim == a_sim),
            "Away Win": np.mean(h_sim < a_sim),
        },
        "Goals": {
            "BTTS": np.mean((h_sim > 0) & (a_sim > 0)),
            "Over 0.5": np.mean(total_goals > 0.5),
            "Over 1.5": np.mean(total_goals > 1.5),
            "Over 2.5": np.mean(total_goals > 2.5),
            "Over 3.5": np.mean(total_goals > 3.5),
            "Under 2.5": np.mean(total_goals <= 2.5),
        },
        "First Half": {
            "FH Over 0.5": np.mean(fh_sim > 0.5),
            "FH Over 1.5": np.mean(fh_sim > 1.5),
            "FH Under 1.5": np.mean(fh_sim <= 1.5),
        },
        "Double Chance": {
            "1X (Home/Draw)": np.mean(h_sim >= a_sim),
            "X2 (Draw/Away)": np.mean(a_sim >= h_sim),
            "12 (No Draw)": np.mean(h_sim != a_sim),
        }
    }
    
    # Display markets in columns
    market_cols = st.columns(len(markets))
    
    for idx, (category, probs) in enumerate(markets.items()):
        with market_cols[idx]:
            st.markdown(f"**{category}**")
            df = pd.DataFrame(probs.items(), columns=["Market", "Prob"])
            df["Prob"] = df["Prob"].apply(lambda x: f"{x:.1%}")
            st.dataframe(df, hide_index=True, use_container_width=True)
    
    # ==============================
    # 🎯 CONFIDENCE METRIC
    # ==============================
    st.subheader("🎯 Prediction Confidence")
    
    outcome_probs = [markets["Match Result"]["Home Win"], 
                     markets["Match Result"]["Draw"], 
                     markets["Match Result"]["Away Win"]]
    
    # Calculate entropy-based confidence
    entropy = -sum(p * np.log(p + 1e-10) for p in outcome_probs if p > 0)
    max_entropy = np.log(3)
    confidence = 1 - (entropy / max_entropy)
    
    conf_col1, conf_col2 = st.columns([3, 1])
    
    with conf_col1:
        st.progress(confidence)
        st.caption(f"Model confidence: **{confidence:.0%}** (higher = more certain outcome)")
        
        if confidence > 0.7:
            st.success("High confidence prediction")
        elif confidence > 0.5:
            st.info("Moderate confidence - consider alternate markets")
        else:
            st.warning("Low confidence - unpredictable match")
    
    with conf_col2:
        # Most likely exact score
        score_probs = {}
        for h in range(6):
            for a in range(6):
                prob = np.mean((h_sim == h) & (a_sim == a))
                score_probs[f"{h}-{a}"] = prob
        
        top_score = max(score_probs, key=score_probs.get)
        st.metric("Most Likely Score", top_score, f"{score_probs[top_score]:.1%}")
    
    # ==============================
    # 🚩 CORNERS
    # ==============================
    if models["corner"]:
        st.subheader("🚩 Corners Market")
        
        # Build corner-specific features
        h_corner_feat = build_corner_features(h_stats, a_stats, True)
        a_corner_feat = build_corner_features(a_stats, h_stats, False)
        
        h_c, h_c_err = predict(models["corner"], h_corner_feat, "Corner Model (Home)")
        a_c, a_c_err = predict(models["corner"], a_corner_feat, "Corner Model (Away)")
        
        if not h_c_err and not a_c_err:
            hc_sim, ac_sim, c_meta = simulate_corners(h_c, a_c, CONFIG.SIMULATION_COUNT)
            total_corners = hc_sim + ac_sim
            
            c_col1, c_col2, c_col3 = st.columns(3)
            
            with c_col1:
                st.metric(f"{h_name} Corners", f"{h_c:.1f}")
                st.caption(f"90% CI: {c_meta['h_ci'][0]:.0f}-{c_meta['h_ci'][1]:.0f}")
            
            with c_col2:
                st.metric(f"{a_name} Corners", f"{a_c:.1f}")
                st.caption(f"90% CI: {c_meta['a_ci'][0]:.0f}-{c_meta['a_ci'][1]:.0f}")
            
            with c_col3:
                st.metric("Total Corners", f"{h_c + a_c:.1f}")
            
            corner_markets = {
                "Over 8.5": np.mean(total_corners > 8.5),
                "Over 9.5": np.mean(total_corners > 9.5),
                "Over 10.5": np.mean(total_corners > 10.5),
                "Under 9.5": np.mean(total_corners < 9.5),
                "Under 10.5": np.mean(total_corners < 10.5),
                "Home > Away": np.mean(hc_sim > ac_sim),
            }
            
            c_df = pd.DataFrame(corner_markets.items(), columns=["Market", "Probability"])
            st.dataframe(c_df.style.format({"Probability": "{:.1%}"}), use_container_width=True)
        else:
            st.error(f"Corner prediction failed: {h_c_err or a_c_err}")
    else:
        st.info("⚠️ Corner model not loaded - corner markets unavailable")
    
    # ==============================
    # 📥 EXPORT
    # ==============================
    st.markdown("---")
    
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "match": f"{h_name} vs {a_name}",
        "context": {
            "home_form": h_form,
            "away_form": a_form,
            "home_injuries": h_inj,
            "away_injuries": a_inj,
            "h2h_edge": h2h,
            "weather": weather,
            "importance": importance
        },
        "predictions": {
            "xg": {"home": round(h_xg, 3), "away": round(a_xg, 3)},
            "most_likely_score": top_score,
            "confidence": round(confidence, 3)
        },
        "markets": {
            cat: {k: round(v, 4) for k, v in probs.items()} 
            for cat, probs in markets.items()
        }
    }
    
    if models["corner"] and not h_c_err:
        export_data["predictions"]["corners"] = {
            "home": round(h_c, 2),
            "away": round(a_c, 2),
            "total": round(h_c + a_c, 2)
        }
    
    json_str = json.dumps(export_data, indent=2)
    
    col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 2])
    
    with col_dl1:
        st.download_button(
            "📥 Download JSON",
            json_str,
            f"prediction_{h_name}_{a_name}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    with col_dl2:
        # CSV export for markets
        csv_df = pd.DataFrame([
            {"Market": f"{cat} - {market}", "Probability": f"{prob:.2%}", "Decimal Odds": f"{1/prob:.2f}"}
            for cat, probs in markets.items()
            for market, prob in probs.items()
        ])
        st.download_button(
            "📊 Download CSV",
            csv_df.to_csv(index=False),
            f"markets_{h_name}_{a_name}.csv",
            mime="text/csv"
        )
    
    with col_dl3:
        st.caption(f"Simulation based on {CONFIG.SIMULATION_COUNT:,} Monte Carlo runs")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ==============================
# 🔧 CONFIG
# ==============================
st.set_page_config(layout="wide", page_title="HighStakes | Match Engine")
st.title("🛡️ HighStakes | Intelligent Match Engine")

# ==============================
# 🧠 CONFIGURATION
# ==============================
@dataclass
class Config:
    """Centralized configuration"""
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
# 🧠 VALIDATION
# ==============================
def validate_team_stats(stats: Dict, team_name: str) -> List[str]:
    """Validate input statistics"""
    errors = []
    
    if not 0 <= stats.get("pos", 0) <= 100:
        errors.append(f"{team_name}: Possession must be 0-100%")
    
    if stats.get("gpg", 0) > 5:
        errors.append(f"{team_name}: Goals/game ({stats['gpg']}) seems unrealistic")
    
    if stats.get("con_pg", 0) > 5:
        errors.append(f"{team_name}: Conceded/game ({stats['con_pg']}) seems unrealistic")
    
    if stats.get("sot", 0) > 20:
        errors.append(f"{team_name}: Shots on target ({stats['sot']}) seems unrealistic")
    
    if stats.get("bc", 0) > 10:
        errors.append(f"{team_name}: Big chances ({stats['bc']}) seems unrealistic")
    
    negative_fields = ["sot", "bc", "bcm", "gpg", "offsides", "fouls", "con_pg", "cs"]
    for field in negative_fields:
        if stats.get(field, 0) < 0:
            errors.append(f"{team_name}: {field} cannot be negative")
    
    return errors

# ==============================
# 🧠 UTIL FUNCTIONS
# ==============================
def normalize(v: float, max_v: float) -> float:
    """Safe normalization with bounds checking"""
    if max_v <= 0:
        return 0.0
    return float(np.clip(v / max_v, 0, 1))

def f_score(form_text: str) -> Tuple[float, List[str]]:
    """
    Calculate form score from WDL string.
    Returns: (score, warnings)
    """
    warnings = []
    
    if not form_text:
        return 1.0, ["No form provided, using neutral 1.0"]
    
    clean = "".join([c for c in form_text.upper() if c in "WDL"])
    invalid_chars = len(form_text) - len(clean)
    
    if invalid_chars > 0:
        warnings.append(f"Ignored {invalid_chars} invalid characters in form")
    
    if not clean:
        return 1.0, ["No valid WDL characters found, using neutral 1.0"]
    
    # Standard football points: 3 for win, 1 for draw, 0 for loss
    points = sum({"W": 3, "D": 1, "L": 0}.get(c, 0) for c in clean)
    max_points = len(clean) * 3
    form_ratio = points / max_points
    
    # Map 0-1 to 0.5-1.3 range (wider impact for form)
    score = 0.5 + (form_ratio * 0.8)
    
    return float(np.clip(score, 0.5, 1.3)), warnings

def player_attack_boost(xi_text: str, key_players: List[str] = None) -> Tuple[float, Dict]:
    """
    Calculate attack boost from starting XI with quality weighting.
    Returns: (boost, metadata)
    """
    metadata = {"total_players": 0, "key_players_found": 0, "boost_breakdown": ""}
    
    if not xi_text or not xi_text.strip():
        return 0.0, metadata
    
    key_players = key_players or CONFIG.KEY_PLAYERS
    players = [p.strip().upper() for p in xi_text.split(",") if p.strip()]
    
    metadata["total_players"] = len(players)
    
    if not players:
        return 0.0, metadata
    
    # Count key players
    key_count = 0
    found_names = []
    for player in players:
        for key in key_players:
            if key in player or player in key:
                key_count += 1
                found_names.append(player.title())
                break
    
    metadata["key_players_found"] = key_count
    metadata["found_names"] = found_names
    
    # Base boost: 0.5% per player + 2% per key player
    base_boost = len(players) * 0.005
    key_boost = key_count * 0.02
    
    total_boost = min(base_boost + key_boost, CONFIG.MAX_ATTACK_BOOST)
    
    metadata["boost_breakdown"] = f"Base: {base_boost:.1%} + Key: {key_boost:.1%}"
    
    return float(total_boost), metadata

# ==============================
# 🧠 FEATURE ENGINE
# ==============================
def build_goal_features(team: Dict, opp: Dict, context: Dict, is_home: bool) -> np.ndarray:
    """
    Build feature vector for goal prediction model.
    Features: [attack, defense, tempo, context_factor, home]
    """
    # Attack metrics
    attack = (
        normalize(team["sot"], 10) * 0.6 +
        normalize(team["bc"], 5) * 1.2 -
        normalize(team["bcm"], 5) * 0.4 +
        normalize(team["gpg"], 4) * 0.8
    )
    
    # Defense metrics (from opponent's perspective)
    defense = (
        normalize(opp["con_pg"], 3) * 1.0 +
        (1 - normalize(opp["cs"], 20)) * 0.8
    )
    
    # Tempo/control metrics
    tempo = (
        (team["pos"] / 100) * 0.6 +
        normalize(team["offsides"], 5) * 0.1 -
        normalize(team["fouls"], 20) * 0.05
    )
    
    # Context factor
    context_factor = (
        context["form"] * 0.5 +
        (1 - context["inj"]) * 0.3 +
        (1 + context["h2h"]) * 0.2
    )
    
    home = 1.0 if is_home else 0.0
    
    return np.array([attack, defense, tempo, context_factor, home], dtype=np.float32)

def build_corner_features(team: Dict, opp: Dict, is_home: bool) -> np.ndarray:
    """
    Build feature vector for corner prediction model.
    Corners have different dynamics than goals.
    Features: [volume, opp_defense, wide_play, home]
    """
    # Attacking volume (crosses, shots from wide)
    volume = (
        normalize(team["sot"], 10) * 0.4 +      # Shots create deflections/corners
        normalize(team["bc"], 5) * 0.3 +        # Big chances often from wide crosses
        normalize(team["pos"], 100) * 0.3      # Possession pressure
    )
    
    # Opponent defensive style
    opp_defense = (
        normalize(opp["con_pg"], 3) * 0.4 +      # Poor defense = clearances for corners
        normalize(opp["fouls"], 20) * 0.3 +      # Desperation defending
        (1 - normalize(opp["cs"], 20)) * 0.3   # Lack of clean sheets
    )
    
    # Wide play indicator (offsides suggest aggressive wide runs)
    wide_play = normalize(team["offsides"], 5) * 0.5 + normalize(team["fouls"], 20) * 0.2
    
    home = 1.0 if is_home else 0.0
    
    return np.array([volume, opp_defense, wide_play, home], dtype=np.float32)

# ==============================
# 🎯 MODEL PREDICTION
# ==============================
def predict(model, features: np.ndarray, model_name: str = "model") -> Tuple[float, Optional[str]]:
    """
    Safe prediction with validation.
    Returns: (prediction, error_message)
    """
    if model is None:
        return 0.0, f"{model_name} not loaded"
    
    # Validate feature count if model exposes feature names
    if hasattr(model, 'feature_names_in_'):
        expected = len(model.feature_names_in_)
        actual = len(features)
        if actual != expected:
            return 0.0, f"{model_name} expects {expected} features, got {actual}"
    
    try:
        pred = model.predict([features])[0]
        # Ensure positive prediction
        pred = max(0.01, float(pred))
        return pred, None
    except Exception as e:
        return 0.0, f"Prediction error: {str(e)}"

# ==============================
# 🎲 SIMULATION
# ==============================
@st.cache_data(show_spinner=False)
def simulate_goals(h_xg: float, a_xg: float, n_sims: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Monte Carlo simulation for goal outcomes.
    Returns: (home_goals, away_goals, first_half_goals, metadata)
    """
    # Random factors for each team
    h_factor = np.random.normal(1, CONFIG.GOAL_NOISE_STD, n_sims)
    a_factor = np.random.normal(1, CONFIG.GOAL_NOISE_STD, n_sims)
    
    # Poisson simulation
    h_goals = np.random.poisson(h_xg * h_factor)
    a_goals = np.random.poisson(a_xg * a_factor)
    
    # First half goals (typically ~44% of total)
    total_xg = h_xg + a_xg
    fh_goals = np.random.poisson(total_xg * CONFIG.FH_GOAL_RATIO, n_sims)
    
    metadata = {
        "h_ci": np.percentile(h_goals, [5, 95]),
        "a_ci": np.percentile(a_goals, [5, 95]),
        "h_mean": np.mean(h_goals),
        "a_mean": np.mean(a_goals),
        "fh_mean": np.mean(fh_goals)
    }
    
    return h_goals, a_goals, fh_goals, metadata

@st.cache_data(show_spinner=False)
def simulate_corners(h_c: float, a_c: float, n_sims: int = 10000) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Monte Carlo simulation for corner outcomes"""
    h_factor = np.random.normal(1, CONFIG.CORNER_NOISE_STD, n_sims)
    a_factor = np.random.normal(1, CONFIG.CORNER_NOISE_STD, n_sims)
    
    h_corners = np.random.poisson(h_c * h_factor)
    a_corners = np.random.poisson(a_c * a_factor)
    
    metadata = {
        "h_ci": np.percentile(h_corners, [5, 95]),
        "a_ci": np.percentile(a_corners, [5, 95]),
        "h_mean": np.mean(h_corners),
        "a_mean": np.mean(a_corners)
    }
    
    return h_corners, a_corners, metadata

# ==============================
# 📦 LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    """Load models with error handling"""
    models = {}
    
    try:
        models["goal"] = joblib.load("model.pkl")
        st.sidebar.success("✅ Goal model loaded")
    except Exception as e:
        st.sidebar.error(f"⚠️ Goal model error: {e}")
        models["goal"] = None
    
    try:
        models["corner"] = joblib.load("corner_model.pkl")
        st.sidebar.success("✅ Corner model loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Corner model not found: {e}")
        models["corner"] = None
    
    return models

models = load_models()

# ==============================
# 🏟️ INPUT FORM
# ==============================
with st.form("match_input"):
    st.markdown("### Match Setup")
    
    col1, col2 = st.columns(2)

    def team_inputs(name: str, is_home: bool):
        color = "🔴" if is_home else "🔵"
        st.markdown(f"**{color} {name}**")
        
        return {
            "sot": st.number_input(f"Shots on Target", 0.0, 20.0, 4.5, key=f"{name}_sot"),
            "bc": st.number_input(f"Big Chances", 0.0, 10.0, 1.5, key=f"{name}_bc"),
            "bcm": st.number_input(f"Big Chances Missed", 0.0, 10.0, 0.8, key=f"{name}_bcm"),
            "gpg": st.number_input(f"Goals/Game", 0.0, 5.0, 1.2, key=f"{name}_gpg"),
            "pos": st.number_input(f"Possession %", 0.0, 100.0, 50.0, key=f"{name}_pos"),
            "offsides": st.number_input(f"Offsides/Game", 0.0, 10.0, 2.0, key=f"{name}_off"),
            "fouls": st.number_input(f"Fouls/Game", 0.0, 25.0, 10.0, key=f"{name}_fouls"),
            "con_pg": st.number_input(f"Conceded/Game", 0.0, 5.0, 1.0, key=f"{name}_con"),
            "cs": st.number_input(f"Clean Sheets", 0.0, 20.0, 5.0, key=f"{name}_cs")
        }

    with col1:
        h_name = st.text_input("Home Team", "ARSENAL", key="home_name")
        h_form = st.text_input("Recent Form (W=Win, D=Draw, L=Loss)", "WWDLW", key="home_form")
        h_xi = st.text_area("Starting XI (comma separated)", "Raya, White, Saliba, Gabriel, Zinchenko, Rice, Odegaard, Havertz, Saka, Martinelli, Trossard", key="home_xi")
        h_stats = team_inputs(h_name, True)

    with col2:
        a_name = st.text_input("Away Team", "NEWCASTLE", key="away_name")
        a_form = st.text_input("Recent Form", "LDWLL", key="away_form")
        a_xi = st.text_area("Starting XI", "Pope, Trippier, Schar, Botman, Burn, Guimaraes, Tonali, Joelinton, Almiron, Isak, Gordon", key="away_xi")
        a_stats = team_inputs(a_name, False)

    st.markdown("---")
    st.subheader("Context Factors")
    
    ctx_col1, ctx_col2, ctx_col3 = st.columns(3)
    
    with ctx_col1:
        h_inj = st.slider("Home Injury Impact", 0.0, 0.6, 0.1, help="0=none, 0.6=severe")
        a_inj = st.slider("Away Injury Impact", 0.0, 0.6, 0.1)
    
    with ctx_col2:
        h2h = st.slider("H2H Edge (Home advantage)", -0.5, 0.5, 0.0, help="Positive favors home")
    
    with ctx_col3:
        weather = st.selectbox("Weather", ["Clear", "Rain", "Wind"], index=0)
        importance = st.selectbox("Match Importance", ["Regular", "Derby", "Title Decider", "Relegation"], index=0)

    submit = st.form_submit_button("🚀 Run Simulation", use_container_width=True)

# ==============================
# 🚀 EXECUTION
# ==============================
if submit:
    # Validation
    all_errors = validate_team_stats(h_stats, h_name) + validate_team_stats(a_stats, a_name)
    
    if all_errors:
        st.error("### ❌ Validation Errors")
        for error in all_errors:
            st.error(error)
        st.stop()
    
    if models["goal"] is None:
        st.error("⚠️ Goal prediction model required. Please upload model.pkl")
        st.stop()
    
    # Calculate form scores
    h_form_score, h_warnings = f_score(h_form)
    a_form_score, a_warnings = f_score(a_form)
    
    # Display warnings
    for w in h_warnings + a_warnings:
        st.warning(w)
    
    # Calculate player boosts
    h_boost, h_boost_meta = player_attack_boost(h_xi, CONFIG.KEY_PLAYERS)
    a_boost, a_boost_meta = player_attack_boost(a_xi, CONFIG.KEY_PLAYERS)
    
    # Build contexts
    h_ctx = {"form": h_form_score, "inj": h_inj, "h2h": h2h}
    a_ctx = {"form": a_form_score, "inj": a_inj, "h2h": -h2h}
    
    # Build features
    h_goal_feat = build_goal_features(h_stats, a_stats, h_ctx, True)
    a_goal_feat = build_goal_features(a_stats, h_stats, a_ctx, False)
    
    # Predict goals
    h_xg, h_err = predict(models["goal"], h_goal_feat, "Goal Model (Home)")
    a_xg, a_err = predict(models["goal"], a_goal_feat, "Goal Model (Away)")
    
    if h_err or a_err:
        st.error(f"Prediction error: {h_err or a_err}")
        st.stop()
    
    # Apply player boosts
    h_xg_raw, a_xg_raw = h_xg, a_xg
    h_xg *= (1 + h_boost)
    a_xg *= (1 + a_boost)
    
    # Run simulations
    h_sim, a_sim, fh_sim, sim_meta = simulate_goals(h_xg, a_xg, CONFIG.SIMULATION_COUNT)
    total_goals = h_sim + a_sim
    
    # ==============================
    # 📊 DISPLAY RESULTS
    # ==============================
    
    st.markdown("---")
    
    # xG Display with confidence intervals
    col_xg1, col_xg2, col_xg3 = st.columns([2, 2, 1])
    
    with col_xg1:
        st.metric(
            f"{h_name} xG", 
            f"{h_xg:.2f}",
            delta=f"+{h_boost:.1%} boost" if h_boost > 0 else None
        )
        st.caption(f"Raw: {h_xg_raw:.2f} | 90% CI: {sim_meta['h_ci'][0]:.0f}-{sim_meta['h_ci'][1]:.0f} goals")
        if h_boost_meta['key_players_found'] > 0:
            st.caption(f"⭐ Key players: {', '.join(h_boost_meta['found_names'][:3])}")
    
    with col_xg2:
        st.metric(
            f"{a_name} xG",
            f"{a_xg:.2f}",
            delta=f"+{a_boost:.1%} boost" if a_boost > 0 else None
        )
        st.caption(f"Raw: {a_xg_raw:.2f} | 90% CI: {sim_meta['a_ci'][0]:.0f}-{sim_meta['a_ci'][1]:.0f} goals")
        if a_boost_meta['key_players_found'] > 0:
            st.caption(f"⭐ Key players: {', '.join(a_boost_meta['found_names'][:3])}")
    
    with col_xg3:
        st.metric("Total xG", f"{h_xg + a_xg:.2f}")
        st.caption(f"Simulated mean: {sim_meta['h_mean'] + sim_meta['a_mean']:.1f}")
    
    # ==============================
    # 📊 MARKET PROBABILITIES
    # ==============================
    st.subheader("📊 Market Probabilities")
    
    markets = {
        "Match Result": {
            "Home Win": np.mean(h_sim > a_sim),
            "Draw": np.mean(h_sim == a_sim),
            "Away Win": np.mean(h_sim < a_sim),
        },
        "Goals": {
            "BTTS": np.mean((h_sim > 0) & (a_sim > 0)),
            "Over 0.5": np.mean(total_goals > 0.5),
            "Over 1.5": np.mean(total_goals > 1.5),
            "Over 2.5": np.mean(total_goals > 2.5),
            "Over 3.5": np.mean(total_goals > 3.5),
            "Under 2.5": np.mean(total_goals <= 2.5),
        },
        "First Half": {
            "FH Over 0.5": np.mean(fh_sim > 0.5),
            "FH Over 1.5": np.mean(fh_sim > 1.5),
            "FH Under 1.5": np.mean(fh_sim <= 1.5),
        },
        "Double Chance": {
            "1X (Home/Draw)": np.mean(h_sim >= a_sim),
            "X2 (Draw/Away)": np.mean(a_sim >= h_sim),
            "12 (No Draw)": np.mean(h_sim != a_sim),
        }
    }
    
    # Display markets in columns
    market_cols = st.columns(len(markets))
    
    for idx, (category, probs) in enumerate(markets.items()):
        with market_cols[idx]:
            st.markdown(f"**{category}**")
            df = pd.DataFrame(probs.items(), columns=["Market", "Prob"])
            df["Prob"] = df["Prob"].apply(lambda x: f"{x:.1%}")
            st.dataframe(df, hide_index=True, use_container_width=True)
    
    # ==============================
    # 🎯 CONFIDENCE METRIC
    # ==============================
    st.subheader("🎯 Prediction Confidence")
    
    outcome_probs = [markets["Match Result"]["Home Win"], 
                     markets["Match Result"]["Draw"], 
                     markets["Match Result"]["Away Win"]]
    
    # Calculate entropy-based confidence
    entropy = -sum(p * np.log(p + 1e-10) for p in outcome_probs if p > 0)
    max_entropy = np.log(3)
    confidence = 1 - (entropy / max_entropy)
    
    conf_col1, conf_col2 = st.columns([3, 1])
    
    with conf_col1:
        st.progress(confidence)
        st.caption(f"Model confidence: **{confidence:.0%}** (higher = more certain outcome)")
        
        if confidence > 0.7:
            st.success("High confidence prediction")
        elif confidence > 0.5:
            st.info("Moderate confidence - consider alternate markets")
        else:
            st.warning("Low confidence - unpredictable match")
    
    with conf_col2:
        # Most likely exact score
        score_probs = {}
        for h in range(6):
            for a in range(6):
                prob = np.mean((h_sim == h) & (a_sim == a))
                score_probs[f"{h}-{a}"] = prob
        
        top_score = max(score_probs, key=score_probs.get)
        st.metric("Most Likely Score", top_score, f"{score_probs[top_score]:.1%}")
    
    # ==============================
    # 🚩 CORNERS
    # ==============================
    if models["corner"]:
        st.subheader("🚩 Corners Market")
        
        # Build corner-specific features
        h_corner_feat = build_corner_features(h_stats, a_stats, True)
        a_corner_feat = build_corner_features(a_stats, h_stats, False)
        
        h_c, h_c_err = predict(models["corner"], h_corner_feat, "Corner Model (Home)")
        a_c, a_c_err = predict(models["corner"], a_corner_feat, "Corner Model (Away)")
        
        if not h_c_err and not a_c_err:
            hc_sim, ac_sim, c_meta = simulate_corners(h_c, a_c, CONFIG.SIMULATION_COUNT)
            total_corners = hc_sim + ac_sim
            
            c_col1, c_col2, c_col3 = st.columns(3)
            
            with c_col1:
                st.metric(f"{h_name} Corners", f"{h_c:.1f}")
                st.caption(f"90% CI: {c_meta['h_ci'][0]:.0f}-{c_meta['h_ci'][1]:.0f}")
            
            with c_col2:
                st.metric(f"{a_name} Corners", f"{a_c:.1f}")
                st.caption(f"90% CI: {c_meta['a_ci'][0]:.0f}-{c_meta['a_ci'][1]:.0f}")
            
            with c_col3:
                st.metric("Total Corners", f"{h_c + a_c:.1f}")
            
            corner_markets = {
                "Over 8.5": np.mean(total_corners > 8.5),
                "Over 9.5": np.mean(total_corners > 9.5),
                "Over 10.5": np.mean(total_corners > 10.5),
                "Under 9.5": np.mean(total_corners < 9.5),
                "Under 10.5": np.mean(total_corners < 10.5),
                "Home > Away": np.mean(hc_sim > ac_sim),
            }
            
            c_df = pd.DataFrame(corner_markets.items(), columns=["Market", "Probability"])
            st.dataframe(c_df.style.format({"Probability": "{:.1%}"}), use_container_width=True)
        else:
            st.error(f"Corner prediction failed: {h_c_err or a_c_err}")
    else:
        st.info("⚠️ Corner model not loaded - corner markets unavailable")
    
    # ==============================
    # 📥 EXPORT
    # ==============================
    st.markdown("---")
    
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "match": f"{h_name} vs {a_name}",
        "context": {
            "home_form": h_form,
            "away_form": a_form,
            "home_injuries": h_inj,
            "away_injuries": a_inj,
            "h2h_edge": h2h,
            "weather": weather,
            "importance": importance
        },
        "predictions": {
            "xg": {"home": round(h_xg, 3), "away": round(a_xg, 3)},
            "most_likely_score": top_score,
            "confidence": round(confidence, 3)
        },
        "markets": {
            cat: {k: round(v, 4) for k, v in probs.items()} 
            for cat, probs in markets.items()
        }
    }
    
    if models["corner"] and not h_c_err:
        export_data["predictions"]["corners"] = {
            "home": round(h_c, 2),
            "away": round(a_c, 2),
            "total": round(h_c + a_c, 2)
        }
    
    json_str = json.dumps(export_data, indent=2)
    
    col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 2])
    
    with col_dl1:
        st.download_button(
            "📥 Download JSON",
            json_str,
            f"prediction_{h_name}_{a_name}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    with col_dl2:
        # CSV export for markets
        csv_df = pd.DataFrame([
            {"Market": f"{cat} - {market}", "Probability": f"{prob:.2%}", "Decimal Odds": f"{1/prob:.2f}"}
            for cat, probs in markets.items()
            for market, prob in probs.items()
        ])
        st.download_button(
            "📊 Download CSV",
            csv_df.to_csv(index=False),
            f"markets_{h_name}_{a_name}.csv",
            mime="text/csv"
        )
    
    with col_dl3:
        st.caption(f"Simulation based on {CONFIG.SIMULATION_COUNT:,} Monte Carlo runs")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ==============================
# 🔧 CONFIG
# ==============================
st.set_page_config(layout="wide", page_title="HighStakes | Match Engine")
st.title("🛡️ HighStakes | Intelligent Match Engine")

# ==============================
# 🧠 CONFIGURATION
# ==============================
@dataclass
class Config:
    """Centralized configuration"""
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
# 🧠 VALIDATION
# ==============================
def validate_team_stats(stats: Dict, team_name: str) -> List[str]:
    """Validate input statistics"""
    errors = []
    
    if not 0 <= stats.get("pos", 0) <= 100:
        errors.append(f"{team_name}: Possession must be 0-100%")
    
    if stats.get("gpg", 0) > 5:
        errors.append(f"{team_name}: Goals/game ({stats['gpg']}) seems unrealistic")
    
    if stats.get("con_pg", 0) > 5:
        errors.append(f"{team_name}: Conceded/game ({stats['con_pg']}) seems unrealistic")
    
    if stats.get("sot", 0) > 20:
        errors.append(f"{team_name}: Shots on target ({stats['sot']}) seems unrealistic")
    
    if stats.get("bc", 0) > 10:
        errors.append(f"{team_name}: Big chances ({stats['bc']}) seems unrealistic")
    
    negative_fields = ["sot", "bc", "bcm", "gpg", "offsides", "fouls", "con_pg", "cs"]
    for field in negative_fields:
        if stats.get(field, 0) < 0:
            errors.append(f"{team_name}: {field} cannot be negative")
    
    return errors

# ==============================
# 🧠 UTIL FUNCTIONS
# ==============================
def normalize(v: float, max_v: float) -> float:
    """Safe normalization with bounds checking"""
    if max_v <= 0:
        return 0.0
    return float(np.clip(v / max_v, 0, 1))

def f_score(form_text: str) -> Tuple[float, List[str]]:
    """
    Calculate form score from WDL string.
    Returns: (score, warnings)
    """
    warnings = []
    
    if not form_text:
        return 1.0, ["No form provided, using neutral 1.0"]
    
    clean = "".join([c for c in form_text.upper() if c in "WDL"])
    invalid_chars = len(form_text) - len(clean)
    
    if invalid_chars > 0:
        warnings.append(f"Ignored {invalid_chars} invalid characters in form")
    
    if not clean:
        return 1.0, ["No valid WDL characters found, using neutral 1.0"]
    
    # Standard football points: 3 for win, 1 for draw, 0 for loss
    points = sum({"W": 3, "D": 1, "L": 0}.get(c, 0) for c in clean)
    max_points = len(clean) * 3
    form_ratio = points / max_points
    
    # Map 0-1 to 0.5-1.3 range (wider impact for form)
    score = 0.5 + (form_ratio * 0.8)
    
    return float(np.clip(score, 0.5, 1.3)), warnings

def player_attack_boost(xi_text: str, key_players: List[str] = None) -> Tuple[float, Dict]:
    """
    Calculate attack boost from starting XI with quality weighting.
    Returns: (boost, metadata)
    """
    metadata = {"total_players": 0, "key_players_found": 0, "boost_breakdown": ""}
    
    if not xi_text or not xi_text.strip():
        return 0.0, metadata
    
    key_players = key_players or CONFIG.KEY_PLAYERS
    players = [p.strip().upper() for p in xi_text.split(",") if p.strip()]
    
    metadata["total_players"] = len(players)
    
    if not players:
        return 0.0, metadata
    
    # Count key players
    key_count = 0
    found_names = []
    for player in players:
        for key in key_players:
            if key in player or player in key:
                key_count += 1
                found_names.append(player.title())
                break
    
    metadata["key_players_found"] = key_count
    metadata["found_names"] = found_names
    
    # Base boost: 0.5% per player + 2% per key player
    base_boost = len(players) * 0.005
    key_boost = key_count * 0.02
    
    total_boost = min(base_boost + key_boost, CONFIG.MAX_ATTACK_BOOST)
    
    metadata["boost_breakdown"] = f"Base: {base_boost:.1%} + Key: {key_boost:.1%}"
    
    return float(total_boost), metadata

# ==============================
# 🧠 FEATURE ENGINE
# ==============================
def build_goal_features(team: Dict, opp: Dict, context: Dict, is_home: bool) -> np.ndarray:
    """
    Build feature vector for goal prediction model.
    Features: [attack, defense, tempo, context_factor, home]
    """
    # Attack metrics
    attack = (
        normalize(team["sot"], 10) * 0.6 +
        normalize(team["bc"], 5) * 1.2 -
        normalize(team["bcm"], 5) * 0.4 +
        normalize(team["gpg"], 4) * 0.8
    )
    
    # Defense metrics (from opponent's perspective)
    defense = (
        normalize(opp["con_pg"], 3) * 1.0 +
        (1 - normalize(opp["cs"], 20)) * 0.8
    )
    
    # Tempo/control metrics
    tempo = (
        (team["pos"] / 100) * 0.6 +
        normalize(team["offsides"], 5) * 0.1 -
        normalize(team["fouls"], 20) * 0.05
    )
    
    # Context factor
    context_factor = (
        context["form"] * 0.5 +
        (1 - context["inj"]) * 0.3 +
        (1 + context["h2h"]) * 0.2
    )
    
    home = 1.0 if is_home else 0.0
    
    return np.array([attack, defense, tempo, context_factor, home], dtype=np.float32)

def build_corner_features(team: Dict, opp: Dict, is_home: bool) -> np.ndarray:
    """
    Build feature vector for corner prediction model.
    Corners have different dynamics than goals.
    Features: [volume, opp_defense, wide_play, home]
    """
    # Attacking volume (crosses, shots from wide)
    volume = (
        normalize(team["sot"], 10) * 0.4 +      # Shots create deflections/corners
        normalize(team["bc"], 5) * 0.3 +        # Big chances often from wide crosses
        normalize(team["pos"], 100) * 0.3      # Possession pressure
    )
    
    # Opponent defensive style
    opp_defense = (
        normalize(opp["con_pg"], 3) * 0.4 +      # Poor defense = clearances for corners
        normalize(opp["fouls"], 20) * 0.3 +      # Desperation defending
        (1 - normalize(opp["cs"], 20)) * 0.3   # Lack of clean sheets
    )
    
    # Wide play indicator (offsides suggest aggressive wide runs)
    wide_play = normalize(team["offsides"], 5) * 0.5 + normalize(team["fouls"], 20) * 0.2
    
    home = 1.0 if is_home else 0.0
    
    return np.array([volume, opp_defense, wide_play, home], dtype=np.float32)

# ==============================
# 🎯 MODEL PREDICTION
# ==============================
def predict(model, features: np.ndarray, model_name: str = "model") -> Tuple[float, Optional[str]]:
    """
    Safe prediction with validation.
    Returns: (prediction, error_message)
    """
    if model is None:
        return 0.0, f"{model_name} not loaded"
    
    # Validate feature count if model exposes feature names
    if hasattr(model, 'feature_names_in_'):
        expected = len(model.feature_names_in_)
        actual = len(features)
        if actual != expected:
            return 0.0, f"{model_name} expects {expected} features, got {actual}"
    
    try:
        pred = model.predict([features])[0]
        # Ensure positive prediction
        pred = max(0.01, float(pred))
        return pred, None
    except Exception as e:
        return 0.0, f"Prediction error: {str(e)}"

# ==============================
# 🎲 SIMULATION
# ==============================
@st.cache_data(show_spinner=False)
def simulate_goals(h_xg: float, a_xg: float, n_sims: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Monte Carlo simulation for goal outcomes.
    Returns: (home_goals, away_goals, first_half_goals, metadata)
    """
    # Random factors for each team
    h_factor = np.random.normal(1, CONFIG.GOAL_NOISE_STD, n_sims)
    a_factor = np.random.normal(1, CONFIG.GOAL_NOISE_STD, n_sims)
    
    # Poisson simulation
    h_goals = np.random.poisson(h_xg * h_factor)
    a_goals = np.random.poisson(a_xg * a_factor)
    
    # First half goals (typically ~44% of total)
    total_xg = h_xg + a_xg
    fh_goals = np.random.poisson(total_xg * CONFIG.FH_GOAL_RATIO, n_sims)
    
    metadata = {
        "h_ci": np.percentile(h_goals, [5, 95]),
        "a_ci": np.percentile(a_goals, [5, 95]),
        "h_mean": np.mean(h_goals),
        "a_mean": np.mean(a_goals),
        "fh_mean": np.mean(fh_goals)
    }
    
    return h_goals, a_goals, fh_goals, metadata

@st.cache_data(show_spinner=False)
def simulate_corners(h_c: float, a_c: float, n_sims: int = 10000) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Monte Carlo simulation for corner outcomes"""
    h_factor = np.random.normal(1, CONFIG.CORNER_NOISE_STD, n_sims)
    a_factor = np.random.normal(1, CONFIG.CORNER_NOISE_STD, n_sims)
    
    h_corners = np.random.poisson(h_c * h_factor)
    a_corners = np.random.poisson(a_c * a_factor)
    
    metadata = {
        "h_ci": np.percentile(h_corners, [5, 95]),
        "a_ci": np.percentile(a_corners, [5, 95]),
        "h_mean": np.mean(h_corners),
        "a_mean": np.mean(a_corners)
    }
    
    return h_corners, a_corners, metadata

# ==============================
# 📦 LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    """Load models with error handling"""
    models = {}
    
    try:
        models["goal"] = joblib.load("model.pkl")
        st.sidebar.success("✅ Goal model loaded")
    except Exception as e:
        st.sidebar.error(f"⚠️ Goal model error: {e}")
        models["goal"] = None
    
    try:
        models["corner"] = joblib.load("corner_model.pkl")
        st.sidebar.success("✅ Corner model loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Corner model not found: {e}")
        models["corner"] = None
    
    return models

models = load_models()

# ==============================
# 🏟️ INPUT FORM
# ==============================
with st.form("match_input"):
    st.markdown("### Match Setup")
    
    col1, col2 = st.columns(2)

    def team_inputs(name: str, is_home: bool):
        color = "🔴" if is_home else "🔵"
        st.markdown(f"**{color} {name}**")
        
        return {
            "sot": st.number_input(f"Shots on Target", 0.0, 20.0, 4.5, key=f"{name}_sot"),
            "bc": st.number_input(f"Big Chances", 0.0, 10.0, 1.5, key=f"{name}_bc"),
            "bcm": st.number_input(f"Big Chances Missed", 0.0, 10.0, 0.8, key=f"{name}_bcm"),
            "gpg": st.number_input(f"Goals/Game", 0.0, 5.0, 1.2, key=f"{name}_gpg"),
            "pos": st.number_input(f"Possession %", 0.0, 100.0, 50.0, key=f"{name}_pos"),
            "offsides": st.number_input(f"Offsides/Game", 0.0, 10.0, 2.0, key=f"{name}_off"),
            "fouls": st.number_input(f"Fouls/Game", 0.0, 25.0, 10.0, key=f"{name}_fouls"),
            "con_pg": st.number_input(f"Conceded/Game", 0.0, 5.0, 1.0, key=f"{name}_con"),
            "cs": st.number_input(f"Clean Sheets", 0.0, 20.0, 5.0, key=f"{name}_cs")
        }

    with col1:
        h_name = st.text_input("Home Team", "ARSENAL", key="home_name")
        h_form = st.text_input("Recent Form (W=Win, D=Draw, L=Loss)", "WWDLW", key="home_form")
        h_xi = st.text_area("Starting XI (comma separated)", "Raya, White, Saliba, Gabriel, Zinchenko, Rice, Odegaard, Havertz, Saka, Martinelli, Trossard", key="home_xi")
        h_stats = team_inputs(h_name, True)

    with col2:
        a_name = st.text_input("Away Team", "NEWCASTLE", key="away_name")
        a_form = st.text_input("Recent Form", "LDWLL", key="away_form")
        a_xi = st.text_area("Starting XI", "Pope, Trippier, Schar, Botman, Burn, Guimaraes, Tonali, Joelinton, Almiron, Isak, Gordon", key="away_xi")
        a_stats = team_inputs(a_name, False)

    st.markdown("---")
    st.subheader("Context Factors")
    
    ctx_col1, ctx_col2, ctx_col3 = st.columns(3)
    
    with ctx_col1:
        h_inj = st.slider("Home Injury Impact", 0.0, 0.6, 0.1, help="0=none, 0.6=severe")
        a_inj = st.slider("Away Injury Impact", 0.0, 0.6, 0.1)
    
    with ctx_col2:
        h2h = st.slider("H2H Edge (Home advantage)", -0.5, 0.5, 0.0, help="Positive favors home")
    
    with ctx_col3:
        weather = st.selectbox("Weather", ["Clear", "Rain", "Wind"], index=0)
        importance = st.selectbox("Match Importance", ["Regular", "Derby", "Title Decider", "Relegation"], index=0)

    submit = st.form_submit_button("🚀 Run Simulation", use_container_width=True)

# ==============================
# 🚀 EXECUTION
# ==============================
if submit:
    # Validation
    all_errors = validate_team_stats(h_stats, h_name) + validate_team_stats(a_stats, a_name)
    
    if all_errors:
        st.error("### ❌ Validation Errors")
        for error in all_errors:
            st.error(error)
        st.stop()
    
    if models["goal"] is None:
        st.error("⚠️ Goal prediction model required. Please upload model.pkl")
        st.stop()
    
    # Calculate form scores
    h_form_score, h_warnings = f_score(h_form)
    a_form_score, a_warnings = f_score(a_form)
    
    # Display warnings
    for w in h_warnings + a_warnings:
        st.warning(w)
    
    # Calculate player boosts
    h_boost, h_boost_meta = player_attack_boost(h_xi, CONFIG.KEY_PLAYERS)
    a_boost, a_boost_meta = player_attack_boost(a_xi, CONFIG.KEY_PLAYERS)
    
    # Build contexts
    h_ctx = {"form": h_form_score, "inj": h_inj, "h2h": h2h}
    a_ctx = {"form": a_form_score, "inj": a_inj, "h2h": -h2h}
    
    # Build features
    h_goal_feat = build_goal_features(h_stats, a_stats, h_ctx, True)
    a_goal_feat = build_goal_features(a_stats, h_stats, a_ctx, False)
    
    # Predict goals
    h_xg, h_err = predict(models["goal"], h_goal_feat, "Goal Model (Home)")
    a_xg, a_err = predict(models["goal"], a_goal_feat, "Goal Model (Away)")
    
    if h_err or a_err:
        st.error(f"Prediction error: {h_err or a_err}")
        st.stop()
    
    # Apply player boosts
    h_xg_raw, a_xg_raw = h_xg, a_xg
    h_xg *= (1 + h_boost)
    a_xg *= (1 + a_boost)
    
    # Run simulations
    h_sim, a_sim, fh_sim, sim_meta = simulate_goals(h_xg, a_xg, CONFIG.SIMULATION_COUNT)
    total_goals = h_sim + a_sim
    
    # ==============================
    # 📊 DISPLAY RESULTS
    # ==============================
    
    st.markdown("---")
    
    # xG Display with confidence intervals
    col_xg1, col_xg2, col_xg3 = st.columns([2, 2, 1])
    
    with col_xg1:
        st.metric(
            f"{h_name} xG", 
            f"{h_xg:.2f}",
            delta=f"+{h_boost:.1%} boost" if h_boost > 0 else None
        )
        st.caption(f"Raw: {h_xg_raw:.2f} | 90% CI: {sim_meta['h_ci'][0]:.0f}-{sim_meta['h_ci'][1]:.0f} goals")
        if h_boost_meta['key_players_found'] > 0:
            st.caption(f"⭐ Key players: {', '.join(h_boost_meta['found_names'][:3])}")
    
    with col_xg2:
        st.metric(
            f"{a_name} xG",
            f"{a_xg:.2f}",
            delta=f"+{a_boost:.1%} boost" if a_boost > 0 else None
        )
        st.caption(f"Raw: {a_xg_raw:.2f} | 90% CI: {sim_meta['a_ci'][0]:.0f}-{sim_meta['a_ci'][1]:.0f} goals")
        if a_boost_meta['key_players_found'] > 0:
            st.caption(f"⭐ Key players: {', '.join(a_boost_meta['found_names'][:3])}")
    
    with col_xg3:
        st.metric("Total xG", f"{h_xg + a_xg:.2f}")
        st.caption(f"Simulated mean: {sim_meta['h_mean'] + sim_meta['a_mean']:.1f}")
    
    # ==============================
    # 📊 MARKET PROBABILITIES
    # ==============================
    st.subheader("📊 Market Probabilities")
    
    markets = {
        "Match Result": {
            "Home Win": np.mean(h_sim > a_sim),
            "Draw": np.mean(h_sim == a_sim),
            "Away Win": np.mean(h_sim < a_sim),
        },
        "Goals": {
            "BTTS": np.mean((h_sim > 0) & (a_sim > 0)),
            "Over 0.5": np.mean(total_goals > 0.5),
            "Over 1.5": np.mean(total_goals > 1.5),
            "Over 2.5": np.mean(total_goals > 2.5),
            "Over 3.5": np.mean(total_goals > 3.5),
            "Under 2.5": np.mean(total_goals <= 2.5),
        },
        "First Half": {
            "FH Over 0.5": np.mean(fh_sim > 0.5),
            "FH Over 1.5": np.mean(fh_sim > 1.5),
            "FH Under 1.5": np.mean(fh_sim <= 1.5),
        },
        "Double Chance": {
            "1X (Home/Draw)": np.mean(h_sim >= a_sim),
            "X2 (Draw/Away)": np.mean(a_sim >= h_sim),
            "12 (No Draw)": np.mean(h_sim != a_sim),
        }
    }
    
    # Display markets in columns
    market_cols = st.columns(len(markets))
    
    for idx, (category, probs) in enumerate(markets.items()):
        with market_cols[idx]:
            st.markdown(f"**{category}**")
            df = pd.DataFrame(probs.items(), columns=["Market", "Prob"])
            df["Prob"] = df["Prob"].apply(lambda x: f"{x:.1%}")
            st.dataframe(df, hide_index=True, use_container_width=True)
    
    # ==============================
    # 🎯 CONFIDENCE METRIC
    # ==============================
    st.subheader("🎯 Prediction Confidence")
    
    outcome_probs = [markets["Match Result"]["Home Win"], 
                     markets["Match Result"]["Draw"], 
                     markets["Match Result"]["Away Win"]]
    
    # Calculate entropy-based confidence
    entropy = -sum(p * np.log(p + 1e-10) for p in outcome_probs if p > 0)
    max_entropy = np.log(3)
    confidence = 1 - (entropy / max_entropy)
    
    conf_col1, conf_col2 = st.columns([3, 1])
    
    with conf_col1:
        st.progress(confidence)
        st.caption(f"Model confidence: **{confidence:.0%}** (higher = more certain outcome)")
        
        if confidence > 0.7:
            st.success("High confidence prediction")
        elif confidence > 0.5:
            st.info("Moderate confidence - consider alternate markets")
        else:
            st.warning("Low confidence - unpredictable match")
    
    with conf_col2:
        # Most likely exact score
        score_probs = {}
        for h in range(6):
            for a in range(6):
                prob = np.mean((h_sim == h) & (a_sim == a))
                score_probs[f"{h}-{a}"] = prob
        
        top_score = max(score_probs, key=score_probs.get)
        st.metric("Most Likely Score", top_score, f"{score_probs[top_score]:.1%}")
    
    # ==============================
    # 🚩 CORNERS
    # ==============================
    if models["corner"]:
        st.subheader("🚩 Corners Market")
        
        # Build corner-specific features
        h_corner_feat = build_corner_features(h_stats, a_stats, True)
        a_corner_feat = build_corner_features(a_stats, h_stats, False)
        
        h_c, h_c_err = predict(models["corner"], h_corner_feat, "Corner Model (Home)")
        a_c, a_c_err = predict(models["corner"], a_corner_feat, "Corner Model (Away)")
        
        if not h_c_err and not a_c_err:
            hc_sim, ac_sim, c_meta = simulate_corners(h_c, a_c, CONFIG.SIMULATION_COUNT)
            total_corners = hc_sim + ac_sim
            
            c_col1, c_col2, c_col3 = st.columns(3)
            
            with c_col1:
                st.metric(f"{h_name} Corners", f"{h_c:.1f}")
                st.caption(f"90% CI: {c_meta['h_ci'][0]:.0f}-{c_meta['h_ci'][1]:.0f}")
            
            with c_col2:
                st.metric(f"{a_name} Corners", f"{a_c:.1f}")
                st.caption(f"90% CI: {c_meta['a_ci'][0]:.0f}-{c_meta['a_ci'][1]:.0f}")
            
            with c_col3:
                st.metric("Total Corners", f"{h_c + a_c:.1f}")
            
            corner_markets = {
                "Over 8.5": np.mean(total_corners > 8.5),
                "Over 9.5": np.mean(total_corners > 9.5),
                "Over 10.5": np.mean(total_corners > 10.5),
                "Under 9.5": np.mean(total_corners < 9.5),
                "Under 10.5": np.mean(total_corners < 10.5),
                "Home > Away": np.mean(hc_sim > ac_sim),
            }
            
            c_df = pd.DataFrame(corner_markets.items(), columns=["Market", "Probability"])
            st.dataframe(c_df.style.format({"Probability": "{:.1%}"}), use_container_width=True)
        else:
            st.error(f"Corner prediction failed: {h_c_err or a_c_err}")
    else:
        st.info("⚠️ Corner model not loaded - corner markets unavailable")
    
    # ==============================
    # 📥 EXPORT
    # ==============================
    st.markdown("---")
    
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "match": f"{h_name} vs {a_name}",
        "context": {
            "home_form": h_form,
            "away_form": a_form,
            "home_injuries": h_inj,
            "away_injuries": a_inj,
            "h2h_edge": h2h,
            "weather": weather,
            "importance": importance
        },
        "predictions": {
            "xg": {"home": round(h_xg, 3), "away": round(a_xg, 3)},
            "most_likely_score": top_score,
            "confidence": round(confidence, 3)
        },
        "markets": {
            cat: {k: round(v, 4) for k, v in probs.items()} 
            for cat, probs in markets.items()
        }
    }
    
    if models["corner"] and not h_c_err:
        export_data["predictions"]["corners"] = {
            "home": round(h_c, 2),
            "away": round(a_c, 2),
            "total": round(h_c + a_c, 2)
        }
    
    json_str = json.dumps(export_data, indent=2)
    
    col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 2])
    
    with col_dl1:
        st.download_button(
            "📥 Download JSON",
            json_str,
            f"prediction_{h_name}_{a_name}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    with col_dl2:
        # CSV export for markets
        csv_df = pd.DataFrame([
            {"Market": f"{cat} - {market}", "Probability": f"{prob:.2%}", "Decimal Odds": f"{1/prob:.2f}"}
            for cat, probs in markets.items()
            for market, prob in probs.items()
        ])
        st.download_button(
            "📊 Download CSV",
            csv_df.to_csv(index=False),
            f"markets_{h_name}_{a_name}.csv",
            mime="text/csv"
        )
    
    with col_dl3:
        st.caption(f"Simulation based on {CONFIG.SIMULATION_COUNT:,} Monte Carlo runs")
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
