# streamlit_app.py
from datetime import date
import streamlit as st
from predict import predict_from_text

st.set_page_config(page_title="NBA-ML", page_icon="🏀", layout="centered")
st.title("🏀 NBA-ML Scoreboard")

with st.container():
    colA, colB = st.columns(2)
    home = colA.text_input("Home team", "Golden State Warriors")
    away = colB.text_input("Away team", "Los Angeles Lakers")

    col1, col2, col3 = st.columns([1,1,1])
    use_date = col1.checkbox("Use game date", value=False)
    gdate = col2.date_input("Game date", value=date.today()) if use_date else None
    season = col3.text_input("Season (optional, e.g., 2024-25)", "")

if st.button("Predict", use_container_width=True):
    parts = [f"{home} vs {away}"]
    if season.strip():
        parts.append(season.strip())
    if use_date and gdate:
        parts.append(gdate.isoformat())
    query = " ".join(parts)

    try:
        res = predict_from_text(query)
        ht = res["prediction"]["home_team"]
        at = res["prediction"]["away_team"]
        p  = float(res["prediction"]["home_win_prob"])
        spread = float(res["prediction"]["predicted_spread_home"])
        winner = res["prediction"]["predicted_winner"]

        st.subheader(f"{ht} vs {at}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Home Win Prob", f"{p*100:.1f}%")
        m2.metric("Predicted Spread", f"{'Home' if spread>=0 else 'Away'} {abs(spread):.1f}")
        m3.metric("Predicted Winner", winner)

        st.progress(min(max(p, 0.0), 1.0))

        ts = res.get("top_scorer", {})
        if ts and ts.get("player"):
            st.markdown(f"**Top Scorer:** {ts['player']} ({ts.get('team','')}) — ~{ts.get('season_ppg',0):.1f} ppg")
            st.caption(ts.get("why",""))

        if res.get("reasons"):
            st.markdown("### Why the model leans this way")
            st.write("\n".join([f"• {r}" for r in res["reasons"]]))

        with st.expander("Raw JSON"):
            st.json(res)

    except Exception as e:
        st.error(str(e))
