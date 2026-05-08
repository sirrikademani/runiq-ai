import os
import re
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="RunIQ AI",
    page_icon="🏃",
    layout="wide"
)

# Strava OAuth config
STRAVA_CLIENT_ID     = os.getenv("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REDIRECT_URI         = "https://2bvtfobfiehd6ah4u2br7z.streamlit.app"
STRAVA_AUTH_URL      = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL     = "https://www.strava.com/oauth/token"
STRAVA_API_URL       = "https://www.strava.com/api/v3"

#Strava OAuth functions
def get_auth_url():
    """Generate Strava OAuth login URL."""
    params = {
        "client_id"    : STRAVA_CLIENT_ID,
        "redirect_uri" : REDIRECT_URI,
        "response_type": "code",
        "scope"        : "activity:read_all",
    }
    param_str = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{STRAVA_AUTH_URL}?{param_str}"

def exchange_token(code: str) -> dict:
    """Exchange auth code for access token."""
    response = requests.post(STRAVA_TOKEN_URL, data={
        "client_id"    : STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "code"         : code,
        "grant_type"   : "authorization_code",
    })
    return response.json()

def fetch_activities(access_token: str, max_pages: int = 5) -> list:
    """Fetch all activities from Strava API."""
    headers    = {"Authorization": f"Bearer {access_token}"}
    activities = []
    for page in range(1, max_pages + 1):
        resp = requests.get(
            f"{STRAVA_API_URL}/athlete/activities",
            headers=headers,
            params={"per_page": 100, "page": page}
        )
        data = resp.json()
        if not data:
            break
        activities.extend(data)
    return activities

def fetch_athlete(access_token: str) -> dict:
    """Fetch athlete profile."""
    headers = {"Authorization": f"Bearer {access_token}"}
    return requests.get(f"{STRAVA_API_URL}/athlete", headers=headers).json()

#In-memory pipeline
def build_dataframe(activities: list) -> pd.DataFrame:
    """Convert raw Strava API activities to clean DataFrame."""
    runs = []
    for a in activities:
        if a.get("type") != "Run":
            continue
        runs.append({
            "date"            : pd.to_datetime(a["start_date"]),
            "name"            : a.get("name", "Run"),
            "distance_km"     : round(a.get("distance", 0) / 1000, 2),
            "distance_mi"     : round(a.get("distance", 0) / 1609.34, 2),
            "moving_time_min" : round(a.get("moving_time", 0) / 60, 2),
            "avg_hr"          : a.get("average_heartrate"),
            "max_hr"          : a.get("max_heartrate"),
            "elevation_gain_m": a.get("total_elevation_gain", 0),
            "calories"        : a.get("calories"),
            "avg_cadence"     : a.get("average_cadence"),
            "training_load"   : a.get("suffer_score"),
        })

    if not runs:
        return pd.DataFrame()

    df = pd.DataFrame(runs).sort_values("date").reset_index(drop=True)

    # Pace
    df["pace_min_per_km"] = df["moving_time_min"] / df["distance_km"]

    # Load fallback
    df["load"] = df["training_load"].fillna(df["moving_time_min"] * 0.5)

    # ATL / CTL / TSB
    daily = df.set_index("date")["load"].resample("D").sum()
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0)
    atl = daily.ewm(span=7,  adjust=False).mean()
    ctl = daily.ewm(span=42, adjust=False).mean()
    tsb = ctl - atl

    date_keys     = df["date"].dt.normalize()
    df["atl"]     = date_keys.map(atl).round(1)
    df["ctl"]     = date_keys.map(ctl).round(1)
    df["tsb"]     = date_keys.map(tsb).round(1)

    # Overtraining
    df["atl_ctl_ratio"]    = (df["atl"] / df["ctl"].replace(0, np.nan)).round(2)
    conditions             = [
        (df["atl_ctl_ratio"] > 1.3) | (df["tsb"] < -20),
        (df["atl_ctl_ratio"] > 1.1) | (df["tsb"] < -10),
    ]
    df["overtraining_risk"] = np.select(conditions, ["high","moderate"], default="low")

    # Text summaries
    def summarize(row):
        hr  = f"HR {row['avg_hr']:.0f}bpm" if pd.notna(row["avg_hr"]) else "no HR"
        cal = f"{row['calories']:.0f} cal"  if pd.notna(row["calories"]) else "unknown cal"
        return (
            f"{row['date'].strftime('%b %d, %Y')} — {row['name']} | "
            f"{row['distance_km']:.2f}km | pace {row['pace_min_per_km']:.2f} min/km | "
            f"{hr} | elev +{row['elevation_gain_m']:.0f}m | {cal} | "
            f"ATL {row['atl']} CTL {row['ctl']} TSB {row['tsb']} | "
            f"risk: {row['overtraining_risk']}"
        )
    df["text_summary"] = df.apply(summarize, axis=1)
    return df

#Tools
def tool_search_runs(question: str, runs: pd.DataFrame, n: int = 5) -> str:
    keywords = question.lower().split()
    results  = []
    for _, row in runs.iterrows():
        score = sum(1 for kw in keywords if kw in row["text_summary"].lower())
        if score > 0:
            results.append((score, row["text_summary"]))
    results = sorted(results, reverse=True)[:n]
    if not results:
        return "No matching runs found."
    return "Relevant runs:\n\n" + "\n".join(f"- {r[1]}" for r in results)

def tool_overtraining_check(runs: pd.DataFrame) -> str:
    latest          = runs.iloc[-1]
    recent          = runs.tail(14)
    high_risk_count = (recent["overtraining_risk"] == "high").sum()
    return f"""
Training Load Analysis:
- ATL (fatigue):  {latest['atl']:.1f}
- CTL (fitness):  {latest['ctl']:.1f}
- TSB (form):     {latest['tsb']:.1f}
- ATL/CTL ratio:  {latest['atl_ctl_ratio']:.2f}
- Risk level:     {latest['overtraining_risk'].upper()}
- High risk runs (last 14): {high_risk_count}/14
- Verdict: {'Very fatigued — rest recommended' if latest['tsb'] < -20 else 'Moderately fatigued' if latest['tsb'] < -10 else 'Good form'}
""".strip()

def tool_predict_race(distance_km: float, runs: pd.DataFrame) -> str:
    recent          = runs[runs["distance_km"] > 1].tail(30)
    best_pace       = recent["pace_min_per_km"].min()
    avg_pace        = recent["pace_min_per_km"].mean()
    fatigue_factors = {5: 0.95, 10: 0.98, 21.1: 1.08, 42.2: 1.20}
    closest         = min(fatigue_factors.keys(), key=lambda x: abs(x - distance_km))
    predicted_pace  = best_pace * fatigue_factors[closest]
    total_min       = predicted_pace * distance_km
    h, m, s         = int(total_min // 60), int(total_min % 60), int((total_min * 60) % 60)
    return f"""
Race Prediction for {distance_km}km:
- Best recent pace:    {best_pace:.2f} min/km
- Predicted pace:      {predicted_pace:.2f} min/km
- Predicted time:      {h}h {m}m {s}s
- Current TSB:         {runs.iloc[-1]['tsb']:.1f} ({'rest first!' if runs.iloc[-1]['tsb'] < -20 else 'okay to race'})
""".strip()

def agent(question: str, runs: pd.DataFrame, client: Groq) -> str:
    routing = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": f"""
You are a running coach AI. User asked: "{question}"
Tools: search_runs, overtraining, predict_race
Reply ONLY with tool name. Use predict_race for race/finish time questions.
"""}],
        max_tokens=10, temperature=0
    )
    tool = routing.choices[0].message.content.strip().lower()

    if "overtraining" in tool:
        result = tool_overtraining_check(runs)
    elif "predict" in tool:
        numbers  = re.findall(r'\d+\.?\d*', question)
        distance = float(numbers[0]) if numbers else 5.0
        if "half"    in question.lower(): distance = 21.1
        elif "full"  in question.lower() or "marathon" in question.lower(): distance = 42.2
        elif "10k"   in question.lower(): distance = 10.0
        elif "5k"    in question.lower(): distance = 5.0
        result = tool_predict_race(distance, runs)
    else:
        result = tool_search_runs(question, runs)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": f"""
You are RunIQ, a friendly running coach AI.
User asked: "{question}"
Data: {result}
Answer in 3-5 sentences. Be encouraging but honest.
"""}],
        max_tokens=300, temperature=0.7
    )
    return response.choices[0].message.content.strip()

#Main app with OAuth flow
def show_dashboard(runs: pd.DataFrame, athlete: dict, client: Groq):
    """Show the full dashboard once user is authenticated."""

    with st.sidebar:
        st.title("🏃 RunIQ AI")
        st.caption(f"Welcome, {athlete.get('firstname', 'Runner')}!")
        if athlete.get("profile"):
            st.image(athlete["profile"], width=80)
        st.divider()
        latest = runs.iloc[-1]
        st.metric("Total Runs",     f"{len(runs)}")
        st.metric("Total Distance", f"{runs['distance_km'].sum():.0f} km")
        st.metric("Avg Pace",       f"{runs['pace_min_per_km'].mean():.2f} min/km")
        st.metric("Current TSB",    f"{latest['tsb']:.1f}")
        st.metric("Fatigue Risk",   latest["overtraining_risk"].upper())
        st.divider()
        st.caption("Try asking:")
        st.caption("• Am I overtraining?")
        st.caption("• What was my longest run?")
        st.caption("• Predict my half marathon time")

    tab1, tab2, tab3 = st.tabs(["💬 AI Coach", "📈 Training Load", "🏅 Pace & Distance"])

    with tab1:
        st.header("Ask your AI running coach")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        if prompt := st.chat_input("Ask anything about your runs..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your runs..."):
                    response = agent(prompt, runs, client)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    with tab2:
        st.header("Training Load — ATL / CTL / TSB")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=runs["date"], y=runs["atl"], name="ATL (fatigue)", line=dict(color="red", width=2)))
        fig.add_trace(go.Scatter(x=runs["date"], y=runs["ctl"], name="CTL (fitness)", line=dict(color="blue", width=2)))
        fig.add_trace(go.Scatter(x=runs["date"], y=runs["tsb"], name="TSB (form)", line=dict(color="green", width=2), fill="tozeroy", fillcolor="rgba(0,255,0,0.05)"))
        fig.add_hline(y=-20, line_dash="dash", line_color="red", annotation_text="Fatigue threshold")
        fig.update_layout(title="Training Load Over Time", xaxis_title="Date", yaxis_title="Load", hovermode="x unified", height=450)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            risk_counts = runs["overtraining_risk"].value_counts().reset_index()
            risk_counts.columns = ["Risk", "Count"]
            fig2 = px.pie(risk_counts, values="Count", names="Risk", title="Risk Distribution",
                          color="Risk", color_discrete_map={"high":"red","moderate":"orange","low":"green"})
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            monthly = runs.copy()
            monthly["month"] = runs["date"].dt.to_period("M").astype(str)
            fig3 = px.bar(monthly.groupby("month")["distance_km"].sum().reset_index(),
                          x="month", y="distance_km", title="Monthly Distance (km)",
                          color="distance_km", color_continuous_scale="blues")
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.header("Pace & Distance Trends")
        fig4 = px.scatter(runs, x="date", y="pace_min_per_km", size="distance_km",
                          color="overtraining_risk",
                          color_discrete_map={"high":"red","moderate":"orange","low":"green"},
                          title="Pace Over Time", hover_data=["name","distance_km","tsb"])
        fig4.update_yaxes(autorange="reversed")
        fig4.update_layout(height=450)
        st.plotly_chart(fig4, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig5 = px.histogram(runs, x="distance_km", nbins=20, title="Distance Distribution")
            st.plotly_chart(fig5, use_container_width=True)
        with col4:
            hr_runs = runs.dropna(subset=["avg_hr"])
            if not hr_runs.empty:
                fig6 = px.scatter(hr_runs, x="pace_min_per_km", y="avg_hr",
                                  color="distance_km", title="Heart Rate vs Pace",
                                  color_continuous_scale="reds")
                fig6.update_xaxes(autorange="reversed")
                st.plotly_chart(fig6, use_container_width=True)

# ── Main app logic ─────────────────────────────────────────────────────────────
client       = Groq(api_key=os.getenv("GROQ_API_KEY"))
access_token = os.getenv("STRAVA_ACCESS_TOKEN")

if "runs" not in st.session_state:
    with st.spinner("Loading your Strava data..."):
        activities = fetch_activities(access_token)
        athlete    = fetch_athlete(access_token)
        runs       = build_dataframe(activities)
        st.session_state["runs"]    = runs
        st.session_state["athlete"] = athlete

show_dashboard(
    st.session_state["runs"],
    st.session_state["athlete"],
    client
)

# Check for OAuth callback code in URL
params = st.query_params
code   = params.get("code")

if "athlete" not in st.session_state:

    if code:
        # Exchange code for token
        with st.spinner("Connecting to Strava..."):
            token_data = exchange_token(code)

        if "access_token" in token_data:
            access_token = token_data["access_token"]
            with st.spinner("Fetching your activities... (this may take a moment)"):
                activities = fetch_activities(access_token)
                athlete    = fetch_athlete(access_token)
                runs       = build_dataframe(activities)

            if runs.empty:
                st.error("No runs found on your Strava account!")
            else:
                st.session_state["athlete"]      = athlete
                st.session_state["runs"]         = runs
                st.session_state["access_token"] = access_token
                st.query_params.clear()
                st.rerun()
        else:
            st.error("Failed to connect to Strava. Please try again.")
            st.json(token_data)
    else:
        # Show login page
        st.title("🏃 RunIQ AI")
        st.subheader("Your personal AI running coach")
        st.write("")
        st.write("Connect your Strava account to get:")
        st.write("✅ AI-powered insights from your run history")
        st.write("✅ Overtraining detection with ATL/CTL/TSB analysis")
        st.write("✅ Race finish time predictions")
        st.write("✅ Interactive training dashboard")
        st.write("")
        auth_url = get_auth_url()
        st.markdown(f"""
        <a href="{auth_url}" target="_self" style="
            display: inline-block;
            background-color: #FC4C02;
            color: white;
            padding: 14px 28px;
            border-radius: 8px;
            text-decoration: none;
            font-size: 18px;
            font-weight: bold;
            font-family: sans-serif;
        ">
            🏃 Connect with Strava
        </a>
        """, unsafe_allow_html=True)
        st.caption("We never store your data. Everything is processed in memory.")

else:
    # Already authenticated — show dashboard
    show_dashboard(
        st.session_state["runs"],
        st.session_state["athlete"],
        client
    )


