import os
import re
import pandas as pd
import lancedb
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Load env
load_dotenv("/Users/sirikademani/runiq-ai/.env")

# Page config
st.set_page_config(
    page_title="RunIQ AI",
    page_icon="🏃",
    layout="wide"
)

# Load data and models (cached so it only loads once)
@st.cache_resource
def load_resources():
    runs   = pd.read_parquet("/Users/sirikademani/runiq-ai/data/runs_clean.parquet")
    db     = lancedb.connect("/Users/sirikademani/runiq-ai/data/lancedb")
    table  = db.open_table("runs")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return runs, table, client

runs, table, client = load_resources()

#The 3 tools
def tool_search_runs(question: str, n: int = 5) -> str:
    # Keyword search over text summaries
    keywords = question.lower().split()
    results  = []
    for _, row in runs.iterrows():
        score = sum(1 for kw in keywords if kw in row["text_summary"].lower())
        if score > 0:
            results.append((score, row["text_summary"]))
    results  = sorted(results, reverse=True)[:n]
    if not results:
        return "No matching runs found."
    return "Found relevant runs:\n\n" + "\n".join(f"- {r[1]}" for r in results)

def tool_overtraining_check() -> str:
    recent         = runs.sort_values("date").tail(14)
    latest         = runs.iloc[-1]
    high_risk_count = (recent["overtraining_risk"] == "high").sum()
    return f"""
Training Load Analysis (last 14 runs):
- Current ATL (fatigue):    {latest['atl']:.1f}
- Current CTL (fitness):    {latest['ctl']:.1f}
- Current TSB (form):       {latest['tsb']:.1f}
- ATL/CTL ratio:            {latest['atl_ctl_ratio']:.2f}
- Current risk level:       {latest['overtraining_risk'].upper()}
- High risk runs (last 14): {high_risk_count}/14
- TSB interpretation:       {'Very fatigued — rest recommended' if latest['tsb'] < -20 else 'Moderately fatigued' if latest['tsb'] < -10 else 'Good form'}
""".strip()

def tool_predict_race(distance_km: float) -> str:
    recent         = runs[runs["distance_km"] > 1].sort_values("date").tail(30)
    avg_pace       = recent["pace_min_per_km"].mean()
    best_pace      = recent["pace_min_per_km"].min()
    fatigue_factors = {5: 0.95, 10: 0.98, 21.1: 1.08, 42.2: 1.20}
    closest        = min(fatigue_factors.keys(), key=lambda x: abs(x - distance_km))
    factor         = fatigue_factors[closest]
    predicted_pace = best_pace * factor
    total_minutes  = predicted_pace * distance_km
    hours          = int(total_minutes // 60)
    minutes        = int(total_minutes % 60)
    seconds        = int((total_minutes * 60) % 60)
    return f"""
Race Prediction for {distance_km}km:
- Best recent pace:      {best_pace:.2f} min/km
- Avg recent pace:       {avg_pace:.2f} min/km
- Predicted race pace:   {predicted_pace:.2f} min/km
- Predicted finish time: {hours}h {minutes}m {seconds}s
- Current TSB:           {runs.iloc[-1]['tsb']:.1f} ({'rest before racing!' if runs.iloc[-1]['tsb'] < -20 else 'okay to race'})
""".strip()

#Agent
def agent(user_question: str) -> str:
    routing_prompt = f"""
You are a running coach AI. A user asked: "{user_question}"
You have 3 tools: search_runs, overtraining, predict_race.
Reply with ONLY the tool name. If question mentions race distance or finish time use predict_race.
"""
    routing = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": routing_prompt}],
        max_tokens=10,
        temperature=0
    )
    tool_choice = routing.choices[0].message.content.strip().lower()

    if "overtraining" in tool_choice:
        tool_result = tool_overtraining_check()
    elif "predict" in tool_choice:
        numbers  = re.findall(r'\d+\.?\d*', user_question)
        distance = float(numbers[0]) if numbers else 5.0
        if "half"    in user_question.lower(): distance = 21.1
        elif "full"  in user_question.lower() or "marathon" in user_question.lower(): distance = 42.2
        elif "10k"   in user_question.lower(): distance = 10.0
        elif "5k"    in user_question.lower(): distance = 5.0
        tool_result = tool_predict_race(distance)
    else:
        tool_result = tool_search_runs(user_question)

    answer_prompt = f"""
You are RunIQ, a friendly running coach AI.
User asked: "{user_question}"
Data: {tool_result}
Give a clear, helpful, conversational answer in 3-5 sentences. Be encouraging but honest.
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": answer_prompt}],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

#Sidebar stats
with st.sidebar:
    st.title("🏃 RunIQ AI")
    st.caption("Your personal AI running coach")
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
    st.caption("• When was my best training week?")

#Main dashboard tabs
tab1, tab2, tab3 = st.tabs(["💬 AI Coach", "📈 Training Load", "🏅 Pace & Distance"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab1:
    st.header("Ask your AI running coach")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    if prompt := st.chat_input("Ask anything about your runs..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your runs..."):
                response = agent(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ── Tab 2: Training Load ──────────────────────────────────────────────────────
with tab2:
    st.header("Training Load — ATL / CTL / TSB")

    fig_load = go.Figure()
    fig_load.add_trace(go.Scatter(
        x=runs["date"], y=runs["atl"],
        name="ATL (fatigue)", line=dict(color="red", width=2)
    ))
    fig_load.add_trace(go.Scatter(
        x=runs["date"], y=runs["ctl"],
        name="CTL (fitness)", line=dict(color="blue", width=2)
    ))
    fig_load.add_trace(go.Scatter(
        x=runs["date"], y=runs["tsb"],
        name="TSB (form)", line=dict(color="green", width=2),
        fill="tozeroy", fillcolor="rgba(0,255,0,0.05)"
    ))
    fig_load.add_hline(y=-20, line_dash="dash", line_color="red",
                       annotation_text="Fatigue threshold")
    fig_load.update_layout(
        title="Training Load Over Time",
        xaxis_title="Date", yaxis_title="Load",
        hovermode="x unified", height=450
    )
    st.plotly_chart(fig_load, use_container_width=True)

    # Risk breakdown
    col1, col2 = st.columns(2)
    with col1:
        risk_counts = runs["overtraining_risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        fig_risk = px.pie(risk_counts, values="Count", names="Risk",
                          title="Overtraining Risk Distribution",
                          color="Risk",
                          color_discrete_map={"high":"red","moderate":"orange","low":"green"})
        st.plotly_chart(fig_risk, use_container_width=True)

    with col2:
        monthly = runs.copy()
        monthly["month"] = runs["date"].dt.to_period("M").astype(str)
        monthly_dist = monthly.groupby("month")["distance_km"].sum().reset_index()
        fig_monthly = px.bar(monthly_dist, x="month", y="distance_km",
                             title="Monthly Distance (km)",
                             color="distance_km",
                             color_continuous_scale="blues")
        st.plotly_chart(fig_monthly, use_container_width=True)

# ── Tab 3: Pace & Distance ────────────────────────────────────────────────────
with tab3:
    st.header("Pace & Distance Trends")

    fig_pace = px.scatter(
        runs, x="date", y="pace_min_per_km",
        size="distance_km", color="overtraining_risk",
        color_discrete_map={"high":"red","moderate":"orange","low":"green"},
        title="Pace Over Time (bubble size = distance)",
        labels={"pace_min_per_km": "Pace (min/km)", "date": "Date"},
        hover_data=["Activity Name", "distance_km", "atl", "tsb"]
    )
    fig_pace.update_yaxes(autorange="reversed")  # lower pace = faster
    fig_pace.update_layout(height=450)
    st.plotly_chart(fig_pace, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig_dist = px.histogram(
            runs, x="distance_km", nbins=20,
            title="Run Distance Distribution",
            color_discrete_sequence=["#636EFA"]
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col4:
        fig_hr = px.scatter(
            runs.dropna(subset=["Average Heart Rate"]),
            x="pace_min_per_km", y="Average Heart Rate",
            color="distance_km",
            title="Heart Rate vs Pace",
            labels={"pace_min_per_km": "Pace (min/km)", "Average Heart Rate": "Avg HR"},
            color_continuous_scale="reds"
        )
        fig_hr.update_xaxes(autorange="reversed")
        st.plotly_chart(fig_hr, use_container_width=True)