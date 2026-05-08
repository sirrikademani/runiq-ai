# 🏃 RunIQ AI — Your Personal AI Running Coach

A GenAI-powered running coach that analyzes your Strava data, detects overtraining, and predicts race finish times — all through a conversational AI interface.

🔗 **Live App:** https://2bvtfobfiehd6ah4u2br7z.streamlit.app

---

## What it does

- 💬 **AI Chat Coach** — Ask anything about your running history in natural language
- 📈 **Overtraining Detector** — Calculates ATL, CTL, and TSB training load metrics and flags injury risk
- 🏅 **Race Time Predictor** — Predicts your 5K, 10K, half marathon, and full marathon finish times from recent training
- 📊 **Training Dashboard** — Visualizes pace trends, monthly distance, heart rate vs pace, and load over time

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data | Strava export (CSV) |
| Vector search | LanceDB |
| LLM | Groq (Llama 3.1) |
| Agent | Custom tool routing |
| Dashboard | Streamlit + Plotly |
| Deployment | Streamlit Cloud |

---

## Architecture
User question
↓
Orchestrator (Groq LLM)
↓ routes to
┌─────────────┬──────────────────┬─────────────────┐
│  RAG Search │  Overtraining    │  Race Predictor │
│  (LanceDB)  │  (ATL/CTL/TSB)   │  (Riegel model) │
└─────────────┴──────────────────┴─────────────────┘
↓
Streamlit Dashboard

---

## How to run locally

**1. Clone the repo**
```bash
git clone https://github.com/sirrikademani/runiq-ai.git
cd runiq-ai
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Add your Groq API key**
```bash
echo "GROQ_API_KEY=your_key_here" > .env
```
Get a free key at https://console.groq.com

**4. Add your Strava data**
- Export your data from Strava → Settings → Download Archive
- Place `activities.csv` in the `data/` folder
- Run the pipeline notebook: `src/pipeline.ipynb`

**5. Run the app**
```bash
streamlit run app.py
```

---

## Data Pipeline

The pipeline (`src/pipeline.ipynb`) processes raw Strava CSV data through:

1. **Filtering** — Runs only
2. **Unit conversion** — Pace in min/km, distance in km and miles
3. **ATL/CTL/TSB** — Exponential weighted training load metrics
4. **Overtraining flag** — Based on ATL/CTL ratio and TSB thresholds
5. **Text summaries** — Natural language summaries per run for vector search
6. **LanceDB** — Embeds and stores run summaries for semantic retrieval

---

## Key Concepts

- **ATL (Acute Training Load)** — 7-day exponential average — represents fatigue
- **CTL (Chronic Training Load)** — 42-day exponential average — represents fitness
- **TSB (Training Stress Balance)** — CTL minus ATL — represents form
- **Overtraining risk** — HIGH when ATL/CTL > 1.3 or TSB < -20

---

## Sample Questions to Ask

- *"Am I overtraining?"*
- *"What was my longest run?"*
- *"Predict my half marathon finish time"*
- *"When was my best training week?"*
- *"How has my pace improved over time?"*
- *"Should I run a race this weekend?"*

---
## Built by
**Siri Kademani** — Data Scientist  
[GitHub](https://github.com/sirrikademani)

---

## How it was built
This project was built using **Claude (Anthropic)** as an AI pair programmer — 
guiding architecture decisions, debugging errors, and writing code collaboratively. 
All data, design decisions, and domain knowledge (running metrics, training load 
theory, deployment) were driven by the author.

This reflects a modern AI-assisted development workflow where the human provides 
direction, context, and judgment — and AI accelerates execution.
