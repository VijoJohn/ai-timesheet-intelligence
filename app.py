import os
import pandas as pd
from groq import Groq

# ---------------------------
# Configuration
# ---------------------------

MODEL = "llama3-8b-8192"
TEMPERATURE = 0.3
TOP_P = 0.9
TOT_BRANCHES = 3

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------
# Shared Agent State
# ---------------------------

state = {
    "timesheets": None,
    "anomalies": None,
    "project_totals": None,
    "risk_summary": None,
    "insight": None
}

# ---------------------------
# Agent 1 — Data Agent
# ---------------------------

def data_agent():

    df = pd.read_csv("data/timesheets.csv")

    state["timesheets"] = df

    return df


# ---------------------------
# Agent 2 — Policy Agent
# ---------------------------

def policy_agent():

    df = state["timesheets"]

    MAX_HOURS = 8

    df["anomaly"] = df["hours"] > MAX_HOURS

    state["anomalies"] = df[df["anomaly"] == True]

    return state["anomalies"]


# ---------------------------
# Tool — Project Effort Calculator
# ---------------------------

def project_effort_tool():

    df = state["timesheets"]

    totals = df.groupby("project")["hours"].sum().to_dict()

    state["project_totals"] = totals

    return totals


# ---------------------------
# Agent 3 — Risk Agent
# ---------------------------

def risk_agent():

    anomalies = state["anomalies"]

    if anomalies.empty:
        state["risk_summary"] = "Low"
        return "Low"

    max_hours = anomalies["hours"].max()

    if max_hours > 11:
        risk = "High"
    elif max_hours > 8:
        risk = "Medium"
    else:
        risk = "Low"

    state["risk_summary"] = risk

    return risk


# ---------------------------
# Agent 4 — Insight Agent (LLM + ToT)
# ---------------------------

def insight_agent():

    anomalies = state["anomalies"]
    project_totals = state["project_totals"]
    risk = state["risk_summary"]

    prompt = f"""
You are a finance operations analyst.

Timesheet anomalies:
{anomalies.to_dict()}

Project total effort:
{project_totals}

Risk level:
{risk}

Explain the financial implications and what finance teams should review.
"""

    candidates = []

    for i in range(TOT_BRANCHES):

        response = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            messages=[{"role": "user", "content": prompt}]
        )

        candidates.append(response.choices[0].message.content)

    best = max(candidates, key=len)

    state["insight"] = best

    return best


# ---------------------------
# Agent 5 — Reflection Agent
# ---------------------------

def reflection_agent():

    insight = state["insight"]

    prompt = f"""
Evaluate the following financial analysis.

If it is weak or incomplete, improve it.

Analysis:
{insight}
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        top_p=0.9,
        messages=[{"role": "user", "content": prompt}]
    )

    final = response.choices[0].message.content

    state["insight"] = final

    return final


# ---------------------------
# Orchestrator
# ---------------------------

def run_workflow():

    print("\n--- Agentic AI Finance Control Workflow ---\n")

    data_agent()

    policy_agent()

    project_effort_tool()

    risk_agent()

    insight_agent()

    final_output = reflection_agent()

    print("Timesheet Data:\n")
    print(state["timesheets"])

    print("\nRisk Level:", state["risk_summary"])

    print("\nAI Finance Insight:\n")
    print(final_output)


# ---------------------------
# Run
# ---------------------------

if __name__ == "__main__":
    run_workflow()
