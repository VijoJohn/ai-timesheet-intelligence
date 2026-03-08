# AI Timesheet Intelligence

Prototype demonstrating how Agentic AI can automate
timesheet anomaly detection for finance workflows.

## Problem

Timesheets impact billing accuracy, project profitability,
and revenue reporting. Finance teams often review entries
manually to identify anomalies.

This prototype demonstrates how AI agents can monitor
timesheet data and surface only high-risk exceptions.

## Agentic Workflow

Operational Systems (Timesheets / ERP)
↓
Data Agent
↓
Policy Agent
↓
Tool: Project Effort Calculator
↓
Risk Agent
↓
LLM Insight Agent
↓
Reflection Agent
↓
Finance Insight

## Technology Stack

Python  
Groq LLM (Llama3)  
Pandas  

## Run

Install dependencies

pip install -r requirements.txt

Set Groq API key

export GROQ_API_KEY=your_api_key_here

Run the workflow

python app.py
