import pandas as pd

# Load dataset
df = pd.read_csv("data/timesheets.csv")

# Finance policy
MAX_HOURS = 8

# Detect anomalies
df["anomaly"] = df["hours"] > MAX_HOURS

# Risk classification
def risk_level(hours):
    if hours > 11:
        return "High"
    elif hours > 8:
        return "Medium"
    else:
        return "Low"

df["risk"] = df["hours"].apply(risk_level)

print("Timesheet Analysis\n")
print(df)

print("\nFlagged Exceptions\n")
print(df[df["anomaly"] == True])
