import pandas as pd
import numpy as np

# Load your internal session file (already has arrival_time)
df_int = pd.read_csv("session_behaviour_clusters.csv", parse_dates=["arrival_time"])

# Add a date column
df_int["date"] = df_int["arrival_time"].dt.date

# Count internal EV sessions per day
daily_counts = df_int.groupby("date").size()

# Convert to probabilities
count_dist = daily_counts.value_counts(normalize=True).sort_index()

print("Daily internal EV count distribution:")
print(count_dist)

# Save for simulator
count_dist.to_csv("internal_daily_count_distribution.csv")
