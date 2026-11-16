import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("session_behaviour_clusters.csv", parse_dates=['arrival_time', 'doneChargingTime'])
df['arrival_time'] = pd.to_datetime(df['arrival_time'])
df['arrival_hour'] = df['arrival_time'].dt.hour + df['arrival_time'].dt.minute/60

df['doneChargingTime'] = pd.to_datetime(df['doneChargingTime'])
df['parking_duration'] = (df['doneChargingTime'] - df['arrival_time']).dt.total_seconds() / 3600

# arrival_hour and parking_duration are already in the CSV
# cluster is already in the CSV

# Make sure we have date column (day granularity)
df['date'] = df['arrival_time'].dt.date

# Total number of days in the dataset
n_days = df['date'].nunique()
print("Number of days in dataset:", n_days)

arrival_dists = {}  # will store λ_g(t) as DataFrames

for g in sorted(df['cluster'].unique()):
    df_g = df[df['cluster'] == g]

    # Histogram of arrival_hour in 24 bins: [0–1), [1–2), ..., [23–24)
    counts, bin_edges = np.histogram(df_g['arrival_hour'], bins=np.arange(0, 25, 1))

    # Expected arrivals per hour per day
    lambda_per_hour = counts / n_days   # shape (24,)

    hours = np.arange(24)
    arrival_df = pd.DataFrame({
        'hour': hours,
        'lambda_per_day': lambda_per_hour
    })
    arrival_dists[g] = arrival_df

    print(f"\nCluster {g} arrival distribution (λ_g(t)):")
    print(arrival_df)

    # Optional: plot
    plt.figure(figsize=(5,3))
    plt.bar(hours, lambda_per_hour, width=0.8)
    plt.xlabel("Hour of day")
    plt.ylabel("Expected arrivals per day")
    plt.title(f"Cluster {g} – Arrival rate λ_g(t)")
    plt.xticks(range(0,24,2))
    plt.tight_layout()
    plt.show()
