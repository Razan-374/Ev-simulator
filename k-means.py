
import pandas as pd

# Load your data
df = pd.read_csv("session_behaviour_clusters.csv")
def build_prebooking_distributions(df):
    df = df.dropna(subset=["arrival_time", "doneChargingTime"]).copy()
    all_days = pd.date_range(
        start=df["arrival_time"].dt.normalize().min(),
        end=df["doneChargingTime"].dt.normalize().max(),
        freq="D",
    )

    k_dist_by_type = {}
    rem_samples_by_type = {}

    for dtype in ["weekday", "friday", "weekend"]:
        ks = []
        rems = []

        for day_start in all_days:
            wd = day_start.weekday()
            day_type = "friday" if wd == 4 else ("weekend" if wd >= 5 else "weekday")
            if day_type != dtype:
                continue

            mask = (df["arrival_time"] < day_start) & (df["doneChargingTime"] > day_start)
            active = df.loc[mask]
            ks.append(len(active))
            if len(active):
                rems.extend((active["doneChargingTime"] - day_start).dt.total_seconds() / 60.0)

        k_series = pd.Series(ks)
        k_dist_by_type[dtype] = (k_series.value_counts(normalize=True).sort_index())
        rem_samples_by_type[dtype] = np.array(rems, dtype=float)

    return k_dist_by_type, rem_samples_by_type
