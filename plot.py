import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

def plot_metric(df, metric: str, out_file: str):
    """
    Draw a bar plot for a single metric (e.g., avg_watch_fraction or genre_diversity).
    Saves the figure to the given path.
    """
    plt.figure(figsize=(7, 4))
    bars = plt.bar(df["approach"], df[metric], color="skyblue")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} per Approach")
    plt.xticks(rotation=30)
    plt.tight_layout()

    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f"{height:.2f}",
                 ha='center', va='bottom', fontsize=8)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved plot: {out_file}")

def plot_engagement_sweep(df, out_file: str):
    """
    Plot engagement rate as a function of the threshold (sweep) for each approach.
    """
    # Find all columns that match 'engagement_rate_*'
    engagement_cols = [col for col in df.columns if re.match(r"engagement_rate_", col)]
    # Extract the thresholds as float values from column names
    thresholds = [float(col.split("_")[-1]) for col in engagement_cols]
    thresholds, engagement_cols = zip(*sorted(zip(thresholds, engagement_cols)))

    plt.figure(figsize=(7, 5))
    for _, row in df.iterrows():
        approach = row["approach"]
        rates = [row[col] for col in engagement_cols]
        plt.plot(thresholds, rates, marker='o', label=approach)
    plt.xlabel("Engagement threshold")
    plt.ylabel("Engagement rate")
    plt.title("Engagement Rate vs. Threshold (per Approach)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved sweep plot: {out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="eval/metrics.csv", help="Path to metrics CSV")
    parser.add_argument("--metrics", nargs="*", default=[
        "avg_return", "avg_watch_fraction", "mean_watch_fraction", "genre_diversity"
    ], help="Metrics to plot (bar plots)")
    parser.add_argument("--sweep", action="store_true", help="Plot engagement rate sweep if possible")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Plot bar for each metric requested
    for metric in args.metrics:
        if metric not in df.columns:
            print(f"⚠️ Warning: Column '{metric}' not found in CSV. Skipping.")
            continue
        out_file = f"eval/plot_{metric}.png"
        plot_metric(df, metric, out_file)

    # Plot engagement sweep if requested and columns exist
    engagement_cols = [col for col in df.columns if col.startswith("engagement_rate_")]
    if args.sweep and engagement_cols:
        plot_engagement_sweep(df, "eval/plot_engagement_sweep.png")
    elif args.sweep:
        print("⚠️ No engagement_rate_* columns found for sweep plot.")

if __name__ == "__main__":
    main()
