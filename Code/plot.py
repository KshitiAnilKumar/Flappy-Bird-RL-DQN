import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

df = pd.read_csv("checkpoints/training_metrics.csv")

window = 100  # rolling window size

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("Flappy Bird DQN", fontsize=13, y=0.01)

# --- Score per Episode ---
ax1.fill_between(df["episode"], df["score"], alpha=0.3, color="steelblue")
ax1.plot(df["episode"], df["score"].rolling(window).mean(), color="steelblue", linewidth=1)
ax1.set_title("Score per Episode")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Score (pipes passed)")
ax1.grid(True, linestyle="--", alpha=0.5)

# --- Total Reward per Episode ---
rolling_mean = df["reward"].rolling(window).mean()
rolling_std  = df["reward"].rolling(window).std()

ax2.fill_between(df["episode"], df["reward"], alpha=0.25, color="orange")
ax2.fill_between(df["episode"],
                 rolling_mean - rolling_std,
                 rolling_mean + rolling_std,
                 alpha=0.3, color="orange")
ax2.plot(df["episode"], rolling_mean, color="orange", linewidth=2)
ax2.set_title("Total Reward per Episode")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Total Reward")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("score_plot.png", dpi=150, bbox_inches="tight")
print("Saved!")