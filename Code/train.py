

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import imageio.v2 as imageio

import numpy as np

from environment import FlappyBirdEnv
from dqn_agent import DQNAgent



# Training settings
MAX_EPISODES = 20_000
MAX_STEPS_EP = 6_000
SAVE_EVERY = 500
PRINT_EVERY = 20
WARMUP_EPISODES = 200
SCORE_WINDOW = 50

CHECKPOINT_DIR = Path("checkpoints")
RECORDING_DIR = Path("recordings")
CHECKPOINT_DIR.mkdir(exist_ok=True)
RECORDING_DIR.mkdir(exist_ok=True)


# Plotting helpers
def _moving_average(values, window):
    if not values:
        return [], []
    window = min(window, len(values))
    ma = np.convolve(values, np.ones(window) / window, mode="valid")
    episodes = list(range(window, len(values) + 1))
    return episodes, ma


def _save_single_plot(values, ylabel, title, path):
    """Save one clean plot. No custom colors, so matplotlib defaults are used."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        episodes = list(range(1, len(values) + 1))
        ma_ep, ma = _moving_average(values, SCORE_WINDOW)

        plt.figure(figsize=(10, 4))
        plt.plot(episodes, values, alpha=0.35, label=ylabel)
        if len(ma):
            plt.plot(ma_ep, ma, linewidth=2, label=f"MA-{min(SCORE_WINDOW, len(values))}")
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=130)
        plt.close()
    except Exception as exc:
        print(f"[plot] Skipped {path}: {exc}")


def _save_plots(all_rewards, all_scores):
    _save_single_plot(
        all_rewards,
        "Reward",
        "Rewards vs Episodes",
        CHECKPOINT_DIR / "rewards_vs_episodes.png",
    )
    _save_single_plot(
        all_scores,
        "Score",
        "Score vs Episodes",
        CHECKPOINT_DIR / "score_vs_episodes.png",
    )


def _save_metrics_csv(rows):
    if not rows:
        return

    path = CHECKPOINT_DIR / "training_metrics.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "score",
                "reward",
                "best_score",
                "avg_score_50",
                "avg_reward_50",
                "epsilon",
                "phase",
                "loss",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


# Video recording helpers
class VideoRecorder:
    """
    Records the pygame screen to MP4.

    Uses imageio. Install with:
        pip install imageio imageio-ffmpeg
    """

    def __init__(self, enabled: bool, fps: int):
        self.enabled = enabled
        self.fps = int(fps)
        self.path = None
        self.writer = None

        if not enabled:
            return

        try:
            import imageio.v2 as imageio
            self.imageio = imageio
        except Exception as exc:
            print(f"[recording] imageio not available, video recording disabled: {exc}")
            self.enabled = False
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = RECORDING_DIR / f"training_{stamp}.mp4"
        self.writer = self.imageio.get_writer(str(self.path), fps=self.fps, macro_block_size=1)
        print(f"[recording] Saving training video -> {self.path}")

    def add_frame(self, env):
        if not self.enabled or self.writer is None or env._screen is None:
            return

        import pygame

        # pygame gives (width, height, channels). imageio expects (height, width, channels).
        frame = pygame.surfarray.array3d(env._screen)
        frame = np.transpose(frame, (1, 0, 2))
        self.writer.append_data(frame)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            print(f"[recording] Video saved -> {self.path}")


def train(render=False, speed=64, max_episodes=MAX_EPISODES, record=True):
    env = FlappyBirdEnv(render=render, render_fps=speed)
    agent = DQNAgent()

    all_rewards = []
    all_scores = []
    metrics_rows = []
    best_score = 0

    total_steps = 0
    t0 = time.time()
    best_path = CHECKPOINT_DIR / "best.pth"

    recorder = VideoRecorder(enabled=(render and record), fps=speed)

    # Resume if a previous best model exists.
    # Important: delete checkpoints/best.pth if you want a fully fresh run.
    if best_path.exists():
        try:
            agent.load(str(best_path))
            print(f"Resuming from {best_path}")
        except Exception as exc:
            print(f"Could not load checkpoint: {exc}. Starting fresh.")

    print("\nStarting training")
    print(f"Device        : {agent.device}")
    print(f"Render        : {render}")
    print(f"Speed/FPS     : {speed}")
    print(f"Record video  : {render and record}")
    print(f"Max episodes  : {max_episodes}")
    print(f"Warm-up       : {WARMUP_EPISODES} random episodes")
    print("Tip           : for fastest learning, run without --render first.\n")

    try:
        for episode in range(1, max_episodes + 1):
            state = env.reset()
            ep_reward = 0.0
            ep_loss = []
            is_warmup = episode <= WARMUP_EPISODES
            phase = "WARMUP" if is_warmup else "LEARNING"

            for step in range(MAX_STEPS_EP):
                if render:
                    import pygame
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt

                if is_warmup:
                    # Slight flap bias creates more useful early experiences.
                    action = 1 if np.random.random() < 0.45 else 0
                else:
                    action = agent.select_action(state)

                next_state, reward, done, info = env.step(action)

                # Keep raw reward for graphs, but clip for stable Q-learning.
                clipped_reward = float(np.clip(reward, -1.0, 1.0))
                agent.store(state, action, clipped_reward, next_state, done)

                if not is_warmup:
                    loss = agent.learn()
                    if loss is not None:
                        ep_loss.append(loss)

                state = next_state
                ep_reward += reward
                total_steps += 1

                elapsed = max(time.time() - t0, 1e-6)
                avg50 = float(np.mean(all_scores[-SCORE_WINDOW:])) if all_scores else 0.0

                if render:
                    env.set_training_info(
                        episode=episode,
                        max_episodes=max_episodes,
                        best=best_score,
                        avg50=avg50,
                        reward=ep_reward,
                        epsilon=agent.epsilon,
                        phase=phase,
                        speed=speed,
                        iters_per_sec=total_steps / elapsed,
                        recording=recorder.enabled,
                    )
                    # Draw one more frame after updating overlay info.
                    env.render()
                    recorder.add_frame(env)

                if done:
                    break

            score = env.score
            all_rewards.append(float(ep_reward))
            all_scores.append(int(score))

            if score > best_score and not is_warmup:
                best_score = score
                agent.save(str(best_path))

            window = min(SCORE_WINDOW, len(all_scores))
            avg_score = float(np.mean(all_scores[-window:]))
            avg_reward = float(np.mean(all_rewards[-window:]))
            avg_loss = float(np.mean(ep_loss)) if ep_loss else 0.0

            metrics_rows.append(
                {
                    "episode": episode,
                    "score": int(score),
                    "reward": float(ep_reward),
                    "best_score": int(best_score),
                    "avg_score_50": avg_score,
                    "avg_reward_50": avg_reward,
                    "epsilon": float(agent.epsilon),
                    "phase": phase,
                    "loss": avg_loss,
                }
            )

            if episode % PRINT_EVERY == 0:
                elapsed = time.time() - t0
                max_window_score = max(all_scores[-window:])
                print(
                    f"Ep {episode:6d}/{max_episodes}"
                    f" | reward_avg50={avg_reward:8.2f}"
                    f" | score_avg50={avg_score:5.2f}"
                    f" | max50={max_window_score:3d}"
                    f" | best={best_score:3d}"
                    f" | eps={agent.epsilon:.4f}"
                    f" | {phase}"
                    f" | loss={avg_loss:.5f}"
                    f" | steps/s={total_steps / max(elapsed, 1e-6):.1f}"
                )

            if episode % SAVE_EVERY == 0:
                agent.save(str(CHECKPOINT_DIR / f"ep{episode:06d}.pth"))
                _save_plots(all_rewards, all_scores)
                _save_metrics_csv(metrics_rows)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving current model, plots, metrics, and video...")
        agent.save(str(CHECKPOINT_DIR / "interrupted.pth"))

    finally:
        agent.save(str(CHECKPOINT_DIR / "final.pth"))
        _save_plots(all_rewards, all_scores)
        _save_metrics_csv(metrics_rows)
        recorder.close()
        env.close()

    print(f"\nTraining complete. Best score: {best_score}")
    print(f"Rewards graph : {CHECKPOINT_DIR / 'rewards_vs_episodes.png'}")
    print(f"Score graph   : {CHECKPOINT_DIR / 'score_vs_episodes.png'}")
    print(f"Metrics CSV   : {CHECKPOINT_DIR / 'training_metrics.csv'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Show pygame window while training.")
    parser.add_argument("--speed", type=int, default=64, help="Render FPS and recording FPS. Default: 64.")
    parser.add_argument("--episodes", type=int, default=MAX_EPISODES, help="Number of training episodes.")
    parser.add_argument("--no-record", action="store_true", help="Disable MP4 recording even when --render is used.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        render=args.render,
        speed=args.speed,
        max_episodes=args.episodes,
        record=not args.no_record,
    )
