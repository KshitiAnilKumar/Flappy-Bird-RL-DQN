# play.py
"""
Interactive play / watch script.

Usage
─────
    python play.py                  # watch the trained DQN agent play
    python play.py --human          # play yourself (SPACE / ↑ to flap)
    python play.py --model path.pth # watch a specific checkpoint

Controls (human mode)
─────────────────────
    SPACE or ↑   – flap
    Q / ESC      – quit
"""

import sys
import os
import pygame

from environment import FlappyBirdEnv
from dqn_agent   import DQNAgent
from config      import FPS


# ──────────────────────────────────────────────────────────────────── #
def play_human():
    """Let the user control the bird manually."""
    env   = FlappyBirdEnv(render=True)
    state = env.reset()
    done  = False
    total = 0.0

    print("Human mode: SPACE / ↑ to flap.  Q / ESC to quit.")

    while True:
        action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close(); return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    action = 1
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    env.close(); return

        state, reward, done, info = env.step(action)
        total += reward

        if done:
            print(f"Game over!  Score={env.score}  Total reward={total:.2f}")
            pygame.time.wait(1200)
            state = env.reset()
            total = 0.0
            done  = False


# ──────────────────────────────────────────────────────────────────── #
def watch_agent(model_path: str = "checkpoints/best.pth"):
    """Load a trained DQN agent and watch it play."""
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Train the agent first:  python train.py")
        return

    agent = DQNAgent()
    agent.load(model_path)

    env   = FlappyBirdEnv(render=True)
    state = env.reset()
    total = 0.0
    done  = False

    print(f"Watching agent from {model_path}.  ESC / Q to quit.")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close(); return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    env.close(); return

        # Greedy action – no exploration during evaluation
        action               = agent.select_action(state, greedy=True)
        state, reward, done, info = env.step(action)
        total               += reward

        if done:
            print(f"Game over!  Score={env.score}  Total reward={total:.2f}")
            pygame.time.wait(1200)
            state = env.reset()
            total = 0.0
            done  = False


# ──────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    if "--human" in sys.argv:
        play_human()
    else:
        # Allow overriding the checkpoint path
        model_path = "checkpoints/best.pth"
        for arg in sys.argv[1:]:
            if arg.endswith(".pth") and os.path.exists(arg):
                model_path = arg
                break
        watch_agent(model_path)
