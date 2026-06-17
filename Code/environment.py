"""
FlappyBirdEnv – OpenAI Gym-style reinforcement learning environment.

──────────────────────────────────────────────────────────────────────
State vector (7 normalised floats)
──────────────────────────────────────────────────────────────────────
  Index  Symbol              Description
  ─────  ──────────────────  ──────────────────────────────────────────
    0    bird_y_norm         Bird's y position / SCREEN_HEIGHT
    1    bird_vel_norm       Bird's velocity clamped to [-1, +1]
    2    dx_norm             Horizontal distance to next obstacle / SCREEN_WIDTH
    3    gap1_center_norm    Centre y of the primary (or only) gap / SCREEN_HEIGHT
    4    gap2_center_norm    Centre y of the secondary gap (= gap1 for NormalPipe)
    5    is_triple           1.0 if the next obstacle is a TriplePipe, else 0.0
    6    speed_norm          Current scroll speed normalised to [0, 1]

Actions
──────────────────────────────────────────────────────────────────────
    0  – do nothing
    1  – flap
"""

import random
import numpy as np
import pygame

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS,
    SKY_BLUE, WHITE, BROWN,
    BIRD_WIDTH, MAX_FALL_SPEED,
    PIPE_WIDTH, PIPE_SPACING, NORMAL_GAP,
    COIN_SPAWN_CHANCE,
    INITIAL_SPEED, MAX_SPEED, SPEED_PER_SCORE,
    REWARD_ALIVE, REWARD_PASS, REWARD_COIN, REWARD_COLLISION,
    STATE_SIZE,
)
from bird import Bird
from pipes import NormalPipe, TriplePipe
from enemy_bird import EnemyBird
from coin import Coin


class FlappyBirdEnv:
    """
    Main game / RL environment.

    Parameters
    ----------
    render : bool
        Set True to display the pygame window.
        Set False for headless training.
    """

    def __init__(self, render: bool = True, render_fps: int = FPS):
        self._render_on  = render
        self._render_fps = int(render_fps)
        self._screen     = None
        self._clock      = None
        self._font       = None
        self._small_font = None

        if render:
            pygame.init()
            self._screen     = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird – RL  |  SPACE to flap")
            self._clock      = pygame.time.Clock()
            self._font       = pygame.font.SysFont("Arial", 26, bold=True)
            self._small_font = pygame.font.SysFont("Arial", 18)

        self.bird             = None
        self.obstacles        = []
        self.enemy_birds      = []
        self.coins            = []
        self.score            = 0
        self.speed            = INITIAL_SPEED
        self.frame            = 0
        self._next_spawn_x    = 0.0
        self._coins_collected = 0
        self._total_reward    = 0.0
        self.training_info    = {}

    # ══════════════════════════════════════════════════════════════════ #
    # Public Gym-style API
    # ══════════════════════════════════════════════════════════════════ #

    def reset(self) -> np.ndarray:
        self.bird             = Bird()
        self.obstacles        = []
        self.enemy_birds      = []
        self.coins            = []
        self.score            = 0
        self.speed            = INITIAL_SPEED
        self.frame            = 0
        self._next_spawn_x    = float(SCREEN_WIDTH + 80)
        self._coins_collected = 0
        self._total_reward    = 0.0

        for _ in range(2):
            self._spawn_obstacle()

        return self._get_state()

    # ---------------------------------------------------------------- #
    def step(self, action: int):
        if action not in (0, 1):
            raise ValueError(f"Invalid action {action!r}. Must be 0 or 1.")

        reward = REWARD_ALIVE
        done   = False
        info   = {}

        if action == 1:
            self.bird.flap()

        self.bird.update()

        target     = min(INITIAL_SPEED + self.score * SPEED_PER_SCORE, MAX_SPEED)
        self.speed = self.speed * 0.995 + target * 0.005

        # Obstacles
        for obs in self.obstacles:
            obs.update(self.speed)

            bird_left = self.bird.x - BIRD_WIDTH / 2
            if not obs.passed and (obs.x + PIPE_WIDTH) < bird_left:
                obs.passed  = True
                self.score += 1
                reward     += REWARD_PASS
                info.setdefault("events", []).append("pipe_passed")

            if isinstance(obs, NormalPipe):
                if (obs.top_rect.colliderect(self.bird.rect)
                        or obs.bottom_rect.colliderect(self.bird.rect)):
                    reward += REWARD_COLLISION
                    done    = True
                    info.setdefault("events", []).append("hit_pipe")
                elif not done:
                    # FIX (reward shaping): small penalty proportional to how
                    # far the bird is from the gap centre.  This teaches the
                    # agent to actively aim for the middle of each gap rather
                    # than just scraping through the edge.
                    gap_dist = abs(self.bird.y - obs.gap_center) / (NORMAL_GAP / 2)
                    reward  -= 0.02 * min(gap_dist, 1.0)  # max −0.02 per frame

            elif isinstance(obs, TriplePipe):
                if obs.collides(self.bird.rect):
                    reward += REWARD_COLLISION
                    done    = True
                    info.setdefault("events", []).append("hit_triple_pipe")

        # Enemy birds
        for enemy in self.enemy_birds:
            enemy.update(self.speed)
            if enemy.alive and enemy.rect.colliderect(self.bird.rect):
                enemy.alive = False
                reward     += REWARD_COLLISION   # was: reward = REWARD_COLLISION
                done        = True
                info.setdefault("events", []).append("hit_enemy_bird")

        # Coins
        for coin in self.coins:
            if not coin.collected:
                coin.update(self.speed)
                if coin.rect.colliderect(self.bird.rect):
                    coin.collected = True
                    reward        += REWARD_COIN
                    self._coins_collected += 1
                    info.setdefault("events", []).append("coin_collected")

        # Boundary collision
        # FIX: use += not = so the alive reward isn't lost; also ensures
        # REWARD_COLLISION is always applied on top of the frame reward.
        if self.bird.y >= SCREEN_HEIGHT - 20 or self.bird.y <= 0:
            reward += REWARD_COLLISION   # was: reward = REWARD_COLLISION
            done    = True
            info.setdefault("events", []).append("boundary")

        # Cull off-screen
        self.obstacles = [o for o in self.obstacles if not o.is_off_screen()]
        live_pipes     = set(id(o) for o in self.obstacles)
        self.enemy_birds = [e for e in self.enemy_birds if id(e.pipe) in live_pipes]
        self.coins = [c for c in self.coins if not c.is_off_screen() and not c.collected]

        # Spawn new obstacles
        # FIX: was checking `rightmost_x < SCREEN_WIDTH - PIPE_SPACING`, which
        # only fires once a pipe has almost scrolled off-screen and causes long
        # gaps at higher speeds.  Now we spawn whenever the rightmost pipe is
        # within one PIPE_SPACING of the right edge so density stays constant.
        rightmost_x = max((o.x for o in self.obstacles), default=0)
        if rightmost_x < SCREEN_WIDTH + PIPE_SPACING - 50:
            self._spawn_obstacle()

        self.frame         += 1
        self._total_reward += reward
        info["score"]       = self.score
        info["speed"]       = round(self.speed, 2)
        info["coins"]       = self._coins_collected

        if self._render_on:
            self._render()

        return self._get_state(), reward, done, info

    # ---------------------------------------------------------------- #
    def render(self):
        if self._screen is not None:
            self._render()

    def set_training_info(self, **kwargs):
        """Update values shown in the render/recording training overlay."""
        self.training_info.update(kwargs)

    def close(self):
        if self._render_on:
            pygame.quit()

    # ══════════════════════════════════════════════════════════════════ #
    # State vector
    # ══════════════════════════════════════════════════════════════════ #

    def _get_state(self) -> np.ndarray:
        bird_y_norm   = np.clip(self.bird.y / SCREEN_HEIGHT, 0.0, 1.0)
        bird_vel_norm = np.clip(self.bird.velocity / MAX_FALL_SPEED, -1.0, 1.0)

        next_obs = self._next_obstacle()

        if next_obs is None:
            dx_norm   = 1.0
            gap1_norm = 0.5
            gap2_norm = 0.5
            is_triple = 0.0
        else:
            bird_front = self.bird.x + BIRD_WIDTH / 2
            dx         = max(0.0, next_obs.x - bird_front)
            dx_norm    = np.clip(dx / SCREEN_WIDTH, 0.0, 1.0)

            if isinstance(next_obs, TriplePipe):
                gap1_norm = next_obs.gap1_center / SCREEN_HEIGHT
                gap2_norm = next_obs.gap2_center / SCREEN_HEIGHT
                is_triple = 1.0
            else:
                gap1_norm = next_obs.gap_center / SCREEN_HEIGHT
                gap2_norm = gap1_norm
                is_triple = 0.0

        speed_norm = np.clip(
            (self.speed - INITIAL_SPEED) / max(MAX_SPEED - INITIAL_SPEED, 1e-6),
            0.0, 1.0
        )

        state = np.array([
            bird_y_norm,
            bird_vel_norm,
            dx_norm,
            gap1_norm,
            gap2_norm,
            is_triple,
            speed_norm,
        ], dtype=np.float32)

        assert state.shape == (STATE_SIZE,), f"State shape mismatch: {state.shape}"
        return state

    # ══════════════════════════════════════════════════════════════════ #
    # Obstacle / coin spawning
    # ══════════════════════════════════════════════════════════════════ #

    def _spawn_obstacle(self):
        x = self._next_spawn_x
        self._next_spawn_x += PIPE_SPACING + random.uniform(-20, 20)

        if random.random() < 0.28:
            self._spawn_triple(x)
        else:
            self._spawn_normal(x)

    # ---------------------------------------------------------------- #
    def _spawn_normal(self, x: float):
        half_gap   = NORMAL_GAP / 2
        margin     = half_gap + 50
        gap_center = random.uniform(margin, SCREEN_HEIGHT - margin)

        pipe = NormalPipe(x, gap_center)
        self.obstacles.append(pipe)

        if random.random() < COIN_SPAWN_CHANCE:
            coin_y = gap_center + random.uniform(-half_gap * 0.4, half_gap * 0.4)
            coin_x = x + PIPE_WIDTH / 2 + random.uniform(-10, 10)
            self.coins.append(Coin(coin_x, coin_y))

    # ---------------------------------------------------------------- #
    def _spawn_triple(self, x: float):
        pipe = TriplePipe(x)
        self.obstacles.append(pipe)

        gap_center = random.choice([pipe.gap1_center, pipe.gap2_center])
        enemy      = EnemyBird(x + PIPE_WIDTH / 2, gap_center, pipe)
        self.enemy_birds.append(enemy)

    # ---------------------------------------------------------------- #
    def _next_obstacle(self):
        bird_front = self.bird.x + BIRD_WIDTH / 2
        candidates = [
            o for o in self.obstacles
            if (o.x + PIPE_WIDTH) > bird_front and not o.passed
        ]
        return min(candidates, key=lambda o: o.x, default=None)

    # ══════════════════════════════════════════════════════════════════ #
    # Rendering
    # ══════════════════════════════════════════════════════════════════ #

    def _render(self):
        self._screen.fill(SKY_BLUE)
        self._draw_clouds()

        ground_y = SCREEN_HEIGHT - 20
        pygame.draw.rect(self._screen, BROWN, (0, ground_y, SCREEN_WIDTH, 20))
        pygame.draw.rect(self._screen, (80, 50, 20), (0, ground_y, SCREEN_WIDTH, 3))

        for obs in self.obstacles:
            obs.draw(self._screen)

        for coin in self.coins:
            coin.draw(self._screen)

        for enemy in self.enemy_birds:
            enemy.draw(self._screen)

        self.bird.draw(self._screen)
        self._draw_hud()

        pygame.display.flip()
        self._clock.tick(self._render_fps)

    _CLOUD_POSITIONS = [(60, 80), (200, 50), (340, 100), (430, 65)]

    def _draw_clouds(self):
        for cx, cy in self._CLOUD_POSITIONS:
            for dx, dy, r in ((-18, 5, 18), (0, 0, 22), (18, 5, 18)):
                pygame.draw.circle(self._screen, WHITE, (cx + dx, cy + dy), r)

    def _draw_hud(self):
        hud = self._font.render(f"Score: {self.score}", True, WHITE)
        self._screen.blit(hud, (10, 8))

        spd = self._small_font.render(
            f"Speed: {self.speed:.1f}   Coins: {self._coins_collected}",
            True, WHITE
        )
        self._screen.blit(spd, (10, 40))

        # Training/recording overlay. These values are injected by train.py
        # before each rendered frame, so the saved video contains the same
        # details shown in your screenshot.
        if self.training_info:
            panel = pygame.Surface((230, 160), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 165))
            self._screen.blit(panel, (250, 8))

            lines = [
                (f"[ TRAINING ]  {'REC' if self.training_info.get('recording') else ''}", (255, 90, 70)),
                (f"Episode : {self.training_info.get('episode', 0)}/{self.training_info.get('max_episodes', 0)}", WHITE),
                (f"Score   : {self.score}", (120, 255, 120)),
                (f"Best    : {self.training_info.get('best', 0)}", (255, 220, 80)),
                (f"Avg-50  : {self.training_info.get('avg50', 0):.2f}", (210, 210, 255)),
                (f"Reward  : {self.training_info.get('reward', 0):.1f}", WHITE),
                (f"Epsilon : {self.training_info.get('epsilon', 0):.4f}", (210, 210, 255)),
                (f"Phase   : {self.training_info.get('phase', 'LEARNING')}", (120, 255, 120)),
                (f"Speed   : {self.training_info.get('speed', self._render_fps)} fps", WHITE),
                (f"Iter/s  : {self.training_info.get('iters_per_sec', 0):.1f}", WHITE),
            ]

            y = 14
            for text, color in lines:
                surf = self._small_font.render(text, True, color)
                self._screen.blit(surf, (260, y))
                y += 15