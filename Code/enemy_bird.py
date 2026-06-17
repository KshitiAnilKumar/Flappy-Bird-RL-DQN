# enemy_bird.py
"""
EnemyBird – a stationary hostile bird placed inside one of the two gaps
of a TriplePipe.  Touching it ends the episode with a full negative reward.
"""
import pygame
from config import RED, WHITE, BLACK, PIPE_WIDTH, SCREEN_HEIGHT

ENEMY_W = 32
ENEMY_H = 24


class EnemyBird:
    """
    Fixed obstacle bird anchored inside a TriplePipe gap.

    The enemy always follows the parent pipe's x-position so that it
    moves left together with the pipe as the game scrolls.

    Parameters
    ----------
    x          : initial horizontal centre position
    gap_center : vertical centre of the gap it lives in
    pipe       : the TriplePipe instance that owns this enemy
    """

    def __init__(self, x: float, gap_center: float, pipe):
        self.pipe       = pipe        # keeps a reference so x stays synced
        self.gap_center = gap_center  # fixed vertical position
        self.alive      = True        # set to False after collision (cosmetic)

    # ---------------------------------------------------------------- #
    @property
    def x(self) -> float:
        """Always centred on the parent pipe."""
        return self.pipe.x + PIPE_WIDTH / 2

    @property
    def y(self) -> float:
        return self.gap_center

    # ---------------------------------------------------------------- #
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            int(self.x) - ENEMY_W // 2,
            int(self.y) - ENEMY_H // 2,
            ENEMY_W,
            ENEMY_H,
        )

    # ---------------------------------------------------------------- #
    def update(self, speed: float):
        """No logic needed – position is derived from parent pipe."""
        pass

    # ---------------------------------------------------------------- #
    def draw(self, surface: pygame.Surface):
        if not self.alive:
            return

        cx, cy = int(self.x), int(self.y)
        hw, hh = ENEMY_W // 2, ENEMY_H // 2

        # ── Body (red) ──────────────────────────────────────────────── #
        pygame.draw.ellipse(surface, RED,
                            (cx - hw, cy - hh, ENEMY_W, ENEMY_H))
        pygame.draw.ellipse(surface, BLACK,
                            (cx - hw, cy - hh, ENEMY_W, ENEMY_H), 2)

        # ── Eye ─────────────────────────────────────────────────────── #
        pygame.draw.circle(surface, WHITE, (cx - 7, cy - 4), 6)
        pygame.draw.circle(surface, BLACK, (cx - 8, cy - 4), 3)

        # ── Angry eyebrow ────────────────────────────────────────────── #
        pygame.draw.line(surface, BLACK,
                         (cx - 13, cy - 11), (cx - 2, cy - 8), 3)

        # ── Beak (pointing left, towards the player) ─────────────────── #
        beak_base = cx - hw
        pygame.draw.polygon(surface, (210, 100, 0),
                            [(beak_base,     cy - 3),
                             (beak_base - 9, cy + 1),
                             (beak_base,     cy + 5)])
