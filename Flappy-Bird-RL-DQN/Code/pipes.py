"""
Two types of pipe obstacles:

  NormalPipe  – classic top/bottom pipe pair with a single gap.
                Always fixed vertically.

  TriplePipe  – three fixed pipes creating two separate gaps.
                Never moves; always stays at a fixed vertical layout.
"""

import pygame
from config import (
    SCREEN_HEIGHT,
    PIPE_WIDTH, NORMAL_GAP,
    TRIPLE_GAP, TRIPLE_MID_H,
    GREEN, DARK_GREEN,
)


class NormalPipe:
    """
    Standard top + bottom pipe pair with one opening (gap).

    Parameters
    ----------
    x          : float – left edge starting position
    gap_center : float – y-coordinate of the gap's vertical centre
    """

    def __init__(self, x: float, gap_center: float):
        self.x          = x
        self.gap_center = gap_center
        self.passed     = False

    # ---------------------------------------------------------------- #
    @property
    def top_rect(self) -> pygame.Rect:
        """Top pipe: from y=0 down to just above the gap."""
        h = max(0, int(self.gap_center - NORMAL_GAP / 2))
        return pygame.Rect(int(self.x), 0, PIPE_WIDTH, h)

    @property
    def bottom_rect(self) -> pygame.Rect:
        """Bottom pipe: from just below the gap down to screen bottom."""
        top_y = int(self.gap_center + NORMAL_GAP / 2)
        h     = max(0, SCREEN_HEIGHT - top_y)
        return pygame.Rect(int(self.x), top_y, PIPE_WIDTH, h)

    @property
    def gap_top_y(self) -> float:
        return self.gap_center - NORMAL_GAP / 2

    @property
    def gap_bottom_y(self) -> float:
        return self.gap_center + NORMAL_GAP / 2

    # ---------------------------------------------------------------- #
    def update(self, speed: float):
        """Move left at the current game speed."""
        self.x -= speed

    # ---------------------------------------------------------------- #
    def is_off_screen(self) -> bool:
        return self.x + PIPE_WIDTH < 0

    # ---------------------------------------------------------------- #
    def draw(self, surface: pygame.Surface):
        """Draw top pipe, bottom pipe, and cap ridges."""
        self._draw_single_pipe(surface, self.top_rect,    cap_at_bottom=True)
        self._draw_single_pipe(surface, self.bottom_rect, cap_at_bottom=False)

    @staticmethod
    def _draw_single_pipe(surface: pygame.Surface, rect: pygame.Rect,
                          cap_at_bottom: bool):
        if rect.height <= 0:
            return

        pygame.draw.rect(surface, GREEN, rect)
        pygame.draw.rect(surface, DARK_GREEN, rect, 3)

        cap_h = 18
        cap_x = rect.x - 5
        cap_w = PIPE_WIDTH + 10
        cap_y = (rect.bottom - cap_h) if cap_at_bottom else rect.top
        cap   = pygame.Rect(cap_x, cap_y, cap_w, cap_h)

        pygame.draw.rect(surface, GREEN, cap)
        pygame.draw.rect(surface, DARK_GREEN, cap, 3)


class TriplePipe:
    """
    Three-pipe fixed obstacle that creates TWO gaps for the bird to fly through.

    Vertical layout (top → bottom):
    ┌──────────────┐  ← top pipe
    └──────────────┘
         [gap 1]
    ┌──────────────┐  ← middle pipe
    └──────────────┘
         [gap 2]
    ┌──────────────┐  ← bottom pipe
    └──────────────┘

    This obstacle NEVER moves vertically.
    """

    def __init__(self, x: float):
        self.x      = x
        self.passed = False

        q = SCREEN_HEIGHT / 4

        top_pipe_bottom = q - TRIPLE_GAP / 2
        self.top_pipe_h = max(10, int(top_pipe_bottom))

        mid_centre      = SCREEN_HEIGHT / 2
        self.mid_pipe_y = int(mid_centre - TRIPLE_MID_H / 2)
        self.mid_pipe_h = TRIPLE_MID_H

        self.bot_pipe_y = int(3 * q + TRIPLE_GAP / 2)
        self.bot_pipe_h = max(10, SCREEN_HEIGHT - self.bot_pipe_y)

        self.gap1_center = float(self.top_pipe_h) + TRIPLE_GAP / 2
        self.gap2_center = float(self.mid_pipe_y + self.mid_pipe_h) + TRIPLE_GAP / 2

    # ---------------------------------------------------------------- #
    @property
    def top_rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), 0, PIPE_WIDTH, self.top_pipe_h)

    @property
    def mid_rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), self.mid_pipe_y, PIPE_WIDTH, self.mid_pipe_h)

    @property
    def bot_rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), self.bot_pipe_y, PIPE_WIDTH, self.bot_pipe_h)

    # ---------------------------------------------------------------- #
    def update(self, speed: float):
        """Move left at the current game speed."""
        self.x -= speed

    def is_off_screen(self) -> bool:
        return self.x + PIPE_WIDTH < 0

    def collides(self, bird_rect: pygame.Rect) -> bool:
        """True if the bird touches any of the three pipe segments."""
        return any(r.colliderect(bird_rect)
                   for r in (self.top_rect, self.mid_rect, self.bot_rect))

    # ---------------------------------------------------------------- #
    def draw(self, surface: pygame.Surface):
        """Draw all three pipe segments with caps."""
        for rect in (self.top_rect, self.mid_rect, self.bot_rect):
            pygame.draw.rect(surface, GREEN, rect)
            pygame.draw.rect(surface, DARK_GREEN, rect, 3)

        NormalPipe._draw_single_pipe(surface, self.top_rect, cap_at_bottom=True)
        NormalPipe._draw_single_pipe(surface, self.bot_rect, cap_at_bottom=False)