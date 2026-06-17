"""
Coin – a collectible reward token placed in the safe gap of a
NormalPipe.
"""
import pygame
from config import YELLOW, GOLD, BLACK, COIN_RADIUS


class Coin:
    """
    A collectible gold coin.

    Parameters
    ----------
    x, y : float – centre position when spawned
    """

    def __init__(self, x: float, y: float):
        self.x         = x
        self.y         = y
        self.collected = False
        self._spin     = 0.0

    # ---------------------------------------------------------------- #
    def update(self, speed: float):
        """Scroll left and advance spin animation."""
        self.x    -= speed
        self._spin = (self._spin + 5.0) % 360.0

    # ---------------------------------------------------------------- #
    @property
    def rect(self) -> pygame.Rect:
        r = COIN_RADIUS
        return pygame.Rect(int(self.x) - r, int(self.y) - r, r * 2, r * 2)

    def is_off_screen(self) -> bool:
        return self.x + COIN_RADIUS < 0

    # ---------------------------------------------------------------- #
    def draw(self, surface: pygame.Surface):
        if self.collected:
            return

        cx, cy = int(self.x), int(self.y)
        r      = COIN_RADIUS

        pygame.draw.circle(surface, GOLD, (cx, cy), r)

        import math
        shine_w = max(1, int(abs(math.cos(math.radians(self._spin))) * (r - 4)))
        pygame.draw.ellipse(
            surface,
            YELLOW,
            (cx - shine_w, cy - (r - 4), shine_w * 2, (r - 4) * 2)
        )

        pygame.draw.circle(surface, BLACK, (cx, cy), r, 2)