# bird.py
"""
Player bird – simple Euler-integration physics with cosmetic drawing.
"""
import pygame
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    BIRD_WIDTH, BIRD_HEIGHT,
    GRAVITY, FLAP_STRENGTH, MAX_FALL_SPEED,
    WHITE, YELLOW, ORANGE, BLACK,
)


class Bird:
    """
    The agent / player character.

    Attributes
    ----------
    x, y     : float  – centre of the bird in screen-space pixels
    velocity : float  – vertical velocity (positive → downward)
    alive    : bool   – set to False by the environment on collision
    """

    # Bird starts left-of-centre, vertically centred
    START_X = 90
    START_Y = SCREEN_HEIGHT // 2

    def __init__(self):
        self.reset()


    def reset(self):
        """Return the bird to its initial position and zero its velocity."""
        self.x        = float(self.START_X)
        self.y        = float(self.START_Y)
        self.velocity = 0.0
        self.alive    = True

    def flap(self):
        """Apply an upward impulse (action = 1)."""
        self.velocity = FLAP_STRENGTH


    def update(self):
        """Advance physics by one game frame."""
        # Apply gravity, clamped to terminal velocity
        self.velocity = min(self.velocity + GRAVITY, MAX_FALL_SPEED)
        self.y       += self.velocity


    @property
    def rect(self) -> pygame.Rect:
        """Axis-aligned bounding rectangle used for collision detection."""
        hw = BIRD_WIDTH  // 2
        hh = BIRD_HEIGHT // 2
        return pygame.Rect(int(self.x) - hw, int(self.y) - hh,
                           BIRD_WIDTH, BIRD_HEIGHT)


    def draw(self, surface: pygame.Surface):
        """Render the bird as a simple illustrated sprite."""
        cx, cy = int(self.x), int(self.y)
        hw, hh = BIRD_WIDTH // 2, BIRD_HEIGHT // 2


        pygame.draw.ellipse(surface, YELLOW,
                            (cx - hw, cy - hh, BIRD_WIDTH, BIRD_HEIGHT))

        # Wing (small ellipse below centre) 
        # Slight animation: wing droops more when falling
        wing_offset = max(0, int(self.velocity * 0.4))
        pygame.draw.ellipse(surface, ORANGE,
                            (cx - 8, cy + wing_offset, 16, 9))

        # Eye (white sclera + black pupil) #
        pygame.draw.circle(surface, WHITE, (cx + 9, cy - 5), 6)
        pygame.draw.circle(surface, BLACK, (cx + 11, cy - 5), 3)

        # Beak (right side)#
        beak_tip = cx + hw + 8
        pygame.draw.polygon(surface, ORANGE,
                            [(cx + hw, cy - 3),
                             (beak_tip,  cy + 1),
                             (cx + hw, cy + 5)])

        # Outline (thin black rim) #
        pygame.draw.ellipse(surface, BLACK,
                            (cx - hw, cy - hh, BIRD_WIDTH, BIRD_HEIGHT), 2)
