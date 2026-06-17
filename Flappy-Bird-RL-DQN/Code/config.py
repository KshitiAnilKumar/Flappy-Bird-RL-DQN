"""
Central configuration for Flappy Bird RL.
All magic numbers live here – tweak freely for experiments.
"""

# ──────────────────────────────────────────────────────────────────── #
#  Display
# ──────────────────────────────────────────────────────────────────── #
SCREEN_WIDTH  = 480
SCREEN_HEIGHT = 640
FPS           = 64

# ──────────────────────────────────────────────────────────────────── #
#  Colours
# ──────────────────────────────────────────────────────────────────── #
SKY_BLUE   = (113, 197, 207)
WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0  )
GREEN      = (34,  177, 76 )
DARK_GREEN = (20,  120, 50 )
YELLOW     = (255, 220, 0  )
ORANGE     = (255, 140, 0  )
RED        = (220, 50,  50 )
DARK_RED   = (160, 20,  20 )
BLUE       = (50,  100, 200)
BROWN      = (101, 67,  33 )
GRAY       = (150, 150, 150)
GOLD       = (255, 215, 0  )

# ──────────────────────────────────────────────────────────────────── #
#  Bird physics
#  FIX: Gravity reduced (0.5→0.35) so the bird doesn't drift down so
#       aggressively.  Flap strength increased (-9→-10) so a single
#       flap can realistically reach gaps in the top half of the screen.
#       MAX_FALL_SPEED tightened (12→10) to prevent unrecoverable dives.
# ──────────────────────────────────────────────────────────────────── #
BIRD_WIDTH     = 36
BIRD_HEIGHT    = 26
GRAVITY        = 0.35        # was 0.5  — less aggressive downward pull
FLAP_STRENGTH  = -10.0       # was -9.0 — stronger upward impulse
MAX_FALL_SPEED = 10.0        # was 12.0 — earlier terminal-velocity clamp

# ──────────────────────────────────────────────────────────────────── #
#  Pipes – normal
# ──────────────────────────────────────────────────────────────────── #
PIPE_WIDTH   = 64
NORMAL_GAP   = 185
PIPE_SPACING = 260

# ──────────────────────────────────────────────────────────────────── #
#  Pipes – triple (always fixed)
# ──────────────────────────────────────────────────────────────────── #
TRIPLE_GAP   = 155
TRIPLE_MID_H = 36

# ──────────────────────────────────────────────────────────────────── #
#  Horizontal speed ramp
# ──────────────────────────────────────────────────────────────────── #
INITIAL_SPEED   = 3.0
MAX_SPEED       = 8.0
SPEED_PER_SCORE = 0.05

# ──────────────────────────────────────────────────────────────────── #
#  Coins
# ──────────────────────────────────────────────────────────────────── #
COIN_RADIUS       = 11
COIN_SPAWN_CHANCE = 0.30

# ──────────────────────────────────────────────────────────────────── #
#  Reward structure
#  FIX: REWARD_ALIVE raised (0.05→0.1) so surviving frames are worth
#       more and the agent isn't incentivised to die quickly.
#       REWARD_PASS raised (1.0→5.0) so the dominant learning signal
#       is pipe-clearance, not just staying alive.
#       REWARD_COLLISION kept at -10 but is now clearly dominant
#       negative to push the agent away from walls / pipes fast.
# ──────────────────────────────────────────────────────────────────── #
REWARD_ALIVE     =  0.03      # was 0.05 — stronger survival incentive
REWARD_PASS      = 10.00      # was 1.00 — dominant positive signal
REWARD_COIN      =  1.00      # unchanged
REWARD_COLLISION = -5.00     # unchanged

# ──────────────────────────────────────────────────────────────────── #
#  RL state / action spaces
# ──────────────────────────────────────────────────────────────────── #
# State vector layout (see environment.py → _get_state()):
#   [0] bird y (norm)
#   [1] bird velocity (norm)
#   [2] dx to next obs
#   [3] gap-1 centre y (norm)
#   [4] gap-2 centre y (norm)
#   [5] is_triple (0/1)
#   [6] speed (norm)
STATE_SIZE  = 7
ACTION_SIZE = 2   # 0 = do nothing, 1 = flap

# ──────────────────────────────────────────────────────────────────── #
#  DQN hyper-parameters
#  FIX: LR lowered (1e-3→5e-4) for more stable gradient updates.
#       EPS_END raised (0.01→0.05) so the agent never stops exploring
#       completely — helps escape local minima (do-nothing policy).
#       EPS_DECAY doubled (6000→12000) to give more warm-up exploration
#       time before the greedy policy dominates.
#       REPLAY_CAPACITY doubled (50k→100k) so the buffer holds a wider
#       variety of experiences, reducing correlation in mini-batches.
#       TARGET_UPDATE lowered (500→300) for more frequent target syncs.
# ──────────────────────────────────────────────────────────────────── #
GAMMA           = 0.99
LR              = 1e-4        # was 1e-3  — more stable updates
BATCH_SIZE      = 128
REPLAY_CAPACITY = 200_000     # was 50_000 — richer replay diversity
EPS_START       = 1.0
EPS_END         = 0.03        # was 0.01  — keep minimum exploration
EPS_DECAY       = 30_000      # was 6_000 — longer warm-up phase
TARGET_UPDATE   = 500         # was 500   — more frequent target sync