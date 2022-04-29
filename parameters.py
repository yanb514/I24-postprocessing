# -----------------------------
__file__ = 'parameters.py'
__doc__ = """
Contains the code-centric relatively static parameters read and used by the system at startup.
Including initialized data structures
"""
# -----------------------------


RAW_TRAJECTORY_QUEUE_SIZE = 10000
STITCHED_TRAJECTORY_QUEUE_SIZE = 10000
RECONCILED_TRAJECTORY_QUEUE_SIZE = 1000
LOG_MESSAGE_QUEUE_SIZE = 10000

RECONCILIATION_POOL_SIZE = 10
RECONCILIATION_TIMEOUT = 15




## specify parameters
MODE = 'test'
TIME_RANGE = 50 # A moving window range in sec for raw-data-feed
START = 0
END = 480
TIME_OUT = 50 # gracefully shutdown if db has not been updated in TIME_OUT seconds

## roadway parameters
X_MAX = 32800 # in feet
X_MIN = 1000


# stitcher parameters
STITCHER_PARAMS = {
        'TIME_WIN': 10, 
        'THRESH': 3,
        'VARX': 0.05, # TODO: unit conversion
        'VARY': 0.03
        }


# rectification parameters
RECONCILIATION_PARAMS = { # TODO fill in those numbers, unit convert in feet
        'lam1_x':0.0012,
        'lam1_Y':0.0012,
        'lam2_x':1.67e-2,
        'lam2_y':1.67e-2,
        'PH': 100,
        'IH': 5
        }


