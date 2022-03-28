# -----------------------------
__file__ = 'parameters.py'
__doc__ = """
Contains the code-centric relatively static parameters read and used by the system at startup.
"""
# -----------------------------

RAW_TRAJECTORY_QUEUE_SIZE = 10000
STITCHED_TRAJECTORY_QUEUE_SIZE = 10000
RECONCILED_TRAJECTORY_QUEUE_SIZE = 1000
LOG_MESSAGE_QUEUE_SIZE = 10000

RECONCILIATION_POOL_SIZE = 10
RECONCILIATION_TIMEOUT = 15

DATABASE_URL = ""
LOG_URL = ""





## specify parameters
MODE = 'test'
TIME_OUT = 50 # gracefully shutdown if db has not been updated in TIME_OUT seconds

# database parameters
login_info = {
        'username': 'i24-data',
        'password': 'mongodb@i24'
        }
db_name = 'raw_trajectories'
DB_PARAMS = {
        'LOGIN': login_info,
        'MODE': MODE,
        'DB': db_name
        }

# data association parameters
STITCHER_PARAMS = {
        'TIME_WIN': XX,
        'THRESH': XX,
        'VARX': XX,
        'VARY': xx
        }

# rectification parameters
RECONCILIATION_PARAMS = {
        'LAM1_X':XX,
        'LAM1_Y':XX,
        'LAM2_X':XX,
        'LAM2_Y':XX,
        'PH': xx,
        'IH': xx
        }
