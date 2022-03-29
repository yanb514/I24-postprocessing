# -----------------------------
__file__ = 'parameters.py'
__doc__ = """
Contains the code-centric relatively static parameters read and used by the system at startup.
Including initialized data structures
"""
# -----------------------------
import utils.data_structures

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

# stitcher parameters
STITCHER_PARAMS = {
        'TIME_WIN': 300,
        'THRESH': 3,
        'VARX': 0.05, # TODO: unit conversion
        'VARY': 0.03,
        }

# Initialize data structures for bookkeeping
STITCHER_INIT = {
        "curr_fragments" = deque() # fragments in view. list of fragments. should be sorted by end_time
        "past_fragments" = OrderedDict() # set of ids indicate end of fragment ready to be matched
        "path" = {} # latest_fragment_id: previous_id. to store matching assignment
        "start_times_heap" = [] # a heap to order start times of fragments in current TIME_WIN
        }

# rectification parameters
RECONCILIATION_PARAMS = { # TODO fill in those numbers
        'lam1_x':XX,
        'lam1_Y':XX,
        'lam2_x':XX,
        'lam2_y':XX,
        'PH': xx,
        'IH': xx
        }