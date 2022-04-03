# -----------------------------
__file__ = 'parameters.py'
__doc__ = """
Contains the code-centric relatively static parameters read and used by the system at startup.
Including initialized data structures
"""
# -----------------------------
import utils.data_structures
from collections import defaultdict, deque, OrderedDict
RAW_TRAJECTORY_QUEUE_SIZE = 1000
STITCHED_TRAJECTORY_QUEUE_SIZE = 10000
RECONCILED_TRAJECTORY_QUEUE_SIZE = 1000
LOG_MESSAGE_QUEUE_SIZE = 10000

RECONCILIATION_POOL_SIZE = 10
RECONCILIATION_TIMEOUT = 15

DATABASE_URL = ""
LOG_URL = ""





## specify parameters
MODE = 'test'
TIME_RANGE = 50 # A moving window range in sec for raw-data-feed
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
        "curr_fragments": deque(),  # fragments in view. list of fragments. should be sorted by end_time
        "past_fragments": OrderedDict(),  # set of ids indicate end of fragment ready to be matched
        "path": {} # latest_fragment_id: previous_id. to store matching assignment
        }

# rectification parameters
RECONCILIATION_PARAMS = { # TODO fill in those numbers
        'lam1_x':0.0012,
        'lam1_Y':0.0012,
        'lam2_x':1.67e-2,
        'lam2_y':1.67e-2,
        'PH': 100,
        'IH': 5
        }