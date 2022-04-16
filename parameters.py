# -----------------------------
__file__ = 'parameters.py'
__doc__ = """
Contains the code-centric relatively static parameters read and used by the system at startup.
Including initialized data structures
"""
# -----------------------------
import utils.data_structures
from collections import defaultdict, deque, OrderedDict

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
TIME_RANGE = 50 # A moving window range in sec for raw-data-feed
START = 0
END = 480
TIME_OUT = 50 # gracefully shutdown if db has not been updated in TIME_OUT seconds

## roadway parameters
X_MAX = 32800 # in feet
X_MIN = 1000

# database parameters
## specify collection names
GT_COLLECTION = "ground_truth_trajectories"
RAW_COLLECTION = "raw_trajectories_one" # specify raw trajectories collection name that is used for reading
STITCHED_COLLECTION = "stitched_trajectories"
RECONCILED_COLLECTION = "reconciled_trajectories"
login_info = {
        'username': 'i24-data',
        'password': 'mongodb@i24'
        }
#DB_PARAMS = {
#        'LOGIN': login_info
#        }

# stitcher parameters
STITCHER_PARAMS = {
        'TIME_WIN': 10, 
        'THRESH': 3,
        'VARX': 0.05, # TODO: unit conversion
        'VARY': 0.03
#        "X_MAX": X_MAX,
#        "X_MIN": X_MIN
        }

# Initialize data structures for bookkeeping
STITCHER_INIT_E = {
        "curr_fragments": deque(),  # fragments in view. list of fragments. should be sorted by end_time
        "past_fragments": OrderedDict(),  # set of ids indicate end of fragment ready to be matched
        "path": {}, # latest_fragment_id: previous_id. to store matching assignment
        "start_times_heap": []
        }

STITCHER_INIT_W = {
        "curr_fragments": deque(),  # fragments in view. list of fragments. should be sorted by end_time
        "past_fragments": OrderedDict(),  # set of ids indicate end of fragment ready to be matched
        "path": {}, # latest_fragment_id: previous_id. to store matching assignment
        "start_times_heap": []
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


