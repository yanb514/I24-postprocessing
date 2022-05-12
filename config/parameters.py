# -----------------------------
__file__ = 'parameters.py'
__doc__ = """
Contains the code-centric relatively static parameters read and used by the system at startup.
Including initialized data structures
"""

# -----------------------------
# database login credentials
DEFAULT_HOST = '10.2.218.56'
DEFAULT_PORT = 27017
DEFAULT_USERNAME = 'i24-data'
READONLY_USER =  "readonly"
DEFAULT_PASSWORD = 'mongodb@i24'


GT_COLLECTION = "ground_truth_one"
RAW_COLLECTION = "raw_trajectories_one" # specify raw trajectories collection name that is used for reading
# RAW_COLLECTION = "test_collection"
STITCHED_COLLECTION = "stitched_trajectories"
RECONCILED_COLLECTION = "reconciled_trajectories"
DB_NAME = "trajectories"

# BYPASS_VALIDATION = False # False if enforcing schema


# Create indices upon instantiation of DBReader and DBWriter objects
INDICES = ["_id", "ID", "first_timestamp", "last_timestamp"]


# -----------------------------
# define at the manager level
RAW_TRAJECTORY_QUEUE_SIZE = 10000
STITCHED_TRAJECTORY_QUEUE_SIZE = 10000
RECONCILED_TRAJECTORY_QUEUE_SIZE = 1000
# Buffer queue size for live_data_read
MIN_QUEUE_SIZE = 1000

# -----------------------------
# live data feed parameters
RANGE_INCREMENT = 50 # seconds interval to batch query and refill raw_trajectories_queue
BUFFER_TIME = 10 # seconds to 

# -----------------------------
# stitcher algorithm parameters
TIME_WIN = 5
THRESH = 3
VARX = 0.05 # TODO: unit conversion (now it's based on meter)
VARY = 0.03
IDLE_TIME = 1 # if tail_time of a path has not changed after IDLE_TIME, then write to database
# if IDLE_TIME is too short, stitcher tends to under stitch


# -----------------------------
# reconciliation paramaters
RECONCILIATION_POOL_SIZE = 4
RECONCILIATION_TIMEOUT = 15
LAM2_X = 1.67e-2
LAM2_Y = 1.67e-2
LAM1_X = 0.0012
LAM1_Y = 0.0012
PH = 100
IH = 5
