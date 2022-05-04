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

RECONCILIATION_POOL_SIZE = 4
RECONCILIATION_TIMEOUT = 15

# -----------------------------
# data query / write parameters
RANGE_INCREMENT = 50 # seconds interval to batch query and refill raw_trajectories_queue

# -----------------------------
# stitcher algorithm parameters
TIME_WIN = 50
THRESH = 3
VARX = 0.05 # TODO: unit conversion (now it's based on meter)
VARY = 0.03
IDLE_TIME = 1 # if tail_time of a path has not changed after IDLE_TIME, then write to database
# if IDLE_TIME is too short, stitcher tends to under stitch

# For writing raw trajectories as Fragment objects
# change first "ID" to "_id" to query by ObjectId
WANTED_DOC_FIELDS = ["ID", "ID","timestamp","x_position","y_position","direction","last_timestamp","last_timestamp", "first_timestamp"]
FRAGMENT_ATTRIBUTES = ["id","ID","t","x","y","dir","tail_time","last_timestamp","first_timestamp"]


# -----------------------------
# reconciliation pamaters
LAM2_X = 1.67e-2
LAM2_Y = 1.67e-2
LAM1_X = 0.0012
LAM1_Y = 0.0012
PH = 100
IH = 5
