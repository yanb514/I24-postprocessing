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
