DEFAULT_HOST = '10.2.218.56'
DEFAULT_PORT = ''
DEFAULT_USERNAME = 'i24-data'
DEFAULT_PASSWORD = 'mongodb@i24'

GT_COLLECTION = "ground_truth_one"
RAW_COLLECTION = "raw_trajectories_one" # specify raw trajectories collection name that is used for reading
STITCHED_COLLECTION = "stitched_trajectories"
RECONCILED_COLLECTION = "reconciled_trajectories"
DB_NAME = "trajectories"

# Buffer queue size for live_data_read
MIN_QUEUE_SIZE = 1000

# Define data schema
RAW_SCHEMA = []
STITCHED_SCHEMA = []
GT_SCHEMA = []
RECONCILED_SCHEMA = []
