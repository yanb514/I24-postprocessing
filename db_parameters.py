DEFAULT_HOST = '10.2.218.56'
DEFAULT_PORT = 27017
DEFAULT_USERNAME = 'i24-data'
DEFAULT_PASSWORD = 'mongodb@i24'

GT_COLLECTION = "ground_truth_one"
# RAW_COLLECTION = "raw_trajectories_one" # specify raw trajectories collection name that is used for reading
RAW_COLLECTION = "test_collection"
STITCHED_COLLECTION = "stitched_trajectories"
RECONCILED_COLLECTION = "reconciled_trajectories"
DB_NAME = "trajectories"

# Buffer queue size for live_data_read
MIN_QUEUE_SIZE = 1000

# Define data schema
RAW_SCHEMA = ["local_fragment_id", "coarse_vehicle_class", "fine_vehicle_class", 
              "timestamp", "raw_timestamp", "first_timestamp", "last_timestamp",
              "road_segment_id", "x_position", "y_position", 
              "starting_x", "ending_x", 
              "camera_snapshots",
              "flags", "direction"
              "lengths", "widths", "heights"]
STITCHED_SCHEMA = ["fragment_ids"]
GT_SCHEMA = ["fragment_ids", "local_fragment_id", "coarse_vehicle_class", "fine_vehicle_class", 
              "timestamp", "raw_timestamp", "first_timestamp", "last_timestamp",
              "road_segment_id", "x_position", "y_position", 
              "starting_x", "ending_x", 
              "camera_snapshots",
              "flags", "direction"
              "length", "width", "height"]
RECONCILED_SCHEMA = ["coarse_vehicle_class", "fine_vehicle_class", 
              "timestamp", "raw_timestamp", "first_timestamp", "last_timestamp",
              "road_segment_id", "x_position", "y_position", 
              "starting_x", "ending_x", 
              "camera_snapshots",
              "flags", "direction"
              "length", "width", "height"]
