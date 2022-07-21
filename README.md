# I24-postprocessing

A real-time trajectory postprocessing pipeline for I-24 MOTION project.
Project website: https://i24motion.org/


## Postprocessing pipeline overview

This pipeline consists of 4 parallel processes, managed by a mutliprocessing manager:
1. `data_feed.py->live_data_feed()`: continuously read fragments from the raw trajectory collection with MongoDB change stream, and put them into a multiprocessing queue.
2. `min_cost_flow.py-> min_cost_flow_alt_path()`: a fragment stitching algorithm using min-cost-flow formulation. It processes fragment one by one from a multiprocessing queue, and writes the stitched trajectories into another queue. Two identical stitchers are spawn for east and west bound traffic.
3. `reconciliation.py -> reconciliation_pool()`: a trajectory reconciliation algorithm to smooth, impute and rectify trajectories, such that the dynamics (e.g., velocity, acceleration and jerk) are within a reasonable range and they satisfy internal consistency. It creates a multiprocessing pool and asynchronously assign workers to reconcile trajectories independly. The reconciled trajectories are written to another queue for bulk write to database.
4. `reconcilation.ppy -> reconciliation_writer()`: writes processed trajectories to database.

All processes are managed by python multiprocessing framework. A diagram illustrates the pipeline:
![visio](https://user-images.githubusercontent.com/30248823/180301065-05b13405-6627-4215-bf38-d94d8587531e.png)



## Evaluation
Qualitative evaluation of the trajectory qualities can be visualized using the time-space, overhead and video-overlay repositories described below.
Due to the lack standard metrics, we provide statistics-based evaluation in `unsupervised_evaluator.py`.

## Data format: 

```python
{
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["timestamp", "last_timestamp", "x_position"],
        "properties": {
            "configuration_id": {
                "bsonType": "int",
                "description": "A unique ID that identifies what configuration was run. It links to a metadata document that defines all the settings that were used system-wide to generate this trajectory fragment"
                },
            "coarse_vehicle_class": {
                "bsonType": "int",
                "description": "Vehicle class number"
                },
            
            "timestamp": {
                "bsonType": "array",
                "items": {
                    "bsonType": "double"
                    },
                "description": "Corrected timestamp. This timestamp may be corrected to reduce timestamp errors."
                },
            
 
            "road_segment_ids": {
                "bsonType": "array",
                "items": {
                    "bsonType": "int"
                    },
                "description": "Unique road segment ID. This differentiates the mainline from entrance ramps and exit ramps, which get distinct road segment IDs."
                },
            "x_position": {
                "bsonType": "array",
                "items": {
                    "bsonType": "double"
                    },
                "description": "Array of back-center x position along the road segment in feet. The  position x=0 occurs at the start of the road segment."
                },
            "y_position": {
                "bsonType": "array",
                "items": {
                    "bsonType": "double"
                    },
                "description": "array of back-center y position across the road segment in feet. y=0 is located at the left yellow line, i.e., the left-most edge of the left-most lane of travel in each direction."
                },
            
            "length": {
                "bsonType": "double",
                "description": "vehicle length in feet."
                },
            "width": {
                "bsonType": "double",
                "description": "vehicle width in feet"
                },
            "height": {
                "bsonType": "double",
                "description": "vehicle height in feet"
                },
            "direction": {
                "bsonType": "int",
                "description": "-1 if westbound, 1 if eastbound"
                }

            }
        }
    }
```


## Other related I-24 MOTION repositories
### I24_logging
https://github.com/Lab-Work/I24_logging
### I24 Visualization
https://github.com/yanb514/i24-overhead-visualizer
https://github.com/zineanteoh/i24-overlay-visualizer
### I24_transform_module
https://github.com/Lab-Work/i24-transform-module
### I24_database_api
https://github.com/Lab-Work/i24_database_api

