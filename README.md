# I24-postprocessing

A real-time trajectory postprocessing pipeline for I-24 MOTION project, following tracking https://github.com/DerekGloudemans/i24_track. 
Project website: https://i24motion.org/

## System architecture
The postprocessing system architecture is illustrated in this diagram:
![architecture](https://github.com/yanb514/i24_data_document/assets/30248823/70057138-db19-4eb3-84cf-e2ed818b0195)
At the first pass (local), a number of parallel components are initiated, each reads, merges and stitches fragments from an upstream videonode. The output is temporarily written to `temp database`. We start a second pass (master) once all the local components finish processing fragments. At this level, we read from `temp database`, merge and stitch the partially associated fragments across adjacent compute nodes. Once fragments are associated into trajectories, they are passed to the `rectification` module, which is managed by a parallel pool. All the rectified trajectories are finally written to the `trajectory database`. Both the local and master components run independently for the eastbound and westbound traffic.

## How to run
There are two ways to run the architecture. Both are equivalent. I created two ways because one is easier for development testing while the other one is easier to deploy (shrug), don't ask me why.

First way: run `pp1_all_nodes.main()`

Second way: run `pp1_local.main()` and then `pp1_master.main()`


### Configuration setting
Let me show you what you may or may not want to change in the config file. The config setting is located in `parameters.json` in the `config` folder. The general rule of thumb is don't change them if you don't know what they mean. Another rule is that you should not change the parameters that are not specified in this list. Those are not listed here are either algorithmic-specific that are extremely delicate, or they are not important for you to know.

- `description`: give a short description of the run. This information will be saved to database as meta data
- `raw_database`: name of the database where the raw fragments are read. Currently set as "trajectories"
- `reconciled_database`: name of the database where the final results are written to. Currently set as "reconciled"
- `temp_database`: name of the database where temporary results are written to. Currently set as "temp"
- `transform_postproc`: whether to run data transform [LINK] after postprocessing. 1 means yes, 0 means no.
- `raw_collection`: name of the collection from `raw_database` that houses the raw fragments. If not specified here it should be specified as function input (see below)
- `reconciled_collection`: name of the collection from `reconciled_database` that houses the final trajectories. If not specified here it should be specified as function input (see below),
- `temp_collection`: name of the collection from `temp_database` that houses the temporarily stitched fragments. If not specified here it will be assgined the same name as `reconciled_collection` (see below),
- `compute_node_list`: a list of names of the videoprocessing nodes. They should be in order of spatial coverage (doesn't matter increasing or decreasing, as long as we know whether two compute nodes cover adjacent road segments). I suggest leave it blank and let the names to be queried.


## Core algorithms
There are three main algorithms in this postprocessing pipeline:
1. Fragment merging (`merge_fragments` in `merge.py`). This algorithm identifies pair-wise fragments that should be merged into one. The merging criteria is that two fragments should have time-overlap, and should be "close" in the time-space domain by some metrics. The merging operation is "associative", meaning that multiple fragments could be merged into one trajectory if any one of the fragments merges to at least another fragment in the set. Finding the merged sets is equivalent to finding connected components in an undirected graph.
2. Fragment stitching (`min_cost_flow_online_alt_path` in `min_cost_flow.py). This algorithm identifies fragments that should be stitched into one. The stitching criteria states that two fragmetns should NOT have time-overlap, and should be kinematically "close". Due to the sequential order and conflicts restriction, finding the stitched sets is equivalent to finding the min-cost-flow in a directed graph. Details of this algorithm is specified in [CITE]
3. Trajectory rectification (`opt2_l1_constr` in `utils_opt.py`). This step simultaneously imputes missing data, identifies and removes outliers and denoises a single trajectory independent of others. It is formulated as a convex program.

## Some random decisions I just have to make
In order to understand this code base, you should have some knowledge about how MongoDB works and how it stores information. Basically you store NoSQL type of data (e.g., a dictionary). You can have multiple databases, within a database there could be multiple collections. Within each collection you have a bunch of documents all sharing the same schema. The postprocessing pipeline reads a collection from `raw_database`, and writes the temporary results in a collection that lives in `temp_database`. Finally it writes the final outputs in another collection inside `reconciled_database`.

- If `raw_collection` and `reconciled_collection` are specified as function arguments for `pp1_all_nodes.main()`, they will overwrite the ones specified in `parameters.json` file.
- If `raw_collection` is not specified in config or as argument, then it automatically processes the latest collection in the database. You have to specify the `raw_database`, otherwise what are you doing here?
- If `temp_collection` is not specified, it will have the same name as one specified as `reconciled_collection` in config, but in the `temp` database.
- If `reconciled_collection` is not specified, then you give me the permission to append a random verb to the `raw_collection` name. This is *DANGEROUS*. If the random name by some remote chance duplicates with an existing collection name, then the process will hang at `data_feed.initialize()`. Well if that happens you will know. A good rule of thumb is to ALWAYS specify `raw_collection` and `reconciled_collection` either in config file or as function input.
- If `compute_node_list` is not specified in config, it is automatically queried from the database.


## Evaluation
Qualitative evaluation of the trajectory qualities can be visualized using the time-space, overhead and video-overlay repositories described below.

![anim_batch_reconciled_timespace_overhead](https://user-images.githubusercontent.com/30248823/180271610-6baf4307-e4a1-4cb5-ae86-3df0d31e3319.gif)

### Dashboard
![dashboard_1](https://github.com/yanb514/i24_data_document/assets/30248823/eb849345-eaed-4281-92c7-3477b9efcda1)
![dashboard_2](https://github.com/yanb514/i24_data_document/assets/30248823/a43b4c5f-bd6a-4ce9-83ce-9d0cd33984b3)
![dashboard_3](https://github.com/yanb514/i24_data_document/assets/30248823/e9e22b17-11cf-4589-9e65-8715800bfa92)
![dashboard_4](https://github.com/yanb514/i24_data_document/assets/30248823/6e59af8a-4c7d-4c84-ae72-995d96b81ab6)


## Output data format: 
Data documentation see [insert link]

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
