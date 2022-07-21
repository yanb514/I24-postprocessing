# I24-postprocessing

A real-time trajectory postprocessing pipeline for I-24 MOTION project.
Project website: https://i24motion.org/


## Postprocessing pipeline overview

This pipeline consists of 5 parallel processes, managed by a mutliprocessing manager:
1. `data_feed.py->live_data_feed()`: continuously read fragments from the raw trajectory collection with MongoDB change stream, and put them into a multiprocessing queue.
2,3. `min_cost_flow.py-> min_cost_flow_alt_path()`: a fragment stitching algorithm using min-cost-flow formulation. It processes fragment one by one from a multiprocessing queue, and writes the stitched trajectories into another queue. Two identical stitchers are spawn for east and west bound traffic.
4. `reconciliation.py -> reconciliation_pool()`: a trajectory reconciliation algorithm to smooth, impute and rectify trajectories, such that the dynamics (e.g., velocity, acceleration and jerk) are within a reasonable range and they satisfy internal consistency. It creates a multiprocessing pool and asynchronously assign workers to reconcile trajectories independly. The reconciled trajectories are written to another queue for bulk write to database.
5. `reconcilation.ppy -> reconciliation_writer()`: writes processed trajectories to database.

All processes are managed by python multiprocessing framework. A diagram illustrates the pipeline:
![visio](https://user-images.githubusercontent.com/30248823/180301065-05b13405-6627-4215-bf38-d94d8587531e.png)



## Evaluation
Qualitative evaluation of the trajectory qualities can be visualized using the time-space, overhead and video-overlay repositories described below.
Due to the lack standard metrics, we provide statistics-based evaluation in `unsupervised_evaluator.py`.

## Data format: 
1. `Frame #`: frame index (30 fps)
2. `Timestamp`: Unix timestamp
3. `ID`: unique vehicle ID
4. `Object class`: vehicle type
5. **`BBox xmin/ymin/xmax/ymax`: 2D bounding box dimensions
6. ** `vel_x/vel_y`: velocity
7. ** `fbrx/y`: the pixel coordinates of the front bottom right corner w.r.t. each vehicle's traveling direction
8. ** `fblx/y`: ... front bottom left ...
9. ** `bbrx/y`: ... back bottom right ...
10. ** `bblx/y`: ... back bottom left ...
11. ** `btlx/y`: ... back top left ...
13. `fbr_x/_y`: the road cooridates (w.r.t some surveyed points) of the front bottom right corner
14. `direction`: 1: south bound traffic; -1: north bound traffic
15. `camera`: camera field of view
16. `acceleration`: acceleration in m/s^2
17. `speed`: speed in m/s
18. `x/y`: back center positions in road coordinates (in meter)
19. `theta`: angle between the travel direction and the positive x-axis. If direction = 1, theta is close to 0; if direction = -1, theta is close to pi.
20. `width`: width of the vehicle in meter
21. `length`: length of the vehicle in meter
22. `height`: height of the vehicle in meter
23. `lane`: lane index, calculated from y coordinate.
**: information not needed for post-processing

## Other related I-24 MOTION repositories
### I24 Visualization
There are multiple ways to visualize data. The visualization toolbox will be updated. As of now, the best way is to run `animation.py`.

## Evaluation
1. `synth_evaluator`: Calculate multi-object tracking performance metrics, such as TP, FP, FN, FRAG, IDS, MOTA, MOTP. Use this when ground truth data is available.
2. `global_metrics`: Examine global traffic condition, plot histograms of the state, and identify abnormal trajectories. Use this when ground truth data is not available.

### Benchmarking using synthetic data
- Synthetic data that resembles the raw 3D tracking data will be generated in `benchmark_TM.py`
    - Point-trajectory data is generated using TransModler, I added vehicle dimension and upsampled the trajectory to get the state information. Data format follows the one above
    - Manual pollution is added to (1) mask part of the data to create fragments and (2) add noise on the bbox.

