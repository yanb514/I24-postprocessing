# I24-trajectory-generation

Before running this notebook, please do the following
- Download the trajectory data at 
    - (3D tracking) https://vanderbilt.box.com/s/sgb996yj09bmev6yhc7slf053nz74p9q
    - (Synthetic data, for algorithm benchmarking) https://vanderbilt.app.box.com/folder/150598703751
        - Based on the simulation data from TransModler, details see `benchmark_TM.py`

### Postprocessing pipeline
- Data association
    - This step is to reduce fragments, details see `data_association.py`
    - MOTA evaluation
- Data rectification
    - This step is to produce trajectory data that satisfy vehicle dynamics (e.g., velocity/acceleration smoothing)
    - details see `rectification.py`

### Data format: 
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

## Visualization
There are multiple ways to visualize data. The visualization toolbox will be updated. As of now, the best way is to run `animation.py`.

## Evaluation
Currently I'm writing an evaluator for synthetic data (when ground truth is available).
### Benchmarking using synthetic data
- Synthetic data that resembles the raw 3D tracking data will be generated in `benchmark_TM.py`
    - Point-trajectory data is generated using TransModler, I added vehicle dimension and upsampled the trajectory to get the state information. Data format follows the one above
    - [TODO]: implement manual pollution to (1) mask part of the data to create fragments and (2) add noise on the bbox.
- [TODO]: MOT evaluator will be implemented.
