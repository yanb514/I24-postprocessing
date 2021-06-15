# I24-trajectory-generation

Before running this notebook, please do the following
1. Download the 3D bbox trajectory data at https://vanderbilt.box.com/s/sgb996yj09bmev6yhc7slf053nz74p9q
2. Download camera_calibration_points csv file. Click "raw" and save as csv (https://github.com/DerekGloudemans/I24-video-processing/blob/main/config/camera_calibration_points_I24_validation_system.csv)
and add to your local repository such that the directory looks like

```
YOUR LOCAL FOLDER
└───2D-3D May 2021
|   |   camera_calibration_points_I24_validation_system.csv
|   |   record_xxx.csv
|   |   ...
│
└───I24-data-generation
    │   README.md
    |   utils.py
    |   I24_visualization.ipynb
    |   I24_xxx.ipynb
    |   ...
```

As for the roadmap, for the short term we can:
1. Recreate the same plots that I showed you today by playing around this notebook. You’re welcome to add your own modifications to it.
2. Overlay a road drawing at the background (so that we know which lane the cars are at)
3. Make an animation!
4. Add multiple vehicles across different camera view.

Meta for `rectified_record_xxx.csv`: 
1. `Frame #`: frame index
2. `Timestamp`: Unix timestamp
3. `ID`: unique vehicle ID
4. `Object class`: vehicle type
5. `BBox xmin/ymin/xmax/ymax`: 2D bounding box dimensions
6. `fbrx/y`: the pixel coordinates of the front bottom right corner w.r.t. each vehicle's traveling direction
7. `fblx/y`: ... front bottom left ...
8. `bbrx/y`: ... back bottom right ...
9. `bblx/y`: ... back bottom left ...
10. We won't be needing all the top points (`*t**x/y`) for now.
11. `fbrlat/lon`: the GPS coordinates (latitudes and longitudes) of the front bottom right corner
12. `fbr_x/_y`: the road cooridates (w.r.t some surveyed points) of the front bottom right corner --> let's plot based on these coordinates
13. `direction`: 1: south bound traffic; -1: north bound traffic
