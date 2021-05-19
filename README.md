# I24-trajectory-generation

Before running this notebook, please do the following
1. Download the data that Derek shared with you (https://vanderbilt.box.com/s/sgb996yj09bmev6yhc7slf053nz74p9q)
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
    |   I24_xxx.ipynb
    |   ...


```

As for the roadmap, for the short term we can:
1. Recreate the same plots that I showed you today by playing around this notebook. You’re welcome to add your own modifications to it.
2. Overlay a road drawing at the background (so that we know which lane the cars are at)
3. Make an animation!
4. Add multiple vehicles across different camera view.
