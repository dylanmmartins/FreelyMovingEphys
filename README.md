# FreelyMovingEphys

## Setup

### instillation of environments
Install the analysis pipeliene's conda environment from `/FreelyMovingEphys/conda_env/` with the terminal command `conda env create -f /path/to/repository/FreelyMovingEphys/conda_env/environment.yml`. This is used for all deinterlacing before DeepLabCut analyzes new videos, and all analysis after DeepLabCut or Anipose.

Then, install the environment DLC-GPU in order to analyze new videos, instructions [here](https://github.com/DeepLabCut/DeepLabCut/blob/master/conda-environments/README.md) To allow for three-camera configurations of the topdown view, it will be necessary to install Anipose in the DLC-GPU environment, instructions [here](https://github.com/lambdaloop/anipose/blob/master/docs/sphinx/installation.rst).

## Usage

### deinterlacing videos and interpolating timestamps
Before running any analysis, any 30fps eye or world .avi videos must be deinterlaced to bring them to 60fps and corresponding .csv timestamps must be adjusted to match the new video length. To accomplish this, run `deinterlace_for_dlc.py` on a parent directory containing all .avi files, .csv files, and .txt files of notes for a given experiment. The `deinterlace_for_dlc.py` script should be given the nested folders that contain videos and timestamps that need not be deinterlaced (frame rates will be checked and files that do not need to be changed will be copied to the output directory to keep experiments together). The user could run: `python deinterlace_for_dlc.py -d '/path/to/top/of/parent/directory/' -s '/save/location/'`.

### running DeepLabCut on new videos
To analyze new videos using DeepLabCut and/or Anipose, use the script `analyze_new_vids.py`.

### analyzing DeepLabCut outputs
To process DeepLabCut outputs, visualize points, and get out calculations of head angle, eye ellipse parameters, pupil rotation, etc., create a .json file to be used as a config file. The config file should include the path to the parent directory of data (`data_path`), the save path for outputs of the analysis pipeline (`save_path`), the camera views in the experiments (`camera_names`), the threshold to set for pose likelihood (`lik_thresh`), the value by which to correct y coordinates (`coord_correction`), whether or not there are crickets in the experiments (`cricket`), whether or not the outer point and tear duct of the eye was labeled in the eye videos (`tear`), the maximum acceptable number of pixels for radius of the pupil (`pxl_thresh`), the maximum ratio of ellipse shortaxis to longaxis (`ell_thresh`), whether or not to save out videos (`save_vids`), whether or not to save out figures (`save_figs`), whether to use Bonsai timestamps for Flir timestamps, where `True` would have it use Bonsai timestamps (`use_BonsaiTS`), the threshold to set for range in radius (`range_radius`), and the interpolation method to use for interpolating over eye timestmaps with world timestamps (`world_interp_method`). If an argument in the .json file is related to someting that doesn't apply to the user's experiments (e.g. eye camera arguments for experiments that have no eye cameras) that argument doesn't need to be included in the dictionary.

```
{
    "data_path": "/path/to/parent/directory/",
    "save_path": "/save/path/",
    "camera_names": [
        "TOP",
        "REYE",
        "RWORLD"
    ],
    "lik_thresh": 0.99,
    "coord_correction": 0,
    "cricket": true,
    "tear": true,
    "pxl_thresh": 50,
    "ell_thresh": 0.9,
    "save_vids": true,
    "save_figs": true,
    "use_BonsaiTS": true,
    "range_radius": 10,
    "world_interp_method": "linear"
}
```

Once this .json config file exists, batch analysis can be run with the script `extract_params_from_dlc.py` which takes only one argument, the path to the config file. The user could run: `python -W ignore extract_params_from_dlc.py -c '/path/to/pipeline_config.json'`, where `-W ignore` ignores a runtime error always generated by the ellipse calculations. Though there is build-in handling for `'TOP'`, `'LEYE'`, `'REYE'`, `'SIDE'`, `'LWORLD'`, and `'RWORLD'` camera names, any camera name outside of this list can be read in and formatted into a .nc file. It's important that experiments use the following naming system so that they will be recognized: `013120_mouse1_control_LEYE.h5`.

There is also a jupyter notebook that can load in a .json file and run through the different camera views step-by-step for one trial at a time, `test_extract_params_from_dlc.ipynb`.

### opening .nc files and visualizing outputs
Once the analysis is run, there will be plots and videos for each trial and one .nc file for each camera name which contains the data from all trials that had a camear of said name. Using only the .json config file's path that was used to , the outputs of the analysis pipeline can be found, viewed, and interacted with in the jupyter notebook `check_pipeline_outputs.ipynb`. There are also examples of way to index and select data from the loaded data structures.
