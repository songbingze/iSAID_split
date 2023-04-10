# iSAID_split
Split iSAID dataset and its coco-format json annotation files.

1.  **Environment and dependencies installation**
    1. Create the conda environment
    ```
    > conda create -n py_isaid python=3.8
    > pip install pycocotools opencv-python natsort h5py scipy
    ```

2.  **Dataset preparation**
    1. Download the iSAID dataset from [iSAID](https://captain-whu.github.io/iSAID/).
    2. Unzip the dataset to the `data` folder.
    3. Make sure that the final dataset must have this structure:
    ```
    iSAID
    ├── test
    │   └── images
    │       ├── P0006.png
    │       └── ...
    │       └── P0009.png
    ├── train
    │   └── images
    │       ├── P0002_instance_color_RGB.png
    │       ├── P0002_instance_id_RGB.png
    │       ├── P0002.png
    │       ├── ...
    │       ├── P0010_instance_color_RGB.png
    │       ├── P0010_instance_id_RGB.png
    │       └── P0010.png
    └── val
        └── images
            ├── P0003_instance_color_RGB.png
            ├── P0003_instance_id_RGB.png
            ├── P0003.png
            ├── ...
            ├── P0004_instance_color_RGB.png
            ├── P0004_instance_id_RGB.png
            └── P0004.png
    ```
    4. Run the following command to split the dataset:
    ```
    > python split.py --set train,val
    > python split.py --set test
    ```
    Note that "train","val" cannot be split simultaneously with "test". If --set includes "test", the "train" and "val" will be split just image without annotation.

    5. Run the following command to split the dataset json file:
    ```
    python preprocess.py --set train,val
    ```
    6. Make sure that the final dataset after preprocesing must have this structure:
    ```
    iSAID_patches
    ├── test
    │   └── images
    │       ├── P0006_0_0_800_800.png
    │       └── ...
    │       └── P0009_0_0_800_800.png
    ├── train
    │   └── instance_only_filtered_train.json
    │   └── images
    │       ├── P0002_0_0_800_800_instance_color_RGB.png
    │       ├── P0002_0_0_800_800_instance_id_RGB.png
    │       ├── P0002_0_800_800.png
    │       ├── ...
    │       ├── P0010_0_0_800_800_instance_color_RGB.png
    │       ├── P0010_0_0_800_800_instance_id_RGB.png
    │       └── P0010_0_800_800.png
    └── val
        └── instance_only_filtered_val.json
        └── images
            ├── P0003_0_0_800_800_instance_color_RGB.png
            ├── P0003_0_0_800_800_instance_id_RGB.png
            ├── P0003_0_0_800_800.png
            ├── ...
            ├── P0004_0_0_800_800_instance_color_RGB.png
            ├── P0004_0_0_800_800_instance_id_RGB.png
            └── P0004_0_0_800_800.png
    ```

[Here](dataset/iSAID) is a sample dataset to test this project.

If you want to change the folder for reading and saving image, please modify the parameters ```--src```, ```--tar``` in [split.py](split.py) and ```--outdir```, ```--datadir``` in [preprocess.py](preprocess.py).

If you want to change the size and the overlap area of the split please modify to the parameters ```--patch_width```, ```--patch_height```, ```--overlap_area``` in [split.py](split.py).

If you want to delete the images with empty annotation, you can use [clear_isaid.py](clear_isaid.py) to clear the folder by ```python clear_isaid.py --clear-folder <clear_folder_path>```.

# Acknowledgement
This script is developed based on [iSAID_Devkit](https://github.com/CAPTAIN-WHU/iSAID_Devkit).
