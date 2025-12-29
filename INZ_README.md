# PYSLAM

Najpierw zainstalować pyslam, mi szło najlepiej przez conde

## Struktura
- W `/data` są dwa rosbagi - jeden z symulacji `sim_rotated_imu`, jeden z Barki `20251130_1`. Wsyzstkie dane są już wygenerowane, bo rosbagi są za duże, żeby wrzucić je na githuba.

- W `/SCRIPTS_PYTHON` są moję skrypty, którymi generuję dane z rosbagu `rosbag_to_pyslam_gt.py`, robię wykres w matplotlibie `plot_tum.py` i obracam do tego jak evo zrobiło aligned `estimated_to_vo_aligned`. Żeby odpalić: 
    ```
    python3 ./SCRIPTS_PYTHON/rosbag_to_pyslam_gt.py
    ```
    **Ważne:** nie odpalać tego z pyslam enva, bo ros2 nie współpracuje wtedy.

- W `/pyslam` jest logika do slama itd.

- W `/` są skrypty do odpalenia. Nas interesuję `main_vo.py`

## Jak odpalić

Żeby odpalić `main_vo.py`, odpalamy środowisko pyslama. Potem w `/` komenda`./main_vo.py`

Wtedy do katalogu z datasetem zapiszę się predicted groundtruth w formacie TUM. Żeby zapisało się to w odpowiedni dataset trzeba odkmentować/zobaczyć linie **181**, **197** w pliku main_vo.py.

- Jeśli chcemy zmienić algorytm, to zmieniamy go na w plikach `main_vo.py` - linia **109**
    ``` python
    tracker_config = FeatureTrackerConfigs.ORB
    ```
    i w pliku `main_feature_matching.py` linia **158**
    ``` python
    tracker_config = FeatureTrackerConfigs.ORB
    ```

- Jeśli chcemy zmienić useGroundtruthScale to w pliku `pyslam/slam/visual_odometry.py` zmieniamy linie **46**:
    ``` python
    kUseGroundTruthScale = True
    ```

## Jak zmienić dataset

W pliku `config.yaml` wybieramy dataset odkomentowując odpowiednią linijkę:

``` yaml
DATASET:
  # select your dataset (decomment only one of the following lines) 
  #type: EUROC_DATASET  
  #type: KITTI_DATASET
  #type: TUM_DATASET
  #type: ICL_NUIM_DATASET
  #type: REPLICA_DATASET
  #type: TARTANAIR_DATASET
  #type: VIDEO_DATASET
  #type: SCANNET_DATASET
  #type: ROS1BAG_DATASET
  #type: ROS2BAG_DATASET  
  #type: FOLDER_DATASET
  #type: LIVE_DATASET  # Not recommended for current development stage
  
  type: BARKA_DATASET
  #type: SIM_DATASET
```

Jesłi chcemy KITTI to w config.yaml w KITTI_DATASET trzeba podać ścieżke do datasetu