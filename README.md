# Narrated Learning

A research project exploring self-supervised learning from sequential sensory data. The core idea is that temporal sequences — video frames, sensor readings, time series — contain structure that can be exploited to learn useful representations without labels. The project investigates how to discover recurring patterns, build hierarchical features, and make predictions, drawing on ideas from competitive learning and sparse coding.

Developed from October 2016 to September 2022, the project went through several research directions described below.

## Research Directions

### 1. Robot Navigation and World Modeling (2016–2019)

A simulated 2D robot navigates an environment with walls, using ray-cast sensors to perceive its surroundings. The robot learns a lookup-table-based model of its sensory inputs and uses predictor ensembles to anticipate future sensor states given motor commands. A task mode tests whether the learned model can support goal-directed navigation to specified regions.

**Entry point:** `run_demo.py`

### 2. Visual Feature Learning with Winner-Take-All Networks (2021–2022)

Video frames (from nature footage, driving video, etc.) are converted into binary events using a simulated event camera, then fed into competitive learning networks. Multiple variants of Winner-Take-All (WTA) networks learn a set of receptive fields (RFs) — small learned image patches that tile the visual input. Each input frame triggers a competition among RFs, and the closest-matching RF updates its weights toward the input.

Key variants explored:
- **Rate-control WTA** — tracks how often each RF wins and adjusts learning rates to prevent any single RF from dominating
- **Iterative WTA** — runs multiple rounds of competition per input frame
- **Tiled WTA** — divides the input image into spatial tiles, each with its own set of competing RFs
- **Multi-layer hierarchical WTA** — stacks multiple layers where higher layers learn features over the outputs of lower layers
- **Coincidence / dynamic coincidence** — learns based on temporal co-occurrence of events rather than spatial similarity alone
- **Indirect learning** — the most recent direction (Sep 2022), using eligibility traces to update RFs indirectly

**Entry points:**
- `run_demo_perception.py` — single-layer visual learning (most actively developed)
- `run_demo_perception_multilayer.py` — multi-layer hierarchical version
- `run_sweeps_perception.py` — hyperparameter sweeps over network type, RF count, learning rate, etc.

### 3. Multi-layer Sequence Prediction (2019–2021)

Learns hierarchical representations where each layer discretizes its input into bins and builds lookup tables or KNN-based associative memories. Uses temporal context (sequences of recent states) as the basis for clustering and prediction. Tested on video, stereo camera data, and a 2D bouncing-ball physics simulation.

**Entry point:** `run_demo_stocks.py` (despite the name, this is the general-purpose entry point for multi-layer WTA on various inputs including stock data, physics simulation, and video)

### 4. Tile-based Segmentation and Prediction (2019–2020)

Divides video frames into small spatial tiles and learns predictive models per tile — given recent tile states, predict the tile's state several frames ahead. Used dynamic tile allocation and tracked prediction error across training and holdout data.

**Entry point:** `run_demo_segment.py`

### 5. MLP Ensemble Prediction (2017–2018)

An ensemble of small multi-layer perceptrons competes to predict the next sensory frame from recent history. At each step, only the MLP with the lowest prediction error is trained (a WTA rule applied to the predictors themselves). Compared ensemble performance against a single monolithic MLP.

**Entry point:** `run_prediction_experiment.py` (written for Python 2; requires adaptation for Python 3)

## Codebase Structure

```
run_demo.py                         # Robot navigation simulation
run_demo_perception.py              # Single-layer visual feature learning
run_demo_perception_multilayer.py   # Multi-layer visual feature learning
run_demo_stocks.py                  # Multi-layer WTA on time-series / video / physics
run_demo_segment.py                 # Tile-based segmentation and prediction
run_sweeps_perception.py            # Hyperparameter sweep runner
run_prediction_experiment.py        # MLP ensemble prediction (Python 2)

robot_brain_classes/                # ~28 brain implementations (learning algorithms)
  indirect.py                       #   Most recent: indirect RF learning via traces
  rate_control_wta.py               #   Rate-controlled WTA
  rate_control_wta_heirarchy.py     #   Multi-layer rate-controlled WTA
  tiled_multilayer_wta.py           #   Tiled multi-layer WTA
  dynamic_coincidence.py            #   Temporal coincidence learning
  wta_multi_layer_brain.py          #   Multi-layer WTA with bin discretization
  simple_nl_brain.py                #   Lookup-table brain for robot navigation
  ...

brain_components_classes/           # ~28 reusable components (layers, predictors, history buffers)
robot_sensor_classes/               # Input sources (video, stereo camera, physics sim, stock data)
robot_preprocess_classes/           # Event camera simulation (frame differencing → binary events)
visualizer_classes/                 # Real-time visualization of inputs, RFs, and errors
offline_analyses/                   # Post-hoc analysis and sparse feature extraction

robot_environment.py                # 2D simulated environment with walls and ray-casting
robot_model.py                      # Robot kinematics (linear + angular velocity)
task_manager.py                     # Goal-directed task evaluation
sim_folder_manager.py               # Output directory management
performance_evaluator.py            # Error metrics

cython_*/                           # Cython-accelerated distance computation and KNN
cuda_dist_query.py                  # CUDA GPU-accelerated distance queries
```

## Dependencies

- Python 3
- NumPy
- OpenCV (`opencv-python`)
- Matplotlib

Optional, for performance:
- Cython (for accelerated distance computation — build with `install_cython_libraries.sh`)
- PyCUDA + scikit-cuda (for GPU-accelerated distance queries; requires NVIDIA GPU)

A Dockerfile is included (based on `nvidia/cuda:8.0-devel`) that installs all dependencies.

## Running Experiments

### Visual feature learning (most developed path)

```bash
python run_demo_perception.py
```

You will be prompted for a simulation name prefix. The script loads video frames from a configured path (`video_dir` in `get_sensors_params()`), simulates an event camera by thresholding brightness changes between frames, and feeds the resulting binary events into an `IndirectRFBrain`. Learned receptive fields and error plots are saved periodically to a timestamped output directory.

Default configuration:
- Input: 16x16 pixel crops from a sea turtle video
- 32 receptive fields, learning rate 0.01
- Runs for up to 5M steps

To change the video source, number of RFs, or learning rate, edit the parameter dictionaries at the top of the file (`get_sensors_params()`, `get_brain_params()`). To switch to a different brain implementation, change the import and the class instantiated in `init_demo()` — commented-out imports show available alternatives.

### Hyperparameter sweeps

```bash
python run_sweeps_perception.py
```

Runs multiple simulations in parallel (18 processes by default) across different values of a chosen parameter. Edit the `param_sets` list in `run_several_sweeps()` to select which parameter to sweep. Generates comparison plots of final error vs. parameter value.

### Robot navigation

```bash
python run_demo.py
```

Runs the 2D robot simulation with a lookup-table brain. The robot moves randomly while building a model of its environment. Visualization shows the top-down map, ray-cast sensor readings, and learned lookup table entries. Set `ENABLE_TASK_MODE = True` to test goal-directed navigation after learning.

### Multi-layer WTA on various inputs

```bash
python run_demo_stocks.py
```

Runs a multi-layer WTA brain on stock data (default), video, or a bouncing-ball physics simulation. Toggle `USE_VIDEO_IN` and change the sensor class in `init_demo()` to switch input sources.

### Tile-based segmentation

```bash
python run_demo_segment.py
```

Learns per-tile predictive models on video input. Tracks prediction error and switches to holdout evaluation after a configured number of training steps.

### Notes

- Most scripts expect video files to be at paths like `/srv/projects/video-downloads/` or `/home/csaba/projects/video-downloads/`. Adjust the `video_dir` and `video_filename` parameters in each script to point to your own data.
- Output directories (models, plots) default to paths like `/srv/projects/NL-sim/` — adjust `sim_folders_path` in `get_sim_folder_manager_params()` as needed.
- The Cython and CUDA modules are optional performance optimizations. The core learning algorithms run on CPU with NumPy alone, though some brain classes may import CUDA modules that need to be commented out if no GPU is available.
