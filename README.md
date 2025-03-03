# HS-AFM-Simulation

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Description

A computational model of High-Speed Atomic Force Microscopy (HS-AFM), based on statistical data from [IMP's](https://github.com/salilab/imp) Nuclear Pore Complex (NPC) transport module. This simulation enables visualization and validation of NPC dynamics at nanoscale resolution.

## Demo

https://github.com/user-attachments/assets/9ce97b72-0078-4fce-85fd-6a67d7bcb794

## Requirements

### System Requirements

Tested on the following systems:

- Windows 11 Version 10.0.22621 Build 22621
- Debian GNU/Linux 11 (bullseye)

### Dependencies

* Python 3.10 or higher
* Packages listed in [requirements.txt](requirements.txt)


## Installation

1. Clone the repository:

```
git clone https://github.com/xroi/HS-AFM-Simulation.git
cd HS-AFM-Simulation
```

2. Create and activate a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

The typical installation time should be a few minutes.

## Usage

### Quick start

A demo dataset is provided in the `demo_dataset` folder. To run the simulation using this dataset:

```
python src/main.py @demo_dataset/args.txt
```

The typical runtime for the demo should be around 10 seconds on modern hardware.

### Input data

The standard input for the simulation is a sequence of .hdf5 files containing spatial density statistics of the nuclear pore complex over time.

These can be generated using the [npctransport module](https://github.com/salilab/npctransport) in imp.

Parts of the code may be adapted to general spatial density statistics.  

### Command-Line Arguments
Arguments can be specified directly or in a text file prefixed with @:


| Argument | Description |
|--------|-------------|
| `-h, --help` | Show a help message and exit |
| `--npc-simulation, --no-npc-simulation` | If `--npc-simulation`, the NPC simulations run 'live'. If `--no-npc-simulation`, the program uses files in the folder specified with `--existing_files_path` |
| `--input-path INPUT_PATH` | Path to the folder containing hdf5 density map files, used with `--no-npc-simulation` flag |
| `--input-suffix INPUT_SUFFIX` | Files should be in input-path and named `<delta_time_in_ns>.<input-suffix>` |
| `--read-from-gzip, --no-read-from-gzip` | If true, allows reading from gzip compressed HDF5 files |
| `--simulation-start-time-ns SIMULATION_START_TIME_NS` | Start time of the AFM simulation in nanoseconds (NPC will be simulated since 0) |
| `--simulation-end-time-ns SIMULATION_END_TIME_NS` | End time of the AFM simulation in nanoseconds (NPC will be simulated since 0) |
| `--interval-ns INTERVAL_NS` | Interval time of the NPC simulation in nanoseconds |
| `--voxel-size-a VOXEL_SIZE_A` | Size of each voxel in the NPC simulation in Angstrom (same as in NPC simulation configuration file) |
| `--tunnel-radius-a TUNNEL_RADIUS_A` | Radius of the NPC tunnel in Angstrom (same as in NPC simulation configuration file) |
| `--slab-thickness-a SLAB_THICKNESS_A` | Thickness of the NPC slab in Angstrom (same as in NPC simulation configuration file) |
| `--statistics-interval-ns STATISTICS_INTERVAL_NS` | Should be the same as specified in the NPC simulation configuration file |
| `--torus-slab, --no-torus-slab` | Specifies if the slab is a torus (should be the same as in the original simulation) |
| `--separate-n-c, --no-separate-n-c` | Specifies if FG chains are separated into N and C parts |
| `--min-x-coord MIN_X_COORD` | First pixel on the X axis for simulation (inclusive, starting from 0) |
| `--max-x-coord MAX_X_COORD` | Last pixel on the X axis for simulation (non-inclusive, starting from 0) |
| `--min-y-coord MIN_Y_COORD` | First pixel on the Y axis for simulation (inclusive, starting from 0) |
| `--max-y-coord MAX_Y_COORD` | Last pixel on the Y axis for simulation (non-inclusive, starting from 0) |
| `--min-z-coord MIN_Z_COORD` | First pixel on the Z axis for simulation (inclusive, starting from 0) |
| `--max-z-coord MAX_Z_COORD` | Last pixel on the Z axis for simulation (non-inclusive, starting from 0) |
| `--vertical-scanning, --no-vertical-scanning` | If true, scans the lines vertically |
| `--z-func {z_test,z_test2}` | Function to use for Z-axis calculations |
| `--calc-needle-threshold, --no-calc-needle-threshold` | If true, calculates the threshold using the `median_threshold` function in utils.py |
| `--calc-threshold-r-px CALC_THRESHOLD_R_PX` | Radius around the center to get median value from in pixels (for threshold calculation) |
| `--calc-threshold-frac CALC_THRESHOLD_FRAC` | Fraction of the median to take for the threshold |
| `--tip-custom-threshold TIP_CUSTOM_THRESHOLD` | Density threshold below which the needle ignores (used for all Z functions) |
| `--tip-radius-px TIP_RADIUS_PX` | How far around the origin pixel the needle considers for determining pixel height (ball shape) |
| `--fgs-resistance, --no-fgs-resistance` | Whether NUPs offer resistance to the simulated tip (default: True) |
| `--fgs-verticality-weights, --no-fgs-verticality-weights` | Whether NUPs are weighted by verticality score (default: True) |
| `--floaters-resistance, --no-floaters-resistance` | Whether floaters (NTRs and Passive Molecules) offer resistance to the simulated tip |
| `--fgs-sigma-a FGS_SIGMA_A` | Sigma value for the normal distribution used to weigh FG chains |
| `--floaters-sigma-a FLOATERS_SIGMA_A` | Sigma value for the normal distribution used to weigh floaters |
| `--floater-size-factor FLOATER_SIZE_FACTOR` | Factor to multiply floater size in Angstrom (0.125 to be proportional with 8Å FG beads) |
| `--floater-distribution-factor FLOATER_DISTRIBUTION_FACTOR` | Factor to multiply the distribution before adding to floater weight |
| `--floater-general-factor FLOATER_GENERAL_FACTOR` | Factor to multiply the final weight of the floater |
| `--time-per-line-ns TIME_PER_LINE_NS` | Time for a needle to pass a full line (mutually exclusive with `--time-per-pixel-ns`) |
| `--time-per-pixel-ns TIME_PER_PIXEL_NS` | Time for a needle to pass a single pixel (mutually exclusive with `--time-per-line-ns`) |
| `--time-between-scans-ns TIME_BETWEEN_SCANS_NS` | Time for the needle to return to the starting point for the next frame |
| `--output-non-raster-gif, --no-output-non-raster-gif` | Whether to output a non-raster GIF |
| `--output-raster-gif, --no-output-raster-gif` | Whether to output a raster GIF |
| `--output-pickle, --no-output-pickle` | Whether to output a pickle file |
| `--output-post, --no-output-post` | Whether to output post-analysis visualizations |
| `--output-path-prefix OUTPUT_PATH_PREFIX` | Path prefix for output files |
| `--output-gif-color-map OUTPUT_GIF_COLOR_MAP` | Matplotlib color map for the output GIF (e.g., 'gist_gray', 'RdBu', 'gist_rainbow') |
| `--output-resolution-x OUTPUT_RESOLUTION_X` | X-axis resolution of output GIF in pixels (upscaled from original height maps) |
| `--output-resolution-y OUTPUT_RESOLUTION_Y` | Y-axis resolution of output GIF in pixels (upscaled from original height maps) |
| `--output-hdf5, --no-output-hdf5` | Whether to output an HDF5 file |
| `--output-hdf5-path OUTPUT_HDF5_PATH` | Path to output HDF5 file |
| `--progress-bar, --no-progress-bar` | Whether to show a progress bar |
| `--enlarge-floaters, --no-enlarge-floaters` | If true, spreads floater density based on the radius in the type name (in Angstrom) |
| `--n-cores N_CORES` | Number of cores for parallel operation (uses maximum if not specified) |

### Output

The simulation generates the following outputs:

- Pickle file containing a python dictionary with the following keys:
  - `non-rasterized-maps`, a list of numpy arrays containing the height maps for each timepoint.
  - `rasterized-maps`, a list of numpy arrays containing the height maps for each timepoint, post rastering
  - `args`, a dictionary containing the arguments used for the simulation
- Visualizations (optional): `.gif` or `.mp4` visual representations of the simulation.

