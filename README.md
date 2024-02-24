# HS-AFM-Simulation

### Description

A model of High Speed Atomic Force Microscopy (HS-AFM), based on statistics from [imp's](https://github.com/salilab/imp)
Nuclear Pore Complex transport module.
---

### Requirements

* A system with python 3.6 or higher
* packages listed in requirements.txt

---

### Installation

1. Clone the repository:

```
git clone https://github.com/xroi/HS-AFM-Simulation.git
```

2. Install the required packages:

```
pip install -r requirements.txt
```

---

### Usage

This example is demonstrated using the demo dataset, available in the `demo_dataset` folder.
The input for the simulation is a sequence of .hdf5 files containing spatial density statistics at increasing
timepoints.
These can be generated using the [npctransport module](https://github.com/salilab/npctransport) in imp.

To run the simulation, use the following command:

```
python src/main.py @demo_dataset/args.txt
```

args.txt is a file containing the following arguments:

```
  -h, --help            show this help message and exit
  --npc-simulation, --no-npc-simulation
                        In the case of --npc-simulation flag, the npc
                        simulations run 'live'. In the case of --no-npc-
                        simulation flag, the program uses files in the folder
                        specified with --existing_files_path
  --input-path INPUT_PATH
                        Path to the folder containing hdf5 density map files,
                        to be used in the case of --no-npc-simulation flag.
  --input-suffix INPUT_SUFFIX
                        Files should be in input-path and named
                        <delta_time_in_ns>.<input-suffix>
  --read-from-gzip, --no-read-from-gzip
                        If true, allows reading from gzip compressed HDF5
                        files.
  --simulation-start-time-ns SIMULATION_START_TIME_NS
                        Start time of the AFM simulation in nanoseconds. (NPC
                        will be simulated since 0)
  --simulation-end-time-ns SIMULATION_END_TIME_NS
                        End time of the AFM simulation in nanoseconds. (NPC
                        will be simulated since 0)
  --interval-ns INTERVAL_NS
                        Interval time of the NPC simulation in nanoseconds.
  --voxel-size-a VOXEL_SIZE_A
                        Size of each voxel in the NPC simulation in Angstrom.
                        (Should be the same as specified inthe npc simulation
                        configuration file).
  --tunnel-radius-a TUNNEL_RADIUS_A
                        Radius of the NPC tunnel in Angstrom. (Should be the
                        same as specified inthe npc simulation configuration
                        file).
  --slab-thickness-a SLAB_THICKNESS_A
                        Thickness of the NPC slab in angstrom. (Should be the
                        same as specified inthe npc simulation configuration
                        file).
  --statistics-interval-ns STATISTICS_INTERVAL_NS
                        (Should be the same as specified in the npc simulation
                        configuration file).
  --torus-slab, --no-torus-slab
                        Specifies if the slab is a torus. should be the same
                        as in the original simulation.
  --separate-n-c, --no-separate-n-c
                        Specifies if fg chains are seperated into N and C
                        parts.
  --min-x-coord MIN_X_COORD
                        Specifies the first pixel on the X axis on which the
                        simulation is ran (inclusive). Count starting from 0.
  --max-x-coord MAX_X_COORD
                        Specifies the last pixel on the X axis on which the
                        simulation is ran (not inclusive). Count starting from
                        0.
  --min-y-coord MIN_Y_COORD
                        Specifies the first pixel on the Y axis on which the
                        simulation is ran (inclusive). Count starting from 0.
  --max-y-coord MAX_Y_COORD
                        Specifies the last pixel on the Y axis on which the
                        simulation is ran (not inclusive). Count starting from
                        0.
  --min-z-coord MIN_Z_COORD
                        Specifies the first pixel on the Z axis on which the
                        simulation is ran (inclusive). Count starting from 0.
  --max-z-coord MAX_Z_COORD
                        Specifies the last pixel on the Z axis on which the
                        simulation is ran (not inclusive). Count starting from
                        0.
  --vertical-scanning, --no-vertical-scanning
                        If true, scans the lines vertically.
  --z-func {z_test,z_test2}
  --calc-needle-threshold, --no-calc-needle-threshold
                        If true, calculates the threshold using the
                        median_threshold function in utils.py.
  --calc-threshold-r-px CALC_THRESHOLD_R_PX
                        The radius around the center to get the median value
                        from in pixels. (for calculating the threshold if
                        calc_needle_threshold is true)
  --calc-threshold-frac CALC_THRESHOLD_FRAC
                        The fraction of the median to take for the threshold.
                        (for calculating the threshold if
                        calc_needle_threshold is true)
  --tip-custom-threshold TIP_CUSTOM_THRESHOLD
                        The density under which the needle ignores. Used for
                        all z funcs.
  --tip-radius-px TIP_RADIUS_PX
                        Determines how far around the origin pixel the needle
                        considers for determining pixel height. (Assuming ball
                        shape). Should be greater than 1. Only used for z_top
                        z func.
  --floaters-resistance, --no-floaters-resistance
                        Determines whether floaters (NTRs and Passive
                        Molecules) offer resistance to the simulated tip.
  --fgs-sigma-a FGS_SIGMA_A
                        The sigma value for the normal distribution used to
                        weigh fg chains.
  --floaters-sigma-a FLOATERS_SIGMA_A
                        The sigma value for the normal distribution used to
                        weigh floaters.
  --floater-size-factor FLOATER_SIZE_FACTOR
                        Determines the factor by which the floater size in
                        angstrom is multiplied which is then added to the
                        floater weight. (0.125 to be proportional with 8A fg
                        beads)
  --floater-distribution-factor FLOATER_DISTRIBUTION_FACTOR
                        Determines the factor by which the distribution is
                        multiplied which is then added to the floater weight.
  --floater-general-factor FLOATER_GENERAL_FACTOR
                        Determines the factor by which the final weight of the
                        floater is multiplied
  --time-per-line-ns TIME_PER_LINE_NS
                        Determines the amount of time it takes for a needle to
                        pass a full line. Mutually exclusive with time-per-
                        pixel-ns.
  --time-per-pixel-ns TIME_PER_PIXEL_NS
                        Determines the amount of time it takes for a needle to
                        pass a single pixel. Mutually exclusive with time-per-
                        line-ns.
  --time-between-scans-ns TIME_BETWEEN_SCANS_NS
                        Determines the amount of time it takes for the needle
                        to return to the starting point to start the next
                        frame.
  --output-non-raster-gif, --no-output-non-raster-gif
                        Outputs a gif if '--output-non-raster-gif', doesn't if
                        --no-output-non-raster-gif
  --output-raster-gif, --no-output-raster-gif
                        Outputs a gif if '--output-raster-gif', doesn't if
                        --no-output-raster-gif
  --output-pickle, --no-output-pickle
                        Outputs a pickle if '--output-pickle', doesn't if
                        --no-output-pickle
  --output-post, --no-output-post
                        Outputs the post analysis visualizations
  --output-path-prefix OUTPUT_PATH_PREFIX
                        Path to output gif file.
  --output-gif-color-map OUTPUT_GIF_COLOR_MAP
                        Determines the matplotlib color map used for the
                        output gif. Some useful ones are: 'gist_gray' for
                        black and white, 'RdBu' for diverging Red-white-blue,
                        'gist_rainbow' for rainbow.
  --output-resolution-x OUTPUT_RESOLUTION_X
                        x axis Resolution of output gif in pixels. (Up-scaled
                        from original height Maps size)
  --output-resolution-y OUTPUT_RESOLUTION_Y
                        y axis Resolution of output gif in pixels. (Up-scaled
                        from original height Maps size)
  --output-hdf5, --no-output-hdf5
                        Outputs a hdf5 file if '--output-hdf5', doesn't if
                        --no-output-hdf5
  --output-hdf5-path OUTPUT_HDF5_PATH
                        Path to output hdf5 file.
  --progress-bar, --no-progress-bar
                        If true, shows a progress bar.
  --enlarge-floaters, --no-enlarge-floaters
                        If true, spreads floater density based on the radius
                        in the type name (in angstrom)
  --n-cores N_CORES     To be used for parallel operation. If not specified,
                        uses the maximum number of cores.
```

Using the default arugments, the run time for the demo should be around 10 seconds. The output is a pickle file
containing a dictionary with
3 entries: `non-rasterized-maps`, a list of numpy arrays containing the height maps for each timepoint,
`rasterized-maps`, a list of numpy arrays containing the height maps for each timepoint, post rastering,
and `args`, a dictionary containing the arguments used for the simulation.

---

A script file, `LOCAL_npctransport_sequential.sh` is supplied to run the npctransport simulation on the HUJI cluster,
with the required output format. To run the script, use the following command:

```
csh scripts/LOCAL_npctransport_sequential.sh <from-start(1/0)> <start> <step> <output_folder_path> <config_path> <output_statistics_interval>
```

Then supply `output_path` as the `input_path` argument for the HS-AFM simulation.