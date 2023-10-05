# HS-AFM-Simulation

A model of high speed atomic force microscopy, based on statistics from [imp's](https://github.com/salilab/imp)
nuclear pore complex transport module.

## Usage

Get all packages listed in requirements.txt and run:

```
python main.py <args>
```

A full list of arguments and their descriptions can be found in the parameters.py file.

It should be noted that this simulation is ran separately than the npctransport simulation, and uses statistics from
it. Specifically a sequence of .hdf5 files containing the spatial statistics, at different timepoints. To run the
npctransport simulation that outputs the statistics at the required format, on HUJI computers, use the
run_sequential.sh script found in the scripts folder:

```
csh run_sequential.sh <from-start(1/0)> <start> <step> <output_path> <config_path>
```

Then supply <output_path> as the <input_path> argument for the AFM simulations.