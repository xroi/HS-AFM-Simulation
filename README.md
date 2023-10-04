# HS-AFM-Simulation

A model of high speed atomic force microscopy, based on statistics from [imp's](https://github.com/salilab/imp)
nuclear pore complex transport module.

## Usage

```
python main.py <args>
```

A full list of arguments and their descriptions can be found in the parameters.py file.

It should be noted that this simulation is ran separately than the npctransport simulation, and uses statistics from
it. Specifically a sequence of .hdf5 files containing the spatial statistics. At different timepoints. To ran a
simulations that outputs the statistics at the required format, on HUJI computers one can use the run_sequential.sh
script found in the scripts folder like so:

```
csh run_sequential.sh <from-start(1/0)> <start> <step> <output_path> <config_path>
```

Then supply <output_path> as the <input_path> argument for the AFM simulations.