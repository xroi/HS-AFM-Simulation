from IMP.npctransport import *
import sys
import math

# fetch params
# Usage: <cmd> <outfile> <newoutfle>
outfile = sys.argv[1]
newoutfile = sys.argv[2]
with open(outfile, "rb") as f:
    output = Output()
    output.ParseFromString(f.read())
assignment = output.assignment
assignment.is_multiple_hdf5s = True
assignment.simulation_time_ns = 1000
with open(newoutfile, "wb") as nf:
    nf.write(output.SerializeToString())
    print(output.assignment)
