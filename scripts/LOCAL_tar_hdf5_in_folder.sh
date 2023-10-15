#!/bin/csh -f

if ($#argv != 1) then
    echo "Syntax: $0 <folder>"
    exit 0
endif

tar -cvf spatial.tar $1/*.hdf5
