#!/bin/bash -f

set SCRIPT=`readlink -f "$0"`
set DIRNAME=`dirname "$SCRIPT"`

echo $DIRNAME
source $DIRNAME/../venv_new/bin/activate.csh
pip3 install -r requirements.txt