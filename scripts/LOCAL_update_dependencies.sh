#!/bin/bash -f

set SCRIPT=`readlink -f "$0"`
set DIRNAME=`dirname "$SCRIPT"`

source $DIRNAME/../venv_new/bin/activate.csh
pip3 install -r requirements.txt