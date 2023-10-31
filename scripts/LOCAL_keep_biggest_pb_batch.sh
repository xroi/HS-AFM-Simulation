#!/bin/bash

if [ $# != 2 ]; then
  echo "Syntax: $0 <dir_path> <amount>"
  exit 1
fi

for (( i=1 ; i<=${2} ; i++ ));
do
   tree -fQFi --sort=size ${1}/${i}/ | grep \.pb\"$ | tail +3 | xargs rm
done
