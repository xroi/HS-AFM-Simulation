#!/bin/bash

if [ $# != 1 ]; then
  echo "Syntax: $0 <dir_path>"
  exit 1
fi

#tree -fQFi --sort=size ${1}/ | grep \.pb\"$ | tail +2 | cut -d '.' -f 1 | sed -n -e 's/^.*\///p'

ls -av ${1}/*.pb.final.rmf | tail -1 | cut -d '.' -f 1