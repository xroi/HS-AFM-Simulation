#!/bin/bash

if [ $# != 1 ]; then
  echo "Syntax: $0 <dir_path>"
  exit 1
fi

tree -fQFi --sort=size ${1}/ | grep \.pb\"$ | tail +2 | cut -d '.' -f 1