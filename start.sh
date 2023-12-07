#!/bin/bash

echo "Exporting the src folder into the python path"
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
