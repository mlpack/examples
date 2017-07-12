#!/bin/bash
#
# The script that runs `mlpack_lstm_baseline` executable from the script directory
# and runs CopyTask on sequence lengths from 2 to 10.

for i in $(seq 2 10); do
    echo "Evaluating length = $i"
    COMMAND="./mlpack_lstm_baseline -t copy -e 500 -l $i -r 1 -s 128 2> /dev/null | tail -n 1";
    eval "$COMMAND";
done