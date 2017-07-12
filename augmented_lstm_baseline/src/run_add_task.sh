#!/bin/bash
#
# The script that runs `mlpack_lstm_baseline` executable from the script directory
# and runs AddTask on sequence lengths from 1 to 10.

for i in $(seq 1 10); do
    echo "Evaluating bit_length $i"
    COMMAND="./mlpack_lstm_baseline -t add -e 500 -b $i -r 1 -s 128 2> /dev/null | tail -n 1";
    eval "$COMMAND";
done