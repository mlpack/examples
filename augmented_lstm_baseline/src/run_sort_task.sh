for i in $(seq 2 10); do
    (( MAX_LEN = 16 / $i ))
    for j in $(seq 1 $MAX_LEN); do
        echo "Evaluating length = $j, binary_length = $i"
        COMMAND="./mlpack_lstm_baseline -t sort -i 500 -l $j -b $i -r 1 -s 128 2> /dev/null | tail -n 1";
        eval "$COMMAND";
    done
done