for i in $(seq 1 10); do
    echo "Evaluating bit_length $i"
    COMMAND="./mlpack_lstm_baseline -t add -i 500 -b $i -r 1 -s 128 2> /dev/null | tail -n 1";
    eval "$COMMAND";
done