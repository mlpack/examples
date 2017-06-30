for i in $(seq 1 10); do
    echo "Evaluating length = $i"
    COMMAND="./mlpack_lstm_baseline -t copy -i 500 -l $i -r 1 -s 128 2> /dev/null | tail -n 1";
    eval "$COMMAND";
done