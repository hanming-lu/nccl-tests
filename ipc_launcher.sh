#!/bin/bash

# Check if an argument is passed
if [ $# -eq 0 ]; then
    echo "Usage: $0 <argument>"
    exit 1
fi

# Get the argument
arg=$1

# Array of PIDs
pids=()

# Start the primary process
./testipc $arg 0 1 &
pids+=($!)

# Start the secondary processes
for i in {1..7}
do
    ./testipc $arg $i 0 &
    pids+=($!)
done

# Wait for all processes to complete
for pid in "${pids[@]}"
do
    wait $pid
done

echo "All processes completed"


