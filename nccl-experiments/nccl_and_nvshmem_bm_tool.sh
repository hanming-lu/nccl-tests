#!/bin/bash

# File to store results
output_file="all_bm_results.txt"
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Clear the output file if it exists
> $output_file

# Define an array of commands to run
commands=(
    "nvshmrun -n 4 -ppn 4 ./nvsh_gather_customize 1"
    "nvshmrun -n 4 -ppn 4 ./nvsh_gather_customize 2"
    "nvshmrun -n 4 -ppn 4 ./nvsh_gather_customize 4"
    "nvshmrun -n 4 -ppn 4 ./nvsh_gather_customize 8"
    "nvshmrun -n 4 -ppn 4 ./nvsh_gather_customize 16"
    "nvshmrun -n 4 -ppn 4 ./nvsh_gather_customize 32"
    "nvshmrun -n 4 -ppn 4 ./nvsh_gather_customize 64"
    "nvshmrun -n 4 -ppn 4 ./nvsh_gather_customize 128"
    "nvshmrun -n 4 -ppn 4 ./nvsh_gather_customize 256"
    "nvshmrun -n 4 -ppn 4 ./nvsh_gather_customize 512"
    "nvshmrun -n 4 -ppn 4 ./nvsh_gather_customize 1024"
    "nvshmrun -n 4 -ppn 4 ./nvsh_gather_customize 2048"
    "nvshmrun -n 4 -ppn 4 ./nvsh_gather_customize 4096"
    "./nccl_gather 1"
    "./nccl_gather 2"
    "./nccl_gather 4"
    "./nccl_gather 8"
    "./nccl_gather 16"
    "./nccl_gather 32"
    "./nccl_gather 64"
    "./nccl_gather 128"
    "./nccl_gather 256"
    "./nccl_gather 512"
    "./nccl_gather 1024"
    "./nccl_gather 2048"
    "./nccl_gather 4096"
)

# Run each command and append its output to the output file
for cmd in "${commands[@]}"; do
    echo "Running: $cmd" >> $output_file
    $cmd >> $output_file 2>&1
    echo "" >> $output_file
done

echo "All commands have been executed. Results are stored in $output_file."