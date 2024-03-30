#!/bin/bash
./run_benchmarks_dit.sh
sleep 120
./run_benchmarks_no_dit.sh
sleep 120
./run_ct_experiments_dit.sh
sleep 120
./run_ct_experiments_no_dit.sh
