#!/bin/bash
echo 'Starting cows split data generation...'
echo ''

nice -7 python -m scripts.data.generate_data -m \
    data=real/cows

wait

echo ''
echo 'Data generation complete!'
read
