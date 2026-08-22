#!/bin/bash
EFFECTS="steps_1D,steps_2D"

echo 'Starting Brownian gaussian steps data generation...'
echo ''

# Run jobs on all datasets
nice -7 python -m scripts.data.generate_data -m \
    data=synthetic/brownian_gaussian \
    data/synthetic/fixed_effect="choice(${EFFECTS})"

# Wait for both background jobs to complete
wait

echo ''
echo 'Experiment complete!'
read
