#!/bin/bash
EFFECTS="steps_2D"

echo 'Starting GP gaussian steps data generation...'
echo ''

# Run jobs on all datasets
nice -7 python -m scripts.data.generate_data -m \
    data=synthetic/gp_gaussian_irrelevant_x \
    data/synthetic/fixed_effect="choice(${EFFECTS})"

# Wait for both background jobs to complete
wait

echo ''
echo 'Experiment complete!'
read
