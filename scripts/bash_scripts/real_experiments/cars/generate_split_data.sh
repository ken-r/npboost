#!/bin/bash
echo 'Starting cars split data generation...'
echo ''

# Run jobs on all datasets
nice -7 python -m scripts.data.generate_data -m \
    data=real/cars
# Wait for both background jobs to complete
wait

echo ''
echo 'Data generation complete!'
read
