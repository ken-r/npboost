# Synthetic Data Generation: GP Gaussian
EFFECTS="steps_1D,steps_2D" # steps_1D, steps_2D, zero_1D, zero_2D

echo 'Starting synthetic data generation...'
echo ''

data="gp_gaussian" # gp_gaussian, gp_gaussian_very_many_groups, brownian_gaussian, brownian_lognormal, gp_gaussian_irrelevant_x

nice -7 python -m scripts.data.generate_data -m \
    data=synthetic/${data} \
    data/synthetic/fixed_effect="choice(${EFFECTS})"

echo ''
echo 'GP Gaussian data generation complete!'


##########################################################


# Real Data generation: Cars dataset
echo 'Starting cars split data generation...'
echo ''

data="cars" # cars, spotify, cows, bike

nice -7 python -m scripts.data.generate_data -m \
    data=real/${data}

wait

echo ''
echo 'Cars data generation complete!'
read
