#!/bin/bash

screen -dmS "gp_gaussian_save_tabicl_predictions" bash -c '

    nice -7 python -m scripts.experiments.save_dense_tabicl_predictions \
        +data_name=gp_gaussian \
        +fixed_effect_name=steps_1D \
        +split_seed=0 \
        +tasks="[all]" \
        +n_jobs=4 \
        +overwrite=true
'
