# Run the full sweep for an experiment.

# Synthetic Experiment
export SCREEN_DATA="gp_gaussian" # gp_gaussian, gp_gaussian_very_many_groups, brownian_gaussian, brownian_lognormal, gp_gaussian_irrelevant_x
export SCREEN_EFFECT="steps_1D" # steps_1D, steps_2D, zero_1D, zero_2D
export SCREEN_SEED=0 # 0, 1, 2, 3, 4
export SCREEN_TASK="in_context" # in_context, few_shot
export SCREEN_MODEL="npboost" # gbm_group_cat, gbm_no_group, np, anp, npboost, anpboost
export SCREEN_DEVICE="cuda" # cuda, cpu
export SCREEN_WANDB_ENABLED=true # true, false


screen -dmS "synthetic_experiment_sweep" bash -c '
    nice -7 python -m scripts.experiments.run_experiment \
        data=synthetic/${SCREEN_DATA} \
        data/synthetic/fixed_effect="${SCREEN_EFFECT}" \
        wandb.project="sweep_runs" \
        wandb.enabled=${SCREEN_WANDB_ENABLED} \
        data.experiment.split_data_seed=${SCREEN_SEED} \
        task="${SCREEN_TASK}" \
        sweep=${SCREEN_MODEL}_sweep \
        model.validation_metric=rmse \
        device=${SCREEN_DEVICE} || {
            echo "------------------------------------------------"
            echo "There was an error! Check the logs above."
            echo "------------------------------------------------"
            read -p "Press [Enter] to kill the script..."
            exit 1
        }


    echo ""
    echo "Experiment complete!"
    read
    '


##########################################################


# Real Experiment
export SCREEN_DATA="cars" # cars, spotify, cows, bike
export SCREEN_SEED=0 # 0, 1, 2, 3, 4
export SCREEN_TASK="in_context" # in_context, few_shot
export SCREEN_MODEL="npboost" # gbm_group_cat, gbm_no_group, np, anp, npboost, anpboost
export SCREEN_DEVICE="cuda" # cuda, cpu
export SCREEN_WANDB_ENABLED=true # true, false


screen -dmS "real_experiment_sweep" bash -c '
    nice -7 python -m scripts.experiments.run_experiment \
        data=real/${SCREEN_DATA} \
        wandb.project="sweep_runs" \
        wandb.enabled=${SCREEN_WANDB_ENABLED} \
        data.experiment.split_data_seed=${SCREEN_SEED} \
        task="${SCREEN_TASK}" \
        sweep=${SCREEN_MODEL}_sweep \
        model.validation_metric=rmse \
        device=${SCREEN_DEVICE} || {
            echo "------------------------------------------------"
            echo "There was an error! Check the logs above."
            echo "------------------------------------------------"
            read -p "Press [Enter] to kill the script..."
            exit 1
        }


    echo ""
    echo "Experiment complete!"
    read
    '
