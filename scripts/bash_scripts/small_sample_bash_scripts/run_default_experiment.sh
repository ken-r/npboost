# Run an experiment using the default hyperparameters for the model.

# Synthetic Experiment
export SCREEN_DATA="gp_gaussian" # gp_gaussian, gp_gaussian_very_many_groups, brownian_gaussian, brownian_lognormal, gp_gaussian_irrelevant_x
export SCREEN_EFFECT="steps_1D" # steps_1D, steps_2D, zero_1D, zero_2D
export SCREEN_SEED=0 # 0, 1, 2, 3, 4
export SCREEN_TASK="in_context" # in_context, few_shot
export SCREEN_MODEL="npboost" # lme, gplinear, gbm_group_cat, gbm_no_group, np, anp, npboost, anpboost, tabicl_pooled, tabicl_taskwise
export SCREEN_DEVICE="cuda" # cuda, cpu
export SCREEN_WANDB_ENABLED=true # true, false

screen -dmS "default_synthetic_experiment" bash -c '
    nice -7 python -m scripts.experiments.run_experiment \
        data=synthetic/${SCREEN_DATA} \
        data/synthetic/fixed_effect="${SCREEN_EFFECT}" \
        wandb.project="default_runs" \
        wandb.enabled=${SCREEN_WANDB_ENABLED} \
        data.experiment.split_data_seed=${SCREEN_SEED} \
        task="${SCREEN_TASK}" \
        model=${SCREEN_MODEL} \
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
export SCREEN_MODEL="npboost" # lme, gplinear, gbm_group_cat, gbm_no_group, np, anp, npboost, anpboost, tabicl_pooled, tabicl_taskwise
export SCREEN_DEVICE="cuda" # cuda, cpu
export SCREEN_WANDB_ENABLED=true # true, false


screen -dmS "default_real_experiment" bash -c '
    nice -7 python -m scripts.experiments.run_experiment \
        data=real/${SCREEN_DATA} \
        wandb.project="default_runs" \
        wandb.enabled=${SCREEN_WANDB_ENABLED} \
        data.experiment.split_data_seed=${SCREEN_SEED} \
        task="${SCREEN_TASK}" \
        model=${SCREEN_MODEL} \
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
