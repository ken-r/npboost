# Rerun the the best hyperparameter settings for an experiment and
# optionally save the model and/or predictions.

# Synthetic Experiment
export SCREEN_DATA="gp_gaussian" # gp_gaussian, gp_gaussian_very_many_groups, brownian_gaussian, brownian_lognormal, gp_gaussian_irrelevant_x
export SCREEN_EFFECT="steps_1D" # steps_1D, steps_2D, zero_1D, zero_2D
export SCREEN_SEED=0 # 0, 1, 2, 3, 4
export SCREEN_TASK="in_context" # in_context, few_shot
export SCREEN_MODEL="npboost" # lme, gplinear, gbm_group_cat, gbm_no_group, np, anp, npboost, anpboost
export SCREEN_DEVICE="cuda" # cuda, cpu
export SCREEN_SAVE_MODEL=false # true, false
export SCREEN_SAVE_PREDICTIONS=false # true, false

screen -dmS "synthetic_rerun_best" bash -c '
    echo "Running ${SCREEN_MODEL} / ${SCREEN_TASK} / seed ${SCREEN_SEED}..."
    nice -7 python -m scripts.experiments.save_model \
        +wandb_project_name=${SCREEN_DATA} \
        +fixed_effect_name=${SCREEN_EFFECT} \
        +local_project_name=${SCREEN_DATA} \
        +model_name="${SCREEN_MODEL}" \
        +task_name="${SCREEN_TASK}" \
        +data_split_seed=${SCREEN_SEED} \
        +validation_metric=rmse \
        +device=${SCREEN_DEVICE} \
        +save_model=${SCREEN_SAVE_MODEL} \
        +save_predictions=${SCREEN_SAVE_PREDICTIONS} || {
            echo "------------------------------------------------"
            echo "There was an error! Check the logs above."
            echo "------------------------------------------------"
            read -p "Press [Enter] to kill the script..."
            exit 1
        }
    echo "Rerunning complete!"
    read
    '


##########################################################


# Real Experiment
export SCREEN_DATA="cars" # cars, spotify, cows, bike
export SCREEN_EFFECT="steps_1D" # steps_1D, steps_2D, zero_1D, zero_2D
export SCREEN_SEED=0 # 0, 1, 2, 3, 4
export SCREEN_TASK="in_context" # in_context, few_shot
export SCREEN_MODEL="npboost" # lme, gplinear, gbm_group_cat, gbm_no_group, np, anp, npboost, anpboost
export SCREEN_DEVICE="cuda" # cuda, cpu
export SCREEN_SAVE_MODEL=false # true, false
export SCREEN_SAVE_PREDICTIONS=false # true, false

screen -dmS "real_rerun_best" bash -c '
    echo "Running ${SCREEN_MODEL} / ${SCREEN_TASK} / seed ${SCREEN_SEED}..."
    nice -7 python -m scripts.experiments.save_model \
        +wandb_project_name=${SCREEN_DATA} \
        +local_project_name=${SCREEN_DATA} \
        +model_name="${SCREEN_MODEL}" \
        +task_name="${SCREEN_TASK}" \
        +data_split_seed=${SCREEN_SEED} \
        +validation_metric=rmse \
        +device=${SCREEN_DEVICE} \
        +save_model=${SCREEN_SAVE_MODEL} \
        +save_predictions=${SCREEN_SAVE_PREDICTIONS} || {
            echo "------------------------------------------------"
            echo "There was an error! Check the logs above."
            echo "------------------------------------------------"
            read -p "Press [Enter] to kill the script..."
            exit 1
        }
    echo "Rerunning complete!"
    read
    '
