#!/bin/bash
for model in "npboost" "anpboost" "np" "anp"; do
    export SCREEN_MODEL=$model
    screen -dmS "cows_save_seed0_${SCREEN_MODEL}" bash -c '
        for task_name in "in_context" "few_shot"; do
            echo "Running ${SCREEN_MODEL} / ${task_name} / seed 0..."
            CUDA_VISIBLE_DEVICES=0 nice -7 python -m scripts.experiments.save_model \
                +wandb_project_name="cows" \
                +local_project_name="cows" \
                +model_name="${SCREEN_MODEL}" \
                +task_name="${task_name}" \
                +data_split_seed=0 \
                +save_model=true \
                +save_predictions=true \
                +force_wandb_run=true || {
                    echo "------------------------------------------------"
                    echo "There was an error! Check the logs above."
                    echo "------------------------------------------------"
                    read -p "Press [Enter] to kill the script..."
                    exit 1
                }
        done
        echo "Saving ${SCREEN_MODEL} complete!"
        read
        '
done

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

screen -dmS "cows_save_seed0_other_models" bash -c '
    for model in "gbm_group_cat" "gbm_no_group" "gplinear" "lme"; do
        for task_name in "in_context" "few_shot"; do
            echo "Running ${model} / ${task_name} / seed 0..."
            nice -7 python -m scripts.experiments.save_model \
                +wandb_project_name="cows" \
                +local_project_name="cows" \
                +model_name="${model}" \
                +task_name="${task_name}" \
                +data_split_seed=0 \
                +save_model=true \
                +save_predictions=true \
                +force_wandb_run=true || {
                    echo "------------------------------------------------"
                    echo "There was an error! Check the logs above."
                    echo "------------------------------------------------"
                    read -p "Press [Enter] to kill the script..."
                    exit 1
                }
        done
        echo "Saving ${model} complete!"
    done
    echo "Saving all models complete!"
    read
    '
