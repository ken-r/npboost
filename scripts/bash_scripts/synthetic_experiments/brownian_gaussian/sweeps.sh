
cuda_device=1
for model in "npboost" "anpboost" "np" "anp"; do
    for effect in "steps_1D" "steps_2D"; do
        export SCREEN_MODEL=$model
        export SCREEN_EFFECT=$effect
        export SCREEN_CUDA_DEVICE=$cuda_device
        screen -dmS "gp_gaussian_very_many_groups_${model}_${effect}_rmse_012" bash -c '
            for seed in 0 1 2; do
                for task in "in_context" "few_shot"; do
                    echo "Running experiment for model ${SCREEN_MODEL}, effect ${SCREEN_EFFECT}, seed 0, task $task..."
                    CUDA_VISIBLE_DEVICES=$SCREEN_CUDA_DEVICE nice -7 python -m scripts.experiments.run_experiment \
                        data=synthetic/gp_gaussian_very_many_groups \
                        sweep=${SCREEN_MODEL}_sweep \
                        model.validation_metric=rmse \
                        data/synthetic/fixed_effect="${SCREEN_EFFECT}" \
                        data.experiment.split_data_seed=$seed \
                        task="$task" \
                        device=cuda || {
                            echo "------------------------------------------------"
                            echo "There was an error! Check the logs above."
                            echo "------------------------------------------------"
                            read -p "Press [Enter] to kill the script..."
                            exit 1
                        }
                done
            done
            echo "Experiment complete!"
            read
            '
        cuda_device=$(( (cuda_device + 1) % 2 ))
    done
done

cuda_device=0
for model in "npboost" "anpboost" "np" "anp"; do
    for effect in "steps_1D" "steps_2D"; do
        export SCREEN_MODEL=$model
        export SCREEN_EFFECT=$effect
        export SCREEN_CUDA_DEVICE=$cuda_device
        screen -dmS "gp_gaussian_very_many_groups_${model}_${effect}_rmse_34" bash -c '
            for seed in 3 4; do
                for task in "in_context" "few_shot"; do
                    echo "Running experiment for model ${SCREEN_MODEL}, effect ${SCREEN_EFFECT}, seed 0, task $task..."
                    CUDA_VISIBLE_DEVICES=$SCREEN_CUDA_DEVICE nice -7 python -m scripts.experiments.run_experiment \
                        data=synthetic/gp_gaussian_very_many_groups \
                        sweep=${SCREEN_MODEL}_sweep \
                        model.validation_metric=rmse \
                        data/synthetic/fixed_effect="${SCREEN_EFFECT}" \
                        data.experiment.split_data_seed=$seed \
                        task="$task" \
                        device=cuda || {
                            echo "------------------------------------------------"
                            echo "There was an error! Check the logs above."
                            echo "------------------------------------------------"
                            read -p "Press [Enter] to kill the script..."
                            exit 1
                        }
                done
            done
            echo "Experiment complete!"
            read
            '
        cuda_device=$(( (cuda_device + 1) % 2 ))
    done
done

for model in "lme" "gplinear"; do
    export SCREEN_MODEL=$model
    screen -dmS "gp_gaussian_very_many_groups_${model}_rmse" bash -c '
        for effect in "steps_1D" "steps_2D"; do
            for seed in 0 1 2 3 4; do
                for task in "in_context" "few_shot"; do
                    echo "Running experiment for model ${SCREEN_MODEL}, effect $effect, seed $seed, task $task..."
                    nice -7 python -m scripts.experiments.run_experiment \
                        data=synthetic/gp_gaussian_very_many_groups \
                        model=${SCREEN_MODEL} \
                        model.validation_metric=rmse \
                        data/synthetic/fixed_effect=$effect \
                        data.experiment.split_data_seed=$seed \
                        task="$task" \
                        device=cpu || {
                            echo "------------------------------------------------"
                            echo "There was an error! Check the logs above."
                            echo "------------------------------------------------"
                            read -p "Press [Enter] to kill the script..."
                            exit 1
                        }
                done
            done
        done
        echo "Experiment complete!"
        read
            '
done

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1



for model in "gbm_no_group" "gbm_group_cat"; do
    for effect in "steps_1D" "steps_2D"; do
        export SCREEN_MODEL=$model
        export SCREEN_EFFECT=$effect
        export SCREEN_CUDA_DEVICE=$cuda_device
        screen -dmS "gp_gaussian_very_many_groups_${model}_${effect}_rmse" bash -c '
            for seed in 0 1 2 3 4; do
                for task in "in_context" "few_shot"; do
                    echo "Running experiment for model ${SCREEN_MODEL}, effect ${SCREEN_EFFECT}, seed $seed, task $task..."
                    nice -7 python -m scripts.experiments.run_experiment \
                        data=synthetic/gp_gaussian_very_many_groups \
                        sweep=${SCREEN_MODEL}_sweep \
                        model.validation_metric=rmse \
                        data/synthetic/fixed_effect="${SCREEN_EFFECT}" \
                        data.experiment.split_data_seed=$seed \
                        task="$task" \
                        device=cpu || {
                            echo "------------------------------------------------"
                            echo "There was an error! Check the logs above."
                            echo "------------------------------------------------"
                            read -p "Press [Enter] to kill the script..."
                            exit 1
                        }
                done
            done
            echo "Experiment complete!"
            read
                '
    done
done

