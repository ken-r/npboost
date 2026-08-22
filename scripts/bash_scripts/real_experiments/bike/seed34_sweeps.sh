
cuda_device=1
for model in "npboost" "anpboost" "np" "anp"; do
    export SCREEN_MODEL=$model
    export SCREEN_CUDA_DEVICE=$cuda_device
    screen -dmS "bike_${model}_rmse_3" bash -c '
        for task in "in_context" "few_shot"; do
            echo "Running experiment for model $model, seed 3, task $task..."
            CUDA_VISIBLE_DEVICES=1 nice -7 python -m scripts.experiments.run_experiment \
                data=real/bike \
                sweep=${SCREEN_MODEL}_sweep \
                model.validation_metric=rmse \
                data.experiment.split_data_seed=3 \
                task="$task" \
                device=cuda || {
                    echo "------------------------------------------------"
                    echo "There was an error! Check the logs above."
                    echo "------------------------------------------------"
                    read -p "Press [Enter] to kill the script..."
                    exit 1
                }
        done
        echo "Experiment complete!"
        read
        '
    cuda_device=$(( (cuda_device + 1) % 2 ))
done

cuda_device=1
for model in "npboost" "anpboost" "np" "anp"; do
    export SCREEN_MODEL=$model
    export SCREEN_CUDA_DEVICE=$cuda_device
    screen -dmS "bike_${model}_rmse_4" bash -c '
        for task in "in_context" "few_shot"; do
            echo "Running experiment for model $model, seed 4, task $task..."
            CUDA_VISIBLE_DEVICES=1 nice -7 python -m scripts.experiments.run_experiment \
                data=real/bike \
                sweep=${SCREEN_MODEL}_sweep \
                model.validation_metric=rmse \
                data.experiment.split_data_seed=4 \
                task="$task" \
                device=cuda || {
                    echo "------------------------------------------------"
                    echo "There was an error! Check the logs above."
                    echo "------------------------------------------------"
                    read -p "Press [Enter] to kill the script..."
                    exit 1
                }
        done
        echo "Experiment complete!"
        read
        '
    cuda_device=$(( (cuda_device + 1) % 2 ))
done
    
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for model in "gbm_no_group" "gbm_group_cat"; do
    export SCREEN_MODEL=$model
    screen -dmS "bike_${model}_rmse" bash -c '
        for seed in 3 4; do
            for task in "in_context" "few_shot"; do
                echo "Running experiment for model ${SCREEN_MODEL}, seed $seed, task $task..."
                nice -7 python -m scripts.experiments.run_experiment \
                    data=real/bike \
                    sweep=${SCREEN_MODEL}_sweep \
                    model.validation_metric=rmse \
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


for model in "lme" "gplinear"; do
    export SCREEN_MODEL=$model
    screen -dmS "bike_${model}_rmse" bash -c '
        for seed in 3 4; do
            for task in "in_context" "few_shot"; do
                echo "Running experiment for model ${SCREEN_MODEL}, seed $seed, task $task..."
                nice -7 python -m scripts.experiments.run_experiment \
                    data=real/bike \
                    model=${SCREEN_MODEL} \
                    model.validation_metric=rmse \
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
