
cuda_device=0
for model in "npboost" "anpboost" "np" "anp"; do
    for effect in "steps_1D" "steps_2D"; do
        export SCREEN_MODEL=$model
        export SCREEN_EFFECT=$effect
        export SCREEN_CUDA_DEVICE=$cuda_device
        screen -dmS "gp_gaussian_${model}_${effect}_rmse" bash -c '
            for task in "in_context" "few_shot"; do
                echo "Running experiment for model $model, effect $effect, seed 0, task $task..."
                CUDA_VISIBLE_DEVICES=$SCREEN_CUDA_DEVICE nice -7 python -m scripts.experiments.run_experiment \
                    data=synthetic/gp_gaussian \
                    wandb.project="basic_runs" \
                    model=${SCREEN_MODEL} \
                    model.validation_metric=rmse \
                    data/synthetic/fixed_effect="${SCREEN_EFFECT}" \
                    data.experiment.split_data_seed=0 \
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
done
