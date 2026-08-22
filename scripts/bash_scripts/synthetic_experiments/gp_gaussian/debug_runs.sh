

screen -dmS "anpboost_gp_gaussian_debug" bash -c '
    for task in "in_context"; do
        echo "Running experiment for model $model, effect $effect, seed 0, task $task..."
        CUDA_VISIBLE_DEVICES=0 nice -7 python -m scripts.experiments.run_experiment \
            data=synthetic/gp_gaussian \
            wandb.project="debug" \
            model=anpboost \
            model.max_boosting_rounds=10 \
            model.validation_metric=rmse \
            data/synthetic/fixed_effect=steps_1D \
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

screen -dmS "anp_gp_gaussian_debug" bash -c '
    for task in "in_context"; do
        echo "Running experiment for model $model, effect $effect, seed 0, task $task..."
        CUDA_VISIBLE_DEVICES=0 nice -7 python -m scripts.experiments.run_experiment \
            data=synthetic/gp_gaussian \
            wandb.project="debug" \
            model=anp \
            model.max_epochs=20 \
            model.validation_metric=rmse \
            data/synthetic/fixed_effect=steps_1D \
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