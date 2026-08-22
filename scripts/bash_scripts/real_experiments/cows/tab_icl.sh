for model in "tabicl_taskwise" "tabicl_pooled"; do
    export SCREEN_MODEL=$model
    screen -dmS "cows_${model}_rmse" bash -c '
        for seed in 0 1 2 3 4; do
            for task in "in_context" "few_shot"; do
                echo "Running experiment for model ${SCREEN_MODEL}, seed $seed, task $task..."
                CUDA_VISIBLE_DEVICES=1 nice -7 python -m scripts.experiments.run_experiment \
                    data=real/cows \
                    model=${SCREEN_MODEL} \
                    model.validation_metric=rmse \
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
done