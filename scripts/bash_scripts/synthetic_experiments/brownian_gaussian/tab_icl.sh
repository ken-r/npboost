for model in "tabicl_taskwise" "tabicl_pooled"; do
    export SCREEN_MODEL=$model
    screen -dmS "brownian_gaussian_${model}_rmse" bash -c '
        for effect in "steps_1D" "steps_2D"; do
            for seed in 0 1 2 3 4; do
                for task in "in_context" "few_shot"; do
                    echo "Running experiment for model ${SCREEN_MODEL}, effect $effect, seed $seed, task $task..."
                    CUDA_VISIBLE_DEVICES=0 nice -7 python -m scripts.experiments.run_experiment \
                        data=synthetic/brownian_gaussian \
                        model=${SCREEN_MODEL} \
                        model.validation_metric=rmse \
                        data/synthetic/fixed_effect=$effect \
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
        done
        echo "Experiment complete!"
        read
            '
done