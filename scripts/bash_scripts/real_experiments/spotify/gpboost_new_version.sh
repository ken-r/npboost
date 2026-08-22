export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export HYDRA_FULL_ERROR=1

for model in "lme" "gplinear"; do
    export SCREEN_MODEL=$model
    screen -dmS "spotify_${model}_rmse" bash -c '
        for seed in 0 1 2 3 4; do
            for task in "in_context" "few_shot"; do
                echo "Running experiment for model ${SCREEN_MODEL}, seed $seed, task $task..."
                nice -7 python -m scripts.experiments.run_experiment \
                    data=real/spotify \
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