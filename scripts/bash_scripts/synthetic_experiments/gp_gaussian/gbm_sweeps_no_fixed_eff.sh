export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1



for model in "gbm_no_group" "gbm_group_cat"; do
    for effect in "zero_1D" "zero_2D"; do
        export SCREEN_MODEL=$model
        export SCREEN_EFFECT=$effect
        screen -dmS "gp_gaussian_${model}_${effect}_rmse" bash -c '
            for seed in 0 1 2 3 4; do
                for task in "in_context" "few_shot"; do
                    echo "Running experiment for model ${SCREEN_MODEL}, effect ${SCREEN_EFFECT}, seed $seed, task $task..."
                    nice -7 python -m scripts.experiments.run_experiment \
                        data=synthetic/gp_gaussian \
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

