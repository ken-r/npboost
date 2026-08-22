echo 'Starting cars data experiment...'
echo ''


cuda_device=0
for model in "npboost" "anpboost" "np" "anp"; do
    for task in "in_context" "few_shot"; do
        SESSION_NAME="${model}_cars_${task}_rmse_seed0"
        export SCREEN_MODEL="$model"
        export SCREEN_TASK="$task"
        export SCREEN_CUDA_DEVICE="$cuda_device"

        screen -dmS "$SESSION_NAME" bash -c '
            CUDA_VISIBLE_DEVICES=$SCREEN_CUDA_DEVICE nice -7 python -m scripts.experiments.run_experiment \
                data=real/cars \
                sweep=${SCREEN_MODEL}_sweep \
                model.validation_metric=rmse \
                data.experiment.split_data_seed=0 \
                task=$SCREEN_TASK \
                device=cuda || {
                    echo "------------------------------------------------"
                    echo "There was an error! Check the logs above."
                    echo "------------------------------------------------"
                    read -p "Press [Enter] to kill the script..."
                    exit 1
                }
            echo "Experiment complete!"
            read
        '
    done
done
