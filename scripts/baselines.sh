#!/bin/bash

SECRET_PATH=$1
if [[ -z $SECRET_PATH ]]; then
    SECRET_PATH="$HOME/.secrets.yaml"
fi

PYTHONWARNINGS=ignore

for config_path in ../configs/*.yaml; do
    echo "Running config: $config_path"
    config_name=${config_path%.yaml}
    config_name=${config_name##*/}
    result_dir=../results/$config_name
    python -W ignore ../src/run.py run \
        --secret_path $SECRET_PATH \
        --config_path $config_path \
        --model_save_dir $result_dir \
        --result_save_dir $result_dir
done
