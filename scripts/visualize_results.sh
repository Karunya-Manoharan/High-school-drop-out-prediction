#!/bin/bash

EXP_DIR=$1

valid_dirs=( $(find $EXP_DIR -maxdepth 2 -name 'summary.hd5' -printf '%h\n' | sort -u) )

all_paths=""
for valid_dir in ${valid_dirs[@]}; do
    echo $valid_dir
    all_paths+="--paths $valid_dir "
done
python ../src/result_parser.py $all_paths --save_dir $EXP_DIR/visualizations --secret_path "$HOME/.secrets.yaml"
