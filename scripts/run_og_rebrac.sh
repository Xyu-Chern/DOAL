#!/bin/bash

# Check for the correct number of arguments
if [ "$#" -lt 1 ] ; then
    echo "Usage: $0 <agent_name> [exp_name]"
    echo "Example: $0 my_new_agent my_experiment"
    exit 1
fi

AGENT_NAME=$1
seeds=$2

env_names=("antmaze-large-navigate-singletask-v0" "antmaze-giant-navigate-singletask-v0"    "puzzle-3x3-play-singletask-v0" "puzzle-4x4-play-singletask-v0"  )
# Loop through all environments and alpha values
for seed in "${seeds[@]}"; do
    for env_name in "${env_names[@]}"; do
        echo "Running with Agent: $AGENT_NAME, Env: $env_name, Seed: $seed, Additional Args: $@"
        python main.py \
            --agent "agents/$AGENT_NAME.py" \
            --env_name "$env_name" \
            --run_group submit_OG  \
            --noretest \
            --seed "$seed" "$@"
    done
done