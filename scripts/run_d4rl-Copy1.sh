#!/bin/bash

# Check for the correct number of arguments
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <agent_name> [exp_name]"
    echo "Example: $0 my_new_agent my_experiment"
    exit 1
fi

AGENT_NAME=$1
EXP_NAME=""
if [ "$#" -eq 2 ]; then
    EXP_NAME=$2
fi

env_names=("pen-human-v1" "pen-cloned-v1" "pen-expert-v1 " "door-expert-v1" "hammer-expert-v1" "relocate-expert-v1" )
alphas=(0.01 0.03 0.1 0.3 1.0)

# Loop through all environments and alpha values
for env_name in "${env_names[@]}"; do
    seed=$RANDOM
    for alpha in "${alphas[@]}"; do
        echo "Running with Agent: $AGENT_NAME, Env: $env_name, Alpha: $alpha, ExpName: $EXP_NAME"
        python main.py \
            --agent "agents/$AGENT_NAME.py" \
            --env_name "$env_name" \
            --alpha "$alpha" \
            --exp_name "$EXP_NAME" \
            --seed "$seed" \
            --offline_steps 500000
    done
done