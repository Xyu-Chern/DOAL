#!/bin/bash

# This script runs a Python script for multiple environments with a specified alpha_actor value

# Check for the correct number of arguments
if [ "$#" -lt 1 ] ; then
    echo "Usage: $0 <alpha_actor>"
    echo "Example: $0 10"
    exit 1
fi

# Assign command-line argument to variable
ALPHA_ACTOR=$1

# Define the list of environments to loop through
env_names=("antmaze-large-navigate-singletask-v0" "humanoidmaze-medium-navigate-singletask-v0" "antsoccer-arena-navigate-singletask-v0" "cube-single-play-singletask-v0" "scene-play-singletask-v0")

# Loop through all environments
for env_name in "${env_names[@]}"; do
    # Generate random seed for each environment
    seed=$RANDOM
    
    echo "================================================================="
    echo "Running with Env: $env_name, Alpha: $ALPHA_ACTOR, Seed: $seed"
    echo "================================================================="
    
    echo "Running trigflow agent..."
    python main.py \
        --agent "agents/trigflow.py" \
        --env_name "$env_name" \
        --alpha_actor "$ALPHA_ACTOR" \
        --exp_name "alpha_actor_${ALPHA_ACTOR}" \
        --seed "$seed"
    
    echo "Running dtrigflow agent..."
    python main.py \
        --agent "agents/dtrigflow.py" \
        --env_name "$env_name" \
        --alpha_actor "$ALPHA_ACTOR" \
        --exp_name "alpha_actor_${ALPHA_ACTOR}" \
        --seed "$seed"
    
    echo "Completed environment: $env_name"
    echo ""
done

echo "All environments have been processed with alpha_actor: $ALPHA_ACTOR!"