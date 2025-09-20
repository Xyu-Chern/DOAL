#!/bin/bash

# This script runs a Python script for multiple environments with a specified alpha_actor value

# Check for the correct number of arguments
if [ "$#" -lt 1 ] ; then
    echo "Usage: $0 <alpha_actor>"
    echo "Example: $0 10"
    exit 1
fi


# Define the list of environments to loop through
env_names=$1

# Assign command-line argument to variable
ALPHA_ACTOR=$2
# Loop through all environments
    # Generate random seed for each environment
seed=$RANDOM

echo "================================================================="
echo "Running with Env: $env_name, Alpha: $ALPHA_ACTOR, Seed: $seed"
echo "================================================================="

echo "Running trigflow agent with q loss..."
python main.py \
    --agent "agents/trigflow.py" \
    --env_name "$env_name" \
    --alpha_actor "$ALPHA_ACTOR" \
    --exp_name "alpha_actor_${ALPHA_ACTOR}" \
    --seed "$seed" --agent.use_q_loss

echo "Completed environment: $env_name"
echo ""

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


echo "All environments have been processed with alpha_actor: $ALPHA_ACTOR!"