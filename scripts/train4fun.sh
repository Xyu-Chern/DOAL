#!/bin/bash

# This script runs a Python script with different 'alpha' values
# while allowing the agent, a seed, and an optional experiment name
# to be passed as arguments.

# Check for the correct number of arguments
if [ "$#" -lt 1 ] ; then
    echo "Usage: $0 <agent_name> [seed] [additional_args...]"
    echo "Example: $0 my_new_agent 12345 my_experiment"
    exit 1
fi

# Assign command-line arguments to variables for clarity
env_name=$1

# Check if a seed is provided as the second argument
if [ -n "$2" ] ; then
    seed=$2
    # Shift arguments to handle the rest of the optional parameters
    shift 2
else
    seed=$RANDOM
    shift 1
fi


# Loop through all environment names
python main.py --agent agents/trigflow.py --env_name humanoidmaze-medium-navigate-singletask-task1-v0  --run_group fun  --seed 40 "$@"
python main.py --agent agents/trigflow.py --env_name "$env_name"  --run_group fun  --seed "$seed"  "$@" --agent.use_q_loss
python main.py --agent agents/dtrigflow.py --env_name "$env_name"  --run_group fun  --seed "$seed"  "$@"
python main.py --agent agents/diql.py --env_name "$env_name"  --run_group fun  --seed "$seed"  "$@"