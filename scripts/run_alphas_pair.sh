#!/bin/bash

# This script runs a Python script with different 'alpha' values
# while allowing the agent, environment, and an optional experiment name
# to be passed as arguments.

# Check for the correct number of arguments
if [ "$#" -lt 2 ] ; then
    echo "Usage: $0 <env_name> "
    echo "Example: $0 my_new_agent my_new_environment-v1 my_experiment"
    exit 1
fi

# Assign command-line arguments to variables for clarity
ENV_NAME=$1


# pen-expert-v1		
# door-expert-v1		
# hammer-expert-v1		
# relocate-expert-v1	

# Define the list of alpha parameters

if [ "$#" -gt 2 ] ; then
# Loop through all alpha values
    seed=$RANDOM
    alpha_actor=$2
    echo "Running with Agent: trigflow, Env: $ENV_NAME, Alpha: $alpha, ExpName: "
    python main.py \
        --agent "agents/trigflow.py" \
        --env_name "$ENV_NAME" \
        --alpha_actor "$alpha_actor" \
        --exp_name alpha_actor_pair\
        --seed "$seed"
    echo "Running with Agent: trigflow, Env: $ENV_NAME, Alpha: $alpha, ExpName: "
    python main.py \
        --agent "agents/dtrigflow.py" \
        --env_name "$ENV_NAME" \
        --alpha_actor "$alpha_actor" \
        --exp_name alpha_actor_pair \
        --seed "$seed"
fi
if [ "$#" -gt 3 ] ; then
# Loop through all alpha values
    seed=$RANDOM
    alpha_actor=$3
    echo "Running with Agent: trigflow, Env: $ENV_NAME, Alpha: $alpha, ExpName: "
    python main.py \
        --agent "agents/trigflow.py" \
        --env_name "$ENV_NAME" \
        --alpha_actor "$alpha_actor" \
        --exp_name alpha_actor_pair\
        --seed "$seed"
    echo "Running with Agent: trigflow, Env: $ENV_NAME, Alpha: $alpha, ExpName: "
    python main.py \
        --agent "agents/dtrigflow.py" \
        --env_name "$ENV_NAME" \
        --alpha_actor "$alpha_actor" \
        --exp_name alpha_actor_pair \
        --seed "$seed"
fi
if [ "$#" -gt 4 ] ; then
# Loop through all alpha values
    seed=$RANDOM
    alpha_actor=$4
    echo "Running with Agent: trigflow, Env: $ENV_NAME, Alpha: $alpha, ExpName: "
    python main.py \
        --agent "agents/trigflow.py" \
        --env_name "$ENV_NAME" \
        --alpha_actor "$alpha_actor" \
        --exp_name alpha_actor_pair\
        --seed "$seed"
    echo "Running with Agent: trigflow, Env: $ENV_NAME, Alpha: $alpha, ExpName: "
    python main.py \
        --agent "agents/dtrigflow.py" \
        --env_name "$ENV_NAME" \
        --alpha_actor "$alpha_actor" \
        --exp_name alpha_actor_pair \
        --seed "$seed"
fi

if [ "$#" -gt 5 ] ; then
# Loop through all alpha values
    seed=$RANDOM
    alpha_actor=$5
    echo "Running with Agent: trigflow, Env: $ENV_NAME, Alpha: $alpha, ExpName: "
    python main.py \
        --agent "agents/trigflow.py" \
        --env_name "$ENV_NAME" \
        --alpha_actor "$alpha_actor" \
        --exp_name alpha_actor_pair\
        --seed "$seed"
    echo "Running with Agent: trigflow, Env: $ENV_NAME, Alpha: $alpha, ExpName: "
    python main.py \
        --agent "agents/dtrigflow.py" \
        --env_name "$ENV_NAME" \
        --alpha_actor "$alpha_actor" \
        --exp_name alpha_actor_pair \
        --seed "$seed"
fi