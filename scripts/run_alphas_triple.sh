#!/bin/bash

# This script runs a Python script with different 'alpha' values
# while allowing the environment and any number of alpha values
# to be passed as arguments.

# Check for at least two arguments: one for the environment and at least one for alpha_actor
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <env_name> <alpha_actor_value_1> [alpha_actor_value_2]..."
  echo "Example: $0 my_new_environment-v1 0.1 0.2 0.5"
  exit 1
fi

# Assign command-line arguments to variables for clarity
ENV_NAME=$1

# Shift the arguments so that $1 is now the first alpha_actor value
shift

# Loop through all provided alpha_actor values
for ALPHA_ACTOR in "$@"; do
  # Generate a random seed for each run
  seed=$RANDOM

  echo "================================================================="
  echo "Running with Env: $ENV_NAME, Alpha: $ALPHA_ACTOR, Seed: $seed"
  echo "================================================================="

  # Running trigflow agent with q loss
  echo "Running trigflow agent with q loss..."
  python main.py \
    --agent "agents/trigflow.py" \
    --env_name "$ENV_NAME" \
    --alpha_actor "$ALPHA_ACTOR" \
    --exp_name "alpha_actor_${ALPHA_ACTOR}" \
    --seed "$seed" --use_q_loss

  # Running trigflow agent
  echo "Running trigflow agent..."
  python main.py \
    --agent "agents/trigflow.py" \
    --env_name "$ENV_NAME" \
    --alpha_actor "$ALPHA_ACTOR" \
    --exp_name "alpha_actor_${ALPHA_ACTOR}" \
    --seed "$seed"

  # Running dtrigflow agent
  echo "Running dtrigflow agent..."
  python main.py \
    --agent "agents/dtrigflow.py" \
    --env_name "$ENV_NAME" \
    --alpha_actor "$ALPHA_ACTOR" \
    --exp_name "alpha_actor_${ALPHA_ACTOR}" \
    --seed "$seed"
  
  echo "Completed environment: $ENV_NAME with alpha_actor: $ALPHA_ACTOR"
  echo ""
done

echo "All specified alpha_actor values have been processed for environment: $ENV_NAME!"