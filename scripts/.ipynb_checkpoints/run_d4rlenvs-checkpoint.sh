# #!/bin/bash

# # This script runs a Python script with different 'alpha' values
# # while allowing the agent, environment, and an optional experiment name
# # to be passed as arguments.

# # Check for the correct number of arguments
# if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
#     echo "Usage: $0 <agent_name>  [exp_name]"
#     echo "Example: $0 my_new_agent my_experiment"
#     exit 1
# fi

# # Assign command-line arguments to variables for clarity
# AGENT_NAME=$1

# # Check if the third argument exists and assign it
# EXP_NAME=""
# if [ "$#" -eq 2 ]; then
#     EXP_NAME=$2
# fi

# # Define the list of alpha parameters
# env_names=("pen-expert-v1" "door-expert-v1" "hammer-expert-v1" "relocate-expert-v1")

# # Loop through all alpha values
# for env_name in "${env_names[@]}"; do
#     echo "Running with Agent: $AGENT_NAME, Env: $env_name, ExpName: $EXP_NAME"
#     python main.py \
#         --agent_name "$AGENT_NAME" \
#         --env_name "$env_name" \
#         --exp_name "$EXP_NAME" \
#         --seed "$RANDOM"
# done


#!/bin/bash

# This script runs a Python script with different 'alpha' values
# while allowing the environment, and an optional experiment name
# to be passed as arguments.

# Check for the correct number of arguments
if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [exp_name]"
    echo "Example: $0 my_experiment"
    exit 1
fi

# Check if the experiment name argument exists and assign it
EXP_NAME=""
if [ "$#" -eq 1 ]; then
    EXP_NAME=$1
fi

# Define the agent names and environment names
AGENT_NAMES=("iql" "ifql" "fql")
ENV_NAMES=("pen-expert-v1" "door-expert-v1" "hammer-expert-v1" "relocate-expert-v1")

# Loop through all agents and environments
for agent_name in "${AGENT_NAMES[@]}"; do
    for env_name in "${ENV_NAMES[@]}"; do
        echo "Running with Agent: $agent_name, Env: $env_name, ExpName: $EXP_NAME"
        python main.py \
            --agent_name "$agent_name" \
            --env_name "$env_name" \
            --exp_name "$EXP_NAME" \
            --seed "$RANDOM"
    done
done