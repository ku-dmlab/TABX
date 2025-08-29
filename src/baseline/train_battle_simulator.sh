#!/bin/bash

# Battle Simulator MAPPO Grid Search Script - General Version
# Usage: ./train_battle_simulator.sh [experiment_type] [gpu_ids...]
# Example: ./train_battle_simulator.sh quick_test 0 1 2
# Example: ./train_battle_simulator.sh lr_sweep 0 1

# Show help if requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Battle Simulator MAPPO Grid Search Script - General Version"
    echo ""
    echo "Usage: $0 [experiment_type] [gpu_ids...]"
    echo ""
    echo "Available experiment types:"
    echo "  quick_test       - Small search space for testing"
    echo "  lr_sweep         - Learning rate sweep"
    echo "  ppo_sweep        - PPO hyperparameters sweep"
    echo "  env_sweep        - Environment scaling sweep"
    echo "  discount_sweep   - Discount factor and GAE lambda sweep"
    echo "  comprehensive    - Full sweep"
    echo "  custom           - Custom experiment (modify custom pool in script)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run quick_test on GPUs 0,1,2,3"
    echo "  $0 lr_sweep                  # Run lr_sweep on GPUs 0,1,2,3"
    echo "  $0 quick_test 0 1            # Run quick_test on GPUs 0,1"
    echo "  $0 comprehensive 3           # Run comprehensive on GPU 3 only"
    echo ""
    echo "Notes:"
    echo "  - Each GPU runs 1 process by default (modify n_processes_per_gpu to change)"
    echo "  - Experiments are distributed evenly across specified GPUs"
    echo "  - Results are logged to wandb and saved to ./save/ directory"
    echo "  - To add new hyperparameters, just add them to any experiment pool"
    echo "  - The script automatically handles all parameters dynamically"
    exit 0
fi

# Define common parameters
seeds=(0)
scenarios=("8archer_vs_1mammoth_1healer")

# Define hyperparameter pools for different experiment types
# Each pool defines the parameters to search over
# You can add ANY hyperparameter here and it will be automatically processed
declare -A hyperparameter_pools

# Quick test pool - small search space for testing
hyperparameter_pools["quick_test"]="
scenario=(8archer_vs_1mammoth_1healer)
clip_ratio=(0.05 0.1 0.2)
entropy_coef=(0.01 0.1 0.001)
"


# Configuration
n_processes_per_gpu=1
experiment_type="quick_test"  # Default experiment type
devices=(0 1 2 3)  # Default GPU IDs

# Parse command line arguments
if [ $# -gt 0 ]; then
    experiment_type=$1
    shift  # Remove first argument
    
    # Remaining arguments are GPU IDs
    if [ $# -gt 0 ]; then
        devices=("$@")
    fi
fi

# Check if experiment type exists
if [[ ! ${hyperparameter_pools[$experiment_type]+_} ]]; then
    echo "Error: Unknown experiment type '$experiment_type'"
    echo "Available types: ${!hyperparameter_pools[@]}"
    echo ""
    echo "Usage: $0 [experiment_type] [gpu_ids...]"
    echo "Example: $0 quick_test 0 1 2"
    echo "Example: $0 lr_sweep 0 1"
    exit 1
fi

echo "Running experiment type: $experiment_type"
echo "Using GPUs: ${devices[@]}"
echo "Processes per GPU: $n_processes_per_gpu"

# Load the selected hyperparameter pool
pool_config="${hyperparameter_pools[$experiment_type]}"

# Parse the hyperparameter pool configuration dynamically
declare -A param_arrays
param_names=()

while IFS= read -r line; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    
    # Remove leading/trailing whitespace
    line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    
    # Check if line contains parameter definition (param=(...))
    if [[ "$line" == *"=("* && "$line" == *")" ]]; then
        # Extract parameter name (everything before =)
        param_name="${line%%=*}"
        
        # Extract values (everything between ( and ))
        param_values="${line#*=}"
        param_values="${param_values#(}"
        param_values="${param_values%)}"
        
        # Store parameter name
        param_names+=("$param_name")
        
        # Convert values string to array
        IFS=' ' read -ra values_array <<< "$param_values"
        param_arrays["$param_name"]="${values_array[*]}"
        
        echo "Found parameter: $param_name = (${param_arrays["$param_name"]})"
    fi
done <<< "$pool_config"

echo ""
echo "Generating all parameter combinations..."

# Function to generate all combinations recursively
generate_combinations() {
    local current_params=("$@")
    local depth=${#current_params[@]}
    
    if [ $depth -eq ${#param_names[@]} ]; then
        # We've assigned values to all parameters, generate command
        local cmd="uv run battle_simulator_mappo.py --gpu_id=GPU_ID"
        
        for i in "${!param_names[@]}"; do
            local param_name="${param_names[$i]}"
            local param_value="${current_params[$i]}"
            cmd+=" --${param_name}=${param_value}"
        done
        
        commands+=("$cmd")
        return
    fi
    
    # Get the current parameter name and its values
    local current_param="${param_names[$depth]}"
    IFS=' ' read -ra current_values <<< "${param_arrays[$current_param]}"
    
    # Recursively try each value for this parameter
    for value in "${current_values[@]}"; do
        generate_combinations "${current_params[@]}" "$value"
    done
}

# Generate all combinations
commands=()
generate_combinations

echo "Total number of experiments: ${#commands[@]}"

# Calculate estimated time (assuming each experiment takes ~30 minutes)
total_time_minutes=$((${#commands[@]} * 30 / ${#devices[@]} / n_processes_per_gpu))
hours=$((total_time_minutes / 60))
minutes=$((total_time_minutes % 60))
echo "Estimated total time: ${hours}h ${minutes}m (assuming 30min per experiment)"

# Ask for confirmation if many experiments
if [ ${#commands[@]} -gt 50 ]; then
    echo ""
    echo "Warning: This will run ${#commands[@]} experiments!"
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Distribute commands across devices
declare -A device_commands
for device in "${devices[@]}"; do
  device_commands[$device]=""
done

device_index=0
num_devices=${#devices[@]}

for cmd in "${commands[@]}"; do
  device=${devices[$device_index]}
  modified_cmd=${cmd/GPU_ID/$device}
  device_commands[$device]+="$modified_cmd"$'\n'  
  device_index=$(( (device_index + 1) % num_devices ))
done

echo "Starting experiments..."

# Execute commands on each device in parallel
for device in "${devices[@]}"; do
  (
    cmds="${device_commands[$device]}"
    IFS=$'\n' read -d '' -r -a cmds_array <<< "$cmds"
    num_cmds=${#cmds_array[@]}
    echo "GPU $device will run $num_cmds experiments"
    
    i=0
    pids=()
    while [ "${#pids[@]}" -gt 0 ] || [ $i -lt $num_cmds ]; do
      # Start new processes if under the limit
      while [ "${#pids[@]}" -lt $n_processes_per_gpu ] && [ $i -lt $num_cmds ]; do
        cmd=${cmds_array[$i]}
        echo "GPU $device starting: $cmd"
        CUDA_VISIBLE_DEVICES=$device $cmd &
        pids+=($!)
        ((i++))
      done
      
      # Wait for any process to finish
      if [ "${#pids[@]}" -gt 0 ]; then
        sleep 10  # Check every 10 seconds
        new_pids=()
        for pid in "${pids[@]}"; do
          if kill -0 $pid 2>/dev/null; then
            new_pids+=($pid)
          else
            wait $pid 2>/dev/null
            echo "GPU $device: Process $pid completed"
          fi
        done
        pids=("${new_pids[@]}")
      fi
    done
    echo "GPU $device: All experiments completed"
  ) &
done

echo "All GPU processes started. Waiting for completion..."
wait
echo "All experiments completed!"

# Optional: Summary and analysis
echo ""
echo "Experiment Summary:"
echo "==================="
echo "Experiment type: $experiment_type"
echo "Total experiments: ${#commands[@]}"
echo "GPUs used: ${devices[@]}"
echo "Results should be available in wandb and save directories"