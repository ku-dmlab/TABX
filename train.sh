#!/bin/bash

# KCLOUD Transformer Grid Search Script
# Usage: ./train.sh [experiment_type] [gpu_ids...]
# Example: ./train.sh quick_test 0 1 2 3

# Show help if requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "KCLOUD Transformer Grid Search Script"
    echo ""
    echo "Usage: $0 [experiment_type] [gpu_ids...]"
    echo ""
    echo "Available experiment types:"
    echo "  quick_test       - Random search over hyperparameters"
    echo ""
    echo "Examples:"
    echo "  $0 quick_test 0 1 2 3      # Run quick_test on GPUs 0,1,2,3"
    exit 0
fi

# Define common parameters

# Define hyperparameter pools for different experiment types
declare -A hyperparameter_pools

# Quick test pool - random search space
hyperparameter_pools["baseline"]="
tabs.scenario_name=(2F1K2A1H_hard 1K2S_hard 1M2C1P_hard 7F2D1H_hard 2F1K2A1H_normal 1K2S_normal 1M2C1P_normal 7F2D1H_normal)
"

# Configuration
n_processes_per_gpu=1
experiment_type="baseline"  # Default experiment type
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
    
    # Check if line contains parameter definition
    if [[ "$line" == *"=("* && "$line" == *")" ]]; then
        param_name="${line%%=*}"
        param_values="${line#*=}"
        param_values="${param_values#(}"
        param_values="${param_values%)}"
        
        param_names+=("$param_name")
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
        local cmd="uv run /workspaces/TABS/TABS/src/baseline/train_tabs_ppo_mappo.py --gpu_id=GPU_ID"
        
        for i in "${!param_names[@]}"; do
            local param_name="${param_names[$i]}"
            local param_value="${current_params[$i]}"
            cmd+=" --${param_name}=${param_value}"
        done
        
        commands+=("$cmd")
        return
    fi
    
    local current_param="${param_names[$depth]}"
    IFS=' ' read -ra current_values <<< "${param_arrays[$current_param]}"
    
    for value in "${current_values[@]}"; do
        generate_combinations "${current_params[@]}" "$value"
    done
}

# Generate all combinations
commands=()
generate_combinations

echo "Total number of experiments: ${#commands[@]}"

# Calculate estimated time (assuming 30 min per experiment)
total_time_minutes=$((${#commands[@]} * 30 / ${#devices[@]} / n_processes_per_gpu))
hours=$((total_time_minutes / 60))
minutes=$((total_time_minutes % 60))
echo "Estimated total time: ${hours}h ${minutes}m (assuming 30min per experiment)"

# Ask for confirmation
echo ""
echo "Warning: This will run ${#commands[@]} experiments!"
read -p "Do you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
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
      while [ "${#pids[@]}" -lt $n_processes_per_gpu ] && [ $i -lt $num_cmds ]; do
        cmd=${cmds_array[$i]}
        echo "GPU $device starting: $cmd"
        CUDA_VISIBLE_DEVICES=$device $cmd &
        pids+=($!)
        ((i++))
      done
      
      if [ "${#pids[@]}" -gt 0 ]; then
        sleep 10
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

# Summary
echo ""
echo "Experiment Summary:"
echo "==================="
echo "Experiment type: $experiment_type"
echo "Total experiments: ${#commands[@]}"
echo "GPUs used: ${devices[@]}"
echo "Results are available in wandb"