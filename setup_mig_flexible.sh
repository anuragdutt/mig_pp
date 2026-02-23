#!/bin/bash

# --- CONFIGURATION ---
# NVIDIA A100 40GB Specific Constraints
# Total Compute Slices available = 7
MAX_SLICES=7

# Check for Root
if [ "$EUID" -ne 0 ]; then
  echo "Error: Please run as root (sudo ./setup_mig_custom.sh)"
  exit 1
fi

echo "=========================================================="
echo "   NVIDIA A100 (40GB) - Custom MIG Partitioner"
echo "=========================================================="
echo "Total Available Capacity: $MAX_SLICES Compute Slices"
echo ""
echo "Supported Profiles Reference:"
echo " - 1g.5gb   (Costs 1 Slice)"
echo " - 2g.10gb  (Costs 2 Slices)"
echo " - 3g.20gb  (Costs 3 Slices)"
echo " - 4g.20gb  (Costs 4 Slices)"
echo " - 7g.40gb  (Costs 7 Slices)"
echo "=========================================================="
echo ""

# 1. Get Instance Count
read -p "How many MIG instances do you want to create? " INSTANCE_COUNT

if ! [[ "$INSTANCE_COUNT" =~ ^[0-9]+$ ]] || [ "$INSTANCE_COUNT" -lt 1 ]; then
    echo "Error: Please enter a valid number greater than 0."
    exit 1
fi

# 2. Loop to collect profiles
STRATEGY=""
CURRENT_SLICES=0
DECLARED_PROFILES=()

for (( i=1; i<=INSTANCE_COUNT; i++ ))
do
    REMAINING=$((MAX_SLICES - CURRENT_SLICES))
    echo ""
    echo "--- Configuring Instance #$i (Remaining Slices: $REMAINING) ---"
    
    while true; do
        read -p "Enter profile name (e.g., 1g.5gb, 2g.10gb, 3g.20gb): " PROFILE_INPUT
        
        # Determine slice cost based on input
        case $PROFILE_INPUT in
            "1g.5gb") COST=1 ;;
            "2g.10gb") COST=2 ;;
            "3g.20gb") COST=3 ;;
            "4g.20gb") COST=4 ;;
            "7g.40gb") COST=7 ;;
            *) 
                echo "Invalid profile name for A100 40GB. Please try again."
                continue 
                ;;
        esac

        # Check if we have enough space
        if [ $((CURRENT_SLICES + COST)) -le $MAX_SLICES ]; then
            # Valid choice
            if [ -z "$STRATEGY" ]; then
                STRATEGY="$PROFILE_INPUT"
            else
                STRATEGY="$STRATEGY,$PROFILE_INPUT"
            fi
            
            CURRENT_SLICES=$((CURRENT_SLICES + COST))
            DECLARED_PROFILES+=("$PROFILE_INPUT")
            break
        else
            echo "Error: Not enough space! You requested $COST slices but only have $REMAINING left."
        fi
    done
done

# 3. Confirmation
echo ""
echo "=========================================================="
echo "Configuration Ready:"
echo "   Instances: $INSTANCE_COUNT"
echo "   Strategy:  $STRATEGY"
echo "   Total Slices Used: $CURRENT_SLICES / $MAX_SLICES"
echo "=========================================================="
read -p "Proceed with creating these partitions? (y/n): " CONFIRM

if [[ "$CONFIRM" != "y" ]]; then
    echo "Aborting."
    exit 0
fi

# 4. Execution
echo ""
echo "Step 1: Resetting MIG configuration..."
# Disable compute instances (CI) and GPU instances (GI)
nvidia-smi mig -dci > /dev/null 2>&1
nvidia-smi mig -dgi > /dev/null 2>&1

echo "Step 2: Creating new partitions..."
# The -C flag creates Compute Instances automatically for the GPU Instances
nvidia-smi mig -cgi $STRATEGY -C

# 5. Final Verification
if [ $? -eq 0 ]; then
    echo ""
    echo "SUCCESS! The following instances are now active:"
    nvidia-smi -L
else
    echo ""
    echo "FAILED. Please check the error message above."
    echo "Ensure MIG mode is enabled (nvidia-smi -i 0 -mig 1)."
fi