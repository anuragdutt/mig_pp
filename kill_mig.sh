#!/bin/bash

# 1. Check for Root
if [ "$EUID" -ne 0 ]; then
  echo "Error: Please run as root (sudo ./kill_mig.sh)"
  exit 1
fi

echo "--- NVIDIA MIG Teardown Script ---"
echo "Cleaning up GPU 0..."

# 2. Delete Compute Instances (CI)
# We MUST do this first. You cannot delete the partition (GI) 
# if there is a compute processor (CI) configured inside it.
echo "Step 1: Destroying Compute Instances..."
nvidia-smi mig -dci > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "   [OK] Compute Instances removed."
else
    echo "   [INFO] No Compute Instances found or already clean."
fi

# 3. Delete GPU Instances (GI)
# Now we delete the actual memory partitions.
echo "Step 2: Destroying GPU Instances..."
nvidia-smi mig -dgi > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "   [OK] GPU Instances removed."
else
    echo "   [INFO] No GPU Instances found or already clean."
fi

echo "----------------------------------"
echo "Done. Verifying status:"
echo ""

# 4. Show final status
nvidia-smi
