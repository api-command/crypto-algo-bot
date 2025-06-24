#!/bin/bash
# Secure Key Rotation Wrapper
# Usage: ./rotate_keys.sh

# Load environment from .env file
set -a
source ../../.env
set +a

# Validate required variables
if [ -z "$MASTER_PASSWORD" ]; then
    echo "ERROR: MASTER_PASSWORD not set in .env file"
    exit 1
fi

# Execute rotation
export MASTER_PASSWORD
python3 rotate_keys.py

# Security cleanup
unset MASTER_PASSWORD
echo "Key rotation completed successfully"