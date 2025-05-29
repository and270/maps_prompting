#!/bin/bash

echo "[INFO] SWE-bench Placeholder Harness: Starting..."

TEST_FILE=""
OUTPUT_DIR="." # Default to current directory

# Parse command line arguments (simple version)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test_file) TEST_FILE="$2"; shift ;;
        # Add other relevant args if needed for placeholder logic
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$TEST_FILE" ]; then
    echo "[ERROR] Placeholder Harness: --test_file argument is missing."
    exit 1
fi

echo "[INFO] Placeholder Harness: Received test file: $TEST_FILE"

if [ ! -f "$TEST_FILE" ]; then
    echo "[ERROR] Placeholder Harness: Test file $TEST_FILE not found."
    exit 1
fi

# Simulate reading the test file and producing a result
# For this placeholder, we'll just assume success if the file exists
# and extract the instance_id from the first line of the jsonl.
# WARNING: This is a very basic parser, assumes first entry is the relevant one.
# Using grep and sed for simplicity as jq might not be available.
INSTANCE_ID_LINE=$(grep -o '"instance_id": "[^"]*"' "$TEST_FILE" | head -n 1)
INSTANCE_ID=$(echo "$INSTANCE_ID_LINE" | sed -e 's/"instance_id": "\(.*\)"/\1/')


if [ -z "$INSTANCE_ID" ]; then
    echo "[WARN] Placeholder Harness: Could not extract instance_id from $TEST_FILE using grep/sed. Using default."
    INSTANCE_ID="unknown_instance_from_placeholder"
fi

echo "[INFO] Placeholder Harness: Simulating successful patch for instance $INSTANCE_ID."

# Create a dummy results file that evaluate_response would parse
# The actual SWE-bench harness might create a file named instance_results.jsonl or results.json
# Let's assume evaluate_response will look for 'instance_results.jsonl' in the CWD.
# The content should reflect what the parsing logic in evaluate_response (in future step) will expect.
# For example, a list of dictionaries, one for each instance.
cat <<EOF > "${OUTPUT_DIR}/instance_results.jsonl"
[
  {
    "instance_id": "${INSTANCE_ID}",
    "status": "PASS", 
    "generated_patch": "Placeholder patch content",
    "model_name_or_path": "placeholder_model"
  }
]
EOF

echo "[INFO] Placeholder Harness: Dummy results written to ${OUTPUT_DIR}/instance_results.jsonl"
echo "[INFO] SWE-bench Placeholder Harness: Finished."

exit 0
