#!/bin/bash
## Script Description: This script runs a pipeline to process USD models
##                      to decimate the number of faces.
## Author: Po-Hsun Chang
## Contact: pohsun@umich.edu
## Usage: ./decimate_usd_models_face_autorestart.sh "navverse_data/grscenes_commercial/models/" --source gr
# pkill -9 -f "python.*decimate_usd_models_face_blender.py"

# Check if input folder argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_folder> [--source <source>]"
    echo "Example: $0 navverse_data/grscenes_commercial/models/ --source gr"
    exit 1
fi

INPUT_FOLDER="$1"
SOURCE="gr"

# Parse optional arguments
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            SOURCE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Validate input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder does not exist: $INPUT_FOLDER"
    exit 1
fi

echo "Processing USD models in folder: $INPUT_FOLDER with source: $SOURCE"

# Step 1: Run renaming script using IsaacLab
PRE_SCRIPT="make_usd_prim_unique.py"
echo "=== Step 1: Running IsaacLab preprocessing at $(date) ==="
python "$PRE_SCRIPT" --input_folder "$INPUT_FOLDER" --source "$SOURCE"
PRE_STATUS=$?

if [ $PRE_STATUS -ne 0 ]; then
    echo ">>> Step 1: IsaacLab preprocessing failed (exit $PRE_STATUS). Aborting."
    exit 1
fi

echo -e "\n=== Step 1: IsaacLab preprocessing finished successfully ===\n"


# Step 2: Run face decimation in Blender with auto-restart on crash or hang
SCRIPT="decimate_usd_models_face_blender.py"
MAX_IDLE=300   # seconds of no output before killing
MEM_LIMIT="3G"  # memory limit for Blender process
RATIO=0.1  # default decimation ratio
NUM_WORKERS=5

echo -e "\n=== Step 2: Running Blender face decimation with $NUM_WORKERS workers ===\n"

run_worker() {
    local WORKER_ID=$1
    local TOTAL_WORKERS=$2
    local LOG="blender_run_${WORKER_ID}.log"

    echo "--- Worker $WORKER_ID: Starting Blender job at $(date) ---" > "$LOG"
    while true; do
        echo "=== [Worker $WORKER_ID] Starting Blender job at $(date) ==="
        echo "--- Restarting Blender job at $(date) ---" >> "$LOG"
        systemd-run --user --scope -p MemoryMax=$MEM_LIMIT blender --background --python "$SCRIPT" -- \
            --input_folder "$INPUT_FOLDER" --ratio $RATIO --worker_id $WORKER_ID --total_workers $TOTAL_WORKERS --source "$SOURCE" >>"$LOG" 2>&1 &
        local PID=$!

        while kill -0 $PID 2>/dev/null; do
            # last modification time of log (seconds since epoch)
            if [ ! -f "$LOG" ]; then touch "$LOG"; fi
            local LAST_UPDATE=$(stat -c %Y "$LOG")
            local NOW=$(date +%s)
            local AGE=$((NOW - LAST_UPDATE))

            if (( AGE > MAX_IDLE )); then
                echo ">>> [Worker $WORKER_ID] No output for $MAX_IDLE seconds, killing Blender (PID $PID and its children)"
                # Gracefully terminate main process and children
                kill -TERM $PID 2>/dev/null
                pkill -P $PID 2>/dev/null
                # Force kill if still alive
                sleep 2
                kill -KILL $PID 2>/dev/null
                pkill -9 -P $PID 2>/dev/null
                break
            fi
            sleep 2
        done

        wait $PID 2>/dev/null
        local STATUS=$?
        if [ $STATUS -eq 0 ]; then
            echo "=== [Worker $WORKER_ID] Blender finished successfully at $(date) ==="
            break
        else
            echo ">>> [Worker $WORKER_ID] Blender crashed or was killed (exit $STATUS). Restarting..."
        fi
    done
}

# Start workers in background
for ((i=0; i<NUM_WORKERS; i++)); do
    run_worker $i $NUM_WORKERS &
done

# Wait for all workers to finish
wait

# Step 3: Run replacing decimated mesh to renamed usd in IsaacLab
# POST_SCRIPT="replace_usd_models_isaac.py"
# echo -e "\n=== Step 3: Running post-processing in IsaacLab ===\n"
# python "$POST_SCRIPT" --input_folder "$INPUT_FOLDER"
# POST_STATUS=$?

# if [ $POST_STATUS -ne 0 ]; then
#     echo ">>> Step 3: Post-processing script failed (exit $POST_STATUS)."
#     exit 1
# fi

# echo "=== Step 3: Pipeline completed successfully at $(date) ==="
