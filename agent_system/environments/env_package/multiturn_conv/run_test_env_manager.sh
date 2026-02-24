#!/bin/bash
set -euo pipefail

# --- Activate virtualenv ---
source /home/pjy0422/verl-agent/bin/activate

# --- GPU Assignment (use free GPUs 1,2) ---
TARGET_GPU=1
JUDGE_GPU=2

# --- Port Assignment ---
TARGET_PORT=9011
JUDGE_PORT=9012

# --- Model Names ---
TARGET_MODEL="meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL="Qwen/Qwen3Guard-Gen-0.6B"

LOG_DIR="$(dirname "$0")/logs_e2e_test"
mkdir -p "$LOG_DIR"

# ===========================================================
# Helpers
# ===========================================================
kill_port() {
    local port=$1
    local pid
    pid=$(lsof -t -i:$port 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "⚠️  Killing existing process on port $port (PID $pid)"
        kill -9 $pid 2>/dev/null || true
        sleep 1
    fi
}

wait_for_server() {
    local PORT=$1
    local NAME=$2
    local PID=$3
    local TIMEOUT=${4:-300}

    echo "⏳ Waiting for $NAME (Port $PORT)..."

    for ((i=1; i<=TIMEOUT; i++)); do
        if ! kill -0 $PID 2>/dev/null; then
            echo "❌ $NAME process (PID $PID) died!"
            return 1
        fi

        HTTP_STATUS=$(curl -o /dev/null -s -L -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null || echo "000")

        if [ "$HTTP_STATUS" == "200" ] || [ "$HTTP_STATUS" == "307" ]; then
            echo "✅ $NAME is READY (status=$HTTP_STATUS)"
            return 0
        fi
        sleep 1
    done

    echo "❌ $NAME TIMEOUT after ${TIMEOUT}s"
    return 1
}

cleanup() {
    echo "🛑 Cleaning up servers..."
    if [ -n "${TARGET_PID:-}" ]; then kill -TERM -- -$TARGET_PID 2>/dev/null || kill -9 $TARGET_PID 2>/dev/null || true; fi
    if [ -n "${JUDGE_PID:-}" ]; then kill -TERM -- -$JUDGE_PID 2>/dev/null || kill -9 $JUDGE_PID 2>/dev/null || true; fi
    kill_port $TARGET_PORT
    kill_port $JUDGE_PORT
}
trap cleanup EXIT INT TERM

echo "🧹 Pre-flight cleanup..."
kill_port $TARGET_PORT
kill_port $JUDGE_PORT
sleep 1

# ===========================================================
# Launch Servers
# ===========================================================
echo "🚀 Starting Target server..."
setsid bash -c "CUDA_VISIBLE_DEVICES=$TARGET_GPU vllm serve \"$TARGET_MODEL\" \
    --host 0.0.0.0 --port $TARGET_PORT --max-model-len 16384 --dtype bfloat16 \
    --gpu-memory-utilization 0.85 > \"$LOG_DIR/target_test.log\" 2>&1" &
TARGET_PID=$!

echo "🚀 Starting Judge server..."
setsid bash -c "CUDA_VISIBLE_DEVICES=$JUDGE_GPU vllm serve \"$JUDGE_MODEL\" \
    --host 0.0.0.0 --port $JUDGE_PORT --max-model-len 8192 --dtype bfloat16 \
    --gpu-memory-utilization 0.5 > \"$LOG_DIR/judge_test.log\" 2>&1" &
JUDGE_PID=$!

wait_for_server $JUDGE_PORT "Judge" $JUDGE_PID 300
wait_for_server $TARGET_PORT "Target" $TARGET_PID 300

echo "🧪 Running tests..."
python "$(cd "$(dirname "$0")" && pwd)/test_env_manager.py"
