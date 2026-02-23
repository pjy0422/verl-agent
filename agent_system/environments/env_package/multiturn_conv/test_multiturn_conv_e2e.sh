#!/bin/bash
set -euo pipefail

# ==========================================================
# E2E Test: MultiTurnConvEnv with real vLLM servers
#
# Launches:  Attacker (Qwen3-4B) + Target (Llama-3.1-8B) + Judge (Qwen3Guard-0.6B)
# Then runs: test_multiturn_conv_verify.py
# ==========================================================

# --- Activate virtualenv ---
source /home/pjy0422/verl-agent/bin/activate

# --- GPU Assignment (use free GPUs 0,1,2) ---
ATTACKER_GPU=0
TARGET_GPU=1
JUDGE_GPU=2

# --- Port Assignment ---
ATTACKER_PORT=9010
TARGET_PORT=9011
JUDGE_PORT=9012

# --- Model Names ---
ATTACKER_MODEL="Qwen/Qwen3-4B-Instruct-2507"
TARGET_MODEL="meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL="Qwen/Qwen3Guard-Gen-0.6B"

# --- Config ---
MAX_TURNS=3
LOG_DIR="$(dirname "$0")/logs_e2e_test"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo " MultiTurnConvEnv E2E Test"
echo "=============================================="
echo " Attacker : $ATTACKER_MODEL (GPU $ATTACKER_GPU, Port $ATTACKER_PORT)"
echo " Target   : $TARGET_MODEL   (GPU $TARGET_GPU, Port $TARGET_PORT)"
echo " Judge    : $JUDGE_MODEL    (GPU $JUDGE_GPU, Port $JUDGE_PORT)"
echo " Max Turns: $MAX_TURNS"
echo "=============================================="

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
            echo "   Last 20 lines of log:"
            tail -20 "$LOG_DIR/${NAME,,}.log" 2>/dev/null || true
            return 1
        fi

        HTTP_STATUS=$(curl -o /dev/null -s -L -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null || echo "000")

        if [ "$HTTP_STATUS" == "200" ] || [ "$HTTP_STATUS" == "307" ]; then
            echo "✅ $NAME is READY (status=$HTTP_STATUS)"
            return 0
        fi

        if (( i % 10 == 0 )); then
            echo "   ... $NAME still loading ($i/${TIMEOUT}s, status=$HTTP_STATUS)"
        fi
        sleep 1
    done

    echo "❌ $NAME TIMEOUT after ${TIMEOUT}s"
    return 1
}

cleanup() {
    echo ""
    echo "🛑 Cleaning up servers..."

    # Kill process groups (to catch child processes spawned by vLLM)
    if [ -n "${ATTACKER_PID:-}" ]; then
        echo "   Stopping Attacker (PID $ATTACKER_PID)..."
        kill -TERM -- -$ATTACKER_PID 2>/dev/null || kill -9 $ATTACKER_PID 2>/dev/null || true
    fi
    if [ -n "${TARGET_PID:-}" ]; then
        echo "   Stopping Target (PID $TARGET_PID)..."
        kill -TERM -- -$TARGET_PID 2>/dev/null || kill -9 $TARGET_PID 2>/dev/null || true
    fi
    if [ -n "${JUDGE_PID:-}" ]; then
        echo "   Stopping Judge (PID $JUDGE_PID)..."
        kill -TERM -- -$JUDGE_PID 2>/dev/null || kill -9 $JUDGE_PID 2>/dev/null || true
    fi

    sleep 3

    # Double-check: kill any remaining processes on the ports
    echo "   Ensuring ports are freed..."
    kill_port $ATTACKER_PORT
    kill_port $TARGET_PORT
    kill_port $JUDGE_PORT

    # Final verification
    for port in $ATTACKER_PORT $TARGET_PORT $JUDGE_PORT; do
        if lsof -i:$port >/dev/null 2>&1; then
            echo "   ⚠️  Warning: Port $port may still be in use"
        fi
    done

    echo "✅ Cleanup done."
}
trap cleanup EXIT INT TERM

# ===========================================================
# 1. Kill any existing processes on our ports
# ===========================================================
echo ""
echo "🧹 Pre-flight cleanup..."
kill_port $ATTACKER_PORT
kill_port $TARGET_PORT
kill_port $JUDGE_PORT
sleep 1

# ===========================================================
# 2. Launch vLLM Servers
# ===========================================================

# --- Attacker ---
echo ""
echo "🚀 Starting Attacker server..."
setsid bash -c "CUDA_VISIBLE_DEVICES=$ATTACKER_GPU vllm serve \"$ATTACKER_MODEL\" \
    --host 0.0.0.0 \
    --port $ATTACKER_PORT \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --reasoning-parser qwen3 \
    > \"$LOG_DIR/attacker.log\" 2>&1" &
ATTACKER_PID=$!
echo "   PID: $ATTACKER_PID"

# --- Target ---
echo "🚀 Starting Target server..."
setsid bash -c "CUDA_VISIBLE_DEVICES=$TARGET_GPU vllm serve \"$TARGET_MODEL\" \
    --host 0.0.0.0 \
    --port $TARGET_PORT \
    --max-model-len 16384 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    > \"$LOG_DIR/target.log\" 2>&1" &
TARGET_PID=$!
echo "   PID: $TARGET_PID"

# --- Judge (small model) ---
echo "🚀 Starting Judge server..."
setsid bash -c "CUDA_VISIBLE_DEVICES=$JUDGE_GPU vllm serve \"$JUDGE_MODEL\" \
    --host 0.0.0.0 \
    --port $JUDGE_PORT \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.5 \
    > \"$LOG_DIR/judge.log\" 2>&1" &
JUDGE_PID=$!
echo "   PID: $JUDGE_PID"

# ===========================================================
# 3. Wait for all servers
# ===========================================================
echo ""
wait_for_server $JUDGE_PORT    "Judge"    $JUDGE_PID    300
wait_for_server $TARGET_PORT   "Target"   $TARGET_PID   300
wait_for_server $ATTACKER_PORT "Attacker" $ATTACKER_PID 300

echo ""
echo "✨ All 3 servers are UP!"
echo "=============================================="

# ===========================================================
# 4. Run the Python verification script
# ===========================================================
echo ""
echo "🧪 Running E2E verification..."
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "$SCRIPT_DIR/test_multiturn_conv_verify.py" \
    --attacker_port  $ATTACKER_PORT \
    --attacker_model "$ATTACKER_MODEL" \
    --target_port    $TARGET_PORT \
    --target_model   "$TARGET_MODEL" \
    --judge_port     $JUDGE_PORT \
    --judge_model    "$JUDGE_MODEL" \
    --max_turns      $MAX_TURNS \
    --num_envs       2

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=============================================="
    echo "  ✅  ALL E2E TESTS PASSED!"
    echo "=============================================="
else
    echo "=============================================="
    echo "  ❌  E2E TESTS FAILED (exit code: $EXIT_CODE)"
    echo "=============================================="
fi

exit $EXIT_CODE
