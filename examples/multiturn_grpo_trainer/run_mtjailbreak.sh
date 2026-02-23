#!/bin/bash
set -euo pipefail

# ==========================================================
# E2E Training Script: MultiTurnConvEnv GRPO with real vLLM servers
#
# Launches:  Target (Llama-3.1-8B) + Judge (Qwen3Guard-0.6B) in background
# Then runs: verl.trainer.main_ppo for Attacker (actor model)
# ==========================================================

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# --- Clean up any existing Ray cluster ---
echo "🧹 Cleaning up any existing Ray cluster..."
ray stop 2>/dev/null || true
sleep 2

# --- Config ---
train_data_size=2
val_data_size=2
group_size=5
lambda_div=0.0 # diversity penalty coefficient

# Prepare jailbreak behavior data for multiturn_conv environment
echo "🔧 Preparing jailbreak behavior data..."
SCRIPT_DIR="$(dirname "$0")"
python3 "$SCRIPT_DIR/prepare_jailbreak_data.py" \
    --local_dir "$HOME/data/verl-agent/jailbreak" \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

TRAIN_DATA="$HOME/data/verl-agent/jailbreak/train.parquet"
VAL_DATA="$HOME/data/verl-agent/jailbreak/test.parquet"

# --- GPU Assignment ---
# Actor (policy model): GPU 0, 1 (needs 2 GPUs for tensor_model_parallel_size=2)
# Target (victim model): GPU 7
# Judge model: GPU 6
TARGET_GPU=7
JUDGE_GPU=6

# --- Port Assignment ---
TARGET_PORT=9011
JUDGE_PORT=9012

# --- Model Names ---
TARGET_MODEL="meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL="Qwen/Qwen3Guard-Gen-0.6B"
MAX_TURNS=2

LOG_DIR="$(dirname "$0")/logs_mtjailbreak"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo " MultiTurnConvEnv PPO Training "
echo "=============================================="
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

    if [ -n "${TARGET_PID:-}" ]; then
        echo "   Stopping Target (PID $TARGET_PID)..."
        kill -TERM -- -$TARGET_PID 2>/dev/null || kill -9 $TARGET_PID 2>/dev/null || true
    fi
    if [ -n "${JUDGE_PID:-}" ]; then
        echo "   Stopping Judge (PID $JUDGE_PID)..."
        kill -TERM -- -$JUDGE_PID 2>/dev/null || kill -9 $JUDGE_PID 2>/dev/null || true
    fi

    sleep 3

    echo "   Ensuring ports are freed..."
    kill_port $TARGET_PORT
    kill_port $JUDGE_PORT

    echo "✅ Cleanup done."
}
trap cleanup EXIT INT TERM

# ===========================================================
# 1. Kill any existing processes on our ports
# ===========================================================
echo ""
echo "🧹 Pre-flight cleanup..."
kill_port $TARGET_PORT
kill_port $JUDGE_PORT
sleep 1

# ===========================================================
# 2. Launch vLLM Servers for Target and Judge
# ===========================================================

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

echo ""
echo "✨ Target and Judge servers are UP!"
echo "=============================================="

# ===========================================================
# 4. Run Multi-Turn GRPO Training with Hydra Config
# ===========================================================
echo ""
echo "🚀 Starting Multi-Turn GRPO Training (Attacker) with Hydra config..."
echo ""

# The trainer will use GPU 0,1 for actor model (tensor parallel)
# Target uses GPU 7, Judge uses GPU 6
# Set CUDA_VISIBLE_DEVICES to make Ray use GPU 0,1
export CUDA_VISIBLE_DEVICES=0,1
echo "🔧 Setting CUDA_VISIBLE_DEVICES=0,1 for Actor training"

# Note: Config file (mtjailbreak.yaml) should be in verl/trainer/config/ directory
#       It uses algorithm.adv_estimator=multiturn_grpo for multi-turn environments
python3 -m verl.trainer.main_ppo \
    --config-name=mtjailbreak \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    actor_rollout_ref.rollout.name=$ENGINE \
    env.max_steps=$MAX_TURNS \
    env.rollout.n=$group_size \
    env.target.port=$TARGET_PORT \
    env.target.model=$TARGET_MODEL \
    env.judge.port=$JUDGE_PORT \
    env.judge.model=$JUDGE_MODEL \
    $@