#!/bin/bash
set -euo pipefail

# ==========================================================
# Multi-Turn GRPO Training Script (2 GPU version)
# Memory-optimized for Qwen2.5-7B on 2 GPUs
# ==========================================================

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HYDRA_FULL_ERROR=1
export RAY_memory_monitor_refresh_ms=0

# --- Clean up any existing Ray cluster ---
echo "🧹 Cleaning up any existing Ray cluster..."
ray stop 2>/dev/null || true
sleep 2

# --- Config ---
train_data_size=2
val_data_size=2
group_size=5
lambda_div=0.0

# --- Project ---
export project_name="verl_agent_multiturn"
export experiment_name="multiturn_grpo_qwen2.5_7b_2gpu"
export OUTPUT="./outputs/${project_name}/${experiment_name}"
export EVAL_LOG_PATH="./eval_logs/${project_name}/${experiment_name}"
mkdir -p $OUTPUT $EVAL_LOG_PATH ./logs

# Prepare jailbreak behavior data
echo "🔧 Preparing jailbreak behavior data..."
SCRIPT_DIR="$(dirname "$0")"
python3 "$SCRIPT_DIR/prepare_jailbreak_data.py" \
    --local_dir "$HOME/data/verl-agent/jailbreak" \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

TRAIN_DATA="$HOME/data/verl-agent/jailbreak/train.parquet"
VAL_DATA="$HOME/data/verl-agent/jailbreak/test.parquet"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

# --- GPU Assignment ---
# Actor (policy model): GPU 0, 1
# Target (victim model): GPU 7
# Judge model: GPU 6
TARGET_GPU=7
JUDGE_GPU=6
TARGET_PORT=9011
JUDGE_PORT=9012
TARGET_MODEL="meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL="Qwen/Qwen3Guard-Gen-0.6B"
MAX_TURNS=5

LOG_DIR="$(dirname "$0")/logs_mtjailbreak"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo " MultiTurnConvEnv GRPO Training (2 GPU)      "
echo "=============================================="
echo " Actor    : $MODEL_PATH          (GPU 0,1)"
echo " Target   : $TARGET_MODEL        (GPU $TARGET_GPU)"
echo " Judge    : $JUDGE_MODEL         (GPU $JUDGE_GPU)"
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
    local PORT=$1 NAME=$2 PID=$3 TIMEOUT=${4:-300}
    echo "⏳ Waiting for $NAME (Port $PORT)..."
    for ((i=1; i<=TIMEOUT; i++)); do
        if ! kill -0 $PID 2>/dev/null; then
            echo "❌ $NAME process (PID $PID) died!"
            tail -20 "$LOG_DIR/${NAME,,}.log" 2>/dev/null || true
            return 1
        fi
        HTTP_STATUS=$(curl -o /dev/null -s -L -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null || echo "000")
        if [ "$HTTP_STATUS" == "200" ] || [ "$HTTP_STATUS" == "307" ]; then
            echo "✅ $NAME is READY"
            return 0
        fi
        (( i % 10 == 0 )) && echo "   ... $NAME still loading ($i/${TIMEOUT}s)"
        sleep 1
    done
    echo "❌ $NAME TIMEOUT after ${TIMEOUT}s"
    return 1
}

cleanup() {
    echo "🛑 Cleaning up servers..."
    [ -n "${TARGET_PID:-}" ] && kill -TERM -- -$TARGET_PID 2>/dev/null || kill -9 $TARGET_PID 2>/dev/null || true
    [ -n "${JUDGE_PID:-}" ] && kill -TERM -- -$JUDGE_PID 2>/dev/null || kill -9 $JUDGE_PID 2>/dev/null || true
    sleep 3
    kill_port $TARGET_PORT
    kill_port $JUDGE_PORT
    echo "✅ Cleanup done."
}
trap cleanup EXIT INT TERM

# ===========================================================
# 1. Launch vLLM Servers
# ===========================================================
kill_port $TARGET_PORT
kill_port $JUDGE_PORT
sleep 1

echo "🚀 Starting Target server..."
setsid bash -c "CUDA_VISIBLE_DEVICES=$TARGET_GPU vllm serve \"$TARGET_MODEL\" \
    --host 0.0.0.0 --port $TARGET_PORT --max-model-len 16384 \
    --dtype bfloat16 --gpu-memory-utilization 0.85 \
    > \"$LOG_DIR/target.log\" 2>&1" &
TARGET_PID=$!

echo "🚀 Starting Judge server..."
setsid bash -c "CUDA_VISIBLE_DEVICES=$JUDGE_GPU vllm serve \"$JUDGE_MODEL\" \
    --host 0.0.0.0 --port $JUDGE_PORT --max-model-len 8192 \
    --dtype bfloat16 --gpu-memory-utilization 0.5 \
    > \"$LOG_DIR/judge.log\" 2>&1" &
JUDGE_PID=$!

wait_for_server $JUDGE_PORT  "Judge"  $JUDGE_PID  300
wait_for_server $TARGET_PORT "Target" $TARGET_PID 300

echo "✨ All servers UP!"

# ===========================================================
# 2. Run Multi-Turn GRPO Training
# ===========================================================
echo ""
echo "🚀 Starting Multi-Turn GRPO Training..."

export CUDA_VISIBLE_DEVICES=1,2

python3 -m verl.trainer.main_ppo --config-name=mtjailbreak \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    \
    `# === Memory Optimization: Actor ===` \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.use_invalid_action_penalty=true \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    \
    `# === Memory Optimization: Rollout (vLLM) ===` \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.response_length=1024 \
    actor_rollout_ref.rollout.prompt_length=16384 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    \
    `# === Algorithm ===` \
    algorithm.adv_estimator=multiturn_grpo \
    algorithm.gamma=1.0 \
    algorithm.lambda_div=$lambda_div \
    algorithm.use_kl_in_reward=false \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    `# === Environment ===` \
    env.max_steps=$MAX_TURNS \
    env.rollout.n=$group_size \
    env.target.port=$TARGET_PORT \
    env.target.model=$TARGET_MODEL \
    env.judge.port=$JUDGE_PORT \
    env.judge.model=$JUDGE_MODEL \
    \
    `# === Trainer ===` \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.logger='[console,wandb]' \
    trainer.val_before_train=false \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=${OUTPUT} \
    trainer.validation_data_dir=${EVAL_LOG_PATH} \
    trainer.rollout_data_dir=./run_logs/${experiment_name}/train_rollout \
    trainer.resume_mode=disable \
    "$@" 2>&1 | tee ./logs/${project_name}_${experiment_name}.log