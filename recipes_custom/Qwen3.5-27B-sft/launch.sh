#!/usr/bin/env bash
# 在 node-0 上执行此脚本，自动 SSH 启动所有节点
# 用法: bash launch.sh

set -euo pipefail

# ============================================================
# 4 个节点的 IP 地址
# ============================================================
NODES=(
    "10.178.129.2"
    "10.178.128.205"
    "10.178.162.19"
    "10.178.160.28"
)

MASTER_ADDR=${NODES[0]}
MASTER_PORT=23457
NNODES=${#NODES[@]}

# 训练脚本路径（所有节点上必须能访问到，共享存储即可）
SCRIPT_PATH="/llm-align/liuchonghan/verl_lao/recipes_custom/Qwen3.5-27B-sft/run_sft_qwen3.5_27b_megatron.sh"
LOG_DIR="/llm-align/liuchonghan/logs/qwen3.5_27b_sft"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "  Launching ${NNODES}-node training"
echo "  MASTER: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Script: ${SCRIPT_PATH}"
echo "  Logs:   ${LOG_DIR}/"
echo "========================================"

PIDS=()

for i in "${!NODES[@]}"; do
    NODE=${NODES[$i]}
    LOG_FILE="${LOG_DIR}/node_${i}.log"

    CMD="NNODES=${NNODES} NODE_RANK=${i} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} bash ${SCRIPT_PATH}"

    if [ "$i" -eq 0 ]; then
        echo "[node-${i}] Starting locally (${NODE}) -> ${LOG_FILE}"
        bash -c "${CMD}" > "${LOG_FILE}" 2>&1 &
    else
        echo "[node-${i}] SSH to ${NODE} -> ${LOG_FILE}"
        ssh -o StrictHostKeyChecking=no "${NODE}" "${CMD}" > "${LOG_FILE}" 2>&1 &
    fi
    PIDS+=($!)
done

echo ""
echo "All ${NNODES} nodes launched. PIDs: ${PIDS[*]}"
echo "Tailing node-0 log (Ctrl+C to stop tailing, training continues in background):"
echo "----------------------------------------"
sleep 2
tail -f "${LOG_DIR}/node_0.log"
