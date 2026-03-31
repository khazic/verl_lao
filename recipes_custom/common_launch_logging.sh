#!/usr/bin/env bash

init_launch_logging() {
    local script_path="$1"
    local exp_name="$2"
    local node_rank="${3:-0}"
    local launch_dir="${4:-${PWD}}"
    local script_name
    local run_timestamp
    local safe_exp_name

    script_name=$(basename "${script_path}")
    script_name=${script_name%.sh}
    script_name=${script_name%.bash}
    run_timestamp=$(date '+%Y%m%d_%H%M%S')
    safe_exp_name=$(printf '%s' "${exp_name}" | tr '/: ' '___')

    LOG_DIR=${LOG_DIR:-${launch_dir}/logs}
    LOG_PREFIX=${script_name}_${safe_exp_name}_${run_timestamp}
    LOG_FILE=${LOG_DIR}/${LOG_PREFIX}_node${node_rank}.log
    TORCHRUN_WORKER_LOG_DIR=${TORCHRUN_WORKER_LOG_DIR:-${LOG_DIR}/${LOG_PREFIX}_workers_node${node_rank}}
    TORCHRUN_TEE=${TORCHRUN_TEE:-3}
    TORCHRUN_REDIRECTS=${TORCHRUN_REDIRECTS:-3}
    TORCHRUN_LOCAL_RANKS_FILTER=${TORCHRUN_LOCAL_RANKS_FILTER:-0}
    TORCHRUN_LOGGING_ARGS=()

    mkdir -p "${LOG_DIR}"
    exec > >(tee -a "${LOG_FILE}") 2>&1

    echo ">>> 启动目录: ${launch_dir}"
    echo ">>> 日志文件: ${LOG_FILE}"

    if command -v torchrun >/dev/null 2>&1; then
        if torchrun --help 2>/dev/null | grep -q -- '--log-dir'; then
            mkdir -p "${TORCHRUN_WORKER_LOG_DIR}"
            TORCHRUN_LOGGING_ARGS=(
                "--log-dir=${TORCHRUN_WORKER_LOG_DIR}"
                "--redirects=${TORCHRUN_REDIRECTS}"
                "--tee=${TORCHRUN_TEE}"
                "--local-ranks-filter=${TORCHRUN_LOCAL_RANKS_FILTER}"
            )
            echo ">>> worker 日志目录: ${TORCHRUN_WORKER_LOG_DIR}"
        else
            echo ">>> 检测到当前 torchrun 不支持 --log-dir/--tee，跳过 worker 独立日志。"
        fi
    fi
}
