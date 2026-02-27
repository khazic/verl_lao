set -xeuo pipefail


NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-4}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

DATASET_DIR=${DATASET_DIR:-~/dataset}
TRAIN_FILES=${TRAIN_FILES:-${DATASET_DIR}/train.parquet}

MODEL_PATH=${MODEL_PATH:-/llm-align/open_models/Qwen3.5/Qwen3.5-27B}

TP_SIZE=${TP_SIZE:-8}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
ETP_SIZE=${ETP_SIZE:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
MAX_LENGTH=${MAX_LENGTH:-8192}
LR=${LR:-5e-6}
MIN_LR=${MIN_LR:-5e-7}
DTYPE=${DTYPE:-bfloat16}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}

BACKEND=megatron
RESUME_MODE=${RESUME_MODE:-disable}

project_name=verl_sft_qwen3_5_27b
exp_name=qwen3_5_27b-${BACKEND}-tp${TP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}
ckpts_home=${ckpts_home:-/llm-align/liuchonghan/ckpt_verl/sft/${project_name}/${exp_name}}

echo ">>> 节点信息: RANK ${NODE_RANK} / WORLD_SIZE ${NNODES}"
echo ">>> 通信信息: MASTER ${MASTER_ADDR} : ${MASTER_PORT}"

if [ "${NODE_RANK}" -eq 0 ]; then
    mkdir -p "${ckpts_home}"
fi

export WANDB_MODE=${WANDB_MODE:-offline}
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Key Qwen3.5 settings:
#   engine.use_remove_padding=False   - GDN requires bshd format (no THD)
#   engine.vanilla_mbridge=True       - use mbridge (not megatron-bridge)
ENGINE_CONFIG="\
    engine=${BACKEND} \
    optim=${BACKEND} \
    optim.lr=${LR} \
    optim.min_lr=${MIN_LR} \
    optim.lr_warmup_steps=20 \
    optim.weight_decay=0.1 \
    optim.betas='[0.9,0.95]' \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    +optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +optim.override_optimizer_config.optimizer_cpu_offload=True \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.expert_model_parallel_size=${EP_SIZE} \
    engine.expert_tensor_parallel_size=${ETP_SIZE} \
    engine.use_mbridge=True \
    engine.vanilla_mbridge=True \
    engine.dtype=${DTYPE} \
    engine.use_remove_padding=False \
    engine.override_transformer_config.attention_backend=auto \
    +engine.override_transformer_config.recompute_method=uniform \
    +engine.override_transformer_config.recompute_granularity=full \
    +engine.override_transformer_config.recompute_num_layers=1"

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    data.max_length=${MAX_LENGTH} \
    data.pad_mode=no_padding \
    data.truncation=right \
    data.use_dynamic_bsz=False \
    data.max_token_len_per_gpu=${MAX_LENGTH} \
    data.messages_key=messages \
    model.path=${MODEL_PATH} \
    model.use_remove_padding=True \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    ${ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=500 \
    trainer.logger="['console']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE}
