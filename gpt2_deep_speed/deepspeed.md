# 复现步骤

## step1：基础环境安装
- gcc:7.5.0 和 g++：11.4.0 --》 安装指令
sudo apt install build-essential

- python==3.8.17
- setuptools==59.5.0 (必须)
- numpy: numpy==1.22.0
- deepspeed: 0.8.0+bf6b9802
- transformers: 4.30.2
- cuda: cuda-11.3
[cuda安装链接](https://developer.nvidia.com/cuda-toolkit-archive)

- torch 安装:
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# step2: install apex
1. git clone https://github.com/NVIDIA/apex.git
2. cd apex
3. git checkout 22.03
4. update setup.py:
```python
diff --git a/setup.py b/setup.py
index d76e998..f224dae 100644
--- a/setup.py
+++ b/setup.py
@@ -31,6 +31,8 @@ def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
     print(raw_output + "from " + cuda_dir + "/bin\n")

     if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
+        # allow minor diffs
+        if bare_metal_minor != torch_binary_minor: return
         raise RuntimeError(
             "Cuda extensions are being compiled with a version of Cuda that does "
             "not match the version used to compile Pytorch binaries.  "
```
5. pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .  2>&1 | tee build.log
6. cd -

## step3: install Megatron-DeepSpeed
1. git clone https://github.com/bigscience-workshop/Megatron-DeepSpeed.git  (不怎么更新了，最新的就行，或者用：e52bdabbde3c6895aceb76c1bced295c2646121f)
2. cd Megatron-DeepSpeed
3. fix requirements.txt
```python
diff --git a/requirements.txt b/requirements.txt
index da76b5e..d2a390c 100644
--- a/requirements.txt
+++ b/requirements.txt
@@ -1,6 +1,6 @@
 datasets
 nltk
-numpy
+numpy==1.22.0
 parameterized
 pybind11
 regex
@@ -8,7 +8,7 @@ six
 tensorboard
 torch>=1.7
 transformers
-DeepSpeed @ git+https://github.com/microsoft/DeepSpeed.git
+DeepSpeed @ git+https://github.com/microsoft/DeepSpeed.git@v0.8.0
 # versions from HF transformers
 black==21.4b0
 isort>=5.5.4
```
4. pip install -r requirments.txt

## step4: data download
1. cd Megatron-DeepSpeed
2. mkdir -p data
3. download data
- wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -O data/gpt2-vocab.json
- wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -O data/gpt2-merges.txt
- python -c 'from datasets import load_dataset; ds = load_dataset("stas/oscar-en-10k", split="train", keep_in_memory=False); ds.to_json(f"data/oscar-en-10k.jsonl", orient="records", lines=True, force_ascii=False)'

4. 最终data文件夹的数据：
- gpt2-merges.txt
- gpt2-vocab.json
- meg-gpt2-oscar-en-10k_text_document.bin
- meg-gpt2-oscar-en-10k_text_document.idx
- oscar-en-10k.jsonl

## step5: data process
```python
python tools/preprocess_data.py \
    --input data/oscar-en-10k.jsonl \
    --output-prefix data/meg-gpt2-oscar-en-10k \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file data/gpt2-merges.txt \
    --vocab data/gpt2-vocab.json \
    --append-eod \
    --workers 4
```

## step6: Train
1. 如果只有一个GPU，则更改如下两行

```python
N_GPUS=1
TP_SIZE=1
```

2. 训练脚本

```python
CHECKPOINT_PATH=checkpoints/gpt2

VOCAB_FILE=data/gpt2-vocab.json
MERGE_FILE=data/gpt2-merges.txt
DATA_PATH=data/meg-gpt2-oscar-en-10k_text_document
TENSORBOARD_PATH=output_dir/tensorboard

N_GPUS=2
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
TP_SIZE=2
PP_SIZE=1

NLAYERS=2
NHIDDEN=8
NHEADS=2
SEQ_LEN=512
VOCAB_SIZE=50257

SAVE_INTERVAL=50

TRAIN_SAMPLES=10_000

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 2 2 1_000 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --lr-warmup-samples 5 \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples 12 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --embed-layernorm \
    --fp16 \
    --partition-activations \
    --seed 42 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    "

OUTPUT_ARGS=" \
    --exit-interval 100 \
    --log-interval 10 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 100 \
    --eval-iters 10 \
    --checkpoint-activations \
    "

DATA_ARGS=" \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --kill-switch-path /tmp/kill-switch \
    "

ZERO_STAGE=1

config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

ALL_ARGS="$GPT_ARGS $OUTPUT_ARGS $DATA_ARGS $DEEPSPEED_ARGS"

MASTER_ADDR=localhost
MASTER_PORT=6777

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $N_GPUS \
    --nnodes 1 \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "
export CMD=" \
    $LAUNCHER pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --distributed-backend nccl \
    $ALL_ARGS \
    "

echo $CMD

$CMD
```




