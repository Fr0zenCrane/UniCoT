pretrained_path="ByteDance-Seed/BAGEL-7B-MoT"
model_path=$pretrained_path # set this if you want to reproduce UniCoT yourself
# model_path="Fr0zencr4nE/UniCoT-7B-MoT-v0.2" # uncomment this line if you want to further finetune on UniCoT-7B-MoT-v0.2
NUM_GPU=8

torchrun \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPU \
  train/pretrain_unified_navit_lora.py \
  --dataset_config_file ./data/configs/example.yaml \
  --warmup_steps 200 \
  --model_path $model_path \
  --wandb_offline True \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from $model_path \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_workers 1 \
  --expected_num_tokens 49152 \
  --max_num_tokens 52224 \
  --max_num_tokens_per_sample 16384 \
  --num_shard $NUM_GPU \
  --cpu_offload False \
  --text_cond_dropout_prob 0. \
  --vae_cond_dropout_prob 0. \
  --vit_cond_dropout_prob 0. \
  --save_every 200 \
  --lora_rank 256 \
  --lora_alpha 512 \
  --results_dir    ./results/test_lora/ \
  --checkpoint_dir ./results/test_lora/checkpoints
