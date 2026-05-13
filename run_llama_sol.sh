#!/bin/bash
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate tidpo
cd ~/TIDPO
pip install -q "setuptools<70" "transformers>=4.42" "tokenizers>=0.19" sentencepiece protobuf
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/bzhao112/tidpo/hf_cache

METHOD="$1"
SEED="${2:-0}"
echo "Method: $METHOD, Seed: $SEED"

if [ "$METHOD" = "sft" ]; then
    python3 -u train.py model=llama7b loss=sft \
        datasets=[hh] n_epochs=1 n_examples=null lr=5e-6 optimizer=RMSprop \
        max_length=256 max_prompt_length=128 warmup_steps=150 max_grad_norm=10.0 seed=0 \
        eval_every=999999 eval_batch_size=1 n_eval_examples=4 \
        sample_during_eval=false do_first_eval=false wandb.enabled=false \
        activation_checkpointing=false trainer=BasicTrainer \
        batch_size=32 gradient_accumulation_steps=4 \
        local_dirs=[/scratch/bzhao112/tidpo] exp_name=sft_llama
else
    SFT=$(find /scratch/bzhao112/tidpo -name policy.pt -path '*sft_llama*' -path '*LATEST*' | sort | tail -1)
    echo "SFT: $SFT"
    if [ -z "$SFT" ]; then echo "ERROR: No SFT checkpoint"; exit 1; fi

    if [ "$METHOD" = "dpo" ]; then
        python3 -u train.py model=llama7b model.archive=$SFT \
            loss=tidpo loss.use_tidpo=false loss.enable_gradient_attribution=false \
            loss.beta=0.1 loss.alpha=0.5 loss.gamma=0.0 loss.alpha_triplet=0.0 \
            datasets=[hh] n_epochs=1 n_examples=null lr=5e-6 optimizer=RMSprop \
            max_length=256 max_prompt_length=128 warmup_steps=150 max_grad_norm=10.0 seed=$SEED \
            eval_every=20000 eval_batch_size=1 n_eval_examples=256 \
            sample_during_eval=false do_first_eval=false wandb.enabled=false \
            activation_checkpointing=false trainer=BasicTrainer \
            batch_size=32 gradient_accumulation_steps=4 \
            local_dirs=[/scratch/bzhao112/tidpo] exp_name=FULL_dpo_s$SEED

    elif [ "$METHOD" = "tidpo" ]; then
        python3 -u train.py model=llama7b model.archive=$SFT \
            loss=tidpo loss.use_tidpo=true loss.enable_gradient_attribution=true +loss.attribution_method=gradient \
            loss.beta=0.1 loss.alpha=0.5 loss.gamma=0.1 loss.alpha_triplet=0.1 loss.lambda_importance=0.7 \
            datasets=[hh] n_epochs=1 n_examples=null lr=5e-6 optimizer=RMSprop \
            max_length=256 max_prompt_length=128 warmup_steps=150 max_grad_norm=10.0 seed=$SEED \
            eval_every=20000 eval_batch_size=1 n_eval_examples=256 \
            sample_during_eval=false do_first_eval=false wandb.enabled=false \
            activation_checkpointing=false trainer=BasicTrainer \
            batch_size=32 gradient_accumulation_steps=4 \
            local_dirs=[/scratch/bzhao112/tidpo] exp_name=FULL_tidpo_s$SEED

    elif [ "$METHOD" = "saliency" ]; then
        python3 -u train.py model=llama7b model.archive=$SFT \
            loss=saliency_tidpo \
            loss.beta=0.1 loss.alpha=0.5 loss.gamma=0.1 loss.alpha_triplet=0.1 loss.lambda_importance=0.7 \
            datasets=[hh] n_epochs=1 n_examples=null lr=5e-6 optimizer=RMSprop \
            max_length=256 max_prompt_length=128 warmup_steps=150 max_grad_norm=10.0 seed=$SEED \
            eval_every=20000 eval_batch_size=1 n_eval_examples=256 \
            sample_during_eval=false do_first_eval=false wandb.enabled=false \
            activation_checkpointing=false trainer=BasicTrainer \
            batch_size=32 gradient_accumulation_steps=4 \
            local_dirs=[/scratch/bzhao112/tidpo] exp_name=FULL_saliency_s$SEED
    fi
fi
echo "=== $METHOD s$SEED COMPLETE ==="
