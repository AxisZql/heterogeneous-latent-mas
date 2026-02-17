# Vision Wormhole: Public Code Release

Core training and utility code for Vision Wormhole (latent communication in heterogeneous multi-agent VLM systems).

![Vision Wormhole teaser](assets/teaser_wormhole_page2.png)


## Environment Setup

Recommended: Python 3.10+ with a CUDA-enabled PyTorch install.


```bash
pip install transformers==5.0.0 datasets accelerate bitsandbytes einops vllm matplotlib torch torchvision torchaudio tqdm openai wandb nvitop timm
# For LFM model: 
pip install git+https://github.com/huggingface/transformers.git@2a5ba8b53d298ed8421e09831bf96bb6d056a24d pillow
```

## Quick Start

### 1) Train a codec for one model

```bash
python train_vision_latent_mas_codec_new.py \
  --model_name "Qwen/Qwen3-VL-4B-Thinking" \
  --vision_codec_path checkpoints/codec_qwen3vl4b_mixed_cose_ocr_prm800k_large.pt \
  --vision_codec_partial_ckpt_path checkpoints/codec_qwen3vl4b_mixed_cose_ocr_prm800k_large.partial.pt \
  --vision_codec_anchor_texts_path data/vision_codec_anchor_text/mixed_cose_ocr_prm800k.jsonl \
  --vision_codec_dim 512 \
  --vision_codec_tokens 1024 \
  --vision_codec_img_tokens 256 \
  --vision_codec_heads 8 \
  --vision_codec_layers 6 \
  --vision_codec_dropout 0.10 \
  --vision_codec_train_steps 400 \
  --vision_codec_train_batch_size 2 \
  --vision_codec_train_lr 2e-4 \
  --latent_steps 1024 \
  --vision_codec_skip_alignment_if_single 1 \
  --vision_codec_save_per_model 1
```

### 2) Merge codecs across model families

```bash
python merge_vision_codec_checkpoints.py \
  --codec_paths checkpoints/codec_qwen3vl8b_mixed_cose_ocr_prm800k_large.pt checkpoints/codec_gemma3_12b_mixed_cose_ocr_prm800k_large.pt \
  --vision_codec_path checkpoints/codec_qwen3vl8b_gemma3_12b_mixed_cose_ocr_prm800k_large_merged.pt \
  --agent_model_names "Qwen/Qwen3-VL-8B-Thinking,google/gemma-3-12b-it" \
  --ref_model_name "google/gemma-3-12b-it" \
  --vision_codec_anchor_texts_path data/vision_codec_anchor_text/mixed_cose_ocr_prm800k.jsonl \
  --vision_codec_align_max_anchors 300 \
  --vision_codec_align_batch_size 2 \
  --latent_steps 1024
```

### 3) (Optional) Build anchor-text files from datasets

```bash
python scripts/preprocess_dataset.py \
  --datasets "salesforce/cos_e,nvidia/opencodereasoning,openai/prm800k" \
  --splits "validation,split_0,train" \
  --limit_per_dataset 1000 \
  --shuffle \
  --format jsonl \
  --out data/vision_codec_anchor_text/mixed_custom.jsonl
```

