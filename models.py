import os
import csv
import re
import inspect
import importlib
import torch

_mpl_dir = os.environ.get("MPLCONFIGDIR", "")
if not _mpl_dir:
    _mpl_dir = os.path.join("/tmp", "matplotlib")
    try:
        os.makedirs(_mpl_dir, exist_ok=True)
    except Exception:
        _mpl_dir = "/tmp"
    os.environ["MPLCONFIGDIR"] = _mpl_dir

import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _layer_to_kv(layer: Any) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(layer, (tuple, list)) and len(layer) == 2:
        return layer[0], layer[1]
    if hasattr(layer, "keys") and hasattr(layer, "values"):
        return layer.keys, layer.values
    if hasattr(layer, "key_cache") and hasattr(layer, "value_cache"):
        return layer.key_cache, layer.value_cache
    if hasattr(layer, "key") and hasattr(layer, "value"):
        return layer.key, layer.value
    if hasattr(layer, "k_cache") and hasattr(layer, "v_cache"):
        return layer.k_cache, layer.v_cache
    return None


def _cache_to_legacy_list(past_key_values: Any) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if past_key_values is None:
        return []
    if isinstance(past_key_values, (list, tuple)):
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in past_key_values:
            kv = _layer_to_kv(layer)
            if kv is not None:
                out.append(kv)
        return out
    if hasattr(past_key_values, "layers"):
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in past_key_values.layers:
            kv = _layer_to_kv(layer)
            if kv is not None:
                out.append(kv)
        return out
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    return []


def _past_length(past_key_values: Any) -> int:
    legacy = _cache_to_legacy_list(past_key_values)
    if not legacy:
        return 0
    k = legacy[0][0]
    if not torch.is_tensor(k) or k.ndim < 3:
        return 0
    return int(k.shape[-2])


def _slice_past(past_key_values: Any, idx: int) -> Optional[Tuple]:
    if past_key_values is None:
        return None
    legacy = _cache_to_legacy_list(past_key_values)
    if not legacy:
        return None
    sliced = []
    for k, v in legacy:
        if torch.is_tensor(k) and torch.is_tensor(v):
            sliced.append((k[idx : idx + 1].contiguous(), v[idx : idx + 1].contiguous()))
    return tuple(sliced) if sliced else None


def _build_position_ids_from_attention(attention_mask: torch.Tensor) -> torch.Tensor:
    pos = attention_mask.long().cumsum(dim=-1) - 1
    pos = pos.clamp_min(0)
    pos = pos.masked_fill(attention_mask == 0, 0)
    return pos.to(dtype=torch.long)


def _is_multimodal_config(cfg) -> bool:
    model_type = str(getattr(cfg, "model_type", "")).lower()
    class_name = cfg.__class__.__name__.lower()
    if any(key in model_type for key in ("qwen", "gemma3")) and any(key in model_type for key in ("vl", "_vl")):
        return True
    if "lfm2_vl" in model_type or "lfm2vl" in class_name:
        return True
    if "smolvlm" in model_type or "smolvlm" in class_name:
        return True
    if "internvl" in model_type or "internvl" in class_name:
        return True
    if "minicpmv" in model_type or "minicpmv" in class_name:
        return True
    if "minicpm" in model_type and "vl" in model_type:
        return True
    if "gemma3" in model_type and "text" not in model_type:
        return True
    if any(key in model_type for key in ("paligemma", "llava", "idefics", "mllama", "mistral3")):
        return True
    if any(key in class_name for key in ("qwen3vl", "qwen2vl", "gemma3")) and "text" not in class_name:
        return True
    if "paligemma" in class_name:
        return True
    return False


def _is_internvl_config(cfg) -> bool:
    model_type = str(getattr(cfg, "model_type", "")).lower()
    class_name = cfg.__class__.__name__.lower()
    return ("internvl" in model_type) or ("internvl" in class_name)


def _is_smolvlm_config(cfg) -> bool:
    model_type = str(getattr(cfg, "model_type", "")).lower()
    class_name = cfg.__class__.__name__.lower()
    return ("smolvlm" in model_type) or ("smolvlm" in class_name)


def _is_lfm2_vl_config(cfg) -> bool:
    model_type = str(getattr(cfg, "model_type", "")).lower()
    class_name = cfg.__class__.__name__.lower()
    return ("lfm2_vl" in model_type) or ("lfm2vl" in class_name)


def _is_minicpm_v_config(cfg) -> bool:
    model_type = str(getattr(cfg, "model_type", "")).lower()
    class_name = cfg.__class__.__name__.lower()
    return ("minicpmv" in model_type) or ("minicpmv" in class_name) or (
        "minicpm" in model_type and "vl" in model_type
    )


def _load_model_without_meta_init(model_cls, model_name: str, torch_dtype: torch.dtype):
                                                                                       
                                                                                       
                                                
    from transformers.modeling_utils import PreTrainedModel

    original = PreTrainedModel.__dict__["get_init_context"]
    original_fn = original.__func__
    original_mark_tied = PreTrainedModel.mark_tied_weights_as_initialized

    def _patched_get_init_context(cls, dtype: torch.dtype, is_quantized: bool, _is_ds_init_called: bool):
        contexts = original_fn(cls, dtype, is_quantized, _is_ds_init_called)
        filtered = []
        for ctx in contexts:
            if isinstance(ctx, torch.device) and ctx.type == "meta":
                continue
            filtered.append(ctx)
        return filtered

    def _patched_mark_tied(self):
                                                                             
        if not hasattr(self, "all_tied_weights_keys"):
            tied = getattr(self, "_tied_weights_keys", None)
            if isinstance(tied, dict):
                self.all_tied_weights_keys = dict(tied)
            elif isinstance(tied, (list, tuple, set)):
                self.all_tied_weights_keys = {k: None for k in tied}
            else:
                self.all_tied_weights_keys = {}
        return original_mark_tied(self)

    PreTrainedModel.get_init_context = classmethod(_patched_get_init_context)
    PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark_tied
    try:
        return model_cls.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    finally:
        PreTrainedModel.get_init_context = original
        PreTrainedModel.mark_tied_weights_as_initialized = original_mark_tied


def _patch_minicpm_batchfeature_compat(model: Any) -> None:
                                                                                 
                                                                                   
    try:
        from collections import UserDict as _UserDict
        import sys
        from transformers.feature_extraction_utils import BatchFeature

        if not bool(getattr(BatchFeature, "_latentmas_skip_tensor_patch", False)):
            def _patched_batchfeature_init(self, data=None, tensor_type=None, skip_tensor_conversion=None):
                _UserDict.__init__(self, data)
                try:
                    self.convert_to_tensors(
                        tensor_type=tensor_type,
                        skip_tensor_conversion=skip_tensor_conversion,
                    )
                except TypeError as err:
                    if "skip_tensor_conversion" not in str(err):
                        raise
                    self.convert_to_tensors(tensor_type=tensor_type)

            BatchFeature.__init__ = _patched_batchfeature_init
            BatchFeature._latentmas_skip_tensor_patch = True

        def _patch_module(image_module: Any) -> None:
            batch_cls = getattr(image_module, "MiniCPMVBatchFeature", None)
            if batch_cls is None:
                return
            if bool(getattr(batch_cls, "_latentmas_skip_tensor_patch", False)):
                return

            convert_fn = getattr(batch_cls, "convert_to_tensors", None)
            if convert_fn is None:
                return
            sig = inspect.signature(convert_fn)
            if "skip_tensor_conversion" in sig.parameters:
                return

            def _patched_convert_to_tensors(self, tensor_type=None, *args, **kwargs):
                kwargs.pop("skip_tensor_conversion", None)
                return convert_fn(self, tensor_type=tensor_type, *args, **kwargs)

            batch_cls.convert_to_tensors = _patched_convert_to_tensors
            batch_cls._latentmas_skip_tensor_patch = True

        model_module = str(type(model).__module__)
        seen: set[str] = set()

                                                                      
        for name, module in list(sys.modules.items()):
            if not isinstance(name, str):
                continue
            lname = name.lower()
            if "minicpm" in lname and name.endswith("image_processing_minicpmv"):
                seen.add(name)
                _patch_module(module)

                                                                         
        parts = model_module.split(".")
        for i in range(len(parts), 0, -1):
            candidate = ".".join(parts[:i] + ["image_processing_minicpmv"])
            if candidate in seen:
                continue
            try:
                module = importlib.import_module(candidate)
            except Exception:
                continue
            seen.add(candidate)
            _patch_module(module)
    except Exception:
        return


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args

                      
        self.pre_aligned = None

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        is_multimodal = _is_multimodal_config(cfg)
        self.is_internvl = _is_internvl_config(cfg)
        self.is_smolvlm = _is_smolvlm_config(cfg)
        self.is_lfm2_vl = _is_lfm2_vl_config(cfg)
        self.is_minicpm_v = _is_minicpm_v_config(cfg)

        if self.use_vllm:
            if is_multimodal:
                raise ValueError(
                    f"vLLM backend does not support multimodal models (model_type={getattr(cfg, 'model_type', None)}). "
                    "Run with --use_vllm unset."
                )

            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))

            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas": 
                self.vllm_engine = LLM(
                    model=model_name,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=gpu_util,
                    enable_prefix_caching=True,
                    enable_prompt_embeds=True,
                )
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                self.HF_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                ).to(args.device2).eval() 
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            return                                   

                                            
        use_fast_tokenizer = not self.is_internvl
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=use_fast_tokenizer,
                trust_remote_code=True,
            )
        except Exception:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = getattr(processor, "tokenizer", None) or (
                processor if hasattr(processor, "convert_tokens_to_ids") else None
            )
            if self.tokenizer is None:
                raise
        if not bool(getattr(cfg, "is_encoder_decoder", False)):
            self.tokenizer.padding_side = "left"
        _ensure_pad_token(self.tokenizer)

                           
                                                                              
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        with torch.no_grad():
            if self.is_internvl or self.is_minicpm_v:
                model_cls = AutoModelForCausalLM
                self.model = _load_model_without_meta_init(model_cls, model_name, torch_dtype)
            else:
                model_cls = AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
                self.model = model_cls.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                )

        vocab_size = int(self.model.get_input_embeddings().weight.shape[0])
        if len(self.tokenizer) > vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(device)
        self.model.eval()
        if self.is_minicpm_v:
            _patch_minicpm_batchfeature_compat(self.model)
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def _get_text_model(self):
        if self.is_internvl and hasattr(self.model, "language_model"):
            return self.model.language_model
        if self.is_minicpm_v and hasattr(self.model, "llm"):
            return self.model.llm
        return self.model

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if self.is_smolvlm:
                                                                            
                                                                              
            norm_messages: List[Dict[str, Any]] = []
            for message in messages:
                role = str(message.get("role", "user"))
                content = message.get("content", "")
                if isinstance(content, list):
                    norm_content = content
                else:
                    norm_content = [{"type": "text", "text": str(content)}]
                norm_messages.append({"role": role, "content": norm_content})
            messages = norm_messages
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations

    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if self.args.latent_space_realign:
            pass
        else:
            realign_matrix = torch.eye(realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype)

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
        stop_regexes: Optional[List[str]] = None,
        return_metadata: bool = False,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        input_len = int(input_ids.shape[-1])
        if stop_regexes and input_ids.shape[0] > 1:
            texts: List[str] = []
            meta_list: List[Dict] = []
            for idx in range(input_ids.shape[0]):
                sub_ids = input_ids[idx : idx + 1]
                sub_mask = attention_mask[idx : idx + 1] if attention_mask is not None else None
                sub_past = _slice_past(past_key_values, idx)
                sub_texts, _, sub_meta = self.generate_text_batch(
                    sub_ids,
                    sub_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    past_key_values=sub_past,
                    stop_regexes=stop_regexes,
                    return_metadata=True,
                )
                texts.append(sub_texts[0])
                meta_list.append(sub_meta)
            combined_meta = {
                "generated_tokens": [m["generated_tokens"][0] for m in meta_list],
                "stop_reasons": [m["stop_reasons"][0] for m in meta_list],
                "stop_spans": [m["stop_spans"][0] for m in meta_list],
            }
            if return_metadata:
                return texts, None, combined_meta
            return texts, None
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        stopping = None
        stop_checker = None
        if stop_regexes:
            compiled = [re.compile(pattern, re.MULTILINE) for pattern in stop_regexes]

            class _StopOnRegex(StoppingCriteria):
                def __init__(self, tokenizer, patterns, prompt_len):
                    self.tokenizer = tokenizer
                    self.patterns = patterns
                    self.prompt_len = prompt_len
                    self.matched = False
                    self.matched_span = None

                def __call__(self, input_ids, scores, **kwargs):
                    text = self.tokenizer.decode(
                        input_ids[0][self.prompt_len :], skip_special_tokens=True
                    )
                    for pattern in self.patterns:
                        match = pattern.search(text)
                        if match:
                            if match.end() == len(text) and text and text[-1].isdigit():
                                continue
                            self.matched = True
                            self.matched_span = match.group(0)
                            return True
                    return False

            stop_checker = _StopOnRegex(self.tokenizer, compiled, input_len)
            stopping = StoppingCriteriaList([stop_checker])

        text_model = self._get_text_model()
        do_sample = float(temperature) > 0.0
                                                                            
                                                                                    
        if self.is_internvl or self.is_minicpm_v:
            do_sample = False

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": False,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "stopping_criteria": stopping,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)

        outputs = text_model.generate(**gen_kwargs)
        sequences = outputs.sequences
        generations: List[str] = []
        generated_tokens: List[int] = []
        stop_reasons: List[str] = []
        stop_spans: List[Optional[str]] = []
        for idx, length in enumerate(prompt_lengths):
            generated_ids = sequences[idx, input_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
            gen_len = generated_ids.shape[0]
            generated_tokens.append(gen_len)
            span = None
            if stop_regexes:
                for pattern in stop_regexes:
                    match = re.search(pattern, text, re.MULTILINE)
                    if match:
                        span = match.group(0)
                        break
            stop_spans.append(span)
            if span:
                stop_reasons.append("stop_pattern")
            elif gen_len >= max_new_tokens:
                stop_reasons.append("max_tokens")
            else:
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None and generated_ids.numel() > 0 and generated_ids[-1].item() == eos_id:
                    stop_reasons.append("eos")
                else:
                    stop_reasons.append("unknown")
        if return_metadata:
            return (
                generations,
                outputs.past_key_values,
                {
                    "generated_tokens": generated_tokens,
                    "stop_reasons": stop_reasons,
                    "stop_spans": stop_spans,
                },
            )
        return generations, outputs.past_key_values

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
        return_latent_embeds: bool = False,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)
        position_ids = _build_position_ids_from_attention(attention_mask) if self.is_minicpm_v else None

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
                if self.is_minicpm_v:
                    position_ids = _build_position_ids_from_attention(attention_mask)

        text_model = self._get_text_model()
        fwd = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if position_ids is not None:
            fwd["position_ids"] = position_ids
        outputs = text_model(**fwd)
        past = outputs.past_key_values

        last_hidden = outputs.hidden_states[-1][:, -1, :]         

        latent_embeds: List[torch.Tensor] = []

        for step in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else text_model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)

            latent_embeds.append(latent_vec.detach().clone())

            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            step_fwd = {
                "inputs_embeds": latent_embed,
                "attention_mask": latent_mask,
                "past_key_values": past,
                "use_cache": True,
                "output_hidden_states": True,
                "return_dict": True,
            }
            if self.is_minicpm_v:
                step_fwd["position_ids"] = torch.full(
                    (latent_embed.shape[0], 1),
                    past_len,
                    dtype=torch.long,
                    device=latent_embed.device,
                )
            outputs = text_model(**step_fwd)
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        if return_latent_embeds:
            if latent_embeds:
                latent_stack = torch.stack(latent_embeds, dim=1)
            else:
                latent_stack = torch.empty(
                    (input_ids.shape[0], 0, last_hidden.shape[-1]),
                    device=last_hidden.device,
                    dtype=last_hidden.dtype,
                )
            return past, latent_stack
        return past

    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        curr_output_embedding = [] 
        curr_output_embedding.append(outputs.hidden_states[0])                   


        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1)                          
