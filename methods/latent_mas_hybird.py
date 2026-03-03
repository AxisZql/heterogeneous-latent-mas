from typing import Dict, List, Optional, Tuple

from . import default_agents
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
import torch
import argparse
from vllm import SamplingParams
import pdb

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


def transfer_via_realignment(
    hidden_states: torch.Tensor,
    model_from: ModelWrapper,
    model_to: ModelWrapper,
    lambda_reg: float = 1e-5
) -> torch.Tensor:
    """
    Transfer hidden states from Model A to Model B using cross-model realignment.
    
    Following the paper's realignment math (equation 8):
    W_a = (W_out^T @ W_out + λI)^-1 @ W_out^T @ W_in
    
    For cross-model transfer:
    W_cross = (W_out_A^T @ W_out_A + λI)^-1 @ W_out_A^T @ W_in_B
    
    Then: embedding_B = hidden_A @ W_cross
    
    This is should be equivalent to: hidden_A @ W_out_A^+ @ W_in_B
    
    Args:
        hidden_states: [batch, seq_len, dim_A] - Hidden states from model A
        model_from: Source model (Model A)
        model_to: Target model (Model B)
        lambda_reg: Regularization for numerical stability (paper uses 1e-5)
    
    Returns:
        embeddings_B: [batch, seq_len, dim_B] - Input embeddings for model B
    """
    batch_size, seq_len, dim_A = hidden_states.shape
    original_dtype = hidden_states.dtype
    
    # Get weights from both models
    W_out_A = model_from.model.get_output_embeddings().weight  # [vocab_A, dim_A]
    W_in_B = model_to.model.get_input_embeddings().weight      # [vocab_B, dim_B]
    
    dim_B = W_in_B.shape[1]
    
    # Convert to float32 for numerical stability (BFloat16 not supported in linalg.solve)
    W_out_A_f32 = W_out_A.float()
    W_in_B_f32 = W_in_B.float()
    
    # Compute cross-model realignment matrix:
    # W_cross = (W_out_A^T @ W_out_A + λI)^-1 @ W_out_A^T @ W_in_B
    
    # Step 1: Gram matrix with regularization
    gram = torch.matmul(W_out_A_f32.T, W_out_A_f32)  # [dim_A, dim_A]
    reg = lambda_reg * torch.eye(gram.shape[0], device=gram.device, dtype=torch.float32)
    gram_reg = gram + reg
    
    # Step 2: Right-hand side: W_out_A^T @ W_in_B
    # For Qwen models, vocab sizes should be the same (they share tokenizer)
    # But handle dimension mismatch gracefully
    vocab_A, _ = W_out_A.shape
    vocab_B, _ = W_in_B.shape
    
    if vocab_A != vocab_B:
        # Vocabulary size mismatch - use minimum common vocab
        min_vocab = min(vocab_A, vocab_B)
        W_out_A_f32 = W_out_A_f32[:min_vocab, :]
        W_in_B_f32 = W_in_B_f32[:min_vocab, :]
        print(f"[WARNING] Vocab size mismatch: {vocab_A} vs {vocab_B}, using first {min_vocab} tokens")
    
    # Compute W_out_A^T @ W_in_B [dim_A, dim_B]
    rhs = torch.matmul(W_out_A_f32.T, W_in_B_f32)
    
    # Step 3: Solve: (gram_reg) @ W_cross = rhs
    W_cross = torch.linalg.solve(gram_reg, rhs)  # [dim_A, dim_B]
    
    # Step 4: Apply cross-model alignment: embedding_B = hidden_A @ W_cross
    hidden_flat = hidden_states.reshape(-1, dim_A).float()
    embeddings_B_flat = torch.matmul(hidden_flat, W_cross)  # [batch*seq, dim_B]
    
    # Step 5: Normalize to match target embedding scale (critical for quality!)
    # This follows the paper's approach in _apply_latent_realignment
    target_norm = W_in_B_f32.norm(dim=1).mean()
    current_norms = torch.norm(embeddings_B_flat, dim=1, keepdim=True)
    embeddings_B_flat = embeddings_B_flat * (target_norm / (current_norms + 1e-8))
    
    # Reshape and convert back to original dtype
    embeddings_B = embeddings_B_flat.reshape(batch_size, seq_len, dim_B).to(original_dtype)
    
    return embeddings_B


class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        agent_models: Optional[List[str]] = None,  # NEW: Specify model per agent
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.initial_model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'latent_mas_hybrid'
        self.vllm_device = args.device 
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False

        if self.latent_only:
            self.sequential_info_only = True

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=args.max_new_tokens,
        )
        self.task = args.task
        
        # NEW: Agent-to-model mapping
        if agent_models is None:
            # Default: all agents use same model
            self.agent_models = [model.model_name] * len(self.agents)
        else:
            assert len(agent_models) == len(self.agents), "Must specify model for each agent"
            self.agent_models = agent_models
        
        # Load all unique models
        self.models: Dict[str, ModelWrapper] = {model.model_name: model}
        self._load_additional_models()
        
        # For compatibility: self.model points to initial model
        # (used in vLLM path which we haven't updated yet)
        self.model = model

    def _load_additional_models(self):
        """Load any models needed by agents that aren't already loaded."""
        unique_models = set(self.agent_models)
        for model_name in unique_models:
            if model_name not in self.models:
                print(f"Loading additional model: {model_name}")
                # Create new ModelWrapper with same args and device
                # Note: Hybrid method doesn't support vLLM yet, so use_vllm=False
                # Use primary vllm_device (args.device) for all models to avoid multi-GPU issues
                new_model = ModelWrapper(
                    model_name,
                    self.vllm_device,  # Use primary device for all models
                    use_vllm=False,  # Hybrid doesn't support vLLM mixing yet
                    args=self.args
                )
                self.models[model_name] = new_model

    def _capture_hidden_states_from_model(
        self,
        agent_model: ModelWrapper,
        wrapped_ids: torch.Tensor,
        wrapped_mask: torch.Tensor,
        past_kv: Optional[Tuple],
        latent_steps: int
    ) -> Tuple[Tuple, torch.Tensor]:
        """
        Run generate_latent_batch and capture ONLY the RAW latent hidden states.
        
        Returns:
            past_kv: Updated KV cache
            raw_latent_hidden_states: [batch, latent_steps, hidden_dim] - RAW hidden states (NOT aligned embeddings)
        """
        # Encode the input_ids
        input_ids = wrapped_ids.to(agent_model.device)
        attention_mask = wrapped_mask.to(agent_model.device)
        
        if past_kv is not None:
            past_len = _past_length(past_kv)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        
        # Initial forward pass
        outputs = agent_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_dim]
        
        # Generate latent steps and store RAW hidden states (before alignment)
        raw_latent_hidden_list = []
        for _ in range(latent_steps):
            # Store RAW hidden state BEFORE alignment
            raw_latent_hidden_list.append(last_hidden.unsqueeze(1))  # [batch, 1, hidden_dim]
            
            # Apply realignment to create embedding for next forward pass
            latent_vec = agent_model._apply_latent_realignment(last_hidden, agent_model.model)
            latent_embed = latent_vec.unsqueeze(1)  # [batch, 1, hidden_dim]
            
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            
            # Feed aligned embedding to model
            outputs = agent_model.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_dim]
        
        # Concatenate ONLY RAW hidden states: [batch, latent_steps, hidden_dim]
        if latent_steps > 0:
            raw_latent_hidden_states = torch.cat(raw_latent_hidden_list, dim=1)
        else:
            # No latent steps, return empty tensor
            batch_size = wrapped_ids.shape[0]
            hidden_dim = last_hidden.shape[-1]
            raw_latent_hidden_states = torch.zeros((batch_size, 0, hidden_dim), device=last_hidden.device, dtype=last_hidden.dtype)
        
        return past, raw_latent_hidden_states

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        current_model_name: Optional[str] = None  # Track which model owns current KV cache
        
        # NEW approach: Track prompts as TEXT and latent hidden states separately
        cumulative_prompts: List[str] = ["" for _ in range(batch_size)]  # Accumulate all agent prompts as text
        cumulative_latent_hiddens: Optional[torch.Tensor] = None  # Only RAW latent hidden states (not embeddings)
        
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        for agent_idx, agent in enumerate(self.agents):
            # NEW: Get model for this agent
            agent_model_name = self.agent_models[agent_idx]
            agent_model = self.models[agent_model_name]
            
            # NEW: Detect model switch
            model_switched = (current_model_name is not None and 
                            agent_model_name != current_model_name)
            
            if model_switched:
                print(f"\n[HYBRID] Model switch detected: {current_model_name} -> {agent_model_name}")
                print(f"[HYBRID] Transferring context via cross-model realignment...")
                print(f"[HYBRID] Cumulative latent hiddens shape: {cumulative_latent_hiddens.shape if cumulative_latent_hiddens is not None else None}")

            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]


            prompts, input_ids, attention_mask, tokens_batch = agent_model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                wrapped_encoded = agent_model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(agent_model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(agent_model.device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(agent_model.tokenizer.convert_ids_to_tokens(active_ids))

                # NEW: Handle model switching with clean text + latent separation
                if model_switched and cumulative_latent_hiddens is not None:
                    # Step 1: Re-encode all previous prompts with NEW model (no alignment needed!)
                    prev_model = self.models[current_model_name]
                    prompt_texts_batch = cumulative_prompts  # List of accumulated prompt text per item
                    
                    prompt_encoded = agent_model.tokenizer(
                        prompt_texts_batch,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    prompt_ids = prompt_encoded["input_ids"].to(agent_model.device)
                    prompt_mask = prompt_encoded["attention_mask"].to(agent_model.device)
                    
                    # Get native Model B embeddings for prompts
                    with torch.no_grad():
                        prompt_embeds = agent_model.model.get_input_embeddings()(prompt_ids)
                    
                    # Step 2: Transfer ONLY latent HIDDEN STATES via cross-model alignment
                    # This converts: hidden_A -> embedding_B (correct!)
                    transferred_latent_embeds = transfer_via_realignment(
                        cumulative_latent_hiddens, prev_model, agent_model
                    )
                    
                    # Step 3: Concatenate [prompt_embeds, transferred_latent_embeds]
                    combined_embeds = torch.cat([prompt_embeds, transferred_latent_embeds], dim=1)
                    combined_mask = torch.cat([
                        prompt_mask,
                        torch.ones((batch_size, transferred_latent_embeds.shape[1]), dtype=prompt_mask.dtype, device=prompt_mask.device)
                    ], dim=1)
                    
                    # Step 4: Feed combined embeddings through Model B to create KV cache
                    with torch.no_grad():
                        transfer_outputs = agent_model.model(
                            inputs_embeds=combined_embeds,
                            attention_mask=combined_mask,
                            past_key_values=None,
                            use_cache=True,
                            return_dict=True
                        )
                        transfer_past_kv = transfer_outputs.past_key_values
                    
                    # Step 5: Now generate NEW latent thoughts with Model B
                    past_kv, new_latent_hiddens = self._capture_hidden_states_from_model(
                        agent_model,
                        wrapped_ids,
                        wrapped_mask,
                        transfer_past_kv,
                        self.latent_steps
                    )
                    
                    # IMPORTANT: After model switch, RESET to new model's hidden states only
                    # The old context is already in the KV cache via transferred_latent_embeds
                    cumulative_latent_hiddens = new_latent_hiddens
                    current_model_name = agent_model_name
                    
                else:
                    # Same model or first agent: use KV cache directly
                    past_kv, new_latent_hiddens = self._capture_hidden_states_from_model(
                        agent_model,
                        wrapped_ids,
                        wrapped_mask,
                        past_kv,
                        self.latent_steps
                    )
                    
                    if cumulative_latent_hiddens is None:
                        cumulative_latent_hiddens = new_latent_hiddens
                    else:
                        cumulative_latent_hiddens = torch.cat([cumulative_latent_hiddens, new_latent_hiddens], dim=1)

                    if current_model_name is None:
                        current_model_name = agent_model_name
                
                # Update cumulative prompts (append current agent's prompt)
                for idx in range(batch_size):
                    cumulative_prompts[idx] += wrapped_prompts[idx]
                
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:
                # Judger agent - need to handle model switching here too
                if model_switched and cumulative_latent_hiddens is not None:
                    # Same clean approach: re-encode prompts + transfer latents
                    prev_model = self.models[current_model_name]
                    
                    prompt_encoded = agent_model.tokenizer(
                        cumulative_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    prompt_ids = prompt_encoded["input_ids"].to(agent_model.device)
                    prompt_mask = prompt_encoded["attention_mask"].to(agent_model.device)
                    
                    with torch.no_grad():
                        prompt_embeds = agent_model.model.get_input_embeddings()(prompt_ids)
                    
                    # Transfer hidden states -> embeddings (correct!)
                    transferred_latent_embeds = transfer_via_realignment(
                        cumulative_latent_hiddens, prev_model, agent_model
                    )
                    
                    combined_embeds = torch.cat([prompt_embeds, transferred_latent_embeds], dim=1)
                    combined_mask = torch.cat([
                        prompt_mask,
                        torch.ones((batch_size, transferred_latent_embeds.shape[1]), dtype=prompt_mask.dtype, device=prompt_mask.device)
                    ], dim=1)
                    
                    with torch.no_grad():
                        transfer_outputs = agent_model.model(
                            inputs_embeds=combined_embeds,
                            attention_mask=combined_mask,
                            past_key_values=None,
                            use_cache=True,
                            return_dict=True
                        )
                        past_for_decoding = transfer_outputs.past_key_values
                else:
                    past_for_decoding = past_kv if self.latent_steps > 0 else None

                if self.args.think:
                        judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    judger_prompts = prompts
                
                judger_encoded = agent_model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_ids = judger_encoded["input_ids"].to(agent_model.device)
                judger_mask = judger_encoded["attention_mask"].to(agent_model.device)
                judger_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(judger_ids, judger_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    judger_tokens_batch.append(agent_model.tokenizer.convert_ids_to_tokens(active_ids))
                generated_batch, _ = agent_model.generate_text_batch(
                    judger_ids,
                    judger_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                )
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = judger_mask[idx].bool()
                    trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": judger_tokens_batch[idx],
                            "output": final_text,
                        }
                    )

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                
                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')
                # print(f'=========================================')

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    if pred is None or pred == "":
                        ok = False
                        error_msg = f'Failed to extract answer from: {final_text[:100]}...'
                    else:
                        pred_int = int(pred)
                        gold_int = int(gold)
                        ok = (pred_int == gold_int)
                        error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None
            
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results
    
    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        embedding_record = []
        for agent in self.agents:
            
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
                
            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                # to wrap all latent thoughts from previous agents
                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                if self.latent_only:
                    if self.latent_steps > 0:
                        previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]
                
                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:
                
                # A stack of [B, L_i, H]
                past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)
                
                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ) 
                judger_encoded = judger_encoded["input_ids"].to(self.model.HF_device)
                # Get current prompt embedding
                curr_prompt_emb = self.model.embedding_layer(judger_encoded).squeeze(0).to(self.vllm_device)
                
                # assert Qwen model
                assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, "latent_embedding_position is only supported for Qwen models currently."

                # handle latent embedding insertion position    
                len_of_left = []
                for p in judger_prompts:
                    idx = p.find("<|im_start|>user\n")
                    # Get the text up to and including "<|im_start|>user\n"
                    left = p[: idx + len("<|im_start|>user\n")]
                    len_of_left.append(len(self.model.tokenizer(left)['input_ids']))
                    
                B, L, H = curr_prompt_emb.shape
                _, Lp, H = past_embedding.shape  # assume shape consistency
                    
                whole_prompt_emb_list = []
                for i in range(B):
                    insert_idx = len_of_left[i]
                    left_emb = curr_prompt_emb[i, :insert_idx, :]
                    right_emb = curr_prompt_emb[i, insert_idx:, :]
                    combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
                    whole_prompt_emb_list.append(combined)

                # Pad back to max length if needed
                max_len = max(x.shape[0] for x in whole_prompt_emb_list)
                whole_prompt_emb = torch.stack([
                    torch.cat([x, torch.zeros(max_len - x.shape[0], H, device=x.device)], dim=0)
                    for x in whole_prompt_emb_list
                ])

                # else:
                    # Get full prompt embedding from cat with previous ones 
                    # B L H B L H
                    # whole_prompt_emb = torch.cat([past_embedding, curr_prompt_emb], dim=1)
                
                # pdb.set_trace()              
                
                # Use vLLM 
                prompt_embeds_list = [
                    {
                        "prompt_embeds": embeds
                    } for embeds in whole_prompt_emb 
                ]
                
                
                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    self.sampling_params,
                )

                generated_texts = [out.outputs[0].text.strip() for out in outputs]
                    
                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "output": text_out,
                        }
                    )


        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            pred = normalize_answer(extract_gsm8k_answer(final_text))
            gold = item["gold"]
            ok = (pred == gold) if (pred and gold) else False
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]