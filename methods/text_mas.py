from typing import Dict, List, Optional

from . import default_agents
from models import ModelWrapper
from prompts import build_agent_messages_hierarchical_text_mas, build_agent_messages_sequential_text_mas
import re
import time
from utils import (
    extract_answer_with_meta,
    get_stop_regexes,
    normalize_answer,
    extract_markdown_python_block,
    run_with_timeout,
)
import argparse


class TextMASMethod:
    def __init__(
        self,
        model: Optional[ModelWrapper],
        models: Optional[List[ModelWrapper]] = None,
        role_model_map: Optional[Dict[str, int]] = None,
        *,
        max_new_tokens_each: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.model = model
        self.max_new_tokens_each = max_new_tokens_each
        self.max_new_tokens_judger = max_new_tokens_each
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.args = args
        self.method_name = "text_mas"
        self.task = args.task
        self.total_infer_time_sec = 0.0
        self.total_infer_batches = 0
        self.total_infer_items = 0

        if models is not None and models:
            self.models = models
        elif model is not None:
            self.models = [model]
        else:
            raise ValueError("TextMASMethod requires either model or models.")

        self.role_model_map = self._normalize_role_map(role_model_map)
        self.agent_model_indices = self._assign_models_to_agents()

    def _normalize_role_map(self, role_model_map: Optional[Dict[str, int]]) -> Dict[str, int]:
        if not role_model_map:
            return {}
        normalized = {str(k).lower(): int(v) for k, v in role_model_map.items()}
        if "solver" in normalized and "judger" not in normalized:
            normalized["judger"] = normalized["solver"]
        return normalized

    def _assign_models_to_agents(self) -> List[int]:
        mapping = []
        num_models = len(self.models)
        for idx, agent in enumerate(self.agents):
            role = agent.role
            if role in self.role_model_map:
                model_idx = self.role_model_map[role]
            else:
                model_idx = idx % num_models
            if model_idx < 0 or model_idx >= num_models:
                raise ValueError(f"role_model_map index {model_idx} out of range for role {role}.")
            mapping.append(model_idx)
        mapping_desc = ", ".join(
            f"{agent.role}->{self.models[m_idx].model_name}" for agent, m_idx in zip(self.agents, mapping)
        )
        print(f"[text_mas] agent-to-model mapping: {mapping_desc}")
        return mapping

    def _truncate_ctx(self, ctx: str) -> str:
        limit = int(getattr(self.args, "text_mas_context_length", -1) or -1)
        if limit is None or limit <= 0:
            return ctx
        return ctx[:limit]

    def run_batch(self, items: List[Dict]) -> List[Dict]:
        infer_start = time.time()
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        contexts = ["" for _ in range(batch_size)]
        history_contexts = ["" for _ in range(batch_size)]
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        judger_meta = None
        stop_regexes = get_stop_regexes(self.task)

        for agent_idx, agent in enumerate(self.agents):
            model_idx = self.agent_model_indices[agent_idx]
            wrapper = self.models[model_idx]

                                                
            safe_contexts = [self._truncate_ctx(c) for c in contexts]

            if self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_messages_hierarchical_text_mas(
                        role=agent.role,
                        question=item["question"],
                        context=safe_contexts[idx],
                        method=self.method_name,
                        args=self.args,
                        model_name=wrapper.model_name,
                    )
                    for idx, item in enumerate(items)
                ]
            else:
                batch_messages = [
                    build_agent_messages_sequential_text_mas(
                        role=agent.role,
                        question=item["question"],
                        context=safe_contexts[idx],
                        method=self.method_name,
                        args=self.args,
                        model_name=wrapper.model_name,
                    )
                    for idx, item in enumerate(items)
                ]

            prompts, input_ids, attention_mask, tokens_batch = wrapper.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            gen_meta = None
            if wrapper.use_vllm:
                generated_texts = wrapper.vllm_generate_text_batch(
                    prompts,
                    max_new_tokens=self.max_new_tokens_each,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            else:
                if agent.role == "judger" and stop_regexes:
                    generated_texts, _, gen_meta = wrapper.generate_text_batch(
                        input_ids,
                        attention_mask,
                        max_new_tokens=self.max_new_tokens_each,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        stop_regexes=stop_regexes,
                        return_metadata=True,
                    )
                else:
                    generated_texts, _ = wrapper.generate_text_batch(
                        input_ids,
                        attention_mask,
                        max_new_tokens=self.max_new_tokens_each,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
            if agent.role == "judger":
                judger_meta = gen_meta

            agent_name_map_for_prompt_hierarchical = {
                "Planner": "Math Agent",
                "Critic": "Science Agent",
                "Refiner": "Code Agent",
                "Judger": "Task Summrizer",
                "planner": "Math Agent",
                "critic": "Science Agent",
                "refiner": "Code Agent",
                "judger": "Task Summrizer",
            }

            for idx in range(batch_size):
                text_out = generated_texts[idx].strip()

                if self.args.prompt == "hierarchical":
                    formatted_output = f"[{agent_name_map_for_prompt_hierarchical[agent.name]}]:\n{text_out}\n\n"
                else:
                    formatted_output = f"[{agent.name}]:\n{text_out}\n\n"

                if agent.role != "judger":
                    contexts[idx] = f"{contexts[idx]}{formatted_output}"
                    history_contexts[idx] = f"{history_contexts[idx]}{formatted_output}"
                else:
                    final_texts[idx] = text_out

                mask = attention_mask[idx].bool()
                trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
                agent_traces[idx].append(
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "model_name": wrapper.model_name,
                        "model_idx": model_idx,
                        "input": prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": tokens_batch[idx],
                        "output": text_out,
                    }
                )

        infer_end = time.time()
        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]

            if self.task in ["mbppplus", "humanevalplus"]:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                extraction_meta = {"extraction_source": None, "extracted_span": None}
            else:
                pred_raw, extraction_meta = extract_answer_with_meta(final_text, self.task)
                pred = normalize_answer(pred_raw)
                gold = str(item.get("gold", "")).strip()
                error_msg = None

            if self.task in ["aime2024", "aime2025"]:
                try:
                    pred_val = float(pred) if pred is not None else None
                    pred_int = int(pred_val) if pred_val is not None else None
                    gold_int = int(gold)
                    ok = pred_int is not None and abs(pred_val - pred_int) < 1e-6 and pred_int == gold_int
                except (ValueError, TypeError):
                    ok = False
                    error_msg = f"Value error in parsing answer. Pred: {pred}, Gold: {gold}"
            elif self.task not in ["mbppplus", "humanevalplus"]:
                ok = (pred == gold) if (pred and gold) else False

            stop_reason = None
            stop_span = None
            generated_tokens = None
            if judger_meta:
                stop_reason = judger_meta["stop_reasons"][idx]
                stop_span = judger_meta["stop_spans"][idx]
                generated_tokens = judger_meta["generated_tokens"][idx]
            elif stop_regexes:
                for pattern in stop_regexes:
                    match = re.search(pattern, final_text, re.MULTILINE)
                    if match:
                        stop_reason = "stop_pattern"
                        stop_span = match.group(0)
                        break
                if stop_reason is None:
                    stop_reason = "unknown"

            results.append(
                {
                    "question": item["question"],
                    "gold": gold if self.task not in ["mbppplus", "humanevalplus"] else item.get("gold", ""),
                    "solution": item["solution"],
                    "context": history_contexts[idx],
                    "prediction": pred if self.task not in ["mbppplus", "humanevalplus"] else pred,
                    "raw_prediction": final_text,
                    "extraction_source": extraction_meta.get("extraction_source") if self.task not in ["mbppplus", "humanevalplus"] else None,
                    "extracted_span": extraction_meta.get("extracted_span") if self.task not in ["mbppplus", "humanevalplus"] else None,
                    "stop_reason": stop_reason,
                    "stop_span": stop_span,
                    "generated_tokens": generated_tokens,
                    "agents": agent_traces[idx],
                    "correct": ok,
                    "error": error_msg,
                }
            )
        infer_total = infer_end - infer_start
        self.total_infer_time_sec += infer_total
        self.total_infer_batches += 1
        self.total_infer_items += len(items)
        per_item = infer_total / max(1, len(items))
        avg_all = self.total_infer_time_sec / max(1, self.total_infer_items)
        print(
            f"[text_mas] batch_infer_time_sec={infer_total:.4f} per_item_sec={per_item:.4f} batch_size={len(items)} "
            f"total_infer_time_sec={self.total_infer_time_sec:.4f} avg_per_item_sec={avg_all:.4f}"
        )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
