"""Single-request Transformers CPU engine using the upstream verifier implementations."""
from __future__ import annotations

import gc
from time import perf_counter

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from securesd.methods import get_spec, is_sd_method
from securesd.sampling_params import SamplingParams
from securesd.verifier import VerifierMethodRequest, run_verifier_method
from securesd.verifier.eta import compute_eta


class _State:
    def __init__(self, model, prompt):
        self.model = model
        self.cache = None
        self.next_logits = None
        self.advance(prompt)

    def advance(self, tokens):
        output = self.model(
            input_ids=torch.tensor([tokens], dtype=torch.long),
            past_key_values=self.cache, use_cache=True,
            logits_to_keep=1 if self.cache is None else len(tokens),
        )
        self.cache = output.past_key_values
        logits = output.logits.float()
        self.next_logits = logits[:, -1, :]
        return logits

    def crop(self, length):
        self.cache.crop(length)


def _new_metrics():
    return {
        "accepted_suffix_lens_with_recovery": [],
        "prefill_total_time": 0.0, "decode_total_time": 0.0,
        "prefill_total_tokens": 0, "decode_total_tokens": 0,
        "decode_step_times": [], "decode_step_start_offsets_s": [],
        "decode_step_end_offsets_s": [], "decode_step_tokens": [],
        "decode_step_scheduled": [], "decode_step_seq_deltas": [],
        "per_seq_decode": [], "target_step_times": [], "target_verify_times": [],
        "cpu_rounds": [],
    }


class CPULLMEngine:
    def __init__(
        self, model, draft, method, speculate_k=5, method_params=None,
        eta_schedule="step", eta_start=0.0, eta_end=1.0, eta_len=2.0, eta_gamma=1.0,
        max_num_seqs=1, max_model_len=2048, max_steps=None,
        cpu_dtype="bfloat16", cpu_threads=4, seed=0, **_kwargs,
    ):
        if max_num_seqs != 1:
            raise ValueError("CPU backend supports --max_num_seqs 1 only")
        if speculate_k < 1 or cpu_threads < 1 or max_model_len < 1:
            raise ValueError("speculate_k, cpu_threads and max_model_len must be positive")
        if cpu_dtype not in ("float32", "bfloat16"):
            raise ValueError("CPU dtype must be float32 or bfloat16")
        if not 0 <= eta_start <= 1 or not 0 <= eta_end <= 1 or eta_len < 0:
            raise ValueError("eta start/end must be in [0, 1] and eta_len must be nonnegative")
        # Validate schedules even if generation is purely autoregressive.
        compute_eta(eta_schedule, eta_start, eta_end, eta_len, torch.tensor([0]), eta_gamma)
        self.method = method
        self.speculate = is_sd_method(method)
        self.params = method_params or {}
        get_spec(method).validate(self.params)
        self.k = speculate_k
        self.eta_args = (eta_schedule, eta_start, eta_end, eta_len)
        self.eta_gamma = eta_gamma
        self.max_model_len = max_model_len
        self.max_steps = max_steps
        torch.set_num_threads(cpu_threads)
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.eos = self.tokenizer.eos_token_id
        dtype = getattr(torch, cpu_dtype)
        self.target = self.draft = None
        if method != "ar_draft":
            self.target = self._load(model, dtype)
        if self.speculate or method == "ar_draft":
            self.draft = self._load(draft, dtype)
        if self.speculate:
            draft_tokenizer = AutoTokenizer.from_pretrained(draft, local_files_only=True)
            if (self.tokenizer.get_vocab() != draft_tokenizer.get_vocab()
                    or self.tokenizer.eos_token_id != draft_tokenizer.eos_token_id):
                raise ValueError("Target and draft must use the same token-to-id mapping and EOS")
            if self.target.config.vocab_size != self.draft.config.vocab_size:
                raise ValueError("Target and draft output vocab sizes must match")
        for active in (self.target, self.draft):
            if active is not None and max_model_len > active.config.max_position_embeddings:
                raise ValueError("max_model_len exceeds the model context limit")
        self.metrics = _new_metrics()

    @staticmethod
    def _load(path, dtype):
        print(f"[CPU] Loading {path} as {dtype}", flush=True)
        return AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, device_map="cpu", attn_implementation="sdpa",
            local_files_only=True,
        ).eval()

    @staticmethod
    def _sample(logits, temperature):
        if temperature == 0:
            return int(logits.argmax(dim=-1).item())
        return int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).item())

    def _speculate(self, target, draft, prefix, generated, sampling, seq_id):
        base = len(prefix) + len(generated)
        first_target_logits = target.next_logits
        proposed, draft_logits = [], []
        for _ in range(self.k):
            draft_logits.append(draft.next_logits)
            token = self._sample(draft.next_logits, sampling.draft_temperature)
            proposed.append(token)
            draft.advance([token])
        verify_started = perf_counter()
        target_logits = torch.cat([first_target_logits.unsqueeze(1), target.advance(proposed)], dim=1)
        self.metrics["target_verify_times"].append(perf_counter() - verify_started)
        # Positions are zero-based: L=1 strictly verifies the first output token.
        positions = torch.arange(len(generated), len(generated) + self.k + 1).unsqueeze(0)
        eta = compute_eta(*self.eta_args, positions, gamma=self.eta_gamma)
        result = run_verifier_method(VerifierMethodRequest(
            method=self.method, logits_p=target_logits,
            logits_q=torch.stack(draft_logits, dim=1),
            speculations=torch.tensor([[(prefix + generated)[-1]] + proposed]),
            temperatures_target=torch.tensor([sampling.temperature]),
            temperatures_draft=torch.tensor([sampling.draft_temperature]),
            params=self.params, logits_q_next=draft.next_logits,
            eta=eta, eta_bypass=float(eta[0, 0]) if bool((eta == eta[0, 0]).all()) else None,
        ))
        accepted = result.new_suffixes[0][1:]
        recovery = result.recovery_tokens[0]
        self.metrics["accepted_suffix_lens_with_recovery"].append(len(accepted) + 1)
        self.metrics["cpu_rounds"].append({
            "seq_id": seq_id, "output_position": len(generated),
            "proposed_token_ids": proposed, "eta": eta[0, :self.k].tolist(),
            "accepted_draft_tokens": len(accepted), "recovery_token_id": recovery,
        })
        committed = self._truncate(accepted + [recovery], generated, sampling)
        if not self._finished(generated + committed, sampling):
            # Discard rejected proposals before processing the target recovery token.
            target.crop(base + len(accepted))
            draft.crop(base + len(accepted))
            target.advance([recovery])
            draft.advance([recovery])
        return committed

    def _truncate(self, tokens, generated, sampling):
        tokens = tokens[:sampling.max_new_tokens - len(generated)]
        if not sampling.ignore_eos and self.eos in tokens:
            tokens = tokens[:tokens.index(self.eos) + 1]
        return tokens

    def _finished(self, tokens, sampling):
        return len(tokens) >= sampling.max_new_tokens or (
            bool(tokens) and not sampling.ignore_eos and tokens[-1] == self.eos
        )

    @torch.inference_mode()
    def generate(self, prompts, sampling_params, use_tqdm=True):
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        if len(prompts) != len(sampling_params):
            raise ValueError("prompts and sampling_params must have the same length")
        self.metrics = _new_metrics()
        outputs = []
        started = perf_counter()
        iterator = tqdm(list(zip(prompts, sampling_params)), desc="Generating (CPU)", disable=not use_tqdm)
        for seq_id, (prompt, sampling) in enumerate(iterator):
            if isinstance(prompt, str):
                prompt = self.tokenizer.encode(prompt)
            if not prompt or sampling.max_new_tokens < 1:
                raise ValueError("Prompt and max_new_tokens must be nonempty/positive")
            if len(prompt) + sampling.max_new_tokens + (self.k if self.speculate else 0) > self.max_model_len:
                raise ValueError("Prompt + output + speculative lookahead exceeds --max_model_len")
            seq_started = perf_counter()
            if self.speculate:
                target, draft = _State(self.target, prompt), _State(self.draft, prompt)
                active = None
            else:
                target = draft = None
                active = _State(self.draft if self.method == "ar_draft" else self.target, prompt)
            self.metrics["prefill_total_time"] += perf_counter() - seq_started
            self.metrics["prefill_total_tokens"] += len(prompt)
            generated, steps, first = [], 0, None
            while not self._finished(generated, sampling):
                if self.max_steps is not None and steps >= self.max_steps:
                    raise RuntimeError("Generation stopped before completion due to --max_steps")
                steps += 1
                t = perf_counter()
                if self.speculate:
                    committed = self._speculate(target, draft, prompt, generated, sampling, seq_id)
                else:
                    temperature = sampling.draft_temperature if self.method == "ar_draft" else sampling.temperature
                    committed = [self._sample(active.next_logits, temperature)]
                    if not self._finished(generated + committed, sampling):
                        active.advance(committed)
                generated.extend(committed)
                end = perf_counter()
                first = end if first is None else first
                elapsed = end - t
                self.metrics["decode_step_times"].append(elapsed)
                self.metrics["decode_step_start_offsets_s"].append(t - started)
                self.metrics["decode_step_end_offsets_s"].append(end - started)
                self.metrics["decode_step_tokens"].append(len(committed))
                self.metrics["decode_step_scheduled"].append(1)
                self.metrics["decode_step_seq_deltas"].append([(seq_id, len(committed))])
                self.metrics["decode_total_time"] += elapsed
                self.metrics["decode_total_tokens"] += len(committed)
                self.metrics["target_step_times"].append(elapsed)
            end = perf_counter()
            decode_time = end - first
            self.metrics["per_seq_decode"].append({
                "seq_id": seq_id, "ttft_s": first - seq_started,
                "e2e_latency_s": end - seq_started, "decode_tokens": len(generated),
                "tpot_s": decode_time / max(len(generated) - 1, 1),
                "decode_tps": max(len(generated) - 1, 0) / max(decode_time, 1e-12),
                "full_span_tps": len(generated) / max(end - seq_started, 1e-12),
            })
            outputs.append({"text": self.tokenizer.decode(generated), "token_ids": generated})
            del target, draft, active
        return outputs, self.metrics

    def exit(self, hard=False):
        self.target = self.draft = None
        gc.collect()
