import copy
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from securesd.engine.cpu_engine import CPULLMEngine
from securesd.sampling_params import SamplingParams


class TinyTokenizer:
    eos_token_id = 31

    def get_vocab(self):
        return {str(i): i for i in range(32)}

    def decode(self, ids):
        return " ".join(map(str, ids))


class CPUEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        config = Qwen3Config(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, max_position_embeddings=128, eos_token_id=31,
        )
        torch.manual_seed(10)
        cls.target = Qwen3ForCausalLM(config).eval()
        torch.manual_seed(20)
        cls.draft = Qwen3ForCausalLM(config).eval()
        cls.prompt = [1, 2, 3]

    def engine(self, method, eta_len=0, epsilon=0.0, **kwargs):
        models = {"target": self.target, "draft": self.draft}
        with patch("securesd.engine.cpu_engine.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()), \
                patch.object(CPULLMEngine, "_load", side_effect=lambda p, d: copy.deepcopy(models[p]).to(d)):
            return CPULLMEngine(
                "target", "draft", method, speculate_k=3, cpu_dtype="float32", cpu_threads=1,
                max_model_len=128, eta_len=eta_len,
                method_params={"lossy_epsilon": epsilon} if method == "lossy_sd" else {}, **kwargs,
            )

    @torch.inference_mode()
    def reference(self, model, count):
        # Recompute the full prefix: this reference does not share the backend's KV cache logic.
        tokens = list(self.prompt)
        for _ in range(count):
            logits = model(torch.tensor([tokens]), use_cache=False).logits[:, -1, :]
            tokens.append(int(logits.argmax(-1)))
        return tokens[len(self.prompt):]

    def run_engine(self, engine, count=12, **kwargs):
        return engine.generate(
            [self.prompt], SamplingParams(max_new_tokens=count, ignore_eos=True, **kwargs), use_tqdm=False,
        )

    def test_ar_and_sd_match_full_prefix_target_after_cache_rollbacks(self):
        expected = self.reference(self.target, 12)
        ar, _ = self.run_engine(self.engine("ar_target"))
        sd, metrics = self.run_engine(self.engine("sd"))
        self.assertEqual(ar[0]["token_ids"], expected)
        self.assertEqual(sd[0]["token_ids"], expected)
        self.assertTrue(any(r["accepted_draft_tokens"] < 3 for r in metrics["cpu_rounds"]))
        self.assertEqual(sum(metrics["decode_step_tokens"]), 12)

    def test_relaxed_accept_all_uses_draft_and_obeys_output_limit(self):
        output, metrics = self.run_engine(self.engine("lossy_sd", epsilon=1.0), count=8)
        self.assertEqual(output[0]["token_ids"][:3], self.reference(self.draft, 3))
        self.assertEqual(len(output[0]["token_ids"]), 8)
        self.assertTrue(all(r["accepted_draft_tokens"] == 3 for r in metrics["cpu_rounds"]))

    def test_first_token_correction_and_full_correction(self):
        output, metrics = self.run_engine(self.engine("lossy_sd", eta_len=1, epsilon=1.0))
        self.assertEqual(metrics["cpu_rounds"][0]["eta"], [0.0, 1.0, 1.0])
        self.assertEqual(output[0]["token_ids"][0], self.reference(self.target, 1)[0])
        corrected, _ = self.run_engine(self.engine("lossy_sd", eta_len=128, epsilon=1.0))
        self.assertEqual(corrected[0]["token_ids"], self.reference(self.target, 12))

    def test_requests_do_not_share_cache_and_sampling_finishes(self):
        engine = self.engine("sd")
        outputs, _ = engine.generate(
            [self.prompt, self.prompt], SamplingParams(max_new_tokens=8, ignore_eos=True), use_tqdm=False,
        )
        self.assertEqual(outputs[0]["token_ids"], outputs[1]["token_ids"])
        sampled, _ = self.run_engine(self.engine("sd"), temperature=0.7, draft_temperature=0.7)
        self.assertEqual(len(sampled[0]["token_ids"]), 12)

    def test_eos_stops_generation_and_max_steps_raises(self):
        engine = self.engine("sd")
        engine.eos = self.reference(self.target, 1)[0]
        outputs, _ = engine.generate([self.prompt], SamplingParams(max_new_tokens=12), use_tqdm=False)
        self.assertEqual(outputs[0]["token_ids"], [engine.eos])
        with self.assertRaisesRegex(RuntimeError, "max_steps"):
            self.run_engine(self.engine("sd", max_steps=1))


if __name__ == "__main__":
    unittest.main()
