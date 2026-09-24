"""lm-evaluation-harness adapter used for ANKA-50M RMW3."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from lm_eval import utils as lm_eval_utils
from lm_eval.api.model import TemplateLM
from tqdm.auto import tqdm


class ANKAHarnessLM(TemplateLM):
    def __init__(self, model, tokenizer, batch_size: int = 16):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._batch_size = int(batch_size)
        self._max_length = 1024
        self._max_gen_toks = 128
        self._device = next(model.parameters()).device
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2

    @property
    def eot_token_id(self):
        return self.eos_id

    @property
    def prefix_token_id(self):
        return self.bos_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    @property
    def tokenizer_name(self):
        return "anka_english_16384_v1"

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        if not string:
            return []
        return self.tokenizer.encode(string, add_special_tokens=False).ids

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _forward_logits(self, input_ids):
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                output = self.model(input_ids)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        if logits.ndim != 3 or logits.shape[-1] != 16_384:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
        return logits

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        results = [None] * len(requests)
        prepared = []
        for request_index, (_, context, continuation) in enumerate(requests):
            context = list(context) or [self.prefix_token_id]
            continuation = list(continuation)
            if not continuation:
                results[request_index] = (0.0, True)
                continue
            if len(continuation) > self.max_length:
                raise ValueError("Continuation exceeds the 1,024-token context")
            keep = self.max_length + 1 - len(continuation)
            context = context[-keep:] or [self.prefix_token_id]
            sequence = context + continuation
            prepared.append(
                (request_index, sequence[:-1], continuation, len(context) - 1)
            )

        prepared.sort(key=lambda item: len(item[1]), reverse=True)
        starts = range(0, len(prepared), self.batch_size)
        if not disable_tqdm:
            starts = tqdm(
                starts,
                total=math.ceil(len(prepared) / self.batch_size) if prepared else 0,
                desc="ANKA likelihood",
            )

        for start in starts:
            items = prepared[start : start + self.batch_size]
            longest = max(len(item[1]) for item in items)
            inputs = torch.full(
                (len(items), longest), self.pad_id, device=self.device, dtype=torch.long
            )
            for row, (_, input_tokens, _, _) in enumerate(items):
                inputs[row, : len(input_tokens)] = torch.tensor(
                    input_tokens, device=self.device
                )
            logits = self._forward_logits(inputs)
            for row, (request_index, _, targets, target_start) in enumerate(items):
                selected = logits[
                    row, target_start : target_start + len(targets)
                ].float()
                target_tensor = torch.tensor(targets, device=self.device)
                log_probs = F.log_softmax(selected, -1).gather(
                    -1, target_tensor.unsqueeze(-1)
                ).squeeze(-1)
                results[request_index] = (
                    float(log_probs.sum()),
                    bool(selected.argmax(-1).eq(target_tensor).all()),
                )
        if any(result is None for result in results):
            raise RuntimeError("Missing likelihood result")
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        output = []
        iterator = requests if disable_tqdm else tqdm(requests, desc="ANKA rolling")
        for request in iterator:
            text = request.args[0]
            token_ids = self.tok_encode(text)
            if not token_ids:
                output.append(0.0)
                continue
            windows = map(
                lm_eval_utils.make_disjoint_window,
                lm_eval_utils.get_rolling_token_windows(
                    token_list=token_ids,
                    prefix_token=self.prefix_token_id,
                    max_seq_len=self.max_length - 1,
                    context_len=1,
                ),
            )
            scores = self._loglikelihood_tokens(
                [(None, context, continuation) for context, continuation in windows],
                disable_tqdm=True,
            )
            value = float(sum(score[0] for score in scores))
            output.append(value)
            self.cache_hook.add_partial("loglikelihood_rolling", (text,), value)
        return output

    def generate_until(self, requests, disable_tqdm=False):
        outputs = []
        iterator = requests if disable_tqdm else tqdm(requests, desc="ANKA generation")
        for request in iterator:
            prompt, kwargs = request.args
            stop = kwargs.get("until", [])
            stop = [stop] if isinstance(stop, str) else list(stop)
            prompt_ids = self.tok_encode(prompt) or [self.prefix_token_id]
            generated = []
            positions = []
            for _ in range(min(int(kwargs.get("max_gen_toks", 128)), 128)):
                active = (prompt_ids + generated)[-self.max_length :]
                inputs = torch.tensor([active], device=self.device)
                next_id = int(self._forward_logits(inputs)[0, -1].argmax())
                if next_id == self.eos_id:
                    break
                generated.append(next_id)
                decoded = self.tok_decode(generated)
                positions = [decoded.find(value) for value in stop if value in decoded]
                if positions:
                    decoded = decoded[: min(positions)]
                    break
            outputs.append(self.tok_decode(generated) if not positions else decoded)
        return outputs
