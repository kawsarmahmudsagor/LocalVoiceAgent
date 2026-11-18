"""Runs Qwen3.5-9B in process with transformers + bitsandbytes 4-bit. See spec §4.2.

Imported only when LLM_BACKEND=transformers (see factory.py): importing this
module loads torch and transformers.
"""
from __future__ import annotations

import threading
from typing import Callable, Iterator

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from voicebot.config import Settings
from voicebot.llm.base import GenerationParams, LLMError

OOM_HINT = "CUDA out of memory loading the model: set STT_DEVICE=cpu or use LLM_BACKEND=llamacpp"


class StopOnEvent(StoppingCriteria):
    def __init__(self, event: threading.Event) -> None:
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.event.is_set()


def build_inputs(tokenizer, messages: list[dict]):
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                         return_dict=True, return_tensors="pt")


class TransformersBackend:
    name = "transformers"

    def __init__(self, settings: Settings, *, loader: Callable[[], tuple] | None = None) -> None:
        self.settings = settings
        self.model_label = f"{settings.hf_model_id} (4-bit)" if settings.hf_load_in_4bit else settings.hf_model_id
        self._loader = loader
        self._model = None
        self._tokenizer = None

    def start(self) -> None:
        try:
            self._model, self._tokenizer = (self._loader or self._default_loader)()
        except torch.cuda.OutOfMemoryError as e:
            raise LLMError(OOM_HINT) from e

    def _default_loader(self):
        if not torch.cuda.is_available():
            raise LLMError("CUDA is not available: the transformers backend needs an NVIDIA GPU with CUDA torch")
        try:
            import bitsandbytes  # noqa: F401
        except Exception as e:  # ImportError or a broken CUDA setup inside bitsandbytes
            raise LLMError(f"bitsandbytes failed to import ({e}); it is required for HF_LOAD_IN_4BIT") from e
        # Text-only load: Qwen3.5's "qwen3_5" config maps to Qwen3_5ForCausalLM, which
        # skips the vision encoder (less VRAM) and needs no Pillow/torchvision.
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        s = self.settings
        kwargs: dict = {"device_map": "cuda:0"}
        if s.hf_load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            kwargs["dtype"] = torch.bfloat16
        tokenizer = AutoTokenizer.from_pretrained(s.hf_model_id)
        model = AutoModelForCausalLM.from_pretrained(s.hf_model_id, **kwargs)
        model.eval()
        return model, tokenizer

    def stream(self, messages: list[dict], params: GenerationParams,
               stop_event: threading.Event) -> Iterator[str]:
        if self._model is None:
            raise LLMError("transformers backend used before start()")
        model, tok = self._model, self._tokenizer
        inputs = build_inputs(tok, messages).to(model.device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        error: list[BaseException] = []
        abort = threading.Event()  # set if the consumer closes this generator early

        def run() -> None:
            try:
                with torch.inference_mode():
                    model.generate(**inputs, streamer=streamer, max_new_tokens=params.max_tokens, do_sample=True,
                                   temperature=params.temperature, top_p=params.top_p,
                                   stopping_criteria=StoppingCriteriaList([StopOnEvent(stop_event), StopOnEvent(abort)]))
            except BaseException as e:  # surfaced to the caller below
                error.append(e)
            finally:
                streamer.end()

        worker = threading.Thread(target=run, name="hf-generate", daemon=True)
        worker.start()
        try:
            for text in streamer:
                if stop_event.is_set():
                    break
                if text:
                    yield text
        finally:
            abort.set()
            worker.join()
        if error:
            raise LLMError(str(error[0])) from error[0]

    def close(self) -> None:
        self._model = self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
