import os
import re
import json

from yandex_client import yandex_complete

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None


RETRIEVAL_TOKENS = ["[Retrieval]", "[No Retrieval]", "[Continue to Use Evidence]"]
RELEVANCE_TOKENS = ["[Relevant]", "[Irrelevant]"]
GROUNDNESS_TOKENS = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
UTILITY_TOKENS = ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]


class Critic:
    """Critic abstraction.

    Modes:
    - teacher: use YandexGPT prompts to emulate critic labels.
    - local: use fine-tuned local critic model path from CRITIC_MODEL_PATH.
    """

    def __init__(self):
        self.mode = os.getenv("CRITIC_MODE", "teacher").strip().lower()
        self.model_path = os.getenv("CRITIC_MODEL_PATH", "").strip()
        self.max_new_tokens = int(os.getenv("CRITIC_MAX_NEW_TOKENS", "48"))
        self.device = "cpu"

        self._tokenizer = None
        self._model = None

        if self.mode == "local":
            self._load_local_model()

    def _load_local_model(self):
        if not self.model_path:
            raise RuntimeError("CRITIC_MODE=local requires CRITIC_MODEL_PATH")
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers/torch are required for local critic mode")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        self._model.to(self.device)
        self._model.eval()

    @staticmethod
    def _extract_token(raw: str, valid_tokens: list[str], default_token: str) -> str:
        for token in valid_tokens:
            if token in raw:
                return token

        # Accept minor formatting issues from teacher LLM, e.g., Retrieval (without brackets)
        cleaned = raw.strip().lower()
        fallback_map = {t.strip("[]").lower(): t for t in valid_tokens}
        for key, token in fallback_map.items():
            if re.search(rf"\b{re.escape(key)}\b", cleaned):
                return token

        return default_token

    def _run_local(self, instruction: str, input_text: str, valid_tokens: list[str], default_token: str) -> str:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        text = self._tokenizer.decode(out[0], skip_special_tokens=False)
        generated = text[len(prompt):]
        return self._extract_token(generated, valid_tokens, default_token)

    def _run_teacher(self, instruction: str, input_text: str, valid_tokens: list[str], default_token: str) -> str:
        prompt = (
            "Return exactly one token from the allowed tokens list. No explanation.\n\n"
            f"Allowed tokens: {', '.join(valid_tokens)}\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"Input:\n{input_text}\n\n"
            "Output token:"
        )
        raw = yandex_complete(prompt, temperature=0.0, max_tokens=64)
        return self._extract_token(raw, valid_tokens, default_token)

    @staticmethod
    def _extract_json_object(raw: str) -> dict | None:
        raw = (raw or "").strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
        return None

    def _run_teacher_distribution(
        self,
        instruction: str,
        input_text: str,
        valid_tokens: list[str],
        default_token: str,
    ) -> dict[str, float]:
        prompt = (
            "Return JSON only. No markdown.\n"
            "For each allowed token, output a probability between 0 and 1.\n"
            "Probabilities must sum to 1.\n\n"
            f"Allowed tokens: {', '.join(valid_tokens)}\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"Input:\n{input_text}\n\n"
            'JSON format example: {"[TokenA]": 0.6, "[TokenB]": 0.4}'
        )
        raw = yandex_complete(prompt, temperature=0.0, max_tokens=240)
        parsed = self._extract_json_object(raw) or {}

        dist: dict[str, float] = {}
        for tok in valid_tokens:
            val = parsed.get(tok, 0.0)
            try:
                dist[tok] = float(val)
            except Exception:
                dist[tok] = 0.0

        total = sum(max(v, 0.0) for v in dist.values())
        if total <= 0:
            chosen = self._run_teacher(instruction, input_text, valid_tokens, default_token)
            return {tok: (1.0 if tok == chosen else 0.0) for tok in valid_tokens}

        return {tok: max(dist[tok], 0.0) / total for tok in valid_tokens}

    def _predict(self, instruction: str, input_text: str, valid_tokens: list[str], default_token: str) -> str:
        if self.mode == "local":
            return self._run_local(instruction, input_text, valid_tokens, default_token)
        return self._run_teacher(instruction, input_text, valid_tokens, default_token)

    def _predict_distribution(
        self,
        instruction: str,
        input_text: str,
        valid_tokens: list[str],
        default_token: str,
    ) -> dict[str, float]:
        if self.mode == "local":
            chosen = self._run_local(instruction, input_text, valid_tokens, default_token)
            return {tok: (1.0 if tok == chosen else 0.0) for tok in valid_tokens}
        return self._run_teacher_distribution(instruction, input_text, valid_tokens, default_token)

    def need_retrieval(self, question: str) -> str:
        instruction = (
            "Judge whether external retrieval is needed to answer this question faithfully. "
            "Output one token."
        )
        return self._predict(instruction, f"Question: {question}", RETRIEVAL_TOKENS, "[Retrieval]")

    def retrieval_distribution(self, question: str) -> dict[str, float]:
        instruction = (
            "Judge whether external retrieval is needed to answer this question faithfully. "
            "Output probability distribution over allowed tokens."
        )
        return self._predict_distribution(instruction, f"Question: {question}", RETRIEVAL_TOKENS, "[Retrieval]")

    def relevance(self, question: str, evidence: str) -> str:
        instruction = "Judge if evidence is relevant for answering the question. Output one token."
        input_text = f"Question: {question}\nEvidence: {evidence}"
        return self._predict(instruction, input_text, RELEVANCE_TOKENS, "[Irrelevant]")

    def relevance_distribution(self, question: str, evidence: str) -> dict[str, float]:
        instruction = "Judge if evidence is relevant for answering the question."
        input_text = f"Question: {question}\nEvidence: {evidence}"
        return self._predict_distribution(instruction, input_text, RELEVANCE_TOKENS, "[Irrelevant]")

    def groundness(self, question: str, answer: str, evidence: str) -> str:
        instruction = (
            "Judge whether the answer is supported by the evidence only. "
            "Do not use external knowledge. Output one token."
        )
        input_text = f"Question: {question}\nAnswer: {answer}\nEvidence: {evidence}"
        return self._predict(instruction, input_text, GROUNDNESS_TOKENS, "[Partially supported]")

    def groundness_distribution(self, question: str, answer: str, evidence: str) -> dict[str, float]:
        instruction = (
            "Judge whether the answer is supported by the evidence only. "
            "Do not use external knowledge."
        )
        input_text = f"Question: {question}\nAnswer: {answer}\nEvidence: {evidence}"
        return self._predict_distribution(
            instruction,
            input_text,
            GROUNDNESS_TOKENS,
            "[Partially supported]",
        )

    def utility(self, question: str, answer: str) -> str:
        instruction = "Judge answer utility from 1 to 5. Output one token."
        input_text = f"Question: {question}\nAnswer: {answer}"
        return self._predict(instruction, input_text, UTILITY_TOKENS, "[Utility:3]")

    def utility_distribution(self, question: str, answer: str) -> dict[str, float]:
        instruction = "Judge answer utility from 1 to 5."
        input_text = f"Question: {question}\nAnswer: {answer}"
        return self._predict_distribution(instruction, input_text, UTILITY_TOKENS, "[Utility:3]")


def utility_token_to_score(token: str) -> int:
    for i in range(1, 6):
        if token == f"[Utility:{i}]":
            return i
    return 3


def groundness_token_to_score(token: str) -> float:
    if token == "[Fully supported]":
        return 1.0
    if token == "[Partially supported]":
        return 0.5
    return 0.0
