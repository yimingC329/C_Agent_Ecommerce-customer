from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Sequence, cast
from urllib.parse import urlparse

from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from openai import AsyncOpenAI

from app.core.config import ServiceType, settings
from app.core.logger import get_logger
from app.lg_agent.lg_prompts import ROUTER_SYSTEM_PROMPT
from app.lg_agent.lg_states import Router

from .few_shot import build_router_prompt
from .models import (
    CODE_TO_LABEL,
    INTENT_LABELS,
    IntentConfidence,
    IntentLabel,
    confidence_from_logprobs,
    confidence_from_probabilities,
)


logger = get_logger(service="intent_classifier")


def extract_last_user_query(messages: Sequence[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, Mapping):
            if message.get("role") == "user":
                return str(message.get("content", ""))
            continue
        message_type = getattr(message, "type", None)
        role = getattr(message, "role", None)
        if message_type in {"human", "user"} or role == "user":
            return str(getattr(message, "content", ""))
    if messages:
        last = messages[-1]
        if isinstance(last, str):
            return last
        if isinstance(last, Mapping):
            return str(last.get("content", ""))
        return str(getattr(last, "content", ""))
    return ""


def extract_conversation_context(messages: Sequence[Any]) -> List[str]:
    """Return compact text history before the last user turn."""
    last_user_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, Mapping):
            role = str(message.get("role", ""))
        else:
            role = str(getattr(message, "role", "") or getattr(message, "type", ""))
        if role in {"user", "human"}:
            last_user_index = index

    history: List[str] = []
    for message in messages[:last_user_index]:
        if isinstance(message, Mapping):
            role = str(message.get("role", "message"))
            content = message.get("content", "")
        else:
            role = str(
                getattr(message, "role", "")
                or getattr(message, "type", "")
                or "message"
            )
            content = getattr(message, "content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
            content = " ".join(text_parts)
        text = str(content).strip()
        if text:
            history.append(f"{role}: {text}")
    return history[-8:]


def _normalize_router(value: Any, query: str) -> Router:
    if hasattr(value, "model_dump"):
        data = value.model_dump()
    elif isinstance(value, Mapping):
        data = dict(value)
    else:
        raise TypeError(f"Unsupported router output type: {type(value)!r}")

    label = data.get("type")
    if label not in INTENT_LABELS:
        raise ValueError(f"Unknown intent label returned by model: {label!r}")
    return cast(
        Router,
        {
            "type": label,
            "logic": str(data.get("logic", "")).strip(),
            "question": str(data.get("question") or query).strip(),
        },
    )


class IntentClassifier:
    """
    线上路由与离线难负样本挖掘共用的意图分类器。

    structured 模式保持原项目的稳定 JSON 输出。logprob 模式使用
    DeepSeek FIM Completion 的 top-logprobs，只输出 A-E 单字符标签。
    """

    def __init__(
        self,
        *,
        hard_negative_path: str | None = None,
        few_shot_enabled: bool | None = None,
        max_few_shots: int | None = None,
        few_shot_label_weights: Mapping[str, int] | None = None,
        latest_round_ratio: float = 0.5,
        anchor_path: str | None = None,
        canonical_examples: int | None = None,
    ) -> None:
        self.hard_negative_path = (
            hard_negative_path
            if hard_negative_path is not None
            else settings.INTENT_HARD_NEGATIVE_PATH
        )
        self.few_shot_enabled = (
            settings.INTENT_FEWSHOT_ENABLED
            if few_shot_enabled is None
            else few_shot_enabled
        )
        self.max_few_shots = (
            settings.INTENT_FEWSHOT_MAX_EXAMPLES
            if max_few_shots is None
            else max_few_shots
        )
        self.few_shot_label_weights = dict(few_shot_label_weights or {})
        self.latest_round_ratio = latest_round_ratio
        self.anchor_path = (
            anchor_path
            if anchor_path is not None
            else settings.INTENT_CANONICAL_ANCHOR_PATH
        )
        self.canonical_examples = (
            settings.INTENT_CANONICAL_MAX_EXAMPLES
            if canonical_examples is None
            else canonical_examples
        )
        self._logprobs_available: bool | None = None
        self._chat_client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
        self._logprob_client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=self._resolve_logprob_base_url(),
        )

    def router_prompt(self) -> str:
        return build_router_prompt(
            ROUTER_SYSTEM_PROMPT,
            hard_negative_path=self.hard_negative_path,
            max_examples=self.max_few_shots,
            enabled=self.few_shot_enabled,
            label_weights=self.few_shot_label_weights,
            latest_round_ratio=self.latest_round_ratio,
            anchor_path=self.anchor_path,
            canonical_examples=self.canonical_examples,
        )

    def _create_langchain_model(self):
        if settings.AGENT_SERVICE == ServiceType.DEEPSEEK:
            return ChatDeepSeek(
                api_key=settings.DEEPSEEK_API_KEY,
                model_name=settings.DEEPSEEK_MODEL,
                temperature=0,
                tags=["router"],
            )
        return ChatOllama(
            model=settings.OLLAMA_AGENT_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
            tags=["router"],
        )

    async def classify_structured(self, messages: Sequence[Any]) -> Router:
        query = extract_last_user_query(messages)
        model = self._create_langchain_model()
        prompt_messages: List[Any] = [
            {"role": "system", "content": self.router_prompt()}
        ]
        prompt_messages.extend(messages)
        response = await model.with_structured_output(Router).ainvoke(prompt_messages)
        return _normalize_router(response, query)

    async def classify(self, messages: Sequence[Any]) -> tuple[Router, IntentConfidence | None]:
        mode = settings.INTENT_ROUTER_MODE.strip().lower()
        query = extract_last_user_query(messages)
        if mode == "boundary_v2" and settings.AGENT_SERVICE == ServiceType.DEEPSEEK:
            score = await self.score_with_boundary_adjudication(
                query,
                extract_conversation_context(messages),
            )
            return (
                cast(
                    Router,
                    {
                        "type": score.predicted_label,
                        "logic": (
                            "DeepSeek intent classification with specialized "
                            "additional-vs-graphrag boundary adjudication; "
                            f"confidence={score.confidence:.4f}"
                        ),
                        "question": query,
                    },
                ),
                score,
            )
        if mode == "logprob" and settings.AGENT_SERVICE == ServiceType.DEEPSEEK:
            try:
                score = await self.score_with_logprobs(query)
                return (
                    cast(
                        Router,
                        {
                            "type": score.predicted_label,
                            "logic": (
                                "Intent selected from DeepSeek token logprobs; "
                                f"confidence={score.confidence:.4f}"
                            ),
                            "question": query,
                        },
                    ),
                    score,
                )
            except Exception as exc:
                logger.warning(
                    f"Logprob router unavailable, falling back to structured output: {exc}"
                )
        return await self.classify_structured(messages), None

    def _resolve_logprob_base_url(self) -> str:
        override = settings.DEEPSEEK_LOGPROB_BASE_URL.strip()
        if override:
            return override.rstrip("/")

        base_url = settings.DEEPSEEK_BASE_URL.rstrip("/")
        parsed = urlparse(base_url)
        if parsed.netloc == "api.deepseek.com" and not parsed.path.rstrip("/").endswith(
            "/beta"
        ):
            return base_url + "/beta"
        return base_url

    @staticmethod
    def _format_evaluation_input(
        query: str,
        context: Sequence[Any] | None = None,
    ) -> str:
        """构造离线评测输入，使短问题能使用真实对话上下文。"""
        context_lines = [str(item).strip() for item in (context or []) if str(item).strip()]
        if not context_lines:
            return f"<current_user_query>{query}</current_user_query>"
        history = "\n".join(context_lines)
        return (
            "<conversation_context>\n"
            f"{history}\n"
            "</conversation_context>\n"
            f"<current_user_query>{query}</current_user_query>"
        )

    def _label_prompt(
        self,
        query: str,
        context: Sequence[Any] | None = None,
    ) -> str:
        evaluation_input = self._format_evaluation_input(query, context)
        return f"""{self.router_prompt()}

离线评测时使用下列单字符代码：
A = general-query
B = additional-query
C = graphrag-query
D = image-query
E = file-query

现在只能输出 A、B、C、D、E 中的一个字符，不得输出解释。
请只判断 current_user_query 的意图；conversation_context 仅用于补全省略的商品、
订单或问题对象，不要把历史中的客服问题当成当前用户问题。
{evaluation_input}
标签:"""

    async def score_with_logprobs(
        self,
        query: str,
        context: Sequence[Any] | None = None,
    ) -> IntentConfidence:
        if not query.strip():
            raise ValueError("用户问题不能为空")
        if self._logprobs_available is False:
            raise RuntimeError("Current endpoint was already detected without logprobs support")

        model = settings.INTENT_LOGPROB_MODEL.strip() or settings.DEEPSEEK_MODEL
        try:
            response = await self._logprob_client.completions.create(
                model=model,
                prompt=self._label_prompt(query, context),
                max_tokens=1,
                temperature=0,
                logprobs=20,
            )
            choice = response.choices[0]
            top_logprobs = getattr(choice.logprobs, "top_logprobs", None)
            if not top_logprobs:
                raise RuntimeError("API response does not contain top_logprobs")
            first_token_scores = top_logprobs[0]
            if hasattr(first_token_scores, "model_dump"):
                first_token_scores = first_token_scores.model_dump()
            if not isinstance(first_token_scores, Mapping):
                raise RuntimeError("Unexpected top_logprobs response format")

            label_logprobs: Dict[IntentLabel, float] = {}
            for token, value in first_token_scores.items():
                normalized_token = str(token).strip().strip('"').strip("'").upper()
                label = CODE_TO_LABEL.get(normalized_token)
                if label is not None:
                    label_logprobs[label] = max(
                        float(value), label_logprobs.get(label, float("-inf"))
                    )
            if len(label_logprobs) < 2:
                raise RuntimeError(
                    "top_logprobs did not include at least two of labels A-E"
                )
            self._logprobs_available = True
            return confidence_from_logprobs(
                label_logprobs,
                source="deepseek_fim_token_logprobs",
            )
        except Exception:
            self._logprobs_available = False
            raise

    async def score_with_model_report(
        self,
        query: str,
        context: Sequence[Any] | None = None,
    ) -> IntentConfidence:
        """
        Chat Completion 不提供 token logprobs 时的回退方案。

        这些分数是模型自报分数，会在记录中与真实 token logprobs 明确区分。
        """
        evaluation_input = self._format_evaluation_input(query, context)
        prompt = f"""{self.router_prompt()}

请对下面的用户问题进行五类意图评分。只输出 JSON。
代码映射：A=general-query, B=additional-query, C=graphrag-query,
D=image-query, E=file-query。
输出格式：
{{"scores": {{"A": 0.1, "B": 0.1, "C": 0.6, "D": 0.1, "E": 0.1}}}}
五个分数必须为非负数，且总和应当接近 1。

请只判断 current_user_query 的意图；conversation_context 仅用于补全省略的商品、
订单或问题对象，不要把历史中的客服问题当成当前用户问题。
{evaluation_input}
"""
        response = await self._chat_client.chat.completions.create(
            model=settings.DEEPSEEK_MODEL,
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=256,
        )
        content = response.choices[0].message.content or ""
        payload = json.loads(content)
        raw_scores = payload.get("scores")
        if not isinstance(raw_scores, Mapping):
            raise ValueError("Model score response is missing a scores object")
        probabilities: Dict[IntentLabel, float] = {}
        for code, label in CODE_TO_LABEL.items():
            if code not in raw_scores:
                raise ValueError(f"Model score response is missing label code {code}")
            probabilities[label] = float(raw_scores[code])
        return confidence_from_probabilities(
            probabilities,
            source="deepseek_model_reported_probabilities",
        )

    async def score_with_boundary_adjudication(
        self,
        query: str,
        context: Sequence[Any] | None = None,
    ) -> IntentConfidence:
        """Use a second, specialized judge for the ask-vs-retrieve boundary."""
        base = await self.score_with_model_report(query, context)
        boundary_mass = (
            base.label_scores["additional-query"]
            + base.label_scores["graphrag-query"]
        )
        if boundary_mass < 0.45:
            return base

        evaluation_input = self._format_evaluation_input(query, context)
        prompt = f"""你是电商意图路由中的边界裁决器。只区分以下两类：

B = additional-query：系统必须先向用户追问，才能确定要查询的商品、订单、问题对象，
或个性化推荐仍缺少会改变结果的必要条件。
C = graphrag-query：商品/订单/问题对象能从当前问题或对话上下文确定，下一步可以直接
查询数据库、知识库、物流、库存、规格、政策或候选商品。

严格执行：
1. 先从当前问题中找对象，再用上下文补全省略和指代。
2. 明确商品类别、订单商品、物流状态或售后问题，通常已经足够查询，不要因为存在多个
   款式就误判为B；查询后可以返回候选列表。
3. 用户正在回答客服刚刚追问的尺寸、肤色、年龄、容量、品牌等条件时，要把回答与上下文
   合并。若合并后已足够执行查询，选C。
4. 只有无法判断查询对象，或推荐仍缺少真正必要条件时，才选B。
5. 不要把“没提供订单号”机械等同于B；如果系统可通过当前登录用户和明确订单商品查询，选C。

例1：无上下文，用户“多少钱？” -> B（不知道哪件商品）
例2：上下文“用户正在看24号垃圾袋”，用户“多少钱？” -> C
例3：无上下文，用户“想给朋友买裙子，推荐一下” -> B（缺少朋友尺寸和偏好）
例4：上下文“想买口红；客服问肤色”，用户“自然偏白” -> C
例5：上下文“想买儿童绘本；客服问年龄”，用户“3到5岁的” -> C
例6：用户“我的核桃已经发货了吗，我要改地址” -> C（对象和动作明确）

请输出JSON，不要输出额外文字：
{{
  "resolved_object": "从当前问题和上下文解析出的对象；无法解析则写unknown",
  "missing_required_information": "仍必须追问的信息；没有则写none",
  "next_action": "ASK_USER或RETRIEVE",
  "scores": {{"B": 0.2, "C": 0.8}}
}}
scores必须非负且总和接近1。

{evaluation_input}
"""
        response = await self._chat_client.chat.completions.create(
            model=settings.DEEPSEEK_MODEL,
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=300,
        )
        content = response.choices[0].message.content or ""
        payload = json.loads(content)
        raw_scores = payload.get("scores")
        if not isinstance(raw_scores, Mapping):
            raise ValueError("Boundary response is missing a scores object")
        b_score = max(0.0, float(raw_scores.get("B", 0.0)))
        c_score = max(0.0, float(raw_scores.get("C", 0.0)))
        denominator = b_score + c_score
        if denominator <= 0:
            raise ValueError("Boundary B/C scores must sum to a positive value")

        blend_weight = min(1.0, max(0.0, settings.INTENT_BOUNDARY_BLEND_WEIGHT))
        judge_additional = boundary_mass * b_score / denominator
        judge_graphrag = boundary_mass * c_score / denominator
        probabilities = dict(base.label_scores)
        probabilities["additional-query"] = (
            (1.0 - blend_weight) * base.label_scores["additional-query"]
            + blend_weight * judge_additional
        )
        probabilities["graphrag-query"] = (
            (1.0 - blend_weight) * base.label_scores["graphrag-query"]
            + blend_weight * judge_graphrag
        )
        return confidence_from_probabilities(
            cast(Mapping[IntentLabel, float], probabilities),
            source="deepseek_boundary_adjudication_v2",
        )

    async def score_query(
        self,
        query: str,
        *,
        method: str = "auto",
        context: Sequence[Any] | None = None,
    ) -> IntentConfidence:
        selected_method = method.strip().lower()
        if selected_method not in {
            "auto",
            "logprobs",
            "model_report",
            "boundary_v2",
        }:
            raise ValueError(
                "method must be auto, logprobs, model_report, or boundary_v2"
            )

        if selected_method == "boundary_v2":
            return await self.score_with_boundary_adjudication(query, context)

        if selected_method in {"auto", "logprobs"}:
            try:
                return await self.score_with_logprobs(query, context)
            except Exception as exc:
                if selected_method == "logprobs":
                    raise
                logger.warning(
                    f"Token logprobs unavailable; using model-reported scores: {exc}"
                )
        return await self.score_with_model_report(query, context)


@lru_cache(maxsize=1)
def get_intent_classifier() -> IntentClassifier:
    return IntentClassifier()
