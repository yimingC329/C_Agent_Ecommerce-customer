from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Literal, Mapping, Tuple


IntentLabel = Literal[
    "general-query",
    "additional-query",
    "graphrag-query",
    "image-query",
    "file-query",
]

INTENT_LABELS: Tuple[IntentLabel, ...] = (
    "general-query",
    "additional-query",
    "graphrag-query",
    "image-query",
    "file-query",
)

LABEL_TO_CODE: Dict[IntentLabel, str] = {
    "general-query": "A",
    "additional-query": "B",
    "graphrag-query": "C",
    "image-query": "D",
    "file-query": "E",
}
CODE_TO_LABEL: Dict[str, IntentLabel] = {
    code: label for label, code in LABEL_TO_CODE.items()
}


@dataclass(frozen=True)
class IntentConfidence:
    """一次带置信度的意图预测。

    label_scores 统一保存为概率，总和为 1。margin 是前两名
    对数概率之差，confidence = sigmoid(margin)。
    """

    predicted_label: IntentLabel
    label_scores: Dict[IntentLabel, float]
    top_score: float
    second_score: float
    margin: float
    confidence: float
    confidence_source: str

    def to_dict(self) -> dict:
        return asdict(self)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def confidence_from_logprobs(
    logprobs: Mapping[IntentLabel, float],
    *,
    source: str = "token_logprobs",
) -> IntentConfidence:
    """按文档中的 margin + sigmoid 方法计算置信度。"""
    if len(logprobs) < 2:
        raise ValueError("至少需要两个意图标签的 logprob")

    invalid_labels = set(logprobs) - set(INTENT_LABELS)
    if invalid_labels:
        raise ValueError(f"未知意图标签: {sorted(invalid_labels)}")

    ordered = sorted(logprobs.items(), key=lambda item: item[1], reverse=True)
    top_label, top_logprob = ordered[0]
    _, second_logprob = ordered[1]

    # 对只返回部分 top-logprobs 的接口重新归一化，便于记录和对比。
    max_logprob = max(logprobs.values())
    exp_scores = {
        label: math.exp(value - max_logprob) for label, value in logprobs.items()
    }
    denominator = sum(exp_scores.values())
    probabilities: Dict[IntentLabel, float] = {
        label: value / denominator for label, value in exp_scores.items()
    }

    margin = top_logprob - second_logprob
    return IntentConfidence(
        predicted_label=top_label,
        label_scores=probabilities,
        top_score=probabilities[top_label],
        second_score=sorted(probabilities.values(), reverse=True)[1],
        margin=margin,
        confidence=_sigmoid(margin),
        confidence_source=source,
    )


def confidence_from_probabilities(
    probabilities: Mapping[IntentLabel, float],
    *,
    source: str = "model_reported_probabilities",
) -> IntentConfidence:
    """将非负分数归一化后，按对数概率 margin 计算置信度。"""
    if set(probabilities) != set(INTENT_LABELS):
        missing = set(INTENT_LABELS) - set(probabilities)
        extra = set(probabilities) - set(INTENT_LABELS)
        raise ValueError(
            f"意图分数不完整, missing={sorted(missing)}, extra={sorted(extra)}"
        )

    cleaned: Dict[IntentLabel, float] = {}
    for label in INTENT_LABELS:
        value = float(probabilities[label])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} 的分数必须是非负有限数")
        cleaned[label] = value

    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("意图分数总和必须大于 0")

    normalized: Dict[IntentLabel, float] = {
        label: value / total for label, value in cleaned.items()
    }
    epsilon = 1e-12
    logprobs: Dict[IntentLabel, float] = {
        label: math.log(max(value, epsilon)) for label, value in normalized.items()
    }
    result = confidence_from_logprobs(logprobs, source=source)
    return IntentConfidence(
        predicted_label=result.predicted_label,
        label_scores=normalized,
        top_score=normalized[result.predicted_label],
        second_score=sorted(normalized.values(), reverse=True)[1],
        margin=result.margin,
        confidence=result.confidence,
        confidence_source=source,
    )
