from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .models import INTENT_LABELS


def evaluate_round_acceptance(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    min_macro_f1_delta: float = 0.0,
    max_class_f1_drop: float = 0.03,
) -> Dict[str, Any]:
    """Decide whether an iteration is safe enough to replace the last checkpoint."""
    previous_macro = float(previous["macro_f1"])
    current_macro = float(current["macro_f1"])
    macro_delta = current_macro - previous_macro
    class_deltas: Dict[str, float] = {}
    reasons: List[str] = []

    if macro_delta < min_macro_f1_delta:
        reasons.append(
            f"macro_f1_delta={macro_delta:.6f} < required={min_macro_f1_delta:.6f}"
        )

    previous_classes = previous.get("per_class", {})
    current_classes = current.get("per_class", {})
    for label in INTENT_LABELS:
        previous_class = previous_classes.get(label, {})
        current_class = current_classes.get(label, {})
        if int(previous_class.get("support", 0)) <= 0:
            continue
        if int(current_class.get("support", 0)) <= 0:
            continue
        delta = float(current_class.get("f1", 0.0)) - float(
            previous_class.get("f1", 0.0)
        )
        class_deltas[label] = delta
        if delta < -max_class_f1_drop:
            reasons.append(
                f"{label}_f1_delta={delta:.6f} < allowed={-max_class_f1_drop:.6f}"
            )

    return {
        "accepted": not reasons,
        "macro_f1_delta": macro_delta,
        "class_f1_deltas": class_deltas,
        "reasons": reasons,
    }


def classification_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid = [
        item
        for item in records
        if item.get("true_label") in INTENT_LABELS
        and item.get("predicted_label") in INTENT_LABELS
    ]
    if not valid:
        raise ValueError("没有可评测的预测记录")

    confusion = {
        true_label: {predicted: 0 for predicted in INTENT_LABELS}
        for true_label in INTENT_LABELS
    }
    correct = 0
    for item in valid:
        true_label = item["true_label"]
        predicted = item["predicted_label"]
        confusion[true_label][predicted] += 1
        correct += int(true_label == predicted)

    per_class: Dict[str, Dict[str, float | int]] = {}
    f1_values: List[float] = []
    supported_f1_values: List[float] = []
    for label in INTENT_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in INTENT_LABELS if other != label)
        fn = sum(confusion[label][other] for other in INTENT_LABELS if other != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        if support > 0:
            supported_f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "sample_count": len(valid),
        "accuracy": correct / len(valid),
        # 主指标只平均数据集中实际存在的类别。比如纯文本数据只有三类，
        # 不应因为没有 image/file 样本而被强制计入两个 0 分。
        "macro_f1": sum(supported_f1_values) / len(supported_f1_values),
        "macro_f1_all_labels": sum(f1_values) / len(f1_values),
        "evaluated_labels": [
            label for label in INTENT_LABELS if per_class[label]["support"] > 0
        ],
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def select_hard_negatives(
    records: Sequence[Mapping[str, Any]],
    *,
    confidence_threshold: float,
    max_ratio: float,
    label_confidence_threshold: float = 0.0,
    exclude_keys: set[Tuple[str, str]] | None = None,
    excluded_reason_terms: Sequence[str] = (),
    min_query_chars: int = 0,
    require_reviewer_consensus: bool = False,
) -> List[Dict[str, Any]]:
    """
    选择高置信误分类样本，并对 confusion pair 轮询取样以保持多样性。
    """
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not 0 < max_ratio <= 1:
        raise ValueError("max_ratio must be in (0, 1]")
    if not 0 <= label_confidence_threshold <= 1:
        raise ValueError("label_confidence_threshold must be between 0 and 1")
    if min_query_chars < 0:
        raise ValueError("min_query_chars must be >= 0")

    excluded = exclude_keys or set()
    normalized_terms = [term.strip().lower() for term in excluded_reason_terms if term.strip()]
    candidates = []
    for raw_item in records:
        item = dict(raw_item)
        query = str(item.get("query", "")).strip()
        context = item.get("context", [])
        if not isinstance(context, list):
            context = [context]
        quality_text = " ".join(
            [
                query,
                *(str(turn) for turn in context),
                *(
                    str(item.get(field, ""))
                    for field in (
                        "review_reason",
                        "curation_reason",
                        "label_reason",
                    )
                ),
            ]
        ).lower()
        if item.get("true_label") not in INTENT_LABELS:
            continue
        if item.get("predicted_label") not in INTENT_LABELS:
            continue
        if item.get("true_label") == item.get("predicted_label"):
            continue
        if item.get("review_status", "approved") not in {"approved", "ai_reviewed"}:
            continue
        if float(item.get("confidence", 0.0)) < confidence_threshold:
            continue
        if float(item.get("review_confidence", 1.0)) < label_confidence_threshold:
            continue
        if len(query) < min_query_chars:
            continue
        if not any(character.isalpha() for character in query):
            continue
        if any(term in quality_text for term in normalized_terms):
            continue
        if require_reviewer_consensus:
            review_votes = item.get("review_votes")
            if not isinstance(review_votes, Mapping) or len(review_votes) < 2:
                continue
            vote_labels = {
                str(vote.get("label", ""))
                for vote in review_votes.values()
                if isinstance(vote, Mapping)
            }
            if vote_labels != {str(item.get("true_label", ""))}:
                continue
        if item.get("prompt_eligible") is False:
            continue
        key = (
            query,
            str(item.get("true_label", "")),
        )
        if key in excluded:
            continue
        candidates.append(item)
    if not candidates:
        return []

    # “约 3%”使用 ceil，避免小数据集永远挖不出样本。
    limit = min(len(candidates), max(1, math.ceil(len(records) * max_ratio)))
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        groups[(item["true_label"], item["predicted_label"])].append(item)
    for values in groups.values():
        values.sort(key=lambda item: float(item["confidence"]), reverse=True)

    selected: List[Dict[str, Any]] = []
    ordered_keys = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for key in ordered_keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop(0))
    return selected


def merge_hard_negatives(
    existing: Iterable[Mapping[str, Any]],
    new_records: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """按 query + true_label 去重，保留置信度更高的版本。"""
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for raw in [*existing, *new_records]:
        item = dict(raw)
        key = (str(item.get("query", "")).strip(), str(item.get("true_label", "")))
        if not key[0] or key[1] not in INTENT_LABELS:
            continue
        old = merged.get(key)
        if old is None or float(item.get("confidence", 0.0)) >= float(
            old.get("confidence", 0.0)
        ):
            merged[key] = item
    return sorted(
        merged.values(),
        key=lambda item: (
            int(item.get("round", 0)),
            float(item.get("confidence", 0.0)),
        ),
        reverse=True,
    )
