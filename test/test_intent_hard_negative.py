import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from app.lg_agent.intent.evaluation import (
    classification_metrics,
    evaluate_round_acceptance,
    select_hard_negatives,
)
from app.lg_agent.intent.few_shot import (
    build_router_prompt,
    format_hard_negative_few_shots,
    load_canonical_anchors,
    load_approved_hard_negatives,
    write_jsonl,
)
from app.lg_agent.intent.models import INTENT_LABELS, confidence_from_logprobs
from app.lg_agent.intent.classifier import extract_conversation_context


class IntentHardNegativeTests(unittest.TestCase):
    def test_extract_conversation_context_excludes_current_user_turn(self):
        messages = [
            {"role": "user", "content": "我想买口红"},
            {"role": "assistant", "content": "您的肤色怎么样？"},
            {"role": "user", "content": "自然偏白"},
        ]
        self.assertEqual(
            extract_conversation_context(messages),
            ["user: 我想买口红", "assistant: 您的肤色怎么样？"],
        )

    def test_margin_sigmoid_confidence(self):
        result = confidence_from_logprobs(
            {
                "general-query": -0.1,
                "additional-query": -2.1,
                "graphrag-query": -3.0,
            }
        )
        self.assertEqual(result.predicted_label, "general-query")
        self.assertAlmostEqual(result.margin, 2.0)
        self.assertAlmostEqual(result.confidence, 0.880797, places=5)

    def test_selects_only_high_confidence_errors(self):
        records = [
            {
                "query": f"q{index}",
                "true_label": "general-query",
                "predicted_label": "graphrag-query" if index < 4 else "general-query",
                "confidence": 0.95 if index < 2 else 0.6,
            }
            for index in range(100)
        ]
        selected = select_hard_negatives(
            records,
            confidence_threshold=0.8,
            max_ratio=0.03,
        )
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(item["confidence"] >= 0.8 for item in selected))
        self.assertTrue(
            all(item["true_label"] != item["predicted_label"] for item in selected)
        )

        selected_without_existing = select_hard_negatives(
            records,
            confidence_threshold=0.8,
            max_ratio=0.03,
            exclude_keys={("q0", "general-query")},
        )
        self.assertEqual([item["query"] for item in selected_without_existing], ["q1"])

    def test_rejects_low_label_confidence(self):
        records = [
            {
                "query": "high-label-confidence",
                "true_label": "additional-query",
                "predicted_label": "graphrag-query",
                "confidence": 0.95,
                "review_confidence": 0.95,
            },
            {
                "query": "low-label-confidence",
                "true_label": "additional-query",
                "predicted_label": "graphrag-query",
                "confidence": 0.99,
                "review_confidence": 0.70,
            },
        ]
        selected = select_hard_negatives(
            records,
            confidence_threshold=0.8,
            label_confidence_threshold=0.9,
            max_ratio=1.0,
        )
        self.assertEqual([item["query"] for item in selected], ["high-label-confidence"])

    def test_rejects_cross_domain_hard_negative_reasons(self):
        records = [
            {
                "query": "老板有饺子么？",
                "true_label": "general-query",
                "predicted_label": "additional-query",
                "confidence": 0.99,
                "review_confidence": 0.99,
                "review_reason": "餐饮点单场景，属于非电商咨询。",
            },
            {
                "query": "我的货怎么还没发？",
                "true_label": "additional-query",
                "predicted_label": "graphrag-query",
                "review_status": "ai_reviewed",
                "confidence": 0.99,
                "review_confidence": 0.99,
                "review_reason": "未提供订单号，需追问。",
            },
        ]
        selected = select_hard_negatives(
            records,
            confidence_threshold=0.8,
            label_confidence_threshold=0.95,
            max_ratio=1.0,
            excluded_reason_terms=["餐饮", "点单", "非电商"],
        )
        self.assertEqual([item["query"] for item in selected], ["我的货怎么还没发？"])

    def test_quality_gate_scans_context_and_requires_consensus(self):
        records = [
            {
                "query": "有什么推荐吗？",
                "context": ["店员：要点酒吗？"],
                "true_label": "additional-query",
                "predicted_label": "general-query",
                "confidence": 0.99,
                "review_confidence": 0.99,
                "review_votes": {
                    "a": {"label": "additional-query"},
                    "b": {"label": "additional-query"},
                },
            },
            {
                "query": "自然偏白吧。",
                "context": ["想买口红", "您的肤色怎么样？"],
                "true_label": "graphrag-query",
                "predicted_label": "additional-query",
                "confidence": 0.99,
                "review_confidence": 0.99,
                "review_votes": {
                    "a": {"label": "graphrag-query"},
                    "b": {"label": "graphrag-query"},
                },
            },
            {
                "query": "这条信息足够长",
                "true_label": "graphrag-query",
                "predicted_label": "additional-query",
                "confidence": 0.99,
                "review_confidence": 0.99,
                "review_votes": {
                    "a": {"label": "graphrag-query"},
                    "b": {"label": "additional-query"},
                },
            },
        ]
        selected = select_hard_negatives(
            records,
            confidence_threshold=0.8,
            label_confidence_threshold=0.95,
            max_ratio=1.0,
            excluded_reason_terms=["点酒"],
            min_query_chars=5,
            require_reviewer_consensus=True,
        )
        self.assertEqual([item["query"] for item in selected], ["自然偏白吧。"])

    def test_macro_f1_and_confusion_matrix(self):
        records = [
            {"true_label": "general-query", "predicted_label": "general-query"},
            {"true_label": "additional-query", "predicted_label": "general-query"},
            {"true_label": "graphrag-query", "predicted_label": "graphrag-query"},
            {"true_label": "image-query", "predicted_label": "image-query"},
            {"true_label": "file-query", "predicted_label": "file-query"},
        ]
        result = classification_metrics(records)
        self.assertAlmostEqual(result["accuracy"], 0.8)
        self.assertEqual(
            result["confusion_matrix"]["additional-query"]["general-query"],
            1,
        )

    def test_macro_f1_ignores_absent_modalities(self):
        records = [
            {"true_label": "general-query", "predicted_label": "general-query"},
            {"true_label": "additional-query", "predicted_label": "additional-query"},
            {"true_label": "graphrag-query", "predicted_label": "graphrag-query"},
        ]
        result = classification_metrics(records)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["macro_f1_all_labels"], 0.6)
        self.assertEqual(len(result["evaluated_labels"]), 3)

    def test_only_approved_examples_enter_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hard.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "query": "approved query",
                        "true_label": "general-query",
                        "review_status": "approved",
                        "confidence": 0.9,
                    },
                    {
                        "query": "pending query",
                        "true_label": "file-query",
                        "review_status": "pending",
                        "confidence": 0.99,
                    },
                ],
            )
            records = load_approved_hard_negatives(path, max_examples=10)
            prompt = format_hard_negative_few_shots(records)
            self.assertIn("approved query", prompt)
            self.assertNotIn("pending query", prompt)

    def test_review_reason_is_included_in_few_shot_prompt(self):
        prompt = format_hard_negative_few_shots(
            [
                {
                    "query": "推荐一款",
                    "true_label": "additional-query",
                    "review_reason": "缺少预算和使用场景，需要先追问。",
                }
            ]
        )
        self.assertIn("<decision_reason>", prompt)
        self.assertIn("缺少预算", prompt)

    def test_weighted_few_shot_sampling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weighted.jsonl"
            records = []
            for label in (
                "general-query",
                "additional-query",
                "graphrag-query",
            ):
                for index in range(10):
                    records.append(
                        {
                            "query": f"{label}-{index}",
                            "true_label": label,
                            "review_status": "approved",
                            "confidence": 0.9,
                        }
                    )
            write_jsonl(path, records)
            selected = load_approved_hard_negatives(
                path,
                max_examples=16,
                label_weights={
                    "general-query": 1,
                    "additional-query": 2,
                    "graphrag-query": 1,
                    "image-query": 0,
                    "file-query": 0,
                },
            )
            counts = {
                label: sum(item["true_label"] == label for item in selected)
                for label in INTENT_LABELS
            }
            self.assertEqual(counts["general-query"], 4)
            self.assertEqual(counts["additional-query"], 8)
            self.assertEqual(counts["graphrag-query"], 4)

    def test_multi_round_sampling_preserves_older_anchors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rounds.jsonl"
            records = []
            for round_index in (1, 2):
                for label in (
                    "general-query",
                    "additional-query",
                    "graphrag-query",
                ):
                    for index in range(10):
                        records.append(
                            {
                                "query": f"r{round_index}-{label}-{index}",
                                "true_label": label,
                                "round": round_index,
                                "review_status": "approved",
                                "confidence": 0.9,
                            }
                        )
            write_jsonl(path, records)
            selected = load_approved_hard_negatives(path, max_examples=15)
            round_counts = {
                round_index: sum(item["round"] == round_index for item in selected)
                for round_index in (1, 2)
            }
            self.assertEqual(round_counts[2], 3)
            self.assertEqual(round_counts[1], 12)

    def test_multi_round_sampling_balances_confusion_direction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rounds.jsonl"
            records = []
            for index in range(6):
                records.append(
                    {
                        "query": f"to-graph-{index}",
                        "true_label": "additional-query",
                        "predicted_label": "graphrag-query",
                        "round": 2,
                        "review_status": "approved",
                        "confidence": 0.99 - index / 100,
                    }
                )
            records.append(
                {
                    "query": "to-general",
                    "true_label": "additional-query",
                    "predicted_label": "general-query",
                    "round": 2,
                    "review_status": "approved",
                    "confidence": 0.9,
                }
            )
            for label in ("general-query", "additional-query", "graphrag-query"):
                for index in range(6):
                    records.append(
                        {
                            "query": f"old-{label}-{index}",
                            "true_label": label,
                            "predicted_label": "graphrag-query",
                            "round": 1,
                            "review_status": "approved",
                            "confidence": 0.8,
                        }
                    )
            write_jsonl(path, records)
            selected = load_approved_hard_negatives(path, max_examples=15)
            latest = [item for item in selected if item["round"] == 2]
            self.assertEqual(len(latest), 3)
            self.assertEqual(
                {item["predicted_label"] for item in latest},
                {"general-query", "graphrag-query"},
            )

    def test_performance_gate_rejects_regression(self):
        previous = classification_metrics(
            [
                {"true_label": "general-query", "predicted_label": "general-query"},
                {"true_label": "additional-query", "predicted_label": "additional-query"},
                {"true_label": "graphrag-query", "predicted_label": "graphrag-query"},
            ]
        )
        current = classification_metrics(
            [
                {"true_label": "general-query", "predicted_label": "general-query"},
                {"true_label": "additional-query", "predicted_label": "general-query"},
                {"true_label": "graphrag-query", "predicted_label": "general-query"},
            ]
        )
        gate = evaluate_round_acceptance(previous, current)
        self.assertFalse(gate["accepted"])
        self.assertTrue(gate["reasons"])

    def test_latest_example_is_offset_by_older_label_quota(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hard.jsonl"
            records = [
                {
                    "query": "new-additional",
                    "true_label": "additional-query",
                    "predicted_label": "graphrag-query",
                    "round": 3,
                    "review_status": "approved",
                }
            ]
            for label in ("additional-query", "graphrag-query"):
                for index in range(6):
                    records.append(
                        {
                            "query": f"old-{label}-{index}",
                            "true_label": label,
                            "predicted_label": "general-query",
                            "round": 1,
                            "review_status": "approved",
                        }
                    )
            write_jsonl(path, records)
            selected = load_approved_hard_negatives(
                path,
                max_examples=6,
                latest_round_ratio=0.2,
            )
            counts = Counter(item["true_label"] for item in selected)
            self.assertEqual(counts["additional-query"], 3)
            self.assertEqual(counts["graphrag-query"], 3)

    def test_context_is_included_in_few_shot_prompt(self):
        prompt = format_hard_negative_few_shots(
            [
                {
                    "query": "多少钱？",
                    "context": ["顾客：我要24号垃圾袋"],
                    "true_label": "graphrag-query",
                }
            ]
        )
        self.assertIn("<conversation_context>", prompt)
        self.assertIn("24号垃圾袋", prompt)
        self.assertIn("<current_user_query>多少钱？</current_user_query>", prompt)

    def test_prompt_reserves_nine_canonical_and_six_hard_examples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            anchor_path = Path(temp_dir) / "anchors.jsonl"
            hard_path = Path(temp_dir) / "hard.jsonl"
            anchors = []
            hard = []
            for label in ("general-query", "additional-query", "graphrag-query"):
                for index in range(3):
                    anchors.append(
                        {
                            "query": f"anchor-{label}-{index}",
                            "true_label": label,
                            "anchor_type": "canonical",
                            "review_status": "approved",
                        }
                    )
                for round_index in (1, 2):
                    for index in range(3):
                        hard.append(
                            {
                                "query": f"hard-r{round_index}-{label}-{index}",
                                "true_label": label,
                                "predicted_label": "general-query",
                                "round": round_index,
                                "review_status": "approved",
                                "confidence": 0.9,
                            }
                        )
            write_jsonl(anchor_path, anchors)
            write_jsonl(hard_path, hard)
            selected_anchors = load_canonical_anchors(anchor_path, max_examples=9)
            prompt = build_router_prompt(
                "BASE",
                hard_negative_path=hard_path,
                max_examples=15,
                enabled=True,
                anchor_path=anchor_path,
                canonical_examples=9,
                latest_round_ratio=0.5,
            )
            self.assertEqual(len(selected_anchors), 9)
            self.assertEqual(prompt.count("anchor-"), 9)
            self.assertEqual(prompt.count("hard-r2-"), 3)
            self.assertEqual(prompt.count("hard-r1-"), 3)


if __name__ == "__main__":
    unittest.main()
