import json
import tempfile
import unittest
from pathlib import Path

from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.grounding_guard.audit import (
    append_grounding_audit,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.grounding_guard.models import (
    GroundingVerdict,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.grounding_guard.node import (
    SAFE_ABSTENTION,
    create_grounding_guard_node,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.grounding_guard.edges import (
    grounding_guard_router,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.grounding_guard.rules import (
    assess_evidence_rules,
    extract_numbers,
)


class _Judge:
    def __init__(self, verdict=None, error=None):
        self.verdict = verdict
        self.error = error

    async def ainvoke(self, _prompt):
        if self.error:
            raise self.error
        return self.verdict


class _Model:
    def __init__(self, verdict=None, error=None):
        self.judge = _Judge(verdict=verdict, error=error)

    def with_structured_output(self, _schema):
        return self.judge


class GroundingRuleTests(unittest.TestCase):
    def test_numeric_units_are_normalized(self):
        self.assertEqual(extract_numbers("价格18元，3天送达"), {"18", "3"})
        result = assess_evidence_rules(
            "Chai 的价格是18元。",
            [{"records": [{"product": "Chai", "unit_price": 18}]}],
        )
        self.assertEqual(result.unsupported_numbers, [])

    def test_invented_number_is_detected(self):
        result = assess_evidence_rules(
            "Chai 的价格是180元。",
            [{"records": [{"product": "Chai", "unit_price": 18}]}],
        )
        self.assertEqual(result.unsupported_numbers, ["180"])

    def test_ordered_list_numbers_are_not_treated_as_claims(self):
        self.assertEqual(
            extract_numbers("1. 查看商品页\n2、联系人工客服\n订单3"),
            {"3"},
        )

    def test_unsupported_stock_sufficiency_claim_is_detected(self):
        result = assess_evidence_rules(
            "这款商品库存充足。",
            [{"records": [{"UnitsInStock": 479}]}],
        )
        self.assertEqual(result.unsupported_phrases, ["库存充足"])

    def test_unsupported_supply_sufficiency_claim_is_detected(self):
        result = assess_evidence_rules(
            "这款商品货源很充足。",
            [{"records": [{"UnitsInStock": 479}]}],
        )
        self.assertEqual(result.unsupported_phrases, ["货源很充足"])

    def test_empty_evidence_is_detected(self):
        result = assess_evidence_rules("库存20件。", [{"records": []}])
        self.assertFalse(result.has_evidence)

    def test_audit_is_utf8_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            append_grounding_audit(path, {"question": "多少钱？"})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["question"], "多少钱？")
            self.assertIn("timestamp", payload)

    def test_retry_switches_from_graphrag_to_cypher(self):
        route = grounding_guard_router(
            {
                "grounding_decision": "retry",
                "question": "Chai 多少钱？",
                "cyphers": [{"steps": ["execute_graphrag_query"]}],
            }
        )
        self.assertEqual(route.node, "cypher_query")

    def test_retry_switches_from_cypher_to_graphrag(self):
        route = grounding_guard_router(
            {
                "grounding_decision": "retry",
                "question": "Chai 多少钱？",
                "cyphers": [{"steps": ["execute_cypher_query"]}],
            }
        )
        self.assertEqual(route.node, "customer_tools")


class GroundingNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_supported_answer_passes(self):
        model = _Model(
            GroundingVerdict(
                decision="pass",
                groundedness_score=0.95,
                unsupported_claims=[],
                reason="all claims supported",
            )
        )
        node = create_grounding_guard_node(model, max_retries=0)
        result = await node(
            {
                "question": "Chai 多少钱？",
                "summary": "Chai 的价格是18元。",
                "cyphers": [{"records": [{"product": "Chai", "price": 18}]}],
                "grounding_retry_count": 0,
            }
        )
        self.assertEqual(result["grounding_decision"], "pass")
        self.assertEqual(result["summary"], "Chai 的价格是18元。")

    async def test_rule_failure_abstains_without_calling_model(self):
        node = create_grounding_guard_node(_Model(error=AssertionError()), max_retries=0)
        result = await node(
            {
                "question": "有货吗？",
                "summary": "有货，库存20件。",
                "cyphers": [{"records": []}],
                "grounding_retry_count": 0,
            }
        )
        self.assertEqual(result["grounding_decision"], "abstain")
        self.assertEqual(result["summary"], SAFE_ABSTENTION)

    async def test_judge_error_fails_closed(self):
        node = create_grounding_guard_node(
            _Model(error=RuntimeError("model unavailable")), max_retries=0
        )
        result = await node(
            {
                "question": "Chai 是什么？",
                "summary": "Chai 是一种茶。",
                "cyphers": [{"records": [{"product": "Chai", "category": "茶"}]}],
                "grounding_retry_count": 0,
            }
        )
        self.assertEqual(result["grounding_decision"], "abstain")
        self.assertIn("grounding_judge_error", result["grounding_reason"])


if __name__ == "__main__":
    unittest.main()
