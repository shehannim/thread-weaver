import os
import json
import tempfile
import unittest
from unittest.mock import patch

from synth.answer import synthesize_answer, answer_question

class TestSynthesis(unittest.TestCase):
    def setUp(self):
        self.mock_evidence = [
            {"chunk_id": "c1", "doc_id": "d1", "page_number": 10, "chunk_text": "text1", "source_reliability": "high"},
            {"chunk_id": "c2", "doc_id": "d2", "page_number": None, "chunk_text": "text2", "source_reliability": "low"}
        ]
        self.ret_res = {
            "question": "test?",
            "seed_entities": [],
            "paths": [],
            "relations": [
                {"source_entity": "A", "target_entity": "B", "relation_type": "is_a", "source_chunk_id": "c1", "source_reliability": "high"},
                {"source_entity": "B", "target_entity": "C", "relation_type": "has_a", "source_chunk_id": "c2", "source_reliability": "low"}
            ],
            "conflicts": [
                {"source_entity": "X", "target_entity": "Y", "relation_type": "dislikes", "source_chunk_id": "c1", "source_reliability": "high", "negated": True}
            ],
            "evidence": self.mock_evidence,
            "warnings": ["retriever warning"]
        }

    @patch('synth.answer.call_openrouter')
    def test_1_valid_model_json(self, mock_call):
        mock_call.return_value = {"choices": [{"message": {"content": '{"answer": "A is B.", "used_chunk_ids": ["c1"]}'}}]}
        res = synthesize_answer("test?", self.ret_res)
        self.assertEqual(res["answer"], "A is B.")
        self.assertTrue(res["grounded"])
        self.assertFalse(res["fallback_used"])
        self.assertEqual(res["used_chunk_ids"], ["c1"])
        
    @patch('synth.answer.call_openrouter')
    def test_2_valid_used_chunk_ids(self, mock_call):
        mock_call.return_value = {"choices": [{"message": {"content": '{"answer": "ans", "used_chunk_ids": ["c1", "c2"]}'}}]}
        res = synthesize_answer("test?", self.ret_res)
        self.assertEqual(res["used_chunk_ids"], ["c1", "c2"])

    @patch('synth.answer.call_openrouter')
    def test_3_citations_built_locally(self, mock_call):
        mock_call.return_value = {"choices": [{"message": {"content": '{"answer": "ans", "used_chunk_ids": ["c1"]}'}}]}
        res = synthesize_answer("test?", self.ret_res)
        self.assertEqual(res["citations"][0]["doc_id"], "d1")
        self.assertEqual(res["citations"][0]["page_number"], 10)
        
    @patch('synth.answer.call_openrouter')
    def test_4_duplicate_ids_deduplicated(self, mock_call):
        mock_call.return_value = {"choices": [{"message": {"content": '{"answer": "ans", "used_chunk_ids": ["c1", "c1"]}'}}]}
        res = synthesize_answer("test?", self.ret_res)
        self.assertEqual(res["used_chunk_ids"], ["c1"])
        
    @patch('synth.answer.call_openrouter')
    def test_5_unknown_chunk_ids_removed_with_warning(self, mock_call):
        mock_call.return_value = {"choices": [{"message": {"content": '{"answer": "ans", "used_chunk_ids": ["c1", "c_fake"]}'}}]}
        res = synthesize_answer("test?", self.ret_res)
        self.assertEqual(res["used_chunk_ids"], ["c1"])
        self.assertTrue(any("unknown chunk ID" in w for w in res["warnings"]))
        
    @patch('synth.answer.call_openrouter')
    def test_6_malformed_json_fallback(self, mock_call):
        mock_call.return_value = {"choices": [{"message": {"content": 'not json'}}]}
        res = synthesize_answer("test?", self.ret_res)
        self.assertTrue(res["fallback_used"])
        
    @patch('synth.answer.call_openrouter')
    def test_7_markdown_fenced_json(self, mock_call):
        mock_call.return_value = {"choices": [{"message": {"content": '```json\n{"answer": "ans", "used_chunk_ids": ["c1"]}\n```'}}]}
        res = synthesize_answer("test?", self.ret_res)
        self.assertFalse(res["fallback_used"])
        self.assertEqual(res["answer"], "ans")
        
    @patch('synth.answer.call_openrouter')
    def test_8_empty_answer_fallback(self, mock_call):
        mock_call.return_value = {"choices": [{"message": {"content": '{"answer": "", "used_chunk_ids": ["c1"]}'}}]}
        res = synthesize_answer("test?", self.ret_res)
        self.assertTrue(res["fallback_used"])
        
    @patch('synth.answer.call_openrouter')
    def test_9_missing_choices_fallback(self, mock_call):
        mock_call.return_value = {}
        res = synthesize_answer("test?", self.ret_res)
        self.assertTrue(res["fallback_used"])
        
    @patch('synth.answer.call_openrouter')
    def test_10_api_exception_fallback(self, mock_call):
        mock_call.side_effect = Exception("API Down")
        res = synthesize_answer("test?", self.ret_res)
        self.assertTrue(res["fallback_used"])
        self.assertTrue(any("API error" in w for w in res["warnings"]))
        
    @patch('synth.answer.call_openrouter')
    def test_11_12_no_evidence_skips_api_and_not_grounded(self, mock_call):
        ret = dict(self.ret_res)
        ret["evidence"] = []
        res = synthesize_answer("test?", ret)
        mock_call.assert_not_called()
        self.assertFalse(res["grounded"])
        self.assertTrue(res["fallback_used"])
        self.assertEqual(res["answer"], "I could not find sufficient evidence in the indexed document graph to answer this question.")
        
    def test_13_fallback_with_evidence_is_grounded(self):
        res = synthesize_answer("test?", self.ret_res, force_fallback=True)
        self.assertTrue(res["grounded"])
        self.assertTrue(res["fallback_used"])
        
    def test_14_fallback_uses_no_more_than_five_relations(self):
        ret = dict(self.ret_res)
        ret["relations"] = [{"source_entity": str(i), "target_entity": "B", "relation_type": "is", "source_chunk_id": "c1"} for i in range(10)]
        res = synthesize_answer("test?", ret, force_fallback=True)
        self.assertLessEqual(res["answer"].count("--is-->"), 5)
        
    def test_15_page_citation_format(self):
        res = synthesize_answer("test?", self.ret_res, force_fallback=True)
        self.assertIn("[d1, p. 10]", res["answer"])
        
    def test_16_missing_page_citation_uses_chunk_id(self):
        res = synthesize_answer("test?", self.ret_res, force_fallback=True)
        self.assertIn("[d2, c2]", res["answer"])
        
    def test_17_low_reliability_identified(self):
        res = synthesize_answer("test?", self.ret_res, force_fallback=True)
        self.assertIn("(low reliability)", res["answer"])
        
    def test_18_conflicts_surfaced(self):
        res = synthesize_answer("test?", self.ret_res, force_fallback=True)
        self.assertIn("Disputed Findings:", res["answer"])
        self.assertIn("(Disputed)", res["answer"])
        
    @patch('synth.answer.call_openrouter')
    def test_19_20_context_limit_enforced_with_warning(self, mock_call):
        mock_call.return_value = {"choices": [{"message": {"content": '{"answer": "a", "used_chunk_ids": []}'}}]}
        ret = dict(self.ret_res)
        ret["evidence"] = [
            {"chunk_id": "c1", "doc_id": "d1", "chunk_text": "A" * 500},
            {"chunk_id": "c2", "doc_id": "d2", "chunk_text": "B" * 500}
        ]
        res = synthesize_answer("test?", ret, max_context_characters=100)
        self.assertTrue(any("truncated" in w for w in res["warnings"]))
        
    @patch('synth.answer.call_openrouter')
    def test_21_input_not_mutated(self, mock_call):
        mock_call.return_value = {"choices": [{"message": {"content": '{"answer": "a", "used_chunk_ids": []}'}}]}
        ev_copy = list(self.ret_res["evidence"])
        synthesize_answer("test?", self.ret_res, max_context_characters=10)
        self.assertEqual(len(self.ret_res["evidence"]), len(ev_copy))
        
    @patch('synth.answer.call_openrouter')
    def test_22_explicit_model_parameter(self, mock_call):
        mock_call.return_value = {"choices": [{"message": {"content": '{"answer": "a", "used_chunk_ids": []}'}}]}
        synthesize_answer("test?", self.ret_res, model="custom-model")
        mock_call.assert_called_with(messages=unittest.mock.ANY, model="custom-model", temperature=0.1)
        
    @patch.dict(os.environ, {"ANSWER_MODEL": "env-model"})
    @patch('synth.answer.call_openrouter')
    def test_23_answer_model_env(self, mock_call):
        mock_call.return_value = {"choices": [{"message": {"content": '{"answer": "a", "used_chunk_ids": []}'}}]}
        synthesize_answer("test?", self.ret_res)
        mock_call.assert_called_with(messages=unittest.mock.ANY, model="env-model", temperature=0.1)

    @patch('synth.answer.synthesize_answer')
    @patch('synth.answer.GraphRetriever')
    def test_24_convenience_function(self, mock_retriever, mock_synth):
        mock_synth.return_value = {"answer": "synthesis"}
        mock_retriever.return_value.retrieve.return_value = {"evidence": []}
        res = answer_question("q?", graph_path="dummy", chunks_path="dummy")
        self.assertEqual(res["question"], "q?")
        self.assertIn("retrieval", res)
        self.assertEqual(res["synthesis"]["answer"], "synthesis")
        
    @patch('synth.answer.call_openrouter')
    @patch('synth.answer.GraphRetriever')
    def test_25_26_offline_mode_no_api_calls(self, mock_retriever, mock_call):
        mock_retriever.return_value.retrieve.return_value = self.ret_res
        res = answer_question("q?", graph_path="dummy", chunks_path="dummy", offline=True)
        mock_call.assert_not_called()
        self.assertTrue(res["synthesis"]["fallback_used"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
