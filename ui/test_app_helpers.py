import unittest
from ui.app import format_relation_label, format_citation, format_path, truncate_excerpt

class TestAppHelpers(unittest.TestCase):
    def test_1_relation_label_formatting(self):
        self.assertEqual(format_relation_label("located_in"), "located in")
        self.assertEqual(format_relation_label(""), "")
        
    def test_2_page_citation_formatting(self):
        c = {"doc_id": "doc1", "page_number": 42, "chunk_id": "c1"}
        self.assertEqual(format_citation(c), "[doc1, p. 42]")
        
    def test_3_missing_page_citation_formatting(self):
        c = {"doc_id": "doc1", "page_number": None, "chunk_id": "c1"}
        self.assertEqual(format_citation(c), "[doc1, c1]")
        
    def test_4_one_hop_path_formatting(self):
        path = {"nodes": ["A", "B"], "hop_count": 1}
        relations = [{"source_entity": "A", "target_entity": "B", "relation_type": "located_in"}]
        res = format_path(path, relations)
        self.assertEqual(res, "A\n-> located in\nB")
        
    def test_5_multi_hop_path_formatting(self):
        path = {"nodes": ["A", "B", "C"], "hop_count": 2}
        relations = [
            {"source_entity": "A", "target_entity": "B", "relation_type": "located_in"},
            {"source_entity": "B", "target_entity": "C", "relation_type": "is_member"}
        ]
        res = format_path(path, relations)
        self.assertEqual(res, "A\n-> located in\nB\n-> is member\nC")
        
    def test_6_incoming_traversal_remains_visibly_marked(self):
        path = {"nodes": ["A", "B"], "hop_count": 1}
        relations = [{"source_entity": "B", "target_entity": "A", "relation_type": "caused_by"}]
        res = format_path(path, relations)
        self.assertEqual(res, "A\n<- caused by\nB")
        
    def test_7_excerpt_below_limit_remains_unchanged(self):
        txt = "hello world"
        self.assertEqual(truncate_excerpt(txt, 50), "hello world")
        
    def test_8_excerpt_above_limit_is_truncated_and_marked(self):
        txt = "A" * 2000
        res = truncate_excerpt(txt, 1500)
        self.assertEqual(len(res), 1500 + len("... [excerpt truncated]"))
        self.assertTrue(res.endswith("... [excerpt truncated]"))
        self.assertTrue(res.startswith("A" * 1500))
        
    def test_9_null_source_path_is_handled(self):
        c = {"doc_id": "doc1", "page_number": 42, "chunk_id": "c1", "source_path": None}
        self.assertEqual(format_citation(c), "[doc1, p. 42]")
        
    def test_10_empty_path_does_not_crash(self):
        self.assertEqual(format_path({"nodes": []}, []), "")
        self.assertEqual(format_path({}, []), "")

if __name__ == "__main__":
    unittest.main(verbosity=2)
