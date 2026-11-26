"""Standard-library runner contracts; no ML imports, inference, or downloads."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


RUNNER_PATH = Path(__file__).resolve().parents[1] / "training" / "own_decoder_runner.py"
SPEC = importlib.util.spec_from_file_location("own_decoder_runner_contract_test", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerContractTests(unittest.TestCase):
    def test_response_first_token_and_eos_are_supervised(self):
        inputs, labels = runner.next_token_targets([1, 40, 41], [50, 51, 2], 5)
        self.assertEqual(inputs, [1, 40, 41, 50, 51])
        self.assertEqual(labels, [-100, -100, 50, 51, 2])

    def test_context_overflow_and_empty_inputs_refused(self):
        for prompt, target, length in (([1, 40], [50, 2], 2), ([], [2], 2), ([1], [], 2)):
            with self.assertRaises(ValueError):
                runner.next_token_targets(prompt, target, length)

    def test_complete_row_review_invalidated_by_changed_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proof = root / "review.json"
            row = {"id": "training-1", "split": "train", "prompt": "Explique o teste.",
                   "response": "O teste mede uma hipótese.", "source": "synthetic",
                   "bio_vector20": list(range(20)), "evidence": {"mode": "general"}}
            digest = runner.canonical_hash(runner.reviewed_payload(row))
            proof.write_text(json.dumps({"payload_sha256": digest, "approved": True}), encoding="utf-8")
            row["teacher_review"] = {"status": "approved", "reviewer": "test_teacher",
                "payload_sha256": digest, "evidence": str(proof), "evidence_sha256": runner.sha256(proof)}
            data = root / "data.jsonl"
            data.write_text(json.dumps(row) + "\n", encoding="utf-8")
            # This fixture tests integrity only; it does not qualify a teacher or physiological trace.
            with patch.object(runner, "cloud_file", side_effect=Path):
                self.assertEqual(len(runner.read_rows(data, approved=True)), 1)
                altered = copy.deepcopy(row)
                altered["bio_vector20"][0] += 1
                data.write_text(json.dumps(altered) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner.read_rows(data, approved=True)

    def test_core_trace_is_bound_to_prompt_and_collapse_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = Path(temporary) / "trace.json"
            row = {"id": "probe-1", "prompt": "Como está a memória?", "bio_vector20": [0.] * 20}
            trace = {"vector20": row["bio_vector20"], "energy_crit": 5.7, "collapse_checked": True,
                     "prompt_sha256": runner.canonical_hash(runner.messages_for(row))}
            trace_path.write_text(json.dumps(trace), encoding="utf-8")
            row["bio_provenance"] = {"producer": "brothertrue_core", "trace_path": str(trace_path),
                "trace_sha256": runner.sha256(trace_path), "energy_crit": 5.7, "collapse_checked": True}
            with patch.object(runner, "cloud_file", side_effect=Path):
                runner.validate_core_trace(row)
                for field, replacement in (("energy_crit", 6.), ("collapse_checked", False)):
                    altered = copy.deepcopy(row)
                    altered["bio_provenance"][field] = replacement
                    with self.assertRaises(ValueError):
                        runner.validate_core_trace(altered)
                altered = copy.deepcopy(row)
                altered["prompt"] = "Outra pergunta."
                with self.assertRaises(ValueError):
                    runner.validate_core_trace(altered)


if __name__ == "__main__":
    unittest.main()
