import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import project


class XGTests(unittest.TestCase):
    def test_geometry_and_goal_label(self):
        distance, angle = project.shot_geometry(100, 40)
        self.assertAlmostEqual(distance, 20)
        self.assertGreater(angle, 0)
        event = {"id": "shot1", "type": {"name": "Shot"}, "period": 1,
                 "location": [100, 40], "shot": {"type": {"name": "Open Play"},
                 "outcome": {"name": "Goal"}, "body_part": {"name": "Right Foot"}}}
        self.assertEqual(project.event_to_row(event, 1)["goal"], 1)
        event["shot"]["type"]["name"] = "Penalty"
        self.assertIsNone(project.event_to_row(event, 1))

    def test_validation(self):
        example = json.loads((Path(__file__).resolve().parents[1] / "examples" / "shot.json").read_text())
        self.assertEqual(project.validate_shot(example)["distance"], 15)
        for invalid in ({**example, "x": True}, {**example, "y": 81},
                        {**example, "under_pressure": "false"}, {"x": 105}):
            with self.assertRaises(ValueError):
                project.validate_shot(invalid)

    def test_cache_is_reusable_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            path.write_text('{"ok": true}', encoding="utf-8")
            self.assertEqual(project.fetch_json("https://invalid.example", path), {"ok": True})


if __name__ == "__main__":
    unittest.main()
