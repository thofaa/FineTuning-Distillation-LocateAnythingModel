import json
import tempfile
import unittest
from pathlib import Path

from scripts.preprocess_exdark import convert_annotations, normalize_box


class PreprocessExDarkTests(unittest.TestCase):
    def test_normalize_box_uses_1000_scale(self) -> None:
        x1, y1, x2, y2 = normalize_box(100, 50, 300, 150, image_width=400, image_height=200)
        self.assertEqual((x1, y1, x2, y2), (250, 250, 750, 750))

    def test_convert_annotations_writes_jsonl_prompt(self) -> None:
        payload = {
            "image_width": 200,
            "image_height": 100,
            "detections": [
                {
                    "label": "car",
                    "bounding_box": {"x_min": 20, "y_min": 10, "x_max": 100, "y_max": 40},
                    "luminance_level": "very_low",
                    "confidence": 0.95,
                    "reasoning": "dim object silhouette",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "ann.json"
            output_path = Path(td) / "out.jsonl"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            written = convert_annotations(input_path, output_path, image_width=None, image_height=None)
            self.assertEqual(written, 1)

            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertIn("<|object_ref_start|>car<|object_ref_end|>", record["prompt"])
            self.assertIn("<|box_start|>(100,100),(500,400)<|box_end|>", record["prompt"])


if __name__ == "__main__":
    unittest.main()
