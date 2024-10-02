# Copyright 2024-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from argilla_llama_index.helpers import _create_svg, _create_tree_structure


class TestHelpers(unittest.TestCase):
    def test_create_tree_structure(self):
        span_data = [
            {"id_": "A", "parent_id": None, "duration": "10s"},
            {"id_": "B", "parent_id": "A", "duration": "15s"},
            {"id_": "C", "parent_id": "A", "duration": "5s"},
            {"id_": "D", "parent_id": "B", "duration": "20s"},
        ]
        event_data = []
        expected_output = [
            (0, 0, "A", "10s"),
            (1, 1, "B", "15s"),
            (2, 2, "D", "20s"),
            (3, 1, "C", "5s"),
        ]

        result = _create_tree_structure(span_data, event_data)
        self.assertEqual(result, expected_output)

    def test_create_svg(self):
        input_data = [(0, 1, "Node1", "10ms"), (1, 2, "Node2", "20ms")]

        result = _create_svg(input_data)

        self.assertIn('viewBox="0 0 750 90"', result)
        self.assertIn('<g transform="translate(30, 0)">', result)
        self.assertIn('<tspan x="0" y="0">Node1</tspan>', result)
        self.assertIn('<tspan x="0" y="0">10ms</tspan>', result)
        self.assertIn('<g transform="translate(60, 45)">', result)
        self.assertIn('<tspan x="0" y="0">Node2</tspan>', result)
        self.assertIn('<tspan x="0" y="0">20ms</tspan>', result)


if __name__ == "__main__":
    unittest.main()
