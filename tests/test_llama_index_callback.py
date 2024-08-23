import unittest
import argilla as rg

from collections import defaultdict
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from argilla_llama_index import ArgillaCallbackHandler
from argilla_llama_index.helpers import _calc_time, _create_svg, _get_time_diff


class TestArgillaCallbackHandler(unittest.TestCase):
    def setUp(self):
        self.dataset_name = "test_dataset_llama_index"
        self.api_url = "http://localhost:6900"
        self.api_key = "argilla.apikey"

        self.client = rg.Argilla(api_url=self.api_url, api_key=self.api_key)
        self.create_workspace("argilla")

        self.handler = ArgillaCallbackHandler(
            dataset_name=self.dataset_name,
            api_url=self.api_url,
            api_key=self.api_key,
        )

        self.events_data = MagicMock()
        self.data_to_log = MagicMock()
        self.components_to_log = MagicMock()
        self._ignore_components_in_tree = MagicMock()
        self.trace_map = MagicMock()

        self.tree_structure_dict = {
            "root": ["query"],
            "query": ["retrieve", "synthesize"],
            "synthesize": ["llm", "grandchild1"],
        }

    def create_workspace(self, workspace_name):
        workspace = self.client.workspaces(name=workspace_name)
        if workspace is None:
            workspace = rg.Workspace(name=workspace_name)
            workspace.create()

    def test_init(self):
        self.assertEqual(self.handler.dataset_name, self.dataset_name)

    @patch("argilla_llama_index.llama_index_handler.rg.Argilla")
    def test_init_connection_error(self, mock_init):
        mock_init.side_effect = ConnectionError("Connection failed")
        with self.assertRaises(ConnectionError):
            ArgillaCallbackHandler(
                dataset_name=self.dataset_name,
                api_url=self.api_url,
                api_key=self.api_key,
            )

    @patch("argilla_llama_index.llama_index_handler.rg.Argilla.datasets")
    @patch("argilla_llama_index.llama_index_handler.rg.Argilla._validate_connection")
    def test_init_file_not_found_error(self, mock_validate_connection, mock_list):
        mock_list.return_value = []
        mock_validate_connection.return_value = None
        with self.assertRaises(FileNotFoundError):
            ArgillaCallbackHandler(
                dataset_name="test_dataset",
                api_url="http://example.com",
                api_key="test_key",
            )

    def test_check_components_for_tree(self):
        self.handler._check_components_for_tree(self.tree_structure_dict)

    def test_get_events_map_with_names(self):

        trace_map = {"query": ["retrieve"], "llm": []}
        events_map = self.handler._get_events_map_with_names(
            self.events_data, trace_map
        )
        self.assertIsInstance(events_map, dict)
        self.assertEqual(len(events_map), 2)

    def test_extract_and_log_info(self):

        tree_structure_dict = self.handler._check_components_for_tree(
            self.tree_structure_dict
        )
        self.handler._extract_and_log_info(self.events_data, tree_structure_dict)

    def test_start_trace(self):
        self.handler.start_trace()
        self.assertIsNotNone(self.handler._start_time)
        self.assertEqual(self.handler._trace_map, defaultdict(list))

    @patch(
        "argilla_llama_index.llama_index_handler.ArgillaCallbackHandler._create_root_and_other_nodes"
    )
    @patch(
        "argilla_llama_index.llama_index_handler.ArgillaCallbackHandler._extract_and_log_info"
    )
    def test_end_trace(
        self, mock_extract_and_log_info, mock_create_root_and_other_nodes
    ):
        self.handler.start_trace()
        trace_id = "test_trace_id"
        trace_map = {"test_key": ["test_value"]}

        self.handler.end_trace(trace_id=trace_id, trace_map=trace_map)
        self.assertIsNotNone(self.handler._end_time)
        self.assertAlmostEqual(
            self.handler._end_time, datetime.now(), delta=timedelta(seconds=1)
        )
        self.assertEqual(self.handler._trace_map, trace_map)

        mock_create_root_and_other_nodes.assert_called_once_with(trace_map)
        mock_extract_and_log_info.assert_called_once_with(
            self.handler.events_data, trace_map
        )

    def test_on_event_start(self):
        event_type = "event1"
        payload = {}
        event_id = "123"
        parent_id = "456"
        self.handler.on_event_start(event_type, payload, event_id, parent_id)

    def test_on_event_end(self):
        event_type = "event1"
        payload = {}
        event_id = "123"
        self.handler.on_event_end(event_type, payload, event_id)

    def test_get_time_diff(self):
        event_1_time_str = "01/11/2024, 17:01:04.328656"
        event_2_time_str = "01/11/2024, 17:02:07.328523"
        time_diff = _get_time_diff(event_1_time_str, event_2_time_str)
        self.assertIsInstance(time_diff, float)

    def test_calc_time(self):
        id = "event1"
        self.events_data.__getitem__().__getitem__().time = (
            "01/11/2024, 17:01:04.328656"
        )
        self.events_data.__getitem__().__getitem__().time = (
            "01/11/2024, 17:02:07.328523"
        )
        time = _calc_time(self.events_data, id)
        self.assertIsInstance(time, float)

    def test_create_svg(self):
        data = [
            (0, 1, "Node1", "10ms"),
            (1, 2, "Node2", "20ms")
        ]

        result = _create_svg(data)

        self.assertIn('viewBox="0 0 750 108"', result)
        self.assertIn('<g transform="translate(40, 0)">', result)
        self.assertIn('<tspan x="0" y="0">Node1</tspan>', result)
        self.assertIn('<tspan x="0" y="0">10ms</tspan>', result)
        self.assertIn('<g transform="translate(80, 54)">', result)
        self.assertIn('<tspan x="0" y="0">Node2</tspan>', result)
        self.assertIn('<tspan x="0" y="0">20ms</tspan>', result)


if __name__ == "__main__":
    unittest.main()
