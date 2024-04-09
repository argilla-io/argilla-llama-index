from argilla_llama_index.helpers import _calc_time, _get_time_diff

import unittest
from unittest.mock import patch, MagicMock
from argilla_llama_index import ArgillaCallbackHandler


class TestArgillaCallbackHandler(unittest.TestCase):
    def setUp(self):
        self.dataset_name = "test_dataset_llama_index"
        self.workspace_name = "admin"
        self.api_url = "http://localhost:6900"
        self.api_key = "admin.apikey"

        self.handler = ArgillaCallbackHandler(
            dataset_name=self.dataset_name,
            workspace_name=self.workspace_name,
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

    def test_init(self):
        self.assertEqual(self.handler.dataset_name, self.dataset_name)
        self.assertEqual(self.handler.workspace_name, self.workspace_name)

    @patch("argilla_llama_index.llama_index_handler.rg.init")
    def test_init_connection_error(self, mock_init):
        mock_init.side_effect = ConnectionError("Connection failed")
        with self.assertRaises(ConnectionError):
            ArgillaCallbackHandler(
                dataset_name=self.dataset_name,
                workspace_name=self.workspace_name,
                api_url=self.api_url,
                api_key=self.api_key,
            )

    @patch("argilla_llama_index.llama_index_handler.rg.FeedbackDataset.list")
    @patch("argilla_llama_index.llama_index_handler.rg.FeedbackDataset.from_argilla")
    def test_init_file_not_found_error(self, mock_from_argilla, mock_list):
        mock_list.return_value = []
        mock_from_argilla.side_effect = FileNotFoundError("Dataset not found")
        with self.assertRaises(FileNotFoundError):
            ArgillaCallbackHandler(
                dataset_name=self.dataset_name,
                workspace_name=self.workspace_name,
                api_url=self.api_url,
                api_key=self.api_key,
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

    # TODO: Create a test for end_trace

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


if __name__ == "__main__":
    unittest.main()
