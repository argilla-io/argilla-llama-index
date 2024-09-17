#  Copyright 2021-present, the Recognai S.L. team.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import inspect
import unittest
from collections import namedtuple
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, call, patch
from uuid import uuid4

from argilla import (
    Argilla,
    Workspace,
)
from argilla_llama_index.llama_index_handler import ArgillaSpanHandler
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.instrumentation.span.simple import SimpleSpan

CommonData = namedtuple(
    "CommonData",
    [
        "trace_id",
        "root_span_id",
        "id_",
        "bound_args",
        "instance",
        "parent_span_id",
        "tags",
        "span",
        "result",
    ],
)


class TestArgillaSpanHandlerLogToArgilla(unittest.TestCase):
    def setUp(self):
        self.dataset_name = f"test_dataset_llama_index_{uuid4()}"
        self.api_url = "http://localhost:6900"
        self.api_key = "argilla.apikey"
        self.workspace_name = "argilla"
        self.number_of_retrievals = 2
        self.client = Argilla(api_url=self.api_url, api_key=self.api_key)
        self._create_workspace("argilla")

        self.handler = ArgillaSpanHandler(
            dataset_name=self.dataset_name,
            api_url=self.api_url,
            api_key=self.api_key,
            workspace_name=self.workspace_name,
            number_of_retrievals=self.number_of_retrievals,
        )
        self.handler.open_spans = {}
        self.handler.trace_buffer = []
        self.handler.fields_info = {}
        self.handler.completed_spans = []
        self.handler.dropped_spans = []

        self.handler.dataset = MagicMock()
        self.handler.dataset.records.log = MagicMock()
        self.context_root_patcher = patch(
            "argilla_llama_index.llama_index_handler.context_root"
        )
        self.mock_context_root = self.context_root_patcher.start()

    def _create_workspace(self, workspace_name):
        workspace = self.client.workspaces(name=workspace_name)
        if workspace is None:
            workspace = Workspace(name=workspace_name)
            workspace.create()

    def _tearDown(self):
        self.context_root_patcher.stop()

    def _create_common_data(
        self, with_instance=False, with_span=False, with_result=False, **kwargs
    ):
        trace_id = kwargs.get("trace_id", "trace_id")
        root_span_id = kwargs.get("root_span_id", "QueryEngine.query")
        id_ = kwargs.get("id_", root_span_id)
        parent_span_id = kwargs.get("parent_span_id", "parent_span_id")
        tags = kwargs.get("tags", {"tag1": "value1"})
        bound_args = Mock(spec=inspect.BoundArguments)
        bound_args.arguments = kwargs.get(
            "arguments", {"message": "Test query message"}
        )

        instance = None
        span = None
        result = None

        if with_instance:
            instance = Mock(spec=BaseQueryEngine)
            instance.__class__ = BaseQueryEngine
        if with_span:
            span = SimpleSpan(
                id_=id_,
                parent_id=parent_span_id,
                tags=tags,
                start_time=datetime.now() - timedelta(seconds=5),
            )
        if with_result:
            result = Mock()
            result.response = kwargs.get("response", "Test response")

        return CommonData(
            trace_id,
            root_span_id,
            id_,
            bound_args,
            instance,
            parent_span_id,
            tags,
            span,
            result,
        )

    @patch("argilla_llama_index.llama_index_handler.Argilla")
    @patch.object(ArgillaSpanHandler, "_initialize_dataset")
    def test_initialization(self, mock_initialize_dataset, mock_argilla):
        dataset_name = "test_dataset"
        api_url = "http://example.com"
        api_key = "test_key"
        workspace_name = "test_workspace"
        number_of_retrievals = 5

        handler = ArgillaSpanHandler(
            dataset_name=dataset_name,
            api_url=api_url,
            api_key=api_key,
            workspace_name=workspace_name,
            number_of_retrievals=number_of_retrievals,
        )
        mock_argilla.assert_called_once_with(api_key=api_key, api_url=api_url)
        self.assertEqual(handler.dataset_name, dataset_name)
        self.assertEqual(handler.workspace_name, workspace_name)
        self.assertEqual(handler.number_of_retrievals, number_of_retrievals)
        self.assertEqual(handler.trace_buffer, [])
        self.assertEqual(handler.fields_info, {})
        self.assertIsInstance(handler.client, MagicMock)
        mock_initialize_dataset.assert_called_once()

    def test_new_span(self):
        data = self._create_common_data()
        self.mock_context_root.get.return_value = (data.trace_id, data.root_span_id)

        span = self.handler.new_span(
            id_=data.id_,
            bound_args=data.bound_args,
            instance=data.instance,
            parent_span_id=data.parent_span_id,
            tags=data.tags,
        )

        self.assertIsInstance(span, SimpleSpan)
        self.assertEqual(span.id_, data.id_)
        self.assertEqual(span.parent_id, data.parent_span_id)
        self.assertEqual(span.tags, data.tags)

    @patch.object(
        ArgillaSpanHandler, "_parse_output", return_value={"parsed": "output"}
    )
    def test_prepare_to_exit_span(self, mock_parse_output):
        data = self._create_common_data(id_="test_id", with_span=True)
        self.mock_context_root.get.return_value = (data.trace_id, data.root_span_id)
        self.handler.open_spans[data.id_] = data.span

        self.handler.prepare_to_exit_span(data.id_, data.bound_args)

        self.assertIsNotNone(data.span.end_time)
        self.assertAlmostEqual(data.span.duration, 5, delta=0.1)
        self.assertEqual(data.span.metadata, {"parsed": "output"})
        self.assertEqual(len(self.handler.trace_buffer), 1)
        self.assertIn(data.span, self.handler.completed_spans)

    def test_prepare_to_drop_span(self):
        data = self._create_common_data(with_span=True)
        self.mock_context_root.get.return_value = (data.trace_id, data.root_span_id)
        self.handler.open_spans[data.id_] = data.span

        self.handler.prepare_to_drop_span(id_=data.id_, bound_args=data.bound_args)

        self.assertEqual(self.handler.trace_buffer, [])
        self.assertEqual(self.handler.fields_info, {})
        self.mock_context_root.set.assert_called_once_with((None, None))
        self.assertIn(data.span, self.handler.dropped_spans)

    @patch("argilla_llama_index.llama_index_handler._create_tree_structure")
    @patch("argilla_llama_index.llama_index_handler._create_svg")
    @patch("argilla_llama_index.llama_index_handler.chat_to_html")
    def test_log_to_argilla(
        self, mock_chat_to_html, mock_create_svg, mock_create_tree_structure
    ):
        data = self._create_common_data()
        trace_buffer = [
            {
                "id_": "span_1",
                "parent_id": None,
                "tags": {},
                "start_time": 0,
                "end_time": 1,
                "duration": 1,
                "metadata": {},
            }
        ]
        fields_info = {
            "query": "test_query",
            "response": "test_response",
            "retrieved_document_1_text": "doc1",
            "retrieved_document_2_text": "doc2",
        }
        mock_chat_to_html.return_value = "chat_html"
        mock_create_tree_structure.return_value = "tree_structure"
        mock_create_svg.return_value = "svg_tree"

        self.handler._log_to_argilla(
            trace_id=data.trace_id, trace_buffer=trace_buffer, fields_info=fields_info
        )

        self.handler.dataset.records.log.assert_called_once()
        records = self.handler.dataset.records.log.call_args[1]["records"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].id, data.trace_id)
        self.assertIn("chat", records[0].fields)
        self.assertIn("time-details", records[0].fields)
        for i in range(1, self.number_of_retrievals + 1):
            self.assertIn(f"retrieved_document_{i}", records[0].fields)
        self.assertNotIn("retrieved_document_3", records[0].fields)

    def test_parse_output_query_engine(self):
        data = self._create_common_data(with_instance=True, with_result=True)

        out_metadata = self.handler._parse_output(
            instance=data.instance, bound_args=data.bound_args, result=data.result
        )

        self.assertEqual(
            self.handler.fields_info["query"], data.bound_args.arguments["message"]
        )
        self.assertEqual(self.handler.fields_info["response"], data.result.response)
        self.assertEqual(out_metadata, {})

    def test_parse_output_nodes(self):
        node1 = Mock(spec=["metadata", "node", "text"])
        node1.metadata = {
            "file_name": "file1.txt",
            "file_size": 123,
        }
        node1.node = Mock(spec=["start_char_idx", "end_char_idx"])
        node1.node.start_char_idx = 0
        node1.node.end_char_idx = 50
        node1.text = "Document 1 text"

        node2 = Mock(spec=["metadata", "node", "text", "score"])
        node2.metadata = {
            "file_name": "file2.txt",
            "file_type": "text",
            "file_size": 456,
        }
        node2.score = 0.8
        node2.node = Mock(spec=[])
        node2.text = "Document 2 text"

        data = self._create_common_data(
            with_instance=True, with_result=True, arguments={"nodes": [node1, node2]}
        )

        out_metadata = self.handler._parse_output(
            instance=data.instance, bound_args=data.bound_args, result=data.result
        )

        expected_metadata = {
            "retrieved_document_1_file_name": "file1.txt",
            "retrieved_document_1_file_type": "unknown",
            "retrieved_document_1_file_size": 123,
            "retrieved_document_1_score": 0,
            "retrieved_document_1_start_char": 0,
            "retrieved_document_1_end_char": 50,
            "retrieved_document_2_file_name": "file2.txt",
            "retrieved_document_2_file_type": "text",
            "retrieved_document_2_file_size": 456,
            "retrieved_document_2_score": 0.8,
            "retrieved_document_2_start_char": -1,
            "retrieved_document_2_end_char": -1,
        }

        self.assertEqual(out_metadata, expected_metadata)
        self.assertEqual(
            self.handler.fields_info["retrieved_document_1_text"],
            "DOCUMENT SCORE: 0\n\nDocument 1 text",
        )
        self.assertEqual(
            self.handler.fields_info["retrieved_document_2_text"],
            "DOCUMENT SCORE: 0.8\n\nDocument 2 text",
        )

    @patch("argilla_llama_index.llama_index_handler.TermsMetadataProperty")
    @patch("argilla_llama_index.llama_index_handler.IntegerMetadataProperty")
    @patch("argilla_llama_index.llama_index_handler.FloatMetadataProperty")
    def test_add_metadata_properties(
        self,
        mock_float_prop_class,
        mock_int_prop_class,
        mock_terms_prop_class,
    ):
        existing_metadata_property = Mock()
        existing_metadata_property.name = "existing_metadata"
        existing_metadata = [existing_metadata_property]
        self.handler.dataset.settings.metadata = MagicMock()
        self.handler.dataset.settings.metadata.__iter__.return_value = iter(
            existing_metadata
        )
        self.handler.dataset.settings.metadata.add = Mock()
        self.handler.dataset.update = Mock()

        metadata = {
            "new_string_property": "test",
            "new_int_property": 42,
            "new_float_property": 3.14,
        }

        property_classes = [
            ("new_string_property", mock_terms_prop_class),
            ("new_int_property", mock_int_prop_class),
            ("new_float_property", mock_float_prop_class),
        ]
        mock_properties = []
        for prop_name, mock_class in property_classes:
            mock_instance = Mock()
            mock_instance.name = prop_name
            mock_class.return_value = mock_instance
            mock_properties.append((prop_name, mock_class, mock_instance))

        self.handler._add_metadata_properties(metadata)

        for prop_name, mock_class, _ in mock_properties:
            mock_class.assert_called_once_with(name=prop_name)
        expected_calls = [
            call(mock_instance) for _, _, mock_instance in mock_properties
        ]
        self.handler.dataset.settings.metadata.add.assert_has_calls(
            expected_calls, any_order=True
        )
        self.assertEqual(
            self.handler.dataset.settings.metadata.add.call_count, len(expected_calls)
        )
        self.handler.dataset.update.assert_called_once()


if __name__ == "__main__":
    unittest.main()
