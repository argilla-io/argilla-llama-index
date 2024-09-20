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
import logging
import os
import uuid
from contextvars import ContextVar
from datetime import datetime
from itertools import islice
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from argilla import (
    Argilla,
    ChatField,
    Dataset,
    FloatMetadataProperty,
    IntegerMetadataProperty,
    RatingQuestion,
    Record,
    Settings,
    TermsMetadataProperty,
    TextField,
    TextQuestion,
)

# from argilla.markdown import chat_to_html
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.instrumentation.span.simple import SimpleSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler

from argilla_llama_index.helpers import _create_svg, _create_tree_structure

context_root: ContextVar[Union[Tuple[str, str], Tuple[None, None]]] = ContextVar(
    "context_root", default=(None, None)
)


class ArgillaHandler(BaseSpanHandler[SimpleSpan], extra="allow"):
    """
    Span handler that logs predictions to Argilla.

    This handler automatically logs the predictions made with LlamaIndex to Argilla,
    without the need to create a dataset and log the predictions manually. Events relevant
    to the predictions are automatically logged to Argilla as well, including timestamps of
    all the different steps of the retrieval and prediction process.

    Attributes:
        dataset_name (str): The name of the Argilla dataset.
        api_url (str): Argilla API URL.
        api_key (str): Argilla API key.
        number_of_retrievals (int): The number of retrievals to log. By default, it is set to 2.
        workspace_name (str): The name of the Argilla workspace. By default, it will use the first available workspace.

    Usage:
        ```python
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
        import llama_index.core.instrumentation as instrument
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.llms.openai import OpenAI

        from argilla_llama_index import ArgillaHandler

        span_handler = ArgillaHandler(
            dataset_name="query_model",
            api_url="http://localhost:6900",
            api_key="argilla.apikey",
            number_of_retrievals=2,
        )

        dispatcher = instrument.get_dispatcher().add_span_handler(span_handler)

        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.8, openai_api_key=os.getenv("OPENAI_API_KEY"))

        documents = SimpleDirectoryReader("../../data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(similarity_top_k=2)

        response = query_engine.query("What did the author do growing up?")
        ```
    """

    def __init__(
        self,
        dataset_name: str,
        api_url: str,
        api_key: str,
        workspace_name: Optional[str] = None,
        number_of_retrievals: int = 2,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.workspace_name = workspace_name

        if number_of_retrievals < 0:
            raise ValueError(
                "The number of retrievals must be 0 (to show no retrieved documents) or a positive number."
            )
        self.number_of_retrievals = number_of_retrievals

        if (api_url is None and os.getenv("ARGILLA_API_URL") is None) or (
            api_key is None and os.getenv("ARGILLA_API_KEY") is None
        ):
            raise ValueError(
                "Both `api_url` and `api_key` must be set. The current values are: "
                f"`api_url`={api_url} and `api_key`={api_key}."
            )
        self.client = Argilla(api_key=api_key, api_url=api_url)

        self.trace_buffer: List[SimpleSpan] = []
        self.fields_info: Dict[str, Any] = {}

        self._initialize_dataset()

    def _initialize_dataset(self):
        """Create the dataset in Argilla if it does not exist, or update it if it does."""

        self.settings = Settings(
            fields=[
                ChatField(name="chat", title="Chat", use_markdown=False, required=True),
            ]
            + self._add_context_fields(self.number_of_retrievals)
            + [
                TextField(
                    name="time-details", title="Time Details", use_markdown=False
                ),
            ],
            questions=[
                RatingQuestion(
                    name="response-rating",
                    title="Rating for the response",
                    description="How would you rate the quality of the response?",
                    values=[1, 2, 3, 4, 5, 6, 7],
                    required=True,
                ),
                TextQuestion(
                    name="response-feedback",
                    title="Feedback for the response",
                    description="What feedback do you have for the response?",
                    required=False,
                ),
            ]
            + self._add_context_questions(self.number_of_retrievals),
            guidelines="You're asked to rate the quality of the response and provide feedback.",
            allow_extra_metadata=True,
        )

        # Either create a new dataset or use an existing one, updating it if necessary
        try:
            dataset_names = [ds.name for ds in self.client.datasets]
            if self.dataset_name not in dataset_names:
                dataset = Dataset(
                    name=self.dataset_name,
                    workspace=self.workspace_name,
                    settings=self.settings,
                )
                self.dataset = dataset.create()
                logging.info(
                    f"A new dataset with the name '{self.dataset_name}' has been created.",
                )
            else:
                # Update the existing dataset. If the fields and questions do not match,
                # a new dataset will be created with the -updated flag in the name.
                self.dataset = self.client.datasets(
                    name=self.dataset_name,
                    workspace=self.workspace_name,
                )
                if self.number_of_retrievals > 0:
                    required_context_fields = self._add_context_fields(
                        self.number_of_retrievals
                    )
                    required_context_questions = self._add_context_questions(
                        self.number_of_retrievals
                    )
                    existing_fields = list(self.dataset.fields)
                    existing_questions = list(self.dataset.questions)

                    if not (
                        all(
                            element in existing_fields
                            for element in required_context_fields
                        )
                        and all(
                            element in existing_questions
                            for element in required_context_questions
                        )
                    ):
                        self.dataset = Dataset(
                            name=f"{self.dataset_name}-updated",
                            workspace=self.workspace_name,
                            settings=self.settings,
                        )
                        self.dataset = self.dataset.create()
                        logging.info(
                            f"A new dataset with the name '{self.dataset_name}-updated' has been created.",
                        )
        except Exception as e:
            raise FileNotFoundError(
                f"`Dataset` creation or update failed with exception `{e}`."
                f" If the problem persists, please report it to https://github.com/argilla-io/argilla/issues/ "
                f"as an `integration` issue."
            ) from e

        supported_context_fields = ["retrieved_document_scores"] + [
            f"retrieved_document_{i+1}" for i in range(self.number_of_retrievals)
        ]
        supported_fields = ["chat"] + supported_context_fields + ["time-details"]
        if supported_fields != [field.name for field in self.dataset.fields]:
            raise ValueError(
                f"`Dataset` with name={self.dataset_name} had fields that are not supported"
                f"for the `llama-index` integration. Supported fields are {supported_fields}."
                f" Current fields are {[field.name for field in self.dataset.fields]}."
            )

    def _add_context_fields(self, number_of_retrievals: int) -> List[Any]:
        """Create the context fields to be added to the dataset."""
        context_scores = [
            TextField(
                name="retrieved_document_scores",
                title="Retrieved document scores",
                use_markdown=True,
                required=False,
            )
        ]
        context_fields = [
            TextField(
                name=f"retrieved_document_{doc+1}",
                title=f"Retrieved document {doc+1}",
                use_markdown=True,
                required=False,
            )
            for doc in range(number_of_retrievals)
        ]
        return context_scores + context_fields

    def _add_context_questions(self, number_of_retrievals: int) -> List[Any]:
        """Create the context questions to be added to the dataset."""
        rating_questions = [
            RatingQuestion(
                name=f"rating_retrieved_document_{doc + 1}",
                title=f"Rate the relevance of the Retrieved document {doc + 1}, if present.",
                values=list(range(1, 8)),
                description=f"Rate the relevance of the retrieved document {doc + 1}, if present.",
                required=False,
            )
            for doc in range(number_of_retrievals)
        ]
        return rating_questions

    def class_name(cls) -> str:
        """Class name."""
        return "ArgillaHandler"

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[SimpleSpan]:
        """
        Create a new span using the SimpleSpan class. If the span is the root span, it generates a new trace ID.

        Args:
            id_ (str): The unique identifier for the new span.
            bound_args (inspect.BoundArguments): The arguments that were bound to when the span was created.
            instance (Optional[Any], optional): The instance associated with the span, if present. Defaults to None.
            parent_span_id (Optional[str], optional): The identifier of the parent span. Defaults to None.
            tags (Optional[Dict[str, Any]], optional): Additional information about the span. Defaults to None.

        Returns:
            Optional[SimpleSpan]: The newly created SimpleSpan object if the span is successfully created.
        """
        trace_id, root_span_id = context_root.get()

        if not parent_span_id:
            trace_id = str(uuid.uuid4())
            root_span_id = id_
            context_root.set((trace_id, root_span_id))

        return SimpleSpan(id_=id_, parent_id=parent_span_id, tags=tags or {})

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[SimpleSpan]:
        """
        Logic to exit the span. It stores the span information in the trace buffer.
        If the trace has ended and and belongs to specific components, it logs the buffered data to Argilla.

            Args:
                id_ (str): The unique identifier of the span to be exited.
                bound_args (inspect.BoundArguments): The arguments that were bound to the span's function during its invocation.
                instance (Optional[Any], optional):  The instance associated with the span, if applicable.. Defaults to None.
                result (Optional[Any], optional): The output or result produced by the span's execution.. Defaults to None.

            Returns:
                Optional[SimpleSpan]:  The exited SimpleSpan object if the span exists and the trace is active; otherwise, None.
        """
        trace_id, root_span_id = context_root.get()
        if not trace_id:
            return None

        span = self.open_spans[id_]
        span = cast(SimpleSpan, span)
        span.end_time = datetime.now()
        span.duration = round((span.end_time - span.start_time).total_seconds(), 4)
        span.metadata = self._parse_output(instance, bound_args, result)

        self.trace_buffer.append(
            {
                "id_": span.id_,
                "parent_id": span.parent_id,
                "start_time": span.start_time.timestamp(),
                "end_time": span.end_time.timestamp(),
                "duration": span.duration,
                "metadata": span.metadata,
            }
        )

        if id_ == root_span_id and any(
            term.lower() in id_.lower() for term in ["AgentRunner", "QueryEngine"]
        ):
            self._log_to_argilla(
                trace_id=trace_id,
                trace_buffer=self.trace_buffer,
                fields_info=self.fields_info,
            )
            self.trace_buffer.clear()
            self.fields_info.clear()
            context_root.set((None, None))
        elif id_ == root_span_id:
            self.trace_buffer.clear()
            self.fields_info.clear()
            context_root.set((None, None))

        with self.lock:
            self.completed_spans += [span]
        return span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> None:
        """
        Logic to drop the span. If the trace has ended, it clears the data.

        Args:
            id_ (str): The unique identifier of the span to be dropped.
            bound_args (inspect.BoundArguments): The arguments that were bound to the span function during its invocation.
            instance (Optional[Any], optional): The instance associated with the span, if applicable. Defaults to None.
            err (Optional[BaseException], optional): An exception that caused the span to be dropped, if any. Defaults to None.

        Returns:
            None:
        """
        trace_id, root_span_id = context_root.get()
        if not trace_id:
            return None

        if id_ == root_span_id:
            self.trace_buffer.clear()
            self.fields_info.clear()
            context_root.set((None, None))

        if id_ in self.open_spans:
            with self.lock:
                span = self.open_spans[id_]
                self.dropped_spans += [span]
        return None

    def _log_to_argilla(
        self,
        trace_id: str,
        trace_buffer: List[Dict[str, Any]],
        fields_info: Dict[str, Any],
    ) -> None:
        """Logs the data in the trace buffer to Argilla."""
        if not trace_buffer:
            return None

        message = [
            {"role": "user", "content": fields_info["query"]},
            {"role": "assistant", "content": fields_info["response"]},
        ]
        tree_structure = _create_tree_structure(trace_buffer)
        tree = _create_svg(tree_structure)

        fields = {
            "chat": message,
            "time-details": tree,
        }
        if self.number_of_retrievals > 0:
            score_keys = filter(lambda k: k.endswith("_score"), fields_info.keys())
            text_keys = filter(lambda k: k.endswith("_text"), fields_info.keys())

            scores = "\n".join(
                f"{key.replace('_score', '').replace('_', ' ').capitalize()}: {fields_info[key]}"
                for key in islice(score_keys, self.number_of_retrievals)
            )
            fields["retrieved_document_scores"] = scores

            for key in islice(text_keys, self.number_of_retrievals):
                idx = key.split("_")[-2]
                fields[f"retrieved_document_{idx}"] = fields_info[key]

        metadata = self._process_metadata(trace_buffer)
        self._add_metadata_properties(metadata)

        records = [Record(id=trace_id, fields=fields, metadata=metadata)]
        self.dataset.records.log(records=records)

    def _parse_output(
        self,
        instance: Optional[Any],
        bound_args: inspect.BoundArguments,
        result: Optional[Any],
    ) -> Dict[str, Any]:
        """Parse the output of the span to extract metadata."""
        out_metadata = {}

        if isinstance(instance, BaseQueryEngine):
            if message := bound_args.arguments.get("message"):
                self.fields_info["query"] = message
            elif query_bundle := bound_args.arguments.get("query_bundle"):
                self.fields_info["query"] = query_bundle.query_str
            self.fields_info["response"] = result.response

        if nodes := bound_args.arguments.get("nodes"):
            for i, n in enumerate(nodes):
                out_metadata[f"retrieved_document_{i+1}_file_name"] = n.metadata.get(
                    "file_name", "unknown"
                )
                out_metadata[f"retrieved_document_{i+1}_file_type"] = n.metadata.get(
                    "file_type", "unknown"
                )
                out_metadata[f"retrieved_document_{i+1}_file_size"] = n.metadata.get(
                    "file_size", 0
                )

                node = getattr(n, "node", None)
                out_metadata[f"retrieved_document_{i+1}_start_char"] = getattr(
                    node, "start_char_idx", -1
                )
                out_metadata[f"retrieved_document_{i+1}_end_char"] = getattr(
                    node, "end_char_idx", -1
                )

                text = getattr(n, "text", "")
                score = getattr(n, "score", 0)
                out_metadata[f"retrieved_document_{i+1}_score"] = score
                self.fields_info[f"retrieved_document_{i+1}_score"] = score
                self.fields_info[f"retrieved_document_{i+1}_text"] = text
        return out_metadata

    def _process_metadata(self, trace_buffer: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process the metadata to be logged to Argilla."""
        metadata_to_log = {}

        for span in trace_buffer:
            key_prefix = span["id_"].split(".")[0].lower()

            metadata_to_log[f"{key_prefix}_start_time"] = span["start_time"]
            metadata_to_log[f"{key_prefix}_end_time"] = span["end_time"]
            metadata_to_log[f"{key_prefix}_duration"] = span["duration"]
            if span["metadata"]:
                metadata_to_log.update(span["metadata"])

        metadata_to_log["total_duration"] = sum(
            span["duration"] for span in trace_buffer
        )
        metadata_to_log["total_spans"] = len(trace_buffer)

        return metadata_to_log

    def _add_metadata_properties(self, metadata: Dict[str, Any]) -> None:
        """Add metadata properties to the dataset if they do not exist."""
        existing_metadata = [
            existing_metadata.name
            for existing_metadata in self.dataset.settings.metadata
        ]
        for mt in metadata.keys():
            if mt not in existing_metadata:
                if isinstance(metadata[mt], str):
                    self.dataset.settings.metadata.add(TermsMetadataProperty(name=mt))

                elif isinstance(metadata[mt], int):
                    self.dataset.settings.metadata.add(IntegerMetadataProperty(name=mt))

                elif isinstance(metadata[mt], float):
                    self.dataset.settings.metadata.add(FloatMetadataProperty(name=mt))

        self.dataset.update()
