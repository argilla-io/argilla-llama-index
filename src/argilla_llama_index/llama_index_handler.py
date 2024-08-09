import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import argilla as rg
from argilla.markdown import chat_to_html
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import (
    CBEvent,
    CBEventType,
    EventPayload,
)
from packaging.version import parse

from argilla_llama_index.helpers import _calc_time, _create_svg, _get_time_diff


class ArgillaCallbackHandler(BaseCallbackHandler):
    """
    Callback handler that logs predictions to Argilla.

    This handler automatically logs the predictions made with LlamaIndex to Argilla,
    without the need to create a dataset and log the predictions manually. Events relevant
    to the predictions are automatically logged to Argilla as well, including timestamps of
    all the different steps of the retrieval and prediction process.

    Attributes:
        dataset_name (str): The name of the Argilla dataset.
        api_url (str): Argilla API URL.
        api_key (str): Argilla API key.
        number_of_retrievals (int): The number of retrievals to log. By default, it is set to 0.
        workspace_name (str): The name of the Argilla workspace. By default, it will use the first available workspace.
        event_starts_to_ignore (List[CBEventType]): List of event types to ignore at the start of the trace.
        event_ends_to_ignore (List[CBEventType]): List of event types to ignore at the end of the trace.
        handlers (List[BaseCallbackHandler]): List of extra handlers to include.

    Methods:
        start_trace(trace_id: Optional[str] = None) -> None:
            Logic to be executed at the beginning of the tracing process.

        end_trace(trace_id: Optional[str] = None, trace_map: Optional[Dict[str, List[str]]] = None) -> None:
            Logic to be executed at the end of the tracing process.

        on_event_start(event_type: CBEventType, payload: Optional[Dict[str, Any]] = None, event_id: Optional[str] = None, parent_id: str = None) -> str:
            Store event start data by event type. Executed at the start of an event.

        on_event_end(event_type: CBEventType, payload: Optional[Dict[str, Any]] = None, event_id: str = None) -> None:
            Store event end data by event type. Executed at the end of an event.

    Usage:
    ```python
    from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, set_global_handler
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.llms.openai import OpenAI

    set_global_handler("argilla",
        api_url="http://localhost:6900",
        api_key="argilla.apikey",
        dataset_name="query_model",
        number_of_retrievals=2
    )

    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.8, openai_api_key=os.getenv("OPENAI_API_KEY"))

    documents = SimpleDirectoryReader("../../data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    response = query_engine.query("What did the author do growing up?")
    ```
    """

    def __init__(  # noqa: C901
        self,
        dataset_name: str,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        number_of_retrievals: int = 0,
        workspace_name: Optional[str] = None,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
        handlers: Optional[List[BaseCallbackHandler]] = None,
    ) -> None:
        self.event_starts_to_ignore = event_starts_to_ignore or []
        self.event_ends_to_ignore = event_ends_to_ignore or []
        self.handlers = handlers or []
        self.number_of_retrievals = number_of_retrievals

        self.ARGILLA_VERSION = rg.__version__

        if parse(self.ARGILLA_VERSION) < parse("2.0.0"):
            raise ImportError(
                f"The installed `argilla` version is {self.ARGILLA_VERSION} but "
                "`ArgillaCallbackHandler` requires at least version 2.0.0. Please "
                "upgrade `argilla` with `pip install --upgrade argilla`."
            )

        if (api_url is None and os.getenv("ARGILLA_API_URL") is None) or (
            api_key is None and os.getenv("ARGILLA_API_KEY") is None
        ):
            raise ValueError(
                "Both `api_url` and `api_key` must be set. The current values are: "
                f"`api_url`={api_url} and `api_key`={api_key}."
            )

        client = rg.Argilla(api_key=api_key, api_url=api_url)

        self.dataset_name = dataset_name
        self.workspace_name = workspace_name
        self.settings = rg.Settings(
            fields=[
                rg.TextField(
                    name="chat", title="Chat", use_markdown=True, required=True
                ),
                rg.TextField(
                    name="time-details", title="Time Details", use_markdown=True
                ),
            ]
            + self._add_context_fields(number_of_retrievals),
            questions=[
                rg.RatingQuestion(
                    name="response-rating",
                    title="Rating for the response",
                    description="How would you rate the quality of the response?",
                    values=[1, 2, 3, 4, 5, 6, 7],
                    required=True,
                ),
                rg.TextQuestion(
                    name="response-feedback",
                    title="Feedback for the response",
                    description="What feedback do you have for the response?",
                    required=False,
                ),
            ]
            + self._add_context_questions(number_of_retrievals),
            guidelines="You're asked to rate the quality of the response and provide feedback.",
            allow_extra_metadata=True,
        )

        # Either create a new dataset or use an existing one, updating it if necessary
        try:
            dataset_names = [ds.name for ds in client.datasets]

            if self.dataset_name not in dataset_names:
                dataset = rg.Dataset(
                    name=self.dataset_name,
                    workspace=self.workspace_name,
                    settings=self.settings,
                )
                self.dataset = dataset.create()
                self.is_new_dataset_created = True
                logging.info(
                    f"A new dataset with the name '{self.dataset_name}' has been created.",
                )

            else:
                # Update the existing dataset. If the fields and questions do not match,
                # a new dataset will be created with the -updated flag in the name.
                self.dataset = client.datasets(
                    name=self.dataset_name,
                    workspace=self.workspace_name,
                )
                self.is_new_dataset_created = False

                if number_of_retrievals > 0:
                    required_context_fields = self._add_context_fields(
                        number_of_retrievals
                    )
                    required_context_questions = self._add_context_questions(
                        number_of_retrievals
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
                        self.dataset = rg.Dataset(
                            name=f"{self.dataset_name}-updated",
                            workspace=self.workspace_name,
                            settings=self.settings,
                        )
                        self.dataset = self.dataset.create()

        except Exception as e:
            raise FileNotFoundError(
                f"`Dataset` creation or update failed with exception `{e}`."
                f" If the problem persists, please report it to https://github.com/argilla-io/argilla/issues/ "
                f"as an `integration` issue."
            ) from e

        supported_context_fields = [
            f"retrieved_document_{i+1}" for i in range(number_of_retrievals)
        ]
        supported_fields = ["chat", "time-details"] + supported_context_fields
        if supported_fields != [field.name for field in self.dataset.fields]:
            raise ValueError(
                f"`Dataset` with name={self.dataset_name} had fields that are not supported"
                f"for the `llama-index` integration. Supported fields are {supported_fields}."
                f" Current fields are {[field.name for field in self.dataset.fields]}."
            )

        self.events_data: Dict[str, List[CBEvent]] = defaultdict(list)
        self.event_map_id_to_name = {}
        self._ignore_components_in_tree = ["templating"]
        self.components_to_log = set()
        self.event_ids_traced = set()

    def _add_context_fields(self, number_of_retrievals: int) -> List[Any]:
        """Create the context fields to be added to the dataset."""
        context_fields = [
            rg.TextField(
                name=f"retrieved_document_{doc + 1}",
                title=f"Retrieved document {doc + 1}",
                use_markdown=True,
                required=False,
            )
            for doc in range(number_of_retrievals)
        ]
        return context_fields

    def _add_context_questions(self, number_of_retrievals: int) -> List[Any]:
        """Create the context questions to be added to the dataset."""
        rating_questions = [
            rg.RatingQuestion(
                name=f"rating_retrieved_document_{doc + 1}",
                title=f"Rate the relevance of the Retrieved document {doc + 1} (if present)",
                values=list(range(1, 8)),
                description=f"Rate the relevance of the retrieved document {doc + 1}.",
                required=False,
            )
            for doc in range(number_of_retrievals)
        ]
        return rating_questions

    def _create_root_and_other_nodes(self, trace_map: Dict[str, List[str]]) -> None:
        """Create the root node and the other nodes in the tree."""
        self.root_node = self._get_event_name_by_id(trace_map["root"][0])
        self.event_ids_traced = set(trace_map.keys()) - {"root"}
        self.event_ids_traced.update(*trace_map.values())
        for id in self.event_ids_traced:
            self.components_to_log.add(self._get_event_name_by_id(id))

    def _get_event_name_by_id(self, event_id: str) -> str:
        """Get the name of the event by its id."""
        return str(self.events_data[event_id][0].event_type).split(".")[1].lower()

        # TODO: If we have a component more than once, properties currently don't account for those after the first one and get overwritten

    def _add_missing_metadata(
        self,
        dataset: rg.Dataset,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add missing metadata properties to the dataset."""

        for mt in metadata.keys():
            if mt not in [metadata.name for metadata in self.dataset.settings.metadata]:
                if mt.endswith("_time"):
                    self.dataset.settings.metadata.add(
                        rg.FloatMetadataProperty(name=mt, title=mt)
                    )
                    dataset.update()

    def _check_components_for_tree(
        self, tree_structure_dict: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Check whether the components in the tree are in the components to log.
        Removes components that are not in the components to log so that they are not shown in the tree.
        """
        final_components_in_tree = self.components_to_log.copy()
        final_components_in_tree.add("root")
        for component in self._ignore_components_in_tree:
            if component in final_components_in_tree:
                final_components_in_tree.remove(component)
        for key in list(tree_structure_dict.keys()):
            if key.strip("0") not in final_components_in_tree:
                del tree_structure_dict[key]
        for key, value in tree_structure_dict.items():
            if isinstance(value, list):
                tree_structure_dict[key] = [
                    element
                    for element in value
                    if element.strip("0") in final_components_in_tree
                ]
        return tree_structure_dict

    def _get_events_map_with_names(
        self, events_data: Dict[str, List[CBEvent]], trace_map: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Returns a dictionary where trace_map is mapped with the event names instead of the event ids.
        Also returns a set of the event ids that were traced.
        """
        self.event_map_id_to_name = {}
        for event_id in self.event_ids_traced:
            event_name = str(events_data[event_id][0].event_type).split(".")[1].lower()
            while event_name in self.event_map_id_to_name.values():
                event_name = event_name + "0"
            self.event_map_id_to_name[event_id] = event_name
        events_trace_map = {
            self.event_map_id_to_name.get(k, k): [
                self.event_map_id_to_name.get(v, v) for v in values
            ]
            for k, values in trace_map.items()
        }

        return events_trace_map

    def _extract_and_log_info(
        self, events_data: Dict[str, List[CBEvent]], trace_map: Dict[str, List[str]]
    ) -> None:
        """
        Main function that extracts the information from the events and logs it to Argilla.
        We currently log data if the root node is either "agent_step" or "query".
        Otherwise, we do not log anything.
        If we want to account for more root nodes, we just need to add them to the if statement.
        """
        events_trace_map = self._get_events_map_with_names(events_data, trace_map)
        root_node = trace_map.get("root")

        if not root_node or len(root_node) != 1:
            return

        if self.root_node == "agent_step":
            data_to_log = self._process_agent_step(events_data, root_node)
        elif self.root_node == "query":
            data_to_log = self._process_query(events_data, root_node)
        else:
            return

        self.event_ids_traced.remove(root_node[0])
        components_to_log = [
            comp for comp in self.components_to_log if comp != self.root_node
        ]
        number_of_components_used = defaultdict(int)
        retrieval_metadata = {}

        for event_id in self.event_ids_traced:
            event_name = self.event_map_id_to_name[event_id]
            event_name_reduced = (
                event_name.rstrip("0") if event_name.endswith("0") else event_name
            )
            number_of_components_used[event_name_reduced] += event_name.endswith("0")

            if event_name_reduced in components_to_log:
                data_to_log[f"{event_name}_time"] = _calc_time(events_data, event_id)

            if event_name_reduced == "llm":
                payload = events_data[event_id][0].payload
                data_to_log.update(
                    {
                        f"{event_name}_system_prompt": payload.get(
                            EventPayload.MESSAGES
                        )[0].content,
                        f"{event_name}_model_name": payload.get(
                            EventPayload.SERIALIZED
                        )["model"],
                    }
                )

            if event_name_reduced == "retrieve":
                for idx, retrieval_node in enumerate(
                    events_data[event_id][1].payload.get(EventPayload.NODES), 1
                ):
                    if idx > self.number_of_retrievals:
                        break
                    retrieve_dict = retrieval_node.to_dict()
                    retrieval_metadata.update(
                        {
                            f"{event_name}_document_{idx}_score": retrieval_node.score,
                            f"{event_name}_document_{idx}_filename": retrieve_dict[
                                "node"
                            ]["metadata"]["file_name"],
                            f"{event_name}_document_{idx}_text": retrieve_dict["node"][
                                "text"
                            ],
                            f"{event_name}_document_{idx}_start_character": retrieve_dict[
                                "node"
                            ][
                                "start_char_idx"
                            ],
                            f"{event_name}_document_{idx}_end_character": retrieve_dict[
                                "node"
                            ]["end_char_idx"],
                        }
                    )

        metadata_to_log = {
            key: data_to_log[key]
            for key in data_to_log
            if key.endswith("_time") or key not in ["query", "response"]
        }
        metadata_to_log["total_time"] = data_to_log.get(
            "query_time", data_to_log.get("agent_step_time")
        )
        metadata_to_log.update(
            {
                f"number_of_{key}_used": value + 1
                for key, value in number_of_components_used.items()
            }
        )
        metadata_to_log.update(retrieval_metadata)

        self._add_missing_metadata(self.dataset, metadata_to_log)

        tree_structure = self._create_tree_structure(events_trace_map, data_to_log)
        tree = _create_svg(tree_structure)

        message = [
            {"role": "user", "content": data_to_log["query"]},
            {"role": "assistant", "content": data_to_log["response"]},
        ]
        fields = {
            "chat": chat_to_html(message),
            "time-details": tree,
        }

        if self.number_of_retrievals > 0:
            for key in list(retrieval_metadata.keys()):
                if key.endswith("_text"):
                    idx = key.split("_")[-2]
                    fields[f"retrieved_document_{idx}"] = (
                        f"DOCUMENT SCORE: {retrieval_metadata[f'{key[:-5]}_score']}\n\n{retrieval_metadata[key]}"
                    )
                    del metadata_to_log[key]

        valid_metadata_keys = [
            metadata.name for metadata in self.dataset.settings.metadata
        ]
        metadata_to_log = {
            k: v
            for k, v in metadata_to_log.items()
            if k in valid_metadata_keys or not k.endswith("_time")
        }

        self.dataset.records.log(
            records=[
                rg.Record(
                    fields=fields,
                    metadata=metadata_to_log,
                ),
            ]
        )

    def _process_agent_step(
        self, events_data: Dict[str, List[CBEvent]], root_node: str
    ) -> Dict:
        """
        Processes events data for 'agent_step' root node.
        """
        data_to_log = {}

        event_start = events_data[root_node[0]][0]
        data_to_log["query"] = event_start.payload.get(EventPayload.MESSAGES)[0]
        query_start_time = event_start.time

        event_end = events_data[root_node[0]][1]
        data_to_log["response"] = event_end.payload.get(EventPayload.RESPONSE).response
        query_end_time = event_end.time

        data_to_log["agent_step_time"] = _get_time_diff(
            query_start_time, query_end_time
        )

        return data_to_log

    def _process_query(
        self, events_data: Dict[str, List[CBEvent]], root_node: str
    ) -> Dict:
        """
        Processes events data for 'query' root node.
        """
        data_to_log = {}

        event_start = events_data[root_node[0]][0]
        data_to_log["query"] = event_start.payload.get(EventPayload.QUERY_STR)
        query_start_time = event_start.time

        event_end = events_data[root_node[0]][1]
        data_to_log["response"] = event_end.payload.get(EventPayload.RESPONSE).response
        query_end_time = event_end.time

        data_to_log["query_time"] = _get_time_diff(query_start_time, query_end_time)

        return data_to_log

    def _create_tree_structure(
        self, events_trace_map: Dict[str, List[str]], data_to_log: Dict[str, Any]
    ) -> List:
        """Create the tree data to be converted to an SVG."""
        events_trace_map = self._check_components_for_tree(events_trace_map)
        data = []
        data.append(
            (
                0,
                0,
                self.root_node.strip("0").upper(),
                data_to_log[f"{self.root_node}_time"],
            )
        )
        current_row = 1
        for root_child in events_trace_map[self.root_node]:
            data.append(
                (
                    current_row,
                    1,
                    root_child.strip("0").upper(),
                    data_to_log[f"{root_child}_time"],
                )
            )
            current_row += 1
            for child in events_trace_map[root_child]:
                data.append(
                    (
                        current_row,
                        2,
                        child.strip("0").upper(),
                        data_to_log[f"{child}_time"],
                    )
                )
                current_row += 1
        return data

    # The four methods required by the abstract class
    # BaseCallbackHandler executed on the different events.

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """
        Start tracing events.

        Args:
            trace_id (str, optional): The trace_id to start tracing.
        """

        self._trace_map = defaultdict(list)
        self._cur_trace_id = trace_id
        self._start_time = datetime.now()

        # Clear the events and the components prior to running the query.
        # They are usually events related to creating the docs and indexing.
        self.events_data.clear()
        self.components_to_log.clear()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        End tracing events.

        Args:
            trace_id (str, optional): The trace_id to end tracing.
            trace_map (Dict[str, List[str]], optional): The trace_map to end. This map has been obtained from the parent class.
        """

        self._trace_map = trace_map or defaultdict(list)
        self._end_time = datetime.now()
        self._create_root_and_other_nodes(trace_map)
        self._extract_and_log_info(self.events_data, trace_map)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        parent_id: str = None,
    ) -> str:
        """
        Store event start data by event type. Executed at the start of an event.

        Args:
            event_type (CBEventType): The event type to store.
            payload (Dict[str, Any], optional): The payload to store.
            event_id (str, optional): The event id to store.
            parent_id (str, optional): The parent id to store.

        Returns:
            str: The event id.
        """

        event = CBEvent(event_type, payload=payload, id_=event_id)
        self.events_data[event_id].append(event)

        return event.id_

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = None,
    ) -> None:
        """
        Store event end data by event type. Executed at the end of an event.

        Args:
            event_type (CBEventType): The event type to store.
            payload (Dict[str, Any], optional): The payload to store.
            event_id (str, optional): The event id to store.
        """

        event = CBEvent(event_type, payload=payload, id_=event_id)
        self.events_data[event_id].append(event)
