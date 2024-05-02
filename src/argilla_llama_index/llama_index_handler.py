from collections import defaultdict
from datetime import datetime
import os
from packaging.version import parse
from typing import Any, Dict, List, Optional
import warnings

from argilla_llama_index.helpers import _calc_time, _get_time_diff

import argilla as rg
from argilla._constants import DEFAULT_API_KEY, DEFAULT_API_URL
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import (
    CBEventType,
    EventPayload,
    CBEvent,
)


class ArgillaCallbackHandler(BaseCallbackHandler):
    """
    Callback handler that logs predictions to Argilla.

    This handler automatically logs the predictions made with LlamaIndex to Argilla,
    without the need to create a dataset and log the predictions manually. Events relevant
    to the predictions are automatically logged to Argilla as well, including timestamps of
    all the different steps of the retrieval and prediction process.

    Attributes:
        dataset_name (str): The name of the Argilla dataset.
        number_of_retrievals (int): The number of retrievals to log.
        workspace_name (str): The name of the Argilla workspace.
        api_url (str): Argilla API URL.
        api_key (str): Argilla API key.
        event_starts_to_ignore (List[CBEventType]): List of event types to ignore at the start of the trace.
        event_ends_to_ignore (List[CBEventType]): List of event types to ignore at the end of the trace.
        handlers (List[BaseCallbackHandler]): List of extra handlers to include.

    Methods:
        start_trace(trace_id: Optional[str] = None) -> None:
            Logic to be executed at the beggining of the tracing process.

        end_trace(trace_id: Optional[str] = None, trace_map: Optional[Dict[str, List[str]]] = None) -> None:
            Logic to be executed at the end of the tracing process.

        on_event_start(event_type: CBEventType, payload: Optional[Dict[str, Any]] = None, event_id: Optional[str] = None, parent_id: str = None) -> str:
            Store event start data by event type. Executed at the start of an event.
        
        on_event_end(event_type: CBEventType, payload: Optional[Dict[str, Any]] = None, event_id: str = None) -> None:
            Store event end data by event type. Executed at the end of an event.

    Usage:
    ```python
    from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader, set_global_handler
    from llama_index.llms.openai import OpenAI

    set_global_handler("argilla", dataset_name="query_model")

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.8, openai_api_key=os.getenv("OPENAI_API_KEY"))

    service_context = ServiceContext.from_defaults(llm=llm)
    docs = SimpleDirectoryReader("../../data").load_data()
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    query_engine = index.as_query_engine()

    response = query_engine.query("What did the author do growing up?")
    ```
    """

    def __init__(
        self,
        dataset_name: str,
        number_of_retrievals: int = 0,
        workspace_name: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
        handlers: Optional[List[BaseCallbackHandler]] = None,
    ) -> None:

        self.event_starts_to_ignore = event_starts_to_ignore or []
        self.event_ends_to_ignore = event_ends_to_ignore or []
        self.handlers = handlers or []
        self.number_of_retrievals = number_of_retrievals

        try:
            import argilla as rg

            self.ARGILLA_VERSION = rg.__version__

        except ImportError:
            raise ImportError(
                "To use the Argilla callback manager you need to have the `argilla` "
                "Python package installed. Please install it with `pip install argilla`"
            )

        if parse(self.ARGILLA_VERSION) < parse("1.18.0"):
            raise ImportError(
                f"The installed `argilla` version is {self.ARGILLA_VERSION} but "
                "`ArgillaCallbackHandler` requires at least version 1.18.0. Please "
                "upgrade `argilla` with `pip install --upgrade argilla`."
            )

        # Ensure the API URL and API key are set, or assume the default values
        if api_url is None and os.getenv("ARGILLA_API_URL") is None:
            warnings.warn(
                (
                    "Since `api_url` is None, and the environment var `ARGILLA_API_URL` is not"
                    f" set, it will default to `{DEFAULT_API_URL}`, which is the"
                    " default API URL in Argilla Quickstart."
                ),
            )
            api_url = DEFAULT_API_URL

        if api_key is None and os.getenv("ARGILLA_API_KEY") is None:
            warnings.warn(
                (
                    "Since `api_key` is None, and the environment var `ARGILLA_API_KEY` is not"
                    f" set, it will default to `{DEFAULT_API_KEY}`, which is the"
                    " default API key in Argilla Quickstart."
                ),
            )
            api_key = DEFAULT_API_KEY

        
        rg.init(api_key=api_key, api_url=api_url, workspace=workspace_name or rg.get_workspace())

        # Set the Argilla variables
        self.dataset_name = dataset_name
        self.workspace_name = workspace_name or rg.get_workspace()

        # Either create a new dataset or use an existing one, updating it if necessary
        try:
            if self.dataset_name not in [ds.name for ds in rg.FeedbackDataset.list()]:
                dataset = rg.FeedbackDataset(
                    fields=[
                        rg.TextField(name="prompt", required=True),
                        rg.TextField(name="response", required=False),
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

                self.dataset = dataset.push_to_argilla(self.dataset_name)
                self.is_new_dataset_created = True
                warnings.warn(
                    (
                        f"No dataset with the name {self.dataset_name} was found in workspace "
                        f"{self.workspace_name}. A new dataset with the name {self.dataset_name} "
                        "has been created with the question fields `prompt` and `response`"
                        "and the rating question `response-rating` with values 1-7 and text question"
                        " named `response-feedback`."
                    ),
                )

            else:
                # Update the existing dataset. If the fields and questions do not match, the dataset will be updated using the
                # -updated flag in the name.
                if self.dataset_name in [ds.name for ds in rg.FeedbackDataset.list()]:
                    self.dataset = rg.FeedbackDataset.from_argilla(
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
                        existing_fields = [
                            field.to_local() for field in self.dataset.fields
                        ]
                        existing_questions = [
                            question.to_local() for question in self.dataset.questions
                        ]
                        # If the required fields and questions do not match with the existing ones, update the dataset and upload it again with "-updated" added to the name
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
                            local_dataset = self.dataset.pull()

                            fields_to_pop = []
                            for index, field in enumerate(local_dataset.fields):
                                if field.name.startswith("retrieved_document_"):
                                    fields_to_pop.append(index)
                                    fields_to_pop.sort(reverse=True)
                            else:
                                for index in fields_to_pop:
                                    local_dataset.fields.pop(index)

                            questions_to_pop = []
                            for index, question in enumerate(local_dataset.questions):
                                if question.name.startswith(
                                    "rating_retrieved_document_"
                                ):
                                    questions_to_pop.append(index)
                                    questions_to_pop.sort(reverse=True)
                            else:
                                for index in questions_to_pop:
                                    local_dataset.questions.pop(index)

                            for field in required_context_fields:
                                local_dataset.fields.append(field)
                            for question in required_context_questions:
                                local_dataset.questions.append(question)
                            self.dataset = local_dataset.push_to_argilla(
                                self.dataset_name + "-updated"
                            )

        except Exception as e:
            raise FileNotFoundError(
                f"`FeedbackDataset` retrieval and creation both failed with exception `{e}`."
                f" If the problem persists please report it to https://github.com/argilla-io/argilla/issues/ "
                f"as an `integration` issue."
            ) from e

        supported_context_fields = [
            f"retrieved_document_{i+1}" for i in range(number_of_retrievals)
        ]
        supported_fields = [
            "prompt",
            "response",
            "time-details",
        ] + supported_context_fields
        if supported_fields != [field.name for field in self.dataset.fields]:
            raise ValueError(
                f"`FeedbackDataset` with name={self.dataset_name} in the workspace="
                f"{self.workspace_name} had fields that are not supported yet for the"
                f"`llama-index` integration. Supported fields are: {supported_fields},"
                f" and the current `FeedbackDataset` fields are {[field.name for field in self.dataset.fields]}."
            )

        self.events_data: Dict[str, List[CBEvent]] = defaultdict(list)
        self.event_map_id_to_name = {}
        self._ignore_components_in_tree = ["templating"]
        self.components_to_log = set()
        self.event_ids_traced = set()

    def _add_context_fields(self, number_of_retrievals: int) -> List:
        """Create the context fields to be added to the dataset."""
        context_fields = [
            rg.TextField(
                name="retrieved_document_" + str(doc + 1),
                title="Retrieved Document " + str(doc + 1),
                use_markdown=True,
                required=False,
            )
            for doc in range(number_of_retrievals)
        ]
        return context_fields

    def _add_context_questions(self, number_of_retrievals: int) -> List:
        """Create the context questions to be added to the dataset."""
        rating_questions = [
            rg.RatingQuestion(
                name="rating_retrieved_document_" + str(doc + 1),
                title="Rate the relevance of the Retrieved Document "
                + str(doc + 1)
                + " (if present)",
                values=list(range(1, 8)),
                # After https://github.com/argilla-io/argilla/issues/4523 is fixed, we can use the description
                description=None,  # "Rate the relevance of the retrieved document."
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

    def _add_missing_metadata_properties(
        self,
        dataset: rg.FeedbackDataset,
    ) -> None:
        """Add missing metadata properties to the dataset."""
        required_metadata_properties = []
        for property in self.components_to_log:
            metadata_name = f"{property}_time"
            if property == self.root_node:
                metadata_name = "total_time"
            required_metadata_properties.append(metadata_name)

        existing_metadata_properties = [
            property.name for property in dataset.metadata_properties
        ]
        missing_metadata_properties = [
            property
            for property in required_metadata_properties
            if property not in existing_metadata_properties
        ]

        for property in missing_metadata_properties:
            title = " ".join([word.capitalize() for word in property.split("_")])
            if title == "Llm Time":
                title = "LLM Time"
            dataset.add_metadata_property(
                rg.FloatMetadataProperty(name=property, title=title)
            )
            if self.is_new_dataset_created == False:
                warnings.warn(
                    (
                        f"The dataset given was missing some required metadata properties. "
                        f"Missing properties were {missing_metadata_properties}. "
                        f"Properties have been added to the dataset with "
                    ),
                )

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
        root_node = trace_map["root"]
        data_to_log = {}

        if len(root_node) == 1:

            # Create logging data for the root node
            if self.root_node == "agent_step":
                # Event start
                event = events_data[root_node[0]][0]
                data_to_log["query"] = event.payload.get(EventPayload.MESSAGES)[0]
                query_start_time = event.time
                # Event end
                event = events_data[root_node[0]][1]
                data_to_log["response"] = event.payload.get(
                    EventPayload.RESPONSE
                ).response
                query_end_time = event.time
                data_to_log["agent_step_time"] = _get_time_diff(
                    query_start_time, query_end_time
                )

            elif self.root_node == "query":
                # Event start
                event = events_data[root_node[0]][0]
                data_to_log["query"] = event.payload.get(EventPayload.QUERY_STR)
                query_start_time = event.time
                # Event end
                event = events_data[root_node[0]][1]
                data_to_log["response"] = event.payload.get(
                    EventPayload.RESPONSE
                ).response
                query_end_time = event.time
                data_to_log["query_time"] = _get_time_diff(
                    query_start_time, query_end_time
                )

            else:
                return

            # Create logging data for the rest of the components
            self.event_ids_traced.remove(root_node[0])
            number_of_components_used = defaultdict(int)
            components_to_log_without_root_node = self.components_to_log.copy()
            components_to_log_without_root_node.remove(self.root_node)
            retrieval_metadata = {}
            for id in self.event_ids_traced:
                event_name = self.event_map_id_to_name[id]
                if event_name.endswith("0"):
                    event_name_reduced = event_name.strip("0")
                    number_of_components_used[event_name_reduced] += 1
                else:
                    event_name_reduced = event_name

                for component in components_to_log_without_root_node:
                    if event_name_reduced == component:
                        data_to_log[f"{event_name}_time"] = _calc_time(events_data, id)

                if event_name_reduced == "llm":
                    data_to_log[f"{event_name}_system_prompt"] = (
                        events_data[id][0].payload.get(EventPayload.MESSAGES)[0].content
                    )
                    data_to_log[f"{event_name}_model_name"] = events_data[id][
                        0
                    ].payload.get(EventPayload.SERIALIZED)["model"]

                retrieved_document_counter = 1
                if event_name_reduced == "retrieve":
                    for retrieval_node in events_data[id][1].payload.get(
                        EventPayload.NODES
                    ):
                        retrieve_dict = retrieval_node.to_dict()
                        retrieval_metadata[
                            f"{event_name}_document_{retrieved_document_counter}_score"
                        ] = retrieval_node.score
                        retrieval_metadata[
                            f"{event_name}_document_{retrieved_document_counter}_filename"
                        ] = retrieve_dict["node"]["metadata"]["file_name"]
                        retrieval_metadata[
                            f"{event_name}_document_{retrieved_document_counter}_text"
                        ] = retrieve_dict["node"]["text"]
                        retrieval_metadata[
                            f"{event_name}_document_{retrieved_document_counter}_start_character"
                        ] = retrieve_dict["node"]["start_char_idx"]
                        retrieval_metadata[
                            f"{event_name}_document_{retrieved_document_counter}_end_character"
                        ] = retrieve_dict["node"]["end_char_idx"]
                        retrieved_document_counter += 1
                        if retrieved_document_counter > self.number_of_retrievals:
                            break

            metadata_to_log = {}

            for keys in data_to_log.keys():
                if keys == "query_time" or keys == "agent_step_time":
                    metadata_to_log["total_time"] = data_to_log[keys]
                elif keys.endswith("_time"):
                    metadata_to_log[keys] = data_to_log[keys]
                elif keys != "query" and keys != "response":
                    metadata_to_log[keys] = data_to_log[keys]

            if len(number_of_components_used) > 0:
                for key, value in number_of_components_used.items():
                    metadata_to_log[f"number_of_{key}_used"] = value + 1

            metadata_to_log.update(retrieval_metadata)

            tree_structure = self._create_tree_structure(events_trace_map, data_to_log)
            tree = self._create_svg(tree_structure)

            fields = {
                "prompt": data_to_log["query"],
                "response": data_to_log["response"],
                "time-details": tree,
            }

            if self.number_of_retrievals > 0:
                for key, value in list(retrieval_metadata.items()):
                    if key.endswith("_text"):
                        fields[f"retrieved_document_{key[-6]}"] = (
                            f"DOCUMENT SCORE: {retrieval_metadata[key[:-5]+'_score']}\n\n"
                            + value
                        )
                        del metadata_to_log[key]

            self.dataset.add_records(
                records=[
                    {"fields": fields, "metadata": metadata_to_log},
                ]
            )

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

    def _create_svg(self, data: List) -> str:
        """
        Create an SVG file from the data.

        Args:
            data (List): The data to create the SVG file from.

        Returns:
            str: The SVG file.
        """

        # changing only the box height changes all other values as well
        # others can be adjusted individually if needed
        box_height = 47
        box_width = box_height * 8.65
        row_constant = box_height + 7
        indent_constant = 40
        font_size_node_name = box_height * 0.4188
        font_size_time = font_size_node_name - 4
        text_centering = box_height * 0.6341
        node_name_indent = box_height * 0.35
        time_indent = box_height * 7.15

        body = ""
        for each in data:
            row, indent, node_name, node_time = each
            body_raw = f"""
<g transform="translate({indent*indent_constant}, {row*row_constant})">
<rect x=".5" y=".5" width="{box_width}" height="{box_height}" rx="8.49" ry="8.49" style="fill: #24272e; stroke: #afdfe5; stroke-miterlimit: 10;"/>
<text transform="translate({node_name_indent} {text_centering})" style="fill: #fff; font-size: {font_size_node_name}px;"><tspan x="0" y="0">{node_name}</tspan></text>
<text transform="translate({time_indent} {text_centering})" style="fill: #b7d989; font-size: {font_size_time}px; font-style: italic;"><tspan x="0" y="0">{node_time}</tspan></text>
</g>
            """
            body += body_raw
            base = (
                base
            ) = f"""
<?xml version="1.0" encoding="UTF-8"?>
<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 {len(data)*row_constant}">
{body}
</svg>
            """
            base = base.strip()
        return base

    # The four methods required by the abstrac class BaseCallbackHandler.
    # These methods are the one being executed on the different events, by the llama-index
    # BaseCallbackHandler class.

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
        self._add_missing_metadata_properties(self.dataset)
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
