from argilla._constants import DEFAULT_API_KEY, DEFAULT_API_URL
from typing import Any, Dict, List, Optional
from packaging.version import parse
import warnings
import os
from datetime import datetime
from collections import defaultdict
from contextvars import ContextVar

from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.schema import (
    BASE_TRACE_EVENT,
    CBEventType,
    EventPayload,
    CBEvent,
)

global_stack_trace = ContextVar("trace", default=[BASE_TRACE_EVENT])

class ArgillaCallbackHandler(BaseCallbackHandler):

    REPO_URL: str = "https://github.com/argilla-io/argilla"
    ISSUES_URL: str = f"{REPO_URL}/issues"

    def __init__(
        self,
        dataset_name: str,
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

        # Import Argilla 
        try:
            import argilla as rg
            self.ARGILLA_VERSION = rg.__version__

        except ImportError:
            raise ImportError(
                "To use the Argilla callback manager you need to have the `argilla` "
                "Python package installed. Please install it with `pip install argilla`"
            )

        ## Check whether the Argilla version is compatible
        if parse(self.ARGILLA_VERSION) < parse("1.18.0"):
            raise ImportError(
                f"The installed `argilla` version is {self.ARGILLA_VERSION} but "
                "`ArgillaCallbackHandler` requires at least version 1.18.0. Please "
                "upgrade `argilla` with `pip install --upgrade argilla`."
            )
        
        # API_URL and API_KEY
        # Show a warning message if Argilla will assume the default values will be used
        if api_url is None and os.getenv("ARGILLA_API_URL") is None:
            warnings.warn(
                (
                    "Since `api_url` is None, and the env var `ARGILLA_API_URL` is not"
                    f" set, it will default to `{DEFAULT_API_URL}`, which is the"
                    " default API URL in Argilla Quickstart."
                ),
            )
            api_url = DEFAULT_API_URL

        if api_key is None and os.getenv("ARGILLA_API_KEY") is None:
            warnings.warn(
                (
                    "Since `api_key` is None, and the env var `ARGILLA_API_KEY` is not"
                    f" set, it will default to `{DEFAULT_API_KEY}`, which is the"
                    " default API key in Argilla Quickstart."
                ),
            )
            api_key = DEFAULT_API_KEY
    
        # Connect to Argilla with the provided credentials, if applicable
        try:
            rg.init(
                api_key=api_key, 
                api_url=api_url
            )
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Argilla with exception: '{e}'.\n"
                "Please check your `api_key` and `api_url`, and make sure that "
                "the Argilla server is up and running. If the problem persists "
                f"please report it to {self.ISSUES_URL} as an `integration` issue."
            ) from e
        
        # Set the Argilla variables
        self.dataset_name = dataset_name
        self.workspace_name = workspace_name or rg.get_workspace()

        # Retrieve the `FeedbackDataset` from Argilla
        try:
            if self.dataset_name in [ds.name for ds in rg.FeedbackDataset.list()]:
                self.dataset = rg.FeedbackDataset.from_argilla(
                    name=self.dataset_name,
                    workspace=self.workspace_name,
                )
            # If the dataset does not exist, create a new one with the given name
            else:
                dataset = rg.FeedbackDataset(
                    fields=[
                        rg.TextField(name="prompt", required=True),
                        rg.TextField(name="response", required=False),
                        rg.TextField(name="time-details", title="Time Details", use_markdown=True),
                    ],
                    questions=[
                        rg.RatingQuestion(
                            name="response-rating",
                            title="Rating",
                            description="How would you rate the quality of the response?",
                            values=[1, 2, 3, 4, 5],
                            required=True,
                        ),
                        rg.TextQuestion(
                            name="response-feedback",
                            title="Feedback",
                            description="What feedback do you have for the response?",
                            required=False,
                        ),
                    ],
                        guidelines="You're asked to rate the quality of the response and provide feedback.",
                        allow_extra_metadata=True,
                )
                dataset.add_metadata_property(
                    rg.FloatMetadataProperty(
                        name="total_time", title="Total Time"
                    )
                )
                self.dataset = dataset.push_to_argilla(self.dataset_name)
                warnings.warn(
                (
                    f"No dataset with the name {self.dataset_name} was found in workspace "
                    f"{self.workspace_name}. A new dataset with the name {self.dataset_name} "
                    "has been created with the question fields `prompt` and `response`"
                    "and the rating question `response-rating` with values 1-5 and text question"
                    "named `response-feedback`."
                ),
            )

        except Exception as e:
            raise FileNotFoundError(
                f"`FeedbackDataset` retrieval and creation both failed with exception `{e}`."
                f" If the problem persists please report it to {self.ISSUES_URL} "
                f"as an `integration` issue."
            ) from e
        
        supported_fields = ["prompt", "response", "time-details"]
        if supported_fields != [field.name for field in self.dataset.fields]:
            raise ValueError(
                f"`FeedbackDataset` with name={self.dataset_name} in the workspace="
                f"{self.workspace_name} had fields that are not supported yet for the"
                f"`llama-index` integration. Supported fields are: {supported_fields},"
                f" and the current `FeedbackDataset` fields are {[field.name for field in self.dataset.fields]}."
            )
        
        self.field_names = [field.name for field in self.dataset.fields]
        self.events_dict: Dict[str, List[CBEvent]] = defaultdict(list)
        self._events_to_trace: List[CBEventType] = [
            CBEventType.EMBEDDING, 
            CBEventType.LLM, 
            CBEventType.QUERY, 
            CBEventType.RETRIEVE, 
            CBEventType.SYNTHESIZE,
            CBEventType.TEMPLATING]

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Launch a trace."""
        self._trace_map = defaultdict(list)
        self._cur_trace_id = trace_id
        self._start_time = datetime.now()
        self.events_dict.clear()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._trace_map = trace_map or defaultdict(list)
        self._end_time = datetime.now()
        self._extract_and_log_info(self.events_dict, trace_map)

    def _get_events_map_with_names(self, events_dict, trace_map):
        """Get all event names."""
        event_ids_traced = set(trace_map.keys()) - {"root"}
        event_ids_traced.update(*trace_map.values())
        event_map_id_to_name = {}
        for event_id in event_ids_traced:
            event_name = events_dict[event_id][0].event_type
            event_map_id_to_name[event_id] = event_name

        event_map_name_to_id = {value: key for key, value in event_map_id_to_name.items()}
        events_trace_map = {event_map_id_to_name.get(k, k): [event_map_id_to_name.get(v, v) for v in values] for k, values in trace_map.items()}

        return events_trace_map, event_map_id_to_name, event_map_name_to_id, event_ids_traced
    
    def _extract_and_log_info(self, events_dict, trace_map):
        events_trace_map, event_map_id_to_name, event_map_name_to_id, event_ids_traced = self._get_events_map_with_names(events_dict, trace_map)
        root_node = trace_map["root"]
        data_to_log = {}
        if len(root_node) == 1:
            if event_map_id_to_name[root_node[0]] == CBEventType.QUERY:
                # Event start
                event = events_dict[root_node[0]][0]
                data_to_log["query"] = event.payload.get(EventPayload.QUERY_STR)
                query_start_time = event.time
                # Event end
                event = events_dict[root_node[0]][1]
                data_to_log["response"] = event.payload.get(EventPayload.RESPONSE).response
                query_end_time = event.time
                data_to_log["query_time"] = _get_time_diff(query_start_time, query_end_time)
            event_ids_traced.remove(root_node[0])     # remove root id from event_ids_traced
            for id in event_ids_traced:
                if event_map_id_to_name[id] == CBEventType.EMBEDDING:
                    data_to_log["embedding_time"] = _calc_time(events_dict, id)
                if event_map_id_to_name[id] == CBEventType.RETRIEVE:
                    data_to_log["retrieve_time"] = _calc_time(events_dict, id)
                if event_map_id_to_name[id] == CBEventType.LLM:
                    data_to_log["llm_time"] = _calc_time(events_dict, id)
                    data_to_log["system_prompt"] = events_dict[id][0].payload.get(EventPayload.MESSAGES)[0].content
                    data_to_log["model_name"] = events_dict[id][0].payload.get(EventPayload.SERIALIZED)["model"]
                if event_map_id_to_name[id] == CBEventType.SYNTHESIZE:
                    data_to_log["synthesize_time"] = _calc_time(events_dict, id)
                if event_map_id_to_name[id] == CBEventType.TEMPLATING:
                    data_to_log["templating_time"] = _calc_time(events_dict, id)
            
            events_names_traced = list({k: event_map_id_to_name[k] for k in event_ids_traced}.values())
            tree_str = _create_tree(events_trace_map, data_to_log)

            self.dataset.add_records(
            records=[
                {
                    "fields": {
                        "prompt": data_to_log["query"], 
                        "response": data_to_log["response"],
                        "time-details": tree_str
                        },
                        "metadata": {"total_time": data_to_log["query_time"]},
                    },
                ]
            )

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Store event start data by event type.

        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.
            parent_id (str): parent event id.

        """
        event = CBEvent(event_type, payload=payload, id_=event_id)
        self.events_dict[event_id].append(event)

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Run handlers when an event ends."""
        event = CBEvent(event_type, payload=payload, id_=event_id)
        self.events_dict[event_id].append(event)

def _get_time_diff(event_1_time_str: str, event_2_time_str: str) -> float:
    """Get the time difference between two events."""
    time_format = "%m/%d/%Y, %H:%M:%S.%f"

    event_1_time = datetime.strptime(event_1_time_str, time_format)
    event_2_time = datetime.strptime(event_2_time_str, time_format)

    return round((event_2_time - event_1_time).total_seconds(), 4)

def _calc_time(events_dict, id) -> float:
    start_time = events_dict[id][0].time  # Event start
    end_time = events_dict[id][1].time  # Event end
    return _get_time_diff(start_time, end_time)
    
def _create_tree(tree_structure_dict, data_to_log):
    root_node = list(tree_structure_dict.keys())[1]
    def print_tree_structure(node, tree_dict, indent=0, output="", root_node=root_node):
        output += "│   " * indent + "├── " + node.upper() + "\t\t\t           " + str(data_to_log[f"{node}_time"]) + "\n"
        if node in tree_dict:
            for child in tree_dict[node]:
                output = print_tree_structure(child, tree_dict, indent + 1, output)
        return output
        
    tree_structure_str = print_tree_structure(root_node, tree_structure_dict)

    return tree_structure_str