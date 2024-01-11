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
                        rg.TextField(name="prompt"),
                        rg.TextField(name="response"),
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
        self.events_data = {}

    def _log_data(self, event_data: Dict[str, Any]) -> None:
        """Log data to Argilla."""

        times = f"""
        Query time: {event_data["query_time"]}
        \tRetrieve time: {event_data["retrieve_time"]}
        \t\tEmbedding time: {event_data["embedding_time"]}
        \tLLM time: {event_data["synthesize_time"]}
        \nModel name: {event_data["model_name"]}
        """

        self.dataset.add_records(
            records=[
                {
                    "fields": {
                        "prompt": event_data["query"], 
                        "response": event_data["response"],
                        "time-details": times
                        },
                    },
                ]
            )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Launch a trace."""
        self._trace_map = defaultdict(list)
        self._cur_trace_id = trace_id
        self._start_time = datetime.now()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._trace_map = trace_map or defaultdict(list)
        self._end_time = datetime.now()

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

        if event_type == CBEventType.QUERY:
            self.events_data["query"] = event.payload.get(EventPayload.QUERY_STR)
            self.events_data["query_times"] = [event.time]

        if event_type == CBEventType.EMBEDDING:
            self.events_data["embedding_times"] = [event.time]

        if event_type == CBEventType.RETRIEVE:
            self.events_data["retrieve_times"] = [event.time]

        if event_type == CBEventType.LLM:
            self.events_data["system_prompt"] = event.payload.get(EventPayload.MESSAGES)[0].content
            self.events_data["model_name"] = event.payload.get(EventPayload.SERIALIZED)["model"]
            self.events_data["llm_times"] = [event.time]

        if event_type == CBEventType.SYNTHESIZE:
            self.events_data["synthesize_times"] = [event.time]

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Run handlers when an event ends."""
        event = CBEvent(event_type, payload=payload, id_=event_id)

        if event_type == CBEventType.QUERY:
            self.events_data["response"] = event.payload.get(EventPayload.RESPONSE).response
            self.events_data["query_times"].append(event.time)
            self.events_data["query_time"] = _get_time_diff(self.events_data["query_times"][0], self.events_data["query_times"][1])
            self._log_data(self.events_data)
            self.events_data = {}

        if event_type == CBEventType.LLM:
            self.events_data["llm_times"].append(event.time)
            self.events_data["llm_time"] = _get_time_diff(self.events_data["llm_times"][0], self.events_data["llm_times"][1])

        if event_type == CBEventType.RETRIEVE:
            self.events_data["retrieve_times"].append(event.time)
            self.events_data["retrieve_time"] = _get_time_diff(self.events_data["retrieve_times"][0], self.events_data["retrieve_times"][1])

        if event_type == CBEventType.EMBEDDING:
            self.events_data["embedding_times"].append(event.time)
            self.events_data["embedding_time"] = _get_time_diff(self.events_data["embedding_times"][0], self.events_data["embedding_times"][1])

        if event_type == CBEventType.SYNTHESIZE:
            self.events_data["synthesize_times"].append(event.time)
            self.events_data["synthesize_time"] = _get_time_diff(self.events_data["synthesize_times"][0], self.events_data["synthesize_times"][1])


def _get_time_diff(event_1_time_str: str, event_2_time_str: str) -> float:
    """Get the time difference between two events."""
    time_format = "%m/%d/%Y, %H:%M:%S.%f"

    event_1_time = datetime.strptime(event_1_time_str, time_format)
    event_2_time = datetime.strptime(event_2_time_str, time_format)

    return (event_2_time - event_1_time).total_seconds()
    