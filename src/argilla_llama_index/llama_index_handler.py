from argilla._constants import DEFAULT_API_KEY, DEFAULT_API_URL
import argilla as rg
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
    """Callback handler for Argilla.
    
    Args:
        dataset_name: The name of the dataset to log the events to. If the dataset does not exist,
            a new one will be created.
        workspace_name: The name of the workspace to log the events to.
        api_url: The URL of the Argilla server.
        api_key: The API key to use to connect to Argilla.
        event_starts_to_ignore: A list of event types to ignore when they start.
        event_ends_to_ignore: A list of event types to ignore when they end.
        handlers: A list of handlers to run when an event starts or ends.
    
    Raises:
        ImportError: If the `argilla` Python package is not installed or the one installed is not compatible
        ConnectionError: If the connection to Argilla fails
        FileNotFoundError: If the retrieval and creation of the `FeedbackDataset` fails
    
    Example:
        >>> from argilla_llama_index import ArgillaCallbackHandler
        >>> from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
        >>> from llama_index.llms import OpenAI
        >>> from llama_index import set_global_handler
        >>> set_global_handler("argilla", dataset_name="query_model")
        >>> llm = OpenAI(model="gpt-3.5-turbo", temperature=0.8)
        >>> service_context = ServiceContext.from_defaults(llm=llm)
        >>> docs = SimpleDirectoryReader("data").load_data()
        >>> index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        >>> query_engine = index.as_query_engine()
        >>> response = query_engine.query("What did the author do growing up dude?")
    """

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
        """Initialize the Argilla callback handler.
        
        Args:
            dataset_name: The name of the dataset to log the events to. If the dataset does not exist,
                a new one will be created.
            workspace_name: The name of the workspace to log the events to.
            api_url: The URL of the Argilla server.
            api_key: The API key to use to connect to Argilla.
            event_starts_to_ignore: A list of event types to ignore when they start.
            event_ends_to_ignore: A list of event types to ignore when they end.
            handlers: A list of handlers to run when an event starts or ends.
    
        Raises:
            ImportError: If the `argilla` Python package is not installed or the one installed is not compatible
            ConnectionError: If the connection to Argilla fails
            FileNotFoundError: If the retrieval and creation of the `FeedbackDataset` fails
        
        """
        
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

        # Check whether the Argilla version is compatible
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
                self.is_new_dataset_created = False
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

                self.dataset = dataset.push_to_argilla(self.dataset_name)
                self.is_new_dataset_created = True
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
        
        self.events_data: Dict[str, List[CBEvent]] = defaultdict(list)
        self.event_map_id_to_name = {}
        self._ignore_components_in_tree = ["templating"]
        self.components_to_log2 = set()
        self.event_ids_traced = set()

    def _create_root_and_other_nodes(
        self, 
        trace_map: Dict[str, List[str]]
    ) -> None:
        """Create the root node and the other nodes in the tree."""
        self.root_node = self._get_event_name_by_id(trace_map["root"][0])
        self.event_ids_traced = set(trace_map.keys()) - {"root"}
        self.event_ids_traced.update(*trace_map.values())
        for id in self.event_ids_traced:
            self.components_to_log2.add(self._get_event_name_by_id(id))

    def _get_event_name_by_id(
        self, 
        event_id: str
    ) -> str:
        """Get the name of the event by its id."""
        return str(self.events_data[event_id][0].event_type).split(".")[1].lower()
    
        # TODO: If we have a component more than once, properties currently don't account for those after the first one and get overwritten
    def _add_missing_metadata_properties(
        self, 
        dataset: rg.FeedbackDataset, 
    ) -> None:
        """Add missing metadata properties to the dataset."""
        required_metadata_properties = []
        for property in self.components_to_log2:
            metadata_name = f"{property}_time"
            if property == self.root_node:
                metadata_name = "total_time"
            required_metadata_properties.append(metadata_name)

        existing_metadata_properties = [property.name for property in dataset.metadata_properties]
        missing_metadata_properties = [property for property in required_metadata_properties if property not in existing_metadata_properties]

        for property in missing_metadata_properties:
            title= " ".join([word.capitalize() for word in property.split('_')])
            if title == "Llm Time":
                title = "LLM Time"
            dataset.add_metadata_property(
                rg.FloatMetadataProperty(name=property, title=title))
            if not self.is_new_dataset_created:
                warnings.warn(
                    (
                        f"The dataset given was missing some required metadata properties. "
                        f"Missing properties were {missing_metadata_properties}. "
                        f"Properties have been added to the dataset with "
                    ),
                )
                
    def _check_components_for_tree(
        self,
        tree_structure_dict: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Check whether the components in the tree are in the components to log.
        Removes components that are not in the components to log so that they are not shown in the tree.  
        """
        final_components_in_tree = self.components_to_log2.copy()
        final_components_in_tree.add("root")
        for component in self._ignore_components_in_tree:
            if component in final_components_in_tree:
                final_components_in_tree.remove(component)
        for key in list(tree_structure_dict.keys()):
            if key.strip("0") not in final_components_in_tree:
                del tree_structure_dict[key]
        for key, value in tree_structure_dict.items():
            if isinstance(value, list):
                tree_structure_dict[key] = [element for element in value if element.strip("0") in final_components_in_tree]
        return tree_structure_dict
                
    def _create_tree(
        self, 
        tree_structure_dict: Dict[str, List[str]],
        data_to_log: Dict[str, Any]
    ) -> str:
        """
        Create a tree structure of the components used.
        Relies on the Markdown syntax to create the tree structure.
        """
        tree_structure_dict = self._check_components_for_tree(tree_structure_dict)
        root_node = list(tree_structure_dict.keys())[1]
        def print_tree_structure(node, tree_dict, indent=0, output="", root_node=root_node):
            node_time = str(data_to_log[f"{node}_time"])
            output += "│   " * indent + "│--- " + node.upper().strip("0") + "--->" + f"<span style='color:green'>**{node_time}**</span>" + "\n"
            if node in tree_dict:
                for child in tree_dict[node]:
                    output = print_tree_structure(child, tree_dict, indent + 1, output)
            return output
        
        tree_structure_str = print_tree_structure(root_node, tree_structure_dict)

        return tree_structure_str
            
    def _get_events_map_with_names(
        self, 
        events_data: Dict[str, List[CBEvent]],
        trace_map: Dict[str, List[str]]
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

        events_trace_map = {self.event_map_id_to_name.get(k, k): [self.event_map_id_to_name.get(v, v) for v in values] for k, values in trace_map.items()}

        return events_trace_map
    
    def _extract_and_log_info(
        self, 
        events_data: Dict[str, List[CBEvent]],
        trace_map: Dict[str, List[str]]
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
                data_to_log["response"] = event.payload.get(EventPayload.RESPONSE).response
                query_end_time = event.time
                data_to_log["agent_step_time"] = _get_time_diff(query_start_time, query_end_time)

            elif self.root_node == "query":
                # Event start
                event = events_data[root_node[0]][0]
                data_to_log["query"] = event.payload.get(EventPayload.QUERY_STR)
                query_start_time = event.time
                # Event end
                event = events_data[root_node[0]][1]
                data_to_log["response"] = event.payload.get(EventPayload.RESPONSE).response
                query_end_time = event.time
                data_to_log["query_time"] = _get_time_diff(query_start_time, query_end_time)
            
            else:
                return
            
            # Create logging data for the rest of the components
            self.event_ids_traced.remove(root_node[0])

            number_of_components_used = defaultdict(int)
            components_to_log_without_root_node = self.components_to_log2.copy()
            components_to_log_without_root_node.remove(self.root_node)
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
                    data_to_log[f"{event_name}_system_prompt"] = events_data[id][0].payload.get(EventPayload.MESSAGES)[0].content
                    data_to_log[f"{event_name}_model_name"] = events_data[id][0].payload.get(EventPayload.SERIALIZED)["model"]
                        
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
            
            tree_str = self._create_tree(events_trace_map, data_to_log)
            
            self.dataset.add_records(
            records=[
                {
                    "fields": {
                        "prompt": data_to_log["query"], 
                        "response": data_to_log["response"],
                        "time-details": tree_str
                        },
                        "metadata": metadata_to_log
                    },
                ]
            )

    def start_trace(
        self, 
        trace_id: Optional[str] = None
    ) -> None:
        """Launch a trace."""
        self._trace_map = defaultdict(list)
        self._cur_trace_id = trace_id
        self._start_time = datetime.now()
        self.events_data.clear()
        self.components_to_log2.clear()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """End a trace."""
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
        parent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Run handlers when an event starts."""
        event = CBEvent(event_type, payload=payload, id_=event_id)
        self.events_data[event_id].append(event)

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Run handlers when an event ends."""
        event = CBEvent(event_type, payload=payload, id_=event_id)
        self.events_data[event_id].append(event)

def _get_time_diff(
    event_1_time_str: str, 
    event_2_time_str: str
) -> float:
    """Get the time difference between two events."""
    time_format = "%m/%d/%Y, %H:%M:%S.%f"

    event_1_time = datetime.strptime(event_1_time_str, time_format)
    event_2_time = datetime.strptime(event_2_time_str, time_format)

    return round((event_2_time - event_1_time).total_seconds(), 4)

def _calc_time(
    events_data: Dict[str, List[CBEvent]], 
    id: str
) -> float:
    """Calculate the time difference between the start and end of an event using the events_data."""
    start_time = events_data[id][0].time
    end_time = events_data[id][1].time
    return _get_time_diff(start_time, end_time)
    