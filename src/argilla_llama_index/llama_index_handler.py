from collections import defaultdict
import os
from packaging.version import parse
from typing import Any, Dict, List, Optional
import warnings

from argilla._constants import DEFAULT_API_KEY, DEFAULT_API_URL
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import (
    BASE_TRACE_EVENT,
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

        # Import Argilla
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

        # Initialize the Argilla client
        try:
            rg.init(api_key=api_key, api_url=api_url)
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
                        if (
                            all(
                                element in existing_fields
                                for element in required_context_fields
                            )
                            == False
                            or all(
                                element in existing_questions
                                for element in required_context_questions
                            )
                            == False
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
                f" If the problem persists please report it to {self.ISSUES_URL} "
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
