<div align="center">
  <h1>✨🦙 Argilla's LlamaIndex Integration</h1>
  <p><em> Argilla integration into the LlamaIndex workflow</em></p>
</div>

> [!TIP]
> To discuss, get support, or give feedback [join Discord](http://hf.co/join/discord) in #argilla-distilabel-general and #argilla-distilabel-help. You will be able to engage with our amazing community and the core developers of `argilla` and `distilabel`.

This integration allows the user to include the feedback loop that Argilla offers into the LlamaIndex ecosystem. It's based on a callback handler to be run within the LlamaIndex workflow.

Don't hesitate to check out both [LlamaIndex](https://github.com/run-llama/llama_index) and [Argilla](https://github.com/argilla-io/argilla)

## Getting Started

You first need to install argilla-llama-index as follows:

```bash
pip install argilla-llama-index
```

If you already have deployed Argilla, you can skip this step. Otherwise, you can quickly deploy Argilla following [this guide](https://docs.argilla.io/latest/getting_started/quickstart/).

## Basic Usage

To easily log your data into Argilla within your LlamaIndex workflow, you only need to initialize the handler and attach it to the LlamaIndex dispatcher. This ensured that the predictions obtained using LlamaIndex are automatically logged to the Argilla instance.

- `dataset_name`: The name of the dataset. If the dataset does not exist, it will be created with the specified name. Otherwise, it will be updated.
- `api_url`: The URL to connect to the Argilla instance.
- `api_key`: The API key to authenticate with the Argilla instance.
- `number_of_retrievals`: The number of retrieved documents to be logged. Defaults to 0.
- `workspace_name`: The name of the workspace to log the data. By default, the first available workspace.

> For more information about the credentials, check the documentation for [users](https://docs.argilla.io/latest/how_to_guides/user/) and [workspaces](https://docs.argilla.io/latest/how_to_guides/workspace/).

```python
from llama_index.core.instrumentation import get_dispatcher
from argilla_llama_index import ArgillaHandler

argilla_handler = ArgillaHandler(
    dataset_name="query_llama_index",
    api_url="http://localhost:6900",
    api_key="argilla.apikey",
    number_of_retrievals=2,
)
root_dispatcher = get_dispatcher()
root_dispatcher.add_span_handler(argilla_handler)
root_dispatcher.add_event_handler(argilla_handler)
```

Let's log some data into Argilla. With the code below, you can create a basic LlamaIndex workflow. We will use GPT3.5 from OpenAI as our LLM ([OpenAI API key](https://openai.com/blog/openai-api)). Moreover, we will use an example `.txt` file obtained from the [LlamaIndex documentation](https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html).

```python
import os

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# LLM settings
Settings.llm = OpenAI(
  model="gpt-3.5-turbo", temperature=0.8, openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Load the data and create the index
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create the query engine with the same similarity top k as the number of retrievals
query_engine = index.as_query_engine(similarity_top_k=2)
```

Now, let's run the `query_engine` to have a response from the model. The generated response will be logged into Argilla.

```python
response = query_engine.query("What did the author do growing up?")
response
```

![Argilla UI](/docs/assets/UI-screenshot.png)
