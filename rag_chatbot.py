import bs4
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- Model & Embeddings Setup ---

model = init_chat_model("gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# --- Document Loading ---

# Fetch blog post and extract only the relevant HTML sections
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

# --- Chunking ---

# Split the document into overlapping chunks for better retrieval coverage
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # chunk size (characters)
    chunk_overlap=200, # overlap between chunks (characters)
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(all_splits)} sub-documents.")

# --- Indexing ---

# Embed each chunk and store in the vector store
document_ids = vector_store.add_documents(documents=all_splits)
print(document_ids[:3])

"""
Indexing time:  chunk text → embeddings model → vector → stored in vector store
Query time:     user question → embeddings model → vector → closest chunks → LLM
"""

# --- Retrieval Tool ---

# Retrieves the most relevant chunks from the vector store for a given query
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# --- Approach 1: Agent with retrieval tool ---

# The agent decides when to call retrieve_context based on the query
tools = [retrieve_context]
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)

query = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)
for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()

# --- Approach 2: Single-inference with injected context ---

# Retrieval happens before the LLM call — no tool use, just one inference
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject retrieved context directly into the system prompt."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

agent = create_agent(model, tools=[], middleware=[prompt_with_context])

query = "What is task decomposition?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
