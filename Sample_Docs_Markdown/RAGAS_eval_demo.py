from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.transforms import default_transforms
from ragas.testset.transforms.extractors import HeadlinesExtractor
from ragas.testset.transforms.splitters import HeadlineSplitter

# Load environment variables (e.g. OPENAI_API_KEY)
load_dotenv()

# Load markdown documents from the sample docs directory
path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(path, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
docs = loader.load()

# Set up LLM and embedding model using OpenAI
openai_client = OpenAI()
generator_llm = llm_factory("gpt-4o-mini", client=openai_client)
generator_embeddings = OpenAIEmbeddings(client=openai_client)

# Build transform pipeline, skipping headline-based transforms since some docs lack headers
transforms = [
    t for t in default_transforms(docs, generator_llm, generator_embeddings)
    if not isinstance(t, (HeadlinesExtractor, HeadlineSplitter))
]

# Generate synthetic testset of 10 QA pairs from the documents
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10, transforms=transforms)

# Convert to pandas DataFrame for inspection
df = dataset.to_pandas()
