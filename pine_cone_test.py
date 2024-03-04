from pinecone import Pinecone, PodSpec

import os

from llama_index.llms.predibase import PredibaseLLM
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
from llama_index.readers.file import PyMuPDFReader

from predibase import PredibaseClient

from pathlib import Path


# Connect to Predibase
os.environ["PREDIBASE_API_TOKEN"] = "YOUR PREDIBASE API TOKEN"

#Establish llm connection (model_name is only works when model is base model, fine-tuned model weren't working)
predibase_llm  = PredibaseLLM(
    model_name="YOUR MODEL NAME", temperature=0.3, max_new_tokens=512
)

# Get embedding model
embed_model = resolve_embed_model("YOUR EMBEDDING MODEL")

# Set up llama_index settings
Settings.llm = predibase_llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900

# Connect to Pinecone
pc = Pinecone(api_key="YOUR PINECONE KEY")

pinecone_index = pc.Index("YOUR PINECONE INDEX")
#pinecone_index.delete(deleteAll=True) # delete all vectors

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

loader = PyMuPDFReader()
documents = loader.load(file_path="FILEPATH FOR LOAD")
splitter = SentenceSplitter(chunk_size=1024)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter], storage_context=storage_context
)

query_engine = index.as_query_engine()


response = query_engine.query("QUERY FOR PINECONE")
print(response)


