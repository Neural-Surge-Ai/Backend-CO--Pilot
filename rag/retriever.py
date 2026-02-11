import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore



import getpass
import os
from dotenv import load_dotenv

load_dotenv(r"D:\Copilot\ai-copilot\.env")
print(os.getenv("OPENAI_API_KEY"))
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

import openai

class SimpleOpenAIEmbeddings:
    def __init__(self, model="text-embedding-3-large"):
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def embed_documents(self, texts):
        # texts: list[str]
        resp = openai.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


embeddings = SimpleOpenAIEmbeddings(model="text-embedding-3-large")



# Init Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Attach vector store (no upsert now, just read)
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)



retriever = vector_store.as_retriever(
    search_type="similarity",   # default cosine similarity
    search_kwargs={"k": 5}
)

# query="What are the main sections of the website and their content?"

# retriever = vector_store.as_retriever(search_kwargs={"k": 2})
# results = retriever.invoke(query)

# print(results)