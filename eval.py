import warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import umap
import hdbscan
import plotly.express as px
import pandas as pd
import nest_asyncio
nest_asyncio.apply()

import numpy as np 

import os
os.environ['GROQ_API_KEY'] = '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'
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import json
from getpass import getpass 
from urllib.request import urlopen

import phoenix as phoenix
from langchain.chains import RetrievalQA
from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)

from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
from phoenix.trace import DocumentEvaluations, SpanEvaluations
from tqdm import tqdm
from phoenix.otel import register

from openinference.instrumentation.groq import GroqInstrumentor
from groq import Groq

os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "api_key=c0ded98735104aa543b:9e8354a"
os.environ["PHOENIX_CLIENT_HEADERS"] = "api_key=c0ded98735104aa543b:9e8354a"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

phoenix.launch_app()

tracer_provider = register(
  project_name="my-llm-app", # Default is 'default'
) 

GroqInstrumentor().instrument(tracer_provider=tracer_provider)


# Add Phoenix API Key for tracing



client = Groq(
    # This is the default and can be omitted
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs",
        }
    ],
    model="mixtral-8x7b-32768",
)
print(chat_completion.choices[0].message.content)

  
# configure the Phoenix tracer

# defaults to endpoint="http://localhost:4317"


# sentences = ["Music is great",
# "Guitars are a great instrument",
# "Doctors use instuments to measure people's health",
# "It is healthy to ask for help",
# "It is helpful to use machine learning",
# "LLMs are doctors"]

# embed_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# embeddings = embed_model.encode(sentences)
# # print(embeddings)

# umap_reducer = umap.UMAP(n_components=2, random_state=0)
# flat_embeddings = umap_reducer.fit_transform(embeddings)
# # print(flat_embeddings)

# df = pd.DataFrame(flat_embeddings, columns=["x", "y"])
# df["sentence"] = sentences
# # print(df)

# fig = px.scatter(df, x="x", y="y", text="sentence", title="music with umap")
# fig.update_traces(textposition='top center')
# fig.update_layout(hovermode='closest')
# # fig.show()

# clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
# cluster_labels = clusterer.fit_predict(embeddings)
# df['cluster'] = cluster_labels
# # print(df)

# fig = px.scatter(df, x="x", y="y", color="cluster", text="sentence", title="music with umap and HDBSCAN")
# fig.update_traces(textposition='top center')
# fig.update_layout(hovermode='closest')
# fig.show()

# llm = ChatGroq(
#     temperature = 0.1,
#     model = "llama3-70b-8192"
# )

# system = "You are a helpful AI assistant"
# human = "{text}"

# prompt = ChatPromptTemplate.from_messages([
#     ("system", system), 
#     ("human", human)
#     ])
    
# chain = prompt | llm
# response = chain.invoke({"text": "What is your favorite instrument?"})
# # print(response)
# # tracer_provider = register()
# # LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
# session = phoenix.launch_app()

