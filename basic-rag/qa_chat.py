import os
import threading

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes, ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

# from chromadb import Client
# import chromadb

from config import load_configs


# credentials_Krisha_ibm = {
#     "url": "https://us-south.ml.cloud.ibm.com",
#     "apikey": "0CM5mrv9fptwPHmK5dz9uMBPvxd_57FSoWPn8w2uEgo5",
# }

# credentials = {
#     "url": "https://us-south.ml.cloud.ibm.com",
#     "apikey": "89rSsvH8ff2F5DaWLKOxp8optU46sEofJQ3YKTlBq1b2",
# }

# try:
#     # project_id = os.environ["PROJECT_ID"]
#     # project_id = "39308f9d-00ff-402b-9399-5d289deb4788" ### Krishna IBM
#     project_id = "189c5a4e-0d09-40e3-8d78-d050d1f93078"
# except KeyError:
#     project_id = input("Please enter your project_id (hit enter): ")

# collection_name = "itr-docs"
# chroma_client = chromadb.HttpClient(host="localhost", port=8000)
# collection = chroma_client.get_or_create_collection(collection_name)

# embeddings = WatsonxEmbeddings(
#     model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
#     url=credentials["url"],
#     apikey=credentials["apikey"],
#     project_id=project_id,
#     params={"TRUNCATE_INPUT_TOKENS": 500},
# )


# chromadb_store = Chroma(client=chroma_client, collection_name=collection_name, embedding_function=embeddings)


# model_id = ModelTypes.GRANITE_13B_CHAT_V2
# parameters = {
#     GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
#     GenParams.MIN_NEW_TOKENS: 1,
#     GenParams.MAX_NEW_TOKENS: 100,  # old: 100
#     GenParams.STOP_SEQUENCES: ["<|endoftext|>"],
# }
# watsonx_granite = WatsonxLLM(
#     model_id=model_id.value,
#     url=credentials.get("url"),
#     apikey=credentials.get("apikey"),
#     project_id=project_id,
#     params=parameters,
# )

# qa = RetrievalQA.from_chain_type(
#     llm=watsonx_granite, chain_type="stuff", retriever=chromadb_store.as_retriever()
# )
# query = "how to select a bank for refund"
# print(f"Question:\n  {query}")
# answer = qa.invoke(query)["result"]
# print(f"Answer:\n {answer}")

def get_embeddings() -> WatsonxEmbeddings :
    embeddings = WatsonxEmbeddings(
        model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
        url=os.environ.get(
            "WATSONX_AI_URL",
            "https://us-south.ml.cloud.ibm.com",
        ),
        apikey=os.environ.get(
            "WATSONX_AI_APIKEY",
            "0CM5mrv9fptwPHmK5dz9uMBPvxd_57FSoWPn8w2uEgo5",
        ),
        project_id=os.environ.get(
            "WATSONX_AI_PROJECT_ID",
            "39308f9d-00ff-402b-9399-5d289deb4788",
        ),
        params={"TRUNCATE_INPUT_TOKENS": 500},
    )
    return embeddings

def setup_vector_store(embeddings: WatsonxEmbeddings) -> Chroma: 
    collection_name = "itr-docs"
    # chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    # collection = chroma_client.get_or_create_collection(collection_name)
    chromadb_store = Chroma(
        # client=chroma_client,
        persist_directory="/tmp/vector_data",
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    print(f"{type(chromadb_store) = }")
    return chromadb_store


def get_llm() -> WatsonxLLM:
    model_id = ModelTypes.GRANITE_13B_CHAT_V2
    default_parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 100,  # old: 100
        GenParams.STOP_SEQUENCES: ["<|endoftext|>"],
    }
    configs = load_configs()
    parameters = configs.get("model_params", default_parameters)
    watsonx_granite = WatsonxLLM(
        model_id=model_id.value,
        url=os.environ.get(
            "WATSONX_AI_URL",
            "https://us-south.ml.cloud.ibm.com",
        ),
        apikey=os.environ.get(
            "WATSONX_AI_APIKEY",
            "0CM5mrv9fptwPHmK5dz9uMBPvxd_57FSoWPn8w2uEgo5",
        ),
        project_id=os.environ.get(
            "WATSONX_AI_PROJECT_ID",
            "39308f9d-00ff-402b-9399-5d289deb4788",
        ),
        params=parameters,
    )
    return watsonx_granite


class WxAIRetriever:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        with WxAIRetriever._lock:
            if WxAIRetriever._instance is None:
                WxAIRetriever._instance = self
                self.embeddings = get_embeddings()
                self.llm = get_llm()
                self.vector_store = setup_vector_store(self.embeddings)
                self.qa_agent = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(),
                )
        print(f"{type(self.qa_agent)} = ")

    @classmethod
    def get_agent(cls):
        if not cls._instance:
            cls._instance = WxAIRetriever()
        return cls._instance.qa_agent


def get_qa_chain() -> RetrievalQA:
    return WxAIRetriever.get_agent()


def process_query(query: str) -> str:
    qa_chain = get_qa_chain()
    try:
        answer = qa_chain.invoke(query)["result"]
    except Exception as ex:
        message = f"Oops. something went wrong, possibly due to.. \n{ex}"
        answer = message
    return answer
