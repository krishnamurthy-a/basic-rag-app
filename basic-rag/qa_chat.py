import os
import threading

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes, ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI


from config import load_configs

def get_embeddings() -> WatsonxEmbeddings :
    # watsonx_apikey = 
    embeddings = WatsonxEmbeddings(
        model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
        url=os.environ.get(
            "WATSONX_AI_URL",
            "https://us-south.ml.cloud.ibm.com",
        ),
        apikey=os.environ.get(
            "WATSONX_AI_APIKEY",
            "***",
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


def setup_openai_vector_store(embeddings: OpenAIEmbeddings) -> Chroma:
    collection_name = "itr-docs-openai"
    # chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    # collection = chroma_client.get_or_create_collection(collection_name)
    chromadb_store = Chroma(
        # client=chroma_client,
        persist_directory="/tmp/vector_data_openai",
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
            "***",
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


class OpenAIRetriever:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        with OpenAIRetriever._lock:
            if OpenAIRetriever._instance is None:
                OpenAIRetriever._instance = self
                self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
                self.llm = OpenAI(temperature=0.7, max_tokens=200)
                self.vector_store = setup_openai_vector_store(self.embeddings)
                self.qa_agent = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(),
                )
        print(f"{type(self.qa_agent)} = ")

    @classmethod
    def get_agent(cls):
        if not cls._instance:
            cls._instance = OpenAIRetriever()
        return cls._instance.qa_agent


def get_qa_chain() -> RetrievalQA:
    configs = load_configs()
    if configs.get("platform", "watsonx") == "openai":
        if configs.get("openai_apikey", None):
            os.environ["OPENAI_API_KEY"] = configs["openai_apikey"]
            return OpenAIRetriever.get_agent()
        else:
            raise Exception("OpenAI api key is not available.")
    else:
        if configs.get("watsonx_apikey", None):
            os.environ["WATSONX_AI_APIKEY"] = configs["watsonx_apikey"]
            return WxAIRetriever.get_agent()
        else:
            raise Exception("watsonx.ai api key is not available.")


def process_query(query: str) -> str:
    qa_chain = get_qa_chain()
    try:
        answer = qa_chain.invoke(query)["result"]
    except Exception as ex:
        message = f"Oops. something went wrong, possibly due to.. \n{ex}"
        answer = message
    return answer
