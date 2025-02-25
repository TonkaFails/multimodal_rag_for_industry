import pandas as pd
import logging
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from question_answering.rag.single_vector_store.rag_chain import MultimodalRAGChain
from question_answering.rag.single_vector_store.retrieval import ClipRetriever
from typing import List
from utils.model_loading_and_prompting.llava import load_llava_model
from rag_env import IMAGES_DIR, INPUT_DATA, MODEL_TYPE, VECTORSTORE_PATH_CLIP_SINGLE


class MultimodalRAGPipelineClip:
    """
    Initializes the Multimodal RAG pipeline.
    Answers a user query retrieving additional context from texts and images within a document collection.
    Can be used with different models for answer generation (AzureOpenAI and LLaVA).
    Uses CLIP as multimodal embedding model to embed the query, the images, and the texts into a single vector store.
    
    Attributes:
        model_type (str): Type of the model to use for answer synthesis.
        store_path (str): Path to the directory where the vector database is stored.
        model: The model used for answer synthesis loaded based on `model_type`.
        tokenizer: The tokenizer used for tokenization. Can be None.
        text_summarizer (TextSummarizer): Can be used to summarize texts before retrieving them.
        clip_retriever (ClipRetriever): Retrieval using CLIP embeddings for images.
        rag_chain (MultimodalRAGChain): RAG chain performing the QA task.
    """
    def __init__(self, model_type, store_path):
           
        print("Using LLaVA model")
        self.model, self.tokenizer = load_llava_model("llava-hf/llava-v1.6-mistral-7b-hf", "mps")

        # self.text_summarizer = TextSummarizer(model_type="gpt4", cache_path=TEXT_SUMMARIES_CACHE_DIR)
        self.clip_retriever = ClipRetriever(vectorstore_dir=store_path)
        self.rag_chain = MultimodalRAGChain(self.model, self.tokenizer, self.clip_retriever.retriever)
    
    
    def load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        # drop duplicate entries in the texts column of the dataframe
        texts = df.drop_duplicates(subset='text')[["text", "doc_id"]]
        return texts
    
    def summarize_data(self, texts: List[str]) -> List[str]:
        text_summaries = self.text_summarizer.summarize(texts)
        return text_summaries   
    
    def index_data(self, texts_df: pd.DataFrame, images_dir: str):
        self.clip_retriever.add_documents(images_dir=images_dir, texts_df=texts_df)

    def answer_question(self, question: str) -> str:
        return self.rag_chain.run(question)


def main():
    pipeline = MultimodalRAGPipelineClip(model_type=MODEL_TYPE, store_path=VECTORSTORE_PATH_CLIP_SINGLE)
    texts_df = pipeline.load_data(INPUT_DATA)
    pipeline.index_data(texts_df=texts_df, images_dir=IMAGES_DIR)
    
    question = "Where is the red wire?"
    answer = pipeline.answer_question(question)
    relevant_docs = pipeline.rag_chain.retrieved_docs  
    print("Retrieved images:", len(relevant_docs["images"]), ", Retrieved texts:", len(relevant_docs["texts"]))  
    print(answer)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("An error occurred during execution", exc_info=True)
