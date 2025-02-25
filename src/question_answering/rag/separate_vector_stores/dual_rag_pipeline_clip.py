import logging
import os
import pandas as pd
from typing import List
from data_summarization.context_summarization import TextSummarizer
from question_answering.rag.separate_vector_stores.dual_rag_chain import DualMultimodalRAGChain
from question_answering.rag.separate_vector_stores.dual_retrieval import DualClipRetriever
from utils.azure_config import get_azure_config
from utils.model_loading_and_prompting.llava import load_llava_model
from rag_env import EMBEDDING_MODEL_TYPE, IMAGES_DIR, INPUT_DATA, MODEL_TYPE, TEXT_SUMMARIES_CACHE_DIR, VECTORSTORE_PATH_CLIP_SEPARATE


class DualMultimodalRAGPipelineClip:
    """
    Initializes the Multimodal RAG pipeline with separate vector stores and retrievers for texts and images.
    Answers a user query retrieving additional context from texts and images within a document collection.
    Can be used with different models for answer generation (AzureOpenAI and LLaVA).
    Uses CLIP as multimodal embedding model to embed the images.
    Uses a text embedding model to embed the text.
    The query is embedded both with CLIP and with the text embedding model to allow retrieval for each modality.
    
    Attributes:  
        model_type (str): Type of the model to use for answer synthesis.  
        store_path (str): Path to the directory where the vector databases for texts and images are stored.  
        model: The model used for answer synthesis loaded based on `model_type`.  
        tokenizer: The tokenizer used for tokenization. Can be None.
        text_summarizer (TextSummarizer): Can be used to summarize texts before retrieving them.
        dual_retriever (DualClipRetriever): Retrieval using CLIP embeddings for images and text embeddings for texts.
        rag_chain (DualMultimodalRAGChain): RAG chain performing the QA task.
    """
    def __init__(self, model_type, store_path, text_embedding_model):
        
        config = get_azure_config()
        
       
        print("Using LLaVA model for answer generation")
        self.model, self.tokenizer = load_llava_model("llava-hf/llava-v1.6-mistral-7b-hf")

        self.text_summarizer = TextSummarizer(model_type="gpt4", cache_path=TEXT_SUMMARIES_CACHE_DIR)
        self.dual_retriever = DualClipRetriever(store_path=store_path,
                                            text_model_id=model_type,
                                            text_embedding_model=text_embedding_model)

        self.rag_chain = DualMultimodalRAGChain(self.model,
                                            self.tokenizer,
                                            self.dual_retriever.text_retriever,
                                            self.dual_retriever.img_retriever)

    def load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        # drop duplicate entries in the texts column of the dataframe
        texts = df.drop_duplicates(subset='text')[["text", "doc_id"]]
        return texts

    def summarize_data(self, texts: List[str]) -> List[str]:
        text_summaries = self.text_summarizer.summarize(texts)
        return text_summaries
    
    def index_data(self, images_dir: str, texts: List[str], text_filenames: List[str], text_summaries: List[str]=None):
        self.dual_retriever.add_images(images_dir)
        if text_summaries:
            self.dual_retriever.add_texts(text_summaries, texts, text_filenames)
        else:
            self.dual_retriever.add_texts(texts, texts, text_filenames)

    def answer_question(self, question: str) -> str:
        return self.rag_chain.run(question)


def main():
    pipeline = DualMultimodalRAGPipelineClip(model_type=MODEL_TYPE,
                                     store_path=VECTORSTORE_PATH_CLIP_SEPARATE,
                                     text_embedding_model=EMBEDDING_MODEL_TYPE)
    texts_df = pipeline.load_data(INPUT_DATA)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()
    
    pipeline.index_data(images_dir=IMAGES_DIR, texts=texts, text_summaries=texts, text_filenames=texts_filenames)

    question = "I want to change the behaviour of the stations to continue, if a moderate error occurrs. How can I do this?"
    answer = pipeline.answer_question(question)
    relevant_images = pipeline.rag_chain.retrieved_images
    relevant_texts = pipeline.rag_chain.retrieved_texts
    print("Retrieved images:", len(relevant_images), ", Retrieved texts:", len(relevant_texts))  
    print(answer)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("An error occurred during execution", exc_info=True)
