import logging
import os
import pandas as pd
from typing import List, Tuple
from data_summarization.context_summarization import ImageSummarizer, TextSummarizer
from question_answering.rag.single_vector_store.rag_chain import MultimodalRAGChain
from question_answering.rag.single_vector_store.retrieval import SummaryStoreAndRetriever
from utils.azure_config import get_azure_config
from utils.model_loading_and_prompting.llava import load_llava_model
from rag_env import EMBEDDING_MODEL_TYPE, IMG_SUMMARIES_CACHE_DIR, INPUT_DATA, MODEL_TYPE, TEXT_SUMMARIES_CACHE_DIR, VECTORSTORE_PATH_SUMMARIES_SINGLE


class MultimodalRAGPipelineSummaries:
    """
    Initializes the Multimodal RAG pipeline.
    Answers a user query retrieving additional context from texts and images within a document collection.
    Can be used with different models for answer generation (AzureOpenAI and LLaVA).
    Transforms images into textual summaries and embeds the image summaries, the texts, and the query into a single vector store.
    
    Attributes:
        model_type (str): Type of the model to use for answer synthesis.
        store_path (str): Path to the directory where the vector database is stored.
        embedding_model (str): Text embedding model used to embed texts and image summaries.
        model: The model used for answer synthesis loaded based on `model_type`.
        tokenizer: The tokenizer used for tokenization. Can be None.
        text_summarizer (TextSummarizer): Can be used to summarize texts before retrieving them.
        image_summarizer: (ImageSummarizer): Used to generate textual summaries from images.
        sotre_and_retriever (SummaryStoreAndRetriever): Retrieval using CLIP embeddings for images.
        rag_chain (MultimodalRAGChain): RAG chain performing the QA task.
    """
    def __init__(self, model_type, store_path, embedding_model):
        
        config = get_azure_config()
        
        print("Using LLaVA model for answer generation")
        self.model, self.tokenizer = load_llava_model("llava-hf/llava-v1.6-mistral-7b-hf")

        self.text_summarizer = TextSummarizer(model_type="gpt4", cache_path=TEXT_SUMMARIES_CACHE_DIR)
        self.image_summarizer = ImageSummarizer(self.model, self.tokenizer)
        self.store_and_retriever = SummaryStoreAndRetriever(embedding_model=embedding_model,
                                                            store_path=f"{store_path}_{model_type}")
        self.rag_chain = MultimodalRAGChain(self.model, self.tokenizer, self.store_and_retriever.retriever)


    def load_data(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_parquet(path)
        # drop duplicate entries in the texts column of the dataframe
        texts = df.drop_duplicates(subset='text')[["text", "doc_id"]]
        img = df[["doc_id", "image_bytes"]]
        images = img.dropna(subset=['image_bytes'])  
        return texts, images


    def summarize_data(self, texts: List[str], images: List[bytes], cache_file:str) -> Tuple[List[str], List[str], List[str]]:
        # Summarize images
        img_base64_list, image_summaries = self.image_summarizer.summarize(images, cache_file)
        # Summarize texts
        if texts:
            text_summaries = self.text_summarizer.summarize(texts)
            return text_summaries, image_summaries, img_base64_list
        return texts, image_summaries, img_base64_list


    def index_data(self, texts: List[str]=None, text_summaries: List[str]=None,
                   image_summaries: List[str]=None, images_base64: List[str]=None,
                   text_filenames: List[str]=None, image_filenames: List[str]=None):
        if text_summaries:
            print("Adding text summaries to store")
            self.store_and_retriever.add_docs(text_summaries, texts, text_filenames)
        if texts:
            print("Adding texts to store")
            self.store_and_retriever.add_docs(texts, texts, text_filenames)
        if image_summaries:
            print("Adding image summaries to store")
            self.store_and_retriever.add_docs(image_summaries, images_base64, image_filenames)


    def answer_question(self, question: str) -> str:
        return self.rag_chain.run(question)


def main():
    pipeline = MultimodalRAGPipelineSummaries(model_type=MODEL_TYPE,
                                     store_path=VECTORSTORE_PATH_SUMMARIES_SINGLE,
                                     embedding_model=EMBEDDING_MODEL_TYPE)
    
    texts_df, images_df = pipeline.load_data(INPUT_DATA)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()
    images, image_filenames = images_df[["image_bytes"]]["image_bytes"].tolist(), images_df[["doc_id"]]["doc_id"].tolist()
    img_base64_list, image_summaries = pipeline.image_summarizer.summarize(images, IMG_SUMMARIES_CACHE_DIR)

    pipeline.index_data(texts, image_summaries=image_summaries, 
                        images_base64=img_base64_list, 
                        image_filenames=image_filenames, 
                        text_filenames=texts_filenames)

    question = "I want to change the behaviour of the stations to continue, if a moderate error occurrs. How can I do this?"
    answer = pipeline.answer_question(question)
    relevant_docs = pipeline.rag_chain.retrieved_docs  
    print("Retrieved images:", len(relevant_docs["images"]), ", Retrieved texts:", len(relevant_docs["texts"]))  
    print(answer)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("An error occurred during execution", exc_info=True)
