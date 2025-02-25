import os
import uuid
from typing import List, Tuple
from langchain.retrievers import MultiVectorRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain.storage import LocalFileStore
from utils.azure_config import get_azure_config
from rag_env import IMAGES_DIR


class DualSummaryStoreAndRetriever:
    """
    A class providing two separate document stores and vector stores
    to contain texts and images respectively, together with their emebeddings.
    The class also provides a text retriever and an image retriever
    to find documents that are relevant for a query.
    Retrieval is performed using the embeddings in the vector store, but the documents contained in the
    document store are returned. This allows image retrieval via the image summaries, while still ensuring
    that the original images associated with the summaries are returned.
  
    Attributes:
        embedding_model (str): Model used to embed the texts and image summaries.
        store_path (str): Path where the vector and document stores should be saved.
        img_docstore (LocalFileStore): Document store containing images.
        text_docstore (LocalFileStore): Document store containing texts.
        img_vectorstore (Chroma): Vector store containing embedded image summaries.
        text_vectorstore (Chroma): Vector store containing embedded texts.
        img_retriever (MultiVectorRetriever): Retriever that encommpasses both a vector store and a document store for image retrieval.
        text_retriever (MultiVectorRetriever): Retriever that encommpasses both a vector store and a document store for text retrieval.
    """
    def __init__(self, embedding_model, store_path=None, model_id=None):
       
        print("Using BGE embeddings")
        model_name = "BAAI/bge-m3"
        model_kwargs = {"device": "mps"}
        encode_kwargs = {"normalize_embeddings": True, "batch_size": 1, "show_progress_bar":True}
        self.embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        
        self.store_path = store_path
        img_vectorstore_dir = os.path.join(self.store_path, rf"image_only_{model_id}_vectorstore_{embedding_model}")
        img_docstore_dir = os.path.join(self.store_path, rf"image_only_{model_id}_docstore_{embedding_model}")
        text_vectorstore_dir = os.path.join(self.store_path, rf"text_only_{model_id}_vectorstore_{embedding_model}")
        text_docstore_dir = os.path.join(self.store_path, rf"text_only_{model_id}_docstore_{embedding_model}")
        
        self.img_docstore = LocalFileStore(img_docstore_dir)
        self.text_docstore = LocalFileStore(text_docstore_dir)
        self.id_key = "doc_id"
        self.doc_ids = []

        # Initialize the image vectorstore with the embedding function
        self.img_vectorstore = Chroma(
            persist_directory=img_vectorstore_dir,
            embedding_function=self.embeddings,
            collection_name=f"mm_rag_with_image_summaries_{embedding_model}_embeddings"
        )

        # Initialize the image vectorstore with the embedding function
        self.text_vectorstore = Chroma(
            persist_directory=text_vectorstore_dir,
            embedding_function=self.embeddings,
            collection_name=f"mm_rag_with_image_summaries_{embedding_model}_embeddings"
        )
        
        results_img = self.img_vectorstore.get(include=["embeddings", "documents", "metadatas"])
        results_text = self.text_vectorstore.get(include=["embeddings", "documents", "metadatas"])
        
        # A vector store is considered as already existing if it contains embeddings for both modalities
        self.is_new_vectorstore = bool(results_img["embeddings"]) and bool(results_text["embeddings"])

        if self.is_new_vectorstore:
            print(f"Vectorstore at path {img_vectorstore_dir} already exists")

        else:
            print(f"Creating new vectorstore and docstore at path {img_vectorstore_dir}")

        self.text_retriever = MultiVectorRetriever(
            vectorstore=self.text_vectorstore,
            docstore=self.text_docstore,
            id_key=self.id_key,
            search_kwargs={"k": 2}
        )
        self.img_retriever = MultiVectorRetriever(
            vectorstore=self.img_vectorstore,
            docstore=self.img_docstore,
            id_key=self.id_key,
            search_kwargs={"k": 2}
        )
        
        self.retrieved_docs = []
        self.retrieved_imgs = []
        self.retrieved_texts =[]
        

    def add_docs(self, doc_summaries: List[str], doc_contents: List[str], doc_filenames: List[str], modality: str):
        """
        Add documents to the vector store and document store of the selected modality.
        
        :param doc_summaries: Either text or image summaries to be stored in the vector store.
        :param doc_contents: The original texts or images to be stored in the document store.
        :param doc_filenames: File names associated with the respective documents to be stored as additional metadata.
        """
        if not self.is_new_vectorstore:
            print("Adding documents...")
            doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
            summary_docs = [
                Document(page_content=s, metadata={self.id_key: doc_ids[i], "filename": doc_filenames[i]})
                for i, s in enumerate(doc_summaries)
            ]
            
            if modality == "text":
                self.text_vectorstore.add_documents(summary_docs)
                self.text_docstore.mset(list(zip(doc_ids, map(lambda x: str.encode(x), doc_contents))))
            else:
                self.img_vectorstore.add_documents(summary_docs)
                self.img_docstore.mset(list(zip(doc_ids, map(lambda x: str.encode(x), doc_contents))))
        else:
            print("Documents have already been added before, skipping...")
            

    def retrieve(self, query: str, limit: int, retriever) -> Tuple[List[Document], List[Document]]:
        """
        Retrieve the most relevant documents based on the query.
        """
        if retriever == "image":
            self.retrieved_imgs = self.img_retriever.invoke(query, limit=limit)
        else:
            self.retrieved_texts = self.text_retriever.invoke(query, limit=limit)

        return self.retrieved_imgs, self.retrieved_texts




class DualClipRetriever:
    """  
    A class providing a vector store to contain multimodal CLIP embeddings of images
    and a vector store and a vector store and associated document store for texts.
    
    The class also provides a text retriever and an image retriever
    to find documents that are relevant for a query.
  
    Attributes:  
        store_path (str): Path where the vector stores and document store should be saved.
        text_model_id:
        text_embedding_model (str): Model used to embed the texts.
        images_dir (str): Directory containing the images to be embedded.
        img_vectorstore (Chroma): Vector store containing images embedded with CLIP.
        text_vectorstore (Chroma): Vector store containing texts embedded with the desired text_embedding_model.
        text_docstore (LocalFileStore): Document store containing texts.
        img_retriever (MultiVectorRetriever): Retriever that used CLIP embeddings for image retrieval.
        text_retriever (MultiVectorRetriever): Retriever that encommpasses both a vector store and a document store for text retrieval.
    """  
    def __init__(self, store_path, text_model_id, text_embedding_model,
                 images_dir=IMAGES_DIR):
        self.images_dir = images_dir

    
        print("Using BGE embeddings")
        model_name = "BAAI/bge-m3"
        model_kwargs = {"device": "mps"}
        encode_kwargs = {"normalize_embeddings": True, "batch_size": 1, "show_progress_bar":True}
        self.embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

        self.store_path = store_path
        img_vectorstore_dir = os.path.join(self.store_path, rf"image_only_clip/image_only_vectorstore_clip")
        text_vectorstore_dir = os.path.join(self.store_path, rf"text_only_{text_model_id}/text_only_{text_model_id}_vectorstore_{text_embedding_model}")
        text_docstore_dir = os.path.join(self.store_path, rf"text_only_{text_model_id}/text_only_{text_model_id}_docstore_{text_embedding_model}")

        self.text_docstore = LocalFileStore(text_docstore_dir)
        self.id_key = "doc_id"
        self.doc_ids = []

        # Create chroma vectorstore
        self.img_vectorstore = Chroma(
            collection_name="mm_rag_clip_photos",
            embedding_function=OpenCLIPEmbeddings(),
            persist_directory=img_vectorstore_dir
        )


        self.text_vectorstore = Chroma(
            persist_directory=text_vectorstore_dir,
            embedding_function=self.embeddings,
            collection_name=f"mm_rag_with_image_summaries_{text_embedding_model}_embeddings"
        )

        results_img = self.img_vectorstore.get(include=["embeddings", "documents", "metadatas"])
        results_text = self.text_vectorstore.get(include=["embeddings", "documents", "metadatas"])

        self.is_new_vectorstore = bool(results_img["embeddings"]) and bool(results_text["embeddings"])

        if self.is_new_vectorstore:
            print(f"Vectorstore at path {img_vectorstore_dir} already exists")

        else:
            print(f"Creating new vectorstore and docstore at path {img_vectorstore_dir}")

        self.img_retriever = self.img_vectorstore.as_retriever(search_kwargs={"k": 2})

        self.text_retriever = MultiVectorRetriever(
            vectorstore=self.text_vectorstore,
            docstore=self.text_docstore,
            id_key=self.id_key,
            search_kwargs={"k": 2}
        )

        self.retrieved_docs = []
        self.retrieved_imgs = []
        self.retrieved_texts =[]


    def add_images(self, images_dir: str=None):
        """
        Add images to the image vector store.
        
        :param images_dir: Directory containing the images to be embedded.
        """
        if not self.is_new_vectorstore:
            if images_dir:
                image_uris = self.extract_image_uris(images_dir)
                print(f"Found {len(image_uris)} images")
                # Convert the list of URIs to a list of dictionaries
                image_metadatas = [{'filename': self.extract_manual_name(uri)} for uri in image_uris]

                print("Adding images to vectorstore...")
                self.img_vectorstore.add_images(uris=image_uris, metadatas=image_metadatas)
        else:
            print("Documents have already been added before, skipping...")



    def extract_image_uris(self, root_path: str, image_extension: str=".png") -> List[str]:
        image_uris = []
        for subdir, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith(image_extension):
                    image_uris.append(os.path.join(subdir, file))
        return sorted(image_uris)



    def extract_manual_name(self, uri: str) -> str:  
        # Split the URI into parts  
        parts = uri.split('/')
        # The directory name is the second to last element  
        directory_name = parts[-2]
        # Append ".pdf" to the directory name  
        return f"{directory_name}.pdf"


    def add_texts(self, doc_summaries: List[str], doc_contents: List[str], doc_filenames: List[str]):
        """
        Add texts to the text vector store and document store.
        
        :param doc_summaries: Text summaries to be stored in the vector store.
        :param doc_contents: The original texts to be stored in the document store.
        :param doc_filenames: File names associated with the texts to be stored as additional metadata.
        """
        if not self.is_new_vectorstore:
            print("Adding documents...")
            doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
            summary_docs = [
                Document(page_content=s, metadata={self.id_key: doc_ids[i], "filename": doc_filenames[i]})
                for i, s in enumerate(doc_summaries)
            ]
            self.text_vectorstore.add_documents(summary_docs)
            self.text_docstore.mset(list(zip(doc_ids, map(lambda x: str.encode(x), doc_contents))))
        else:
            print("Documents have already been added before, skipping...")


    def retrieve(self, query: str, limit: int, retriever) -> Tuple[List[Document], List[Document]]:
        """
        Retrieve the most relevant documents based on the query.
        """
        if retriever == "image":
            self.retrieved_imgs = self.img_retriever.invoke(query, limit=limit)
        else:
            self.retrieved_texts = self.text_retriever.invoke(query, limit=limit)

        return self.retrieved_imgs, self.retrieved_texts

