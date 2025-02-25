import pandas as pd
from collections import defaultdict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from utils.base64_utils.base64_utils import *
from utils.model_loading_and_prompting.llava import llava_call
from rag_env import REFERENCE_QA


class DualMultimodalRAGChain:
    def __init__(self, model, tokenizer, text_retriever, image_retriever):
        """
        Multi-modal RAG chain with separate vector stores and retrievers for texts and images
        The steps are as follows:
        1. Call the retrievers to find relevant texts and images
        2. Create dictionary with retrieved texts or images
        3. Create a QA prompt using the question and retrieved image/text context
        4. Call the LLM and prompt it with the created prompt
        5. Obtain the generated answer from the model
        
        :param model: The model used for answer generation.
        :param tokenizer: The tokenizer used for tokenization.
        :param text_retriever: The retriever used for text retrieval.
        :param image_retriever: The retriever used for image retrieval.
        :param df: The Dataframe containing the user questions. The names of the files associated with the
        user questions can be used to filter the document collection for retrieval.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever
        self.df = pd.read_excel(REFERENCE_QA)
        
        self.chain = (
            {
                "text_context": self.text_retriever | RunnableLambda(self.split_image_text_types),
                "image_context": self.image_retriever | RunnableLambda(self.split_image_text_types),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self.img_prompt_func)
            | RunnableLambda(self.call_model)
            | StrOutputParser()
        )

        self.retrieved_docs = defaultdict(list)
        self.retrieved_images = []
        self.retrieved_texts = []

        
    def run(self, question):
        return self.chain.invoke(question)
        

    def call_model(self, prompt):
        """  
        Calls the model based on the model type.
        
        :return: A Langchain abstraction (RunnableLambda) to turn the model into a pipe-compatible function for the RAG chain.  
        """ 
        if self.tokenizer:
            return RunnableLambda(self.call_llava)
        else:
            return RunnableLambda(self.model)

        
    def call_llava(self, inputs):
        
        prompt = inputs['prompt']
        image = inputs.get('image', None)
        ans = llava_call(prompt, self.model, self.tokenizer, device="mps", image=image)
        return ans


    def split_image_text_types(self, docs):
        """
        Split base64-encoded images and texts.
        
        :return: A dictionary with separate entries for texts and base64-encoded images.
        """
        b64_images = []
        texts = []
        is_image = False
        for doc in docs:
            # Check if the document is of type Document and extract page_content if so
            if isinstance(doc, Document):
                doc = doc.page_content
            else:
                doc = doc.decode('utf-8')
            if looks_like_base64(doc) and is_image_data(doc):
                is_image = True
                doc = resize_base64_image(doc, size=(1300, 600))
                b64_images.append(doc)
            else:
                texts.append(doc)
                
        if is_image:
            self.retrieved_images = b64_images
            return {"images": b64_images, "texts": []}
        else:
            self.retrieved_texts = texts
            return {"images": [], "texts": texts}


    def img_prompt_func(self, data_dict):
        """
        Constructs a dictionary containing the model-specific prompt and an image.
        """
        
        qa_prompt = """You are an expert AI assistant that answers questions about manuals from the industrial domain.\n
        You will be given some context consisiting of text and/or image(s) that can be photos, screenshots, graphs, charts and other.\n
        Use this information from both text and image (if present) to provide an answer to the user question.\n
        Avoid expressions like: 'according to the text/image provided' and similar, and just answer the question directly."""
        
        prompt = self.llava_qa(data_dict, qa_prompt)
            
        return prompt
    
    
    def llava_qa(self, inputs, qa_prompt):
        """
        Constructs a prompt for question answering using the formatting required by LLaVA.
        """
        
        formatted_texts = "\n".join(inputs["text_context"]["texts"])
    
        prompt = f"[INST]{'<image>' if inputs['image_context']['images'] else ' '}\n{qa_prompt}\nUser-provided question: {inputs['question']}\n\nText:\n{formatted_texts}[/INST]"
       
        if inputs['image_context']['images']:
             # pass always only the first image as llava cannot handle multiple images
            image = inputs['image_context']['images'][0]
            image = decode_image_to_bytes(image)
            image = Image.open(io.BytesIO(image))
        else:
            image = None  
        
        return {"prompt": prompt, "image": image}

        
        
    def azure_qa(self, data_dict, qa_prompt):
        """
        Constructs a prompt for question answering using the formatting required by AzureOpenAI.
        """
        if data_dict["text_context"]["texts"]:
            formatted_texts = "Text:\n\n".join(data_dict["text_context"]["texts"])
        else:
            formatted_texts = ""
        messages = []

        # Adding image(s) to the messages if present
        if data_dict["image_context"]["images"]:
            # if multiple images should be added
            # for image in data_dict["image_context"]["images"]:
            # add only the first image since llava only supports 1 image
            image = data_dict["image_context"]["images"][0]
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

        # Adding the text for answer generation
        text_message = {
            "type": "text",
            "text": (
                f"{qa_prompt}\n"
                f"User-provided question: {data_dict['question']}\n\n"
                f"{formatted_texts}"
            ),
        }
        messages.append(text_message)
        print("Calling model...")
        return [HumanMessage(content=messages)]
