
from collections import defaultdict
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from utils.base64_utils.base64_utils import *
from utils.model_loading_and_prompting.llava import llava_call, load_llava_model
from utils.azure_config import get_azure_config


class CorrectContextQAChain:
    def __init__(self, model_type):
        """
        Multi-modal RAG chain
        """
        
        config = get_azure_config()

        if model_type in config:
            print("Using Azure model")
            azure_llm_config = config[model_type]
            self.model = AzureChatOpenAI(
                openai_api_version=azure_llm_config["openai_api_version"],
                azure_endpoint=azure_llm_config["openai_endpoint"],
                azure_deployment=azure_llm_config["deployment_name"],
                model=azure_llm_config["model_version"],
                api_key=azure_llm_config["openai_api_key"],
                max_tokens=400)
            self.tokenizer = None
            
        else:
            print("Using LLaVA model")
            self.model, self.tokenizer = load_llava_model("llava-hf/llava-v1.6-mistral-7b-hf")
        
        # here we don't have a retriever since we already know which are the relevant documents and pass them directly when invoking the chain
        self.chain = (
                self.img_prompt_func | RunnableLambda(self.call_model) | StrOutputParser()
        )

        
    def run(self, inputs):
        return self.chain.invoke(inputs)
        

    def call_model(self, prompt: str) -> RunnableLambda:
        """
        Calls the model based on the model type.
        
        :return: A Langchain abstraction (RunnableLambda) to turn the model into a pipe-compatible function for the RAG chain.
        """
        if self.tokenizer:
            return RunnableLambda(self.call_llava)
        else:
            return RunnableLambda(self.model)

        
    def call_llava(self, inputs: dict) -> str:
        
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
        
        for doc in docs["context"]["images"]:
            if looks_like_base64(doc) and is_image_data(doc):
                doc = resize_base64_image(doc, size=(1300, 600))
                b64_images.append(doc)
            else:
                texts.append(doc)
                
        self.retrieved_docs = defaultdict(list)
        self.retrieved_docs['images'] = b64_images
        self.retrieved_docs['texts'] = docs["context"]["texts"]

        return self.retrieved_docs


    def img_prompt_func(self, data_dict: dict) -> dict:
        """
        Constructs a dictionary containing the model-specific prompt and an image.
        """
        
        qa_prompt = """You are an expert AI assistant that answers questions about manuals from the industrial domain.\n
        You will be given some context consisiting of text and/or image(s) that can be photos, screenshots, graphs, charts and other.\n
        Use this information from both text and image (if present) to provide an answer to the user question.\n
        Avoid expressions like: 'according to the text/image provided' and similar, and just answer the question directly."""
        
        if type(self.model) == AzureChatOpenAI:
            prompt_dict = self.azure_qa(data_dict, qa_prompt)
        else:
            prompt_dict = self.llava_qa(data_dict, qa_prompt)
            
        return (prompt_dict)
    
    
    def llava_qa(self, inputs: dict, qa_prompt: str) -> dict:
        """
        Constructs a prompt for question answering using the formatting required by LLaVA.
        """
        
        formatted_texts = "\n".join(inputs["context"]["texts"])
    
        prompt = f"[INST]{'<image>' if inputs['context']['images'] else ' '}\n{qa_prompt}\nUser-provided question: {inputs['question']}\n\nText:\n{formatted_texts}[/INST]"
       
        if inputs['context']['images']:
             # pass always only the first image as llava cannot handle multiple images
            image = inputs['context']['images'][0]
            image = decode_image_to_bytes(image)
            image = Image.open(io.BytesIO(image))
        else:
            image = None  
        
        return {"prompt": prompt, "image": image}
    
        
    def azure_qa(self, data_dict, qa_prompt):
        
        formatted_texts = data_dict["context"]["texts"]
        messages = []

        # Adding image(s) to the messages if present
        if data_dict["context"]["images"]:
            # if multiple images should be added
            # for image in data_dict["context"]["images"]:
            # add only the first image since llava only supports 1 image
            image = data_dict["context"]["images"][0]
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
                "Text:\n"
                f"{formatted_texts}"
            ),
        }
        messages.append(text_message)
        return [HumanMessage(content=messages)]
