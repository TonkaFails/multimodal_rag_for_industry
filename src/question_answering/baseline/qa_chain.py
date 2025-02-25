from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from utils.model_loading_and_prompting.llava import load_llava_model, llava_call
from utils.azure_config import get_azure_config


class QAChain:
    def __init__(self, model_type):
        """
        Question Answering Chain: Baseline without RAG (direct model prompting without additional context)
        The steps are as follows:
        1. Create a QA prompt using the question and an instruction to answer the question.
        2. Call the LLM and prompt it with the created prompt
        3. Obtain the generated answer from the model
        
        :param model: The type of model used for answer generation.
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
        
        # here we don't have a retriever since we prompt the model directly, without any additional context
        self.chain = (self.prompt_func | RunnableLambda(self.call_model) | StrOutputParser())
        
        
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

        
    def call_llava(self, inputs):
        
        prompt = inputs['prompt']
        ans = llava_call(prompt, self.model, self.tokenizer, device="mps")
        return ans


    def prompt_func(self, data_dict):
        
        qa_prompt = """You are an expert AI assistant that answers questions about manuals from the industrial domain.\n"""
        
        if type(self.model) == AzureChatOpenAI:
            prompt = self.azure_qa(data_dict, qa_prompt)
        else:
            prompt = self.llava_qa(data_dict, qa_prompt)
            
        return prompt
    
    
    def llava_qa(self, inputs, qa_prompt):
        """
        Constructs a prompt for question answering using the formatting required by LLaVA.
        """
    
        prompt = f"[INST]\n{qa_prompt}\nUser-provided question: {inputs['question']}\n\n[/INST]"
        image = None
        
        return {"prompt": prompt, "image": image}

        
        
    def azure_qa(self, data_dict, qa_prompt):
        """
        Constructs a prompt for question answering using the formatting required by AzureOpenAI.
        """
        
        messages = []

        # Adding the text for answer generation
        text_message = {
            "type": "text",
            "text": (
                f"{qa_prompt}\n"
                f"User-provided question: {data_dict['question']}\n\n"
            ),
        }
        messages.append(text_message)
        print("Calling model...")
        return [HumanMessage(content=messages)]