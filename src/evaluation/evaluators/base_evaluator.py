import os
from abc import abstractmethod
from langchain.chains.transform import TransformChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from langchain.output_parsers import BooleanOutputParser, OutputFixingParser
from transformers import LlavaNextForConditionalGeneration
from evaluation.evaluators.evaluator_interface import EvaluatorInterface
from utils.azure_config import get_azure_config
from utils.model_loading_and_prompting.llava import llava_call


class EvaluationResult(BaseModel):
    """The result of an evaluation for a given metric"""

    grade: str = Field(description="the grade after evaluating the metric (YES or NO)")
    reason: str = Field(description="The reasoning behind the grading decision")


class BaseEvaluator(EvaluatorInterface):
    """  
    A base class for an LLM evaluator.
  
    Attributes: 
        model (str): The model to be used for evaluation.
        tokenizer (LlavaNextProcessor or PreTrainedTokenizerFast): Tokenizer used for tokenization. Can be None.
        model_type (AzureChatOpenAI or LlavaNextForConditionalGeneration): Type of the model to use for evaluation.
        json_parser (JsonOutputParser): Parser used to parse evaluation results to a json object.
        boolean_parser (BooleanOutputParser): Parser used to parse the assigned grade to a boolean value.
        check_grade_chain (TransformChain): Applies the transformation from the LLM output for the grade to a boolean value.
        fix_format_parser (OutputFixingParser): Parser used to fix misformatted json output of an LLM.
    """
    def __init__(self, model, tokenizer=None, **kwargs):
        """  
        Initializes the BaseEvaluator object.
  
        :param model: The model to be used for evaluation.
        :param tokenizer: The tokenizer to be used for tokenization. Can be None.
        
        Keyword Args:
            user_query (str): The user query
            generated_answer (str): The answer produced by the model
            reference_answer (str): The ground truth answer
            context (str): The texts retrieved by the retrieval system
            image (str): The image retrieved by the retrieval system
        """
        self.model = model
        self.model_type = type(self.model)
        self.json_parser = JsonOutputParser(pydantic_object=EvaluationResult)
        self.boolean_parser = BooleanOutputParser()
        self.kwargs = kwargs
        self.check_grade_chain = TransformChain(
            input_variables=["grade", "reason"],
            output_variables=["grade", "reason"],
            transform=self.get_numeric_score
        )
        
        # if a tokenizer is specified, the Evaluator is initialized for evaluation with LLaVA, otherwise GPT4v
        if tokenizer:
            self.tokenizer = tokenizer
            self.config = get_azure_config()
            gpt4v_config = self.config['gpt4']
            fixing_llm = AzureChatOpenAI(
                openai_api_version=gpt4v_config["openai_api_version"],
                azure_endpoint=gpt4v_config["openai_endpoint"],
                azure_deployment=gpt4v_config["deployment_name"],
                model=gpt4v_config["model_version"],
                api_key=os.environ.get("GPT4V_API_KEY"),
                max_tokens=500
            )
        
            self.fix_format_parser = OutputFixingParser.from_llm(parser=self.json_parser, llm=fixing_llm)
        
        else:
            self.tokenizer = None
            
            
    def call_llava(self, inputs: dict) -> str:
        
        prompt = inputs['prompt']
        image = inputs.get('image', None)
        ans = llava_call(prompt, self.model, self.tokenizer, device="mps", image=image)
        return ans
        

    def get_numeric_score(self, inputs: str) -> dict:
        """
        Checks that the obtained grade (YES or NO) can be parsed to a boolean and sets the grade to its integer value (0 ur 1)
        """
        inputs["grade"] = int(self.boolean_parser.parse(inputs["grade"]))
        return inputs

    def run_evaluation(self) -> dict:
        """  
        Performs evaluation for one output of a RAG system.
        Creates an evaluation chain that constructs the prompt, calls the model, fixes possible 
        json formatting errors and checks the validity of the assigned grade.

        :return: A json object with a grade (0 or 1) and a reason for the grading as string.
        """ 
        if self.tokenizer:
            chain = RunnableLambda(self.get_prompt) | RunnableLambda(self.call_llava) | self.fix_format_parser | self.check_grade_chain
        else:
            # GPT4v chain
            chain = RunnableLambda(self.get_prompt) | self.model | self.json_parser | self.check_grade_chain
            # chain = RunnableLambda(self.get_prompt) | self.model | self.fix_format_parser | self.check_grade_chain
        result = chain.invoke(self.kwargs)

        return result

    @abstractmethod
    def get_prompt(self, inputs: dict):
        """
        Construct the prompt for evaluation based on a dictionary containing required input arguments.
        """
        pass
