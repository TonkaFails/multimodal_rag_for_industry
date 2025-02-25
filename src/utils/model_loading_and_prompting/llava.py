import io
from PIL import Image
import pandas as pd
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from typing import Tuple
from rag_env import INPUT_DATA

def format_prompt_with_image(prompt: str) -> str:
    return f"[INST] <image>\n{prompt} [/INST]"

def get_qa_prompt(model_id: str, system_prompt: str, question: str, context: str, image: Image = None) -> str:
    if "vicuna" in model_id:
        prompt = (
            f"USER:{'<image>' if image else ' '}\n"
            f"{system_prompt}\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"ASSISTANT:"
        )
    else:  # mistral
        prompt = (
            f"[INST]{'<image>' if image else ' '}\n"
            f"{system_prompt}\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"[/INST]"
        )
    return prompt

def get_dataset_generation_prompt(model_id: str, system_prompt: str, context: str, image: Image = None):
    if "vicuna" in model_id:
        prompt = (
            f"USER:{'<image>' if image else ' '}\n"
            f"{system_prompt}\n{context}\n\n"
            f"ASSISTANT:"
        )
    else:  # mistral
        prompt = (
            f"[INST]{'<image>' if image else ' '}\n"
            f"{system_prompt}\n{context}\n\n"
            f"[/INST]"
        )
    return prompt

def format_output(raw_output, processor: LlavaNextProcessor, prompt: str) -> str:
    out = processor.decode(raw_output[0], skip_special_tokens=True)
    out_prompt = prompt.replace("<image>", " ").strip()
    formatted_output = out.replace(out_prompt, "").strip()
    return formatted_output

def get_prompt(task: str, model_id: str, system_prompt: str, text: str, image: Image, question: str) -> str:
    if task == "qa":
        prompt = get_qa_prompt(model_id, system_prompt, question, text, image)
    else:
        prompt = get_dataset_generation_prompt(model_id, system_prompt, text, image)
    return prompt

def llava_call(prompt: str,
               model: LlavaNextForConditionalGeneration,
               processor: LlavaNextProcessor,
               device: str,
               image: Image = None) -> str:
    # Process the prompt and image; then move tensors to the target device.
    inputs = processor(prompt, image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    raw_output = model.generate(**inputs, max_new_tokens=300)
    formatted_output = format_output(raw_output, processor, prompt)
    return formatted_output

def load_llava_model(model_id: str, device: str) -> Tuple[LlavaNextForConditionalGeneration, LlavaNextProcessor]:
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    return model, processor

if __name__ == "__main__":
    # Choose the device: MPS if available (Apple Silicon), then CUDA, else CPU.
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    index = 56
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    model, processor = load_llava_model(model_id, device)
    model.eval()

    df = pd.read_parquet(INPUT_DATA)

    qa_system_prompt = (
        "You are an AI assistant that answers questions from the industrial domain based on a given context. "
        "Use the information from the context to answer the question. \nContext:\n"
    )
    qa_system_prompt_img = (
        "You are an AI assistant that answers questions from the industrial domain based on a given context as text and image. "
        "Use both the information from text and image to answer the question. \nContext:\n"
    )

    question = "What are the possible positions of the manual operator and what colors are associated with each position?"
    context = df["text"][index]
    img_bytes = df["image_bytes"][index]
    image = Image.open(io.BytesIO(img_bytes))
    
    img_prompt = get_qa_prompt(model_id, qa_system_prompt_img, question, context, image)
    no_img_prompt = get_qa_prompt(model_id, qa_system_prompt, question, context)

    # Get response from image and text context.
    print("============== Answer with image:")
    llava_response_img = llava_call(img_prompt, model, processor, device, image)
    print(llava_response_img)
    
    # Get response using text only.
    print("============== Answer without image:")
    llava_response_no_img = llava_call(no_img_prompt, model, processor, device)
    print(llava_response_no_img)
