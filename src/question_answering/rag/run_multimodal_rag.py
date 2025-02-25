import os  
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from single_vector_store.rag_pipeline_clip import MultimodalRAGPipelineClip
from separate_vector_stores.dual_rag_pipeline_clip import DualMultimodalRAGPipelineClip
from single_vector_store.rag_pipeline_summaries import MultimodalRAGPipelineSummaries
from separate_vector_stores.dual_rag_pipeline_summaries import DualMultimodalRAGPipelineSummaries
from rag_env import *



def write_to_df(df, user_query, reference_answer, generated_answer, context, image, output_file):
    df.loc[len(df)] = [user_query, reference_answer, generated_answer, context, image]
    df.to_json(output_file, orient="records", indent=2)


def process_dataframe(input_df, pipeline, output_file, output_df=None):
    if not output_df:
        columns = ['user_query', 'reference_answer', 'generated_answer', 'context', 'image']
        output_df= pd.DataFrame(columns=columns)
    for index, row in input_df.iterrows():
        print(f"Processing query no. {index+1}...")
        user_query = input_df["question"][index]
        print("USER QUERY:\n", user_query)
        reference_answer = input_df["reference_answer"][index]
        print("REFERENCE ANSWER:", reference_answer)
        generated_answer = pipeline.answer_question(user_query)
        print("GENERATED ANSWER:\n", generated_answer)
        relevant_images = pipeline.rag_chain.retrieved_images
        relevant_texts = pipeline.rag_chain.retrieved_texts
        print("Retrieved images:", len(relevant_images), ", Retrieved texts:", len(relevant_texts))
        context = "\n".join(relevant_texts) if len(relevant_texts) > 0 else []
        image = relevant_images[0] if len(relevant_images) > 0 else []
        write_to_df(output_df, user_query, reference_answer, generated_answer, context, image, output_file)
    return output_df



def run_pipeline_with_clip_single(model, input_df, vectorstore_path, images_dir, reference_qa, output_dir):
    pipeline = MultimodalRAGPipelineClip(model_type=model, store_path=vectorstore_path)
    texts_df = pipeline.load_data(input_df)
    pipeline.index_data(texts_df=texts_df, images_dir=images_dir)

    df = pd.read_excel(reference_qa)
 
    output_file = os.path.join(output_dir, f"rag_output_{model}_multimodal_clip_single.json")
    output_df = process_dataframe(df, pipeline, output_file)
    return output_df



def run_pipeline_with_clip_dual(model, input_df, vectorstore_path, images_dir, reference_qa, output_dir, text_embedding_model):
    pipeline = DualMultimodalRAGPipelineClip(model_type=model,
                                             store_path=vectorstore_path,
                                             text_embedding_model=text_embedding_model)
    texts_df = pipeline.load_data(input_df)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()
    
    pipeline.index_data(images_dir=images_dir, texts=texts, text_summaries=texts, text_filenames=texts_filenames)

    df = pd.read_excel(reference_qa)
 
    output_file = os.path.join(output_dir, f"rag_output_{model}_multimodal_clip_dual.json")
    output_df = process_dataframe(df, pipeline, output_file)
    return output_df

    
    
def run_pipeline_with_summaries_single(qa_model, embedding_model, vectorstore_path, input_df, reference_qa, output_dir, img_summaries_dir):
    summaries_pipeline = MultimodalRAGPipelineSummaries(model_type=qa_model,
                                                        store_path=vectorstore_path,
                                                        embedding_model=embedding_model)
    
    texts_df, images_df = summaries_pipeline.load_data(input_df)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()
    images, image_filenames = images_df[["image_bytes"]]["image_bytes"].tolist(), images_df[["doc_id"]]["doc_id"].tolist()
    img_base64_list, image_summaries = summaries_pipeline.image_summarizer.summarize(images, img_summaries_dir)
    summaries_pipeline.index_data(image_summaries=image_summaries, 
                                  images_base64=img_base64_list, image_filenames=image_filenames, 
                                  texts = texts, text_filenames=texts_filenames)
    
    df = pd.read_excel(reference_qa)
    
    output_file = os.path.join(output_dir, f"rag_output_{qa_model}_multimodal_summaries_single.json")
    output_df = process_dataframe(df, summaries_pipeline, output_file)
    return output_df  



def run_pipeline_with_summaries_dual(qa_model, embedding_model, vectorstore_path, input_df, reference_qa, output_dir, img_summaries_dir):
    summaries_pipeline = DualMultimodalRAGPipelineSummaries(model_type=qa_model,
                                                        store_path=vectorstore_path, 
                                                        embedding_model=embedding_model)
    
    texts_df, images_df = summaries_pipeline.load_data(input_df)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()
    images, image_filenames = images_df[["image_bytes"]]["image_bytes"].tolist(), images_df[["doc_id"]]["doc_id"].tolist()
    img_base64_list, image_summaries = summaries_pipeline.image_summarizer.summarize(images, img_summaries_dir)
    summaries_pipeline.index_data(image_summaries=image_summaries, 
                                  images_base64=img_base64_list, image_filenames=image_filenames, 
                                  texts = texts, text_filenames=texts_filenames)
    
    df = pd.read_excel(reference_qa)
    
    output_file = os.path.join(output_dir, f"rag_output_{qa_model}_multimodal_summaries_dual.json")
    output_df = process_dataframe(df, summaries_pipeline, output_file)
    return output_df  


  
if __name__ == "__main__":  
    # uncomment one of the following options to run multimodal RAG either with CLIP embedings or with image summaries
    # and either with a single vector store for both modalities or a dedicated one for each modality.
    rag_results_clip_single = run_pipeline_with_clip_single(model=MODEL_TYPE, input_df=INPUT_DATA,
                                                            vectorstore_path=VECTORSTORE_PATH_IMAGE_ONLY,
                                                            images_dir=IMAGES_DIR, reference_qa=REFERENCE_QA,
                                                            output_dir=RAG_OUTPUT_DIR)
    '''
    rag_results_clip_dual = run_pipeline_with_clip_dual(model=MODEL_TYPE, input_df=INPUT_DATA,
                                                        vectorstore_path=VECTORSTORE_PATH_IMAGE_ONLY,
                                                        images_dir=IMAGES_DIR, reference_qa=REFERENCE_QA, 
                                                        output_dir=RAG_OUTPUT_DIR, text_embedding_model=EMBEDDING_MODEL_TYPE)
    
    rag_results_summaries_single = run_pipeline_with_summaries_single(qa_model=MODEL_TYPE,
                                                                      vectorstore_path=VECTORSTORE_PATH_IMAGE_ONLY,
                                                                      embedding_model=EMBEDDING_MODEL_TYPE, input_df=INPUT_DATA,
                                                                      reference_qa=REFERENCE_QA, output_dir=RAG_OUTPUT_DIR,
                                                                      img_summaries_dir=IMG_SUMMARIES_CACHE_DIR)
    
    rag_results_summaries_single = run_pipeline_with_summaries_dual(qa_model=MODEL_TYPE, vectorstore_path=VECTORSTORE_PATH_IMAGE_ONLY,
                                                                    embedding_model=EMBEDDING_MODEL_TYPE, input_df=INPUT_DATA,
                                                                    reference_qa=REFERENCE_QA, output_dir=RAG_OUTPUT_DIR,
                                                                    img_summaries_dir=IMG_SUMMARIES_CACHE_DIR)
    '''