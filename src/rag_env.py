"""
Configuration for the RAG pipeline. Replace the paths with your own actual paths.
The data used here is small sample data to show the expected format, not the real data.
This toy data will not lead to good results.
"""

# model to use for answer synthesis and image summarization. ('gpt4_vision', otherwise LLaVA will be used)
MODEL_TYPE  = 'llava'
# text embedding model, set to 'openai' to use text-embeding-3-small, otherwise bge-m3 will be used
EMBEDDING_MODEL_TYPE = "openai"

# excel file containing questions and reference answers
REFERENCE_QA = r"../../sample_data/reference_qa.xlsx"

# directory containing the pdf files from which to extract texts and images
MANUALS_DIR = "/Users/anton/bachelor/test-anweisungen"

# parquet file where extracted texts and image bytes are stored
INPUT_DATA = r'../../sample_data/extracted_texts_and_imgs.parquet'

# directory where extracted images are stored
IMAGES_DIR = r"../../sample_data/images"

# directories containing csv files with text summaries or image summaries
IMG_SUMMARIES_CACHE_DIR = r"../../sample_data/image_summaries"
TEXT_SUMMARIES_CACHE_DIR = r"../../sample_data/text_summaries"

# directories where vector stores are saved
VECTORSTORE_PATH_CLIP_SINGLE = r"../../sample_data/vec_and_doc_stores/clip"
VECTORSTORE_PATH_CLIP_SEPARATE = r"../../sample_data/vec_and_doc_stores/clip_dual"
VECTORSTORE_PATH_SUMMARIES_SINGLE = r"../../sample_data/vec_and_doc_stores/image_summaries"
VECTORSTORE_PATH_SUMMARIES_SEPARATE = r"../../sample_data/vec_and_doc_stores/image_summaries_dual"
VECTORSTORE_PATH_IMAGE_ONLY = r".../../sample_data/vec_and_doc_stores/image_only"
VECTORSTORE_PATH_TEXT_ONLY = r".../../sample_data/vec_and_doc_stores/text_only"

# directory where the output of a RAG pipeline is stored
RAG_OUTPUT_DIR = r".../../sample_data/rag_outputs"

# directory where the evaluation results for a RAG pipeline are stored
EVAL_RESULTS_PATH = r"../../sample_data/rag_evaluation_results"
