import os
import gc
import re
import json
import torch
import pickle
import string
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax
from collections import OrderedDict
from vllm import LLM, SamplingParams
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


import warnings
warnings.filterwarnings("ignore")

os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'

# ---------------------------------------------------------------------------
# TEXT EMBEDDING
# ---------------------------------------------------------------------------
class TextEmbedder:
    """
    A class for embedding large text documents using a specified embedding model.

    Attributes:
        embedding_model: The embedding model used for generating text embeddings.
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def chunk_text(self, text, max_tokens):
        """
        Splits the input text into smaller chunks, ensuring each chunk 
        does not exceed the specified token limit.

        Args:
            text (str): The input text to be chunked.
            max_tokens (int): The maximum number of tokens per chunk.

        Returns:
            list: A list of text chunks.
        """
        chunks = []
        words = text.split()
        for i in range(0, len(words), max_tokens):
            chunk = ' '.join(words[i:i+max_tokens])
            chunks.append(chunk)
        return chunks

    def average_embeddings(self, embeddings):
        """
        Computes the average of a list of embeddings.

        Args:
            embeddings (list): A list of numerical embeddings.

        Returns:
            The averaged embedding vector, or None if the list is empty.
        """
        if embeddings:
            return sum(embeddings) / len(embeddings)
        else:
            return None

    def generate_embeddings(self, sample, max_tokens=8000):
        """
        Generates an embedding for a large text sample by:
        1. Splitting it into smaller chunks.
        2. Generating embeddings for each chunk.
        3. Averaging the embeddings to obtain a final representation.

        Args:
            sample (str): The input text to be embedded.
            max_tokens (int, optional): The max token size for each chunk. Defaults to 8000.

        Returns:
            The averaged embedding vector representing the entire text.
        """
        chunks = self.chunk_text(sample, max_tokens)
        chunk_embeddings = []
        for chunk in chunks:
            chunk_embedding = self.embedding_model.encode(chunk)
            chunk_embeddings.append(chunk_embedding)
        return self.average_embeddings(chunk_embeddings)

    def create_vector_db(self, examples):
        """
        Creates a vector database by generating embeddings for a list of text examples.

        Args:
            examples (list): A list of text samples to be embedded.

        Returns:
            list: A list of embedding vectors representing the text examples.
        """
        db = [self.generate_embeddings(example) for example in examples]
        return db

# ---------------------------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------------------------
class ModelEvaluator:
    """
    A class for evaluating a model using a nearest-neighbor approach for text-based classification.

    Attributes:
        knn (NearestNeighbors): A k-Nearest Neighbors model for similarity-based retrieval.
        examples (list): A list of example texts used for evaluation.
        test_samples (list): A list of test samples to be evaluated.
        llm (object): The language model used for text processing and inference.
        sampling_params (dict): Parameters for controlling the sampling behavior of the LLM.
        results (list): Stores evaluation results.
        auc_data (list): Stores Area Under Curve (AUC) data for performance analysis.
        pos_token_id (int): Token ID representing the positive class.
        neg_token_id (int): Token ID representing the negative class.
        labels (list): Ground-truth labels for the test samples.
        zeroshot (bool): Indicates whether zero-shot classification is used.
        summary (bool): Indicates whether summary-based evaluation is performed.
        embedder (object): The embedding model used for vector representation.
        cancer_type (str): The specific cancer type being analyzed.
    """
    def __init__(self, vector_db, examples, test_samples, llm, sampling_params, labels, zeroshot, summary, embedder, cancer_type):
        self.knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.knn.fit(vector_db)
        self.examples = examples
        self.test_samples = test_samples
        self.llm = llm
        self.sampling_params = sampling_params
        self.results = []
        self.auc_data = []
        self.pos_token_id = 27592
        self.neg_token_id = 85165
        self.labels = labels
        self.zeroshot = zeroshot
        self.summary = summary
        self.embedder = embedder
        self.cancer_type = cancer_type

    def create_prompt_shots(self, sample):
        """
        Generates a prompt for an oncologist to predict a cancer patient's survival outlook.

        Args:
            sample (int): The index of the example to use for prompt generation.

        Returns:
            str: A formatted text prompt for model evaluation.
        """
        cutoff = '14-month' if self.cancer_type == 'glioma' else '5-year'
        cancer = 'glioma' if self.cancer_type == 'glioma' else 'breast cancer'

        if self.summary:
            ans = 'POSITIVE' if self.labels[sample] == 1 else 'NEGATIVE'
            text_prompt = f'''<|start_header_id|>user<|end_header_id|>You are an oncologist specializing in {cancer} at a major cancer hospital. Your task is to predict the {cutoff}s survival outlook for a {cancer} patient
                based on the following clinical summary, which represents the patient's status at 0.5 years (6 months) post-diagnosis.\n\n''' 
            text_prompt += '''Here is a clinical summary example: \n'''
            text_prompt += str(self.examples[sample])
            text_prompt += f"\nFor this example, the correct answer is {ans}.\n"
        else:
            ans = 'POSITIVE' if self.labels[sample] == 1 else 'NEGATIVE'
            text_prompt = f'''<|start_header_id|>user<|end_header_id|>You are an oncologist specializing in {cancer} at a major cancer hospital. Your task is to predict the {cutoff}s survival outlook for a {cancer} patient
                based on the following clinical notes, which represents the patient's status at 0.5 years (6 months) post-diagnosis.\n\n''' 
            text_prompt += '''Here is the clinical notes example: \n'''
            text_prompt += str(self.examples[sample])
            text_prompt += f"\nFor this example, the correct answer is {ans}.\n"

        return text_prompt

    def create_prompt(self, sample, context=True):
        """
        Generates a structured prompt for an oncologist to classify a cancer patient's survival outlook.

        Args:
            sample (int): The index of the test sample to be used in the prompt.
            context (bool): Determines whether contextual information (previous examples) is included.

        Returns:
            str: A formatted text prompt for model evaluation.
        """
        cutoff = '14 month' if self.cancer_type == 'glioma' else '5 year'
        cancer = 'glioma' if self.cancer_type == 'glioma' else 'breast cancer'
        if self.zeroshot:
            if self.summary:
                text_prompt = f'''
                You are an oncologist at a major cancer hospital, tasked with predicting
                outcomes for patients.
                I am going to provide you with a clinical note summary for a {cancer} patient at 6 months. Here is the summary:
                '''
                text_prompt += str(self.test_samples[sample])
                text_prompt += f'''\nBased on your understanding of the clinical note summary for a {cancer} patient at 6 months, classify the {cutoff}
                    survival outlook for this patient. Please respond with either POSITIVE or NEGATIVE answer only.'''
            else:
                text_prompt = f'''
                You are an oncologist at a major cancer hospital, tasked with predicting
                outcomes for patients.
                I am going to provide you with clinical notes for a {cancer} patient at 6 months. Here is the summary:
                '''
                text_prompt += str(self.test_samples[sample])
                text_prompt += f'''\nBased on your understanding of the clinical notes for a {cancer} patient at 6 months, classify the {cutoff}
                    survival outlook for this patient. Please respond with either POSITIVE or NEGATIVE answer only.'''
            text_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>The correct answer is "
        if self.summary:
            if self.zeroshot:
                text_prompt = f'''<|start_header_id|>user<|end_header_id|>You are an oncologist at a major cancer hospital, tasked with predicting outcomes for patients.
                    I am going to provide you with a clinical note summary for a {cancer} patient at 0.5 years. Here is the summary: '''
                text_prompt += str(self.test_samples[sample])
                text_prompt += f'''\nBased on your understanding of the clinical summary for a {cancer} patient at 0.5 years, classify the {cutoff} survival outlook for this patient.
                    Please respond with either POSITIVE or NEGATIVE'''
                text_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>The correct answer is "
            else:
                text_prompt = f'''<|start_header_id|>user<|end_header_id|>You are an oncologist specializing in {cancer} at a major cancer hospital. Your task is to predict the {cutoff} survival outlook for a {cancer} patient
                    based on a clinical summary, which represents the patient's status at 0.5 years (6 months) post-diagnosis. I am going to provide you previous patient examples to help guide the decision making.\n'''
                text_prompt += "Here is the summary:\n"
                text_prompt += str(self.test_samples[sample])
                text_prompt += f'''\nPlease analyze this clinical summary carefully, considering factors such as tumor progression, treatment response, symptoms, and any relevant biomarkers. Based on this analysis as well as the knowledge from the previous examples, classify the patient's {cutoff} survival outlook. Respond with either 'POSITIVE' (if the patient is likely to survive beyond {cutoff}s) or 'NEGATIVE' (if the patient is unlikely to survive beyond {cutoff}s).'''
                text_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>The correct answer is "
        else:
            if self.zeroshot:
                text_prompt = f'''<|start_header_id|>user<|end_header_id|>You are an oncologist at a major cancer hospital, tasked with predicting outcomes for patients.
                    I am going to provide you with a clinical notes for a {cancer} patient at 0.5 years. Here is the summary: '''
                text_prompt += str(self.test_samples[sample])
                text_prompt += f'''\nBased on your understanding of the clinical notes for a {cancer} patient at 0.5 years, classify the {cutoff} survival outlook for this patient.
                    Please respond with either POSITIVE or NEGATIVE'''
                text_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>The correct answer is "
            else:
                text_prompt = f'''<|start_header_id|>user<|end_header_id|>You are an oncologist specializing in {cancer} at a major cancer hospital. Your task is to predict the {cutoff} survival outlook for a {cancer} patient
                    based on the clinical notes, which represents the patient's status at 0.5 years (6 months) post-diagnosis. I am going to provide you previous patient examples to help guide the decision making.\n'''
                text_prompt += "Here are the clinical notes:\n"
                text_prompt += str(self.test_samples[sample])
                text_prompt += f'''\nPlease analyze the clinical notes carefully, considering factors such as tumor progression, treatment response, symptoms, and any relevant biomarkers. Based on this analysis as well as the knowledge from the previous examples, classify the patient's {cutoff} survival outlook. Respond with either 'POSITIVE' (if the patient is likely to survive beyond {cutoff}s) or 'NEGATIVE' (if the patient is unlikely to survive beyond {cutoff}s).'''
                text_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>ANSWER: "

        return text_prompt

    def generate_output(self):
        """
        Runs the LLM to generate predictions and extracts the binary output.

        - If dynamic prompting (few-shot learning) is enabled, retrieves the most similar examples
        from the vector database and includes them as context in the prompt.
        - If zero-shot learning is enabled, constructs a prompt without additional examples.
        - Executes the LLM model, captures results, and calculates log probabilities for AUC evaluation.

        Returns:
            list: A list of model-generated results.
        """
        for i, note in enumerate(self.test_samples):
            if not self.zeroshot:
                embedding = self.embedder.generate_embeddings(note)
                top = self.knn.kneighbors(embedding.reshape(1, -1))
                sample1 = top[1][0][1]
                sample0 = top[1][0][0]

                text_prompt = self.create_prompt_shots(sample0)
                text_prompt += self.create_prompt_shots(sample1)
                if self.summary:
                    sample4 = top[1][0][4]
                    sample3 = top[1][0][3]
                    sample2 = top[1][0][2]
                    text_prompt += self.create_prompt_shots(sample2)
                    text_prompt += self.create_prompt_shots(sample3)
                    text_prompt += self.create_prompt_shots(sample4)

                text_prompt += self.create_prompt(i, context=False)
            else:
                text_prompt = self.create_prompt(i, context=False)

            torch.cuda.empty_cache()
            output = self.llm.generate(text_prompt, self.sampling_params)
            res = output[0].outputs[0].text
            self.results.append(res)

            correct_answer_token = output[0].outputs[0].token_ids[0]
            wrong_answer_tokens_func = lambda correct_answer_tokens: 27592 if correct_answer_tokens == 85165 else 85165
            wrong_answer_token = wrong_answer_tokens_func(correct_answer_token)

            all_logprobs = output[0].outputs[0].logprobs
            for logprob_dict in all_logprobs:
                if wrong_answer_token in logprob_dict:
                    wrong_answer_logit = logprob_dict[wrong_answer_token].logprob
                if correct_answer_token in logprob_dict:
                    correct_answer_logit = logprob_dict[correct_answer_token].logprob

            new_entry = {'correct_logit': correct_answer_logit, 'wrong_logit': wrong_answer_logit}
            self.auc_data.append(new_entry)
        return self.results

    def compute_metrics(self, y_test):
        """
        Computes and retrieves classification metrics for the generated model responses.

        Args:
            y_test (list or array): The true labels for the test samples.

        Returns:
            tuple: A tuple containing the classification metrics (accuracy, precision, recall, F1-score, AUC).
        """
        preds = []
        df_test = pd.DataFrame()
        df_test['label'] = y_test
        for num, i in enumerate(self.results):
            if 'POS' in i:
                preds.append(1)
            elif 'NEG' in i:
                preds.append(0)
            else:
                preds.append(2)
                print(num)

        df_test['prediction'] = preds
        df_new = df_test[df_test['prediction'] != 2]
        df_new.reset_index(inplace=True)

        accuracy = accuracy_score(df_new['label'], df_new['prediction'])
        prec = precision_score(df_new['label'], df_new['prediction'])
        rec = recall_score(df_new['label'], df_new['prediction'])
        f1 = f1_score(df_new['label'], df_new['prediction'])

        lbls = df_test['label']
        preds_ = df_test['prediction']
        for data, label, pred in zip(self.auc_data, lbls, preds_):
            data["label"] = label
            data['pred'] = pred

        true_labels = []
        predicted_probs = []
        for data in self.auc_data:
            if data['pred'] == 0:
                data['logit_0'] = data['correct_logit']
                data['logit_1'] = data['wrong_logit']
            else:
                data['logit_0'] = data['wrong_logit']
                data['logit_1'] = data['correct_logit']

            logit_0 = data['logit_0']
            logit_1 = data['logit_1']
            truth = data['label']

            logits = np.array([logit_0, logit_1])
            probs_ = softmax_func(logits)

            if truth is not None:
                true_labels.append(truth)
                predicted_probs.append(probs_[1])

        auc = roc_auc_score(true_labels, predicted_probs)

        return accuracy, prec, rec, f1, auc


def softmax_func(logits):
    return softmax(logits)

# ---------------------------------------------------------------------------
# DATA PREPARATION AND PREPROCESSING
# ---------------------------------------------------------------------------
def ingest_data(example_file, test_file, summary):
    """
    Reads CSV files containing clinical notes and labels, processes them, and outputs lists for training and testing data.

    Args:
        example_file (str): Path to the CSV file containing the training data (clinical notes and labels).
        test_file (str): Path to the CSV file containing the test data (clinical notes and labels).
        summary (bool): Flag indicating whether to process summaries (useful for controlling different types of data preprocessing).

    Returns:
        tuple: A tuple containing four elements:
            - examples (list): Preprocessed clinical notes for training.
            - labels (list): Labels corresponding to the training data.
            - test_samples (list): Preprocessed clinical notes for testing.
            - y_test (list): Labels corresponding to the test data.
    """
    df = pd.read_csv(example_file)
    df2 = pd.read_csv(test_file)

    tqdm.pandas()
    df['note'] = df['note'].progress_apply(glioma_preprocess)
    df2['note'] = df2['note'].progress_apply(glioma_preprocess)

    examples = df['note']
    test_samples = df2['note']
    labels = df['label']
    y_test = df2['label']

    return examples, labels, test_samples, y_test


def glioma_preprocess(text):
    """
    Preprocessing function built specifically for the UCSF glioma dataset.
    Takes in a clinical note as a string and outputs a cleaned and preprocessed string.

    The function performs the following preprocessing steps:
    1. Removes duplicated text by splitting and deduplicating.
    2. Strips unnecessary whitespaces and removes extraneous characters.
    3. Removes stars, slashes, and other symbols for deidentification.
    4. Replaces or removes special characters, redundant words, and specific phrases.
    5. Normalizes the text by stripping redundant characters and words, ensuring it is clean and consistent.

    Args:
        text (str): A string representing the raw clinical note.

    Returns:
        str: A preprocessed version of the input string.
    """
    split_on_b = re.split(r"b'|B'|b\"|B\"", text)
    # 1. Remove duplicated text
    duplicates_removed = list(OrderedDict.fromkeys(split_on_b))
    # 2. Strip
    strip = [item.strip().rstrip("'") for item in duplicates_removed]
    # 3. Replace Stars and slashes - deidentification
    replace_stars = [re.sub(r'[*\\/]+', '', item) for item in strip]
    # 4. Replace special characters
    replace_special = [re.sub('--|__|==', '', item) for item in replace_stars]
    replace_special = [re.sub(r'\([^a-zA-Z0-9]*\)', '', item) for item in replace_special]
    replace_special = [re.sub(r' +', ' ', item) for item in replace_special]
    replace_special = [re.sub(r'([.,\s-]){4,}', '', item) for item in replace_special]
    # 4. Remove redundant words and characters
    redundancy = [re.sub(r'DOB:|REFERRING PHYSICIAN:|Date:|name:|date of birth:|TEL:|FAX:|MRN:|DATE OF PROCEDURE:|NA|Referring No|No address on file|DATE OF SERVICE:|Fellow:|OPERATIVE REPORTDATE OF OPERATION:|Patient Rec.#|DIRECTOR|DATE OF OPERATION:|PATIENTRECORD #:|Admission date:|discharge date:|Patient Rec.#:', '', 
                        item, flags=re.IGNORECASE) for item in replace_special]
    redundancy = [item.strip() for item in redundancy]
    redundancy = [re.sub(r'([.,\s-]){4,}', '', item) for item in redundancy]
    redundancy = [re.sub(r'^[^\w]*', '', item) for item in redundancy]
    redundancy = [re.sub(r'DOB:|REFERRING PHYSICIAN:|Date:|name:|date of birth:|TEL:|FAX:|MRN:|DATE OF PROCEDURE:|NA|Referring No|No address on file|DATE OF SERVICE:|Fellow:|OPERATIVE REPORTDATE OF OPERATION:|Patient Rec.#|DIRECTOR|DATE OF OPERATION:|PATIENTRECORD #:|Admission date:|discharge date:|Patient Rec.#:', '', 
                        item, flags=re.IGNORECASE) for item in redundancy]
    redundancy = [re.sub(re.escape("This laboratory is certified under the Clinical Laboratory Improvement Amendments of 1988 (\"CLIA\") as qualified to perform high-complexity clinical testing"), " ", item, flags=re.IGNORECASE)
              for item in redundancy]
    redundancy = [item.strip() for item in redundancy]
    redundancy = [re.sub(r'([.,\s-]){4,}', '', item) for item in redundancy]
    redundancy = [re.sub(r'^[^\w]*', '', item) for item in redundancy]
    redundancy = [item.replace('Dr.', 'Doctor') for item in redundancy]
    preprocessed_data = " ".join(redundancy)
    preprocessed_data = preprocessed_data.strip()
    return preprocessed_data


def main(exp_type, fold_number, large, num_gpus, zeroshot, example_file, test_file, summary, vec_db, cancer_type):
    print("Starting script execution...")
    print("")

    print("Processing data...")
    examples, labels, test_samples, y_test = ingest_data(example_file, test_file, summary)
    print(f"Loaded {len(examples)} examples and {len(test_samples)} test samples.")
    print("")

    print("Loading embedding model...")
    embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    print("Loaded embedding model successfully")
    print("")

    print("Creating vector database...")
    text_embedder = TextEmbedder(embedding_model)
    if vec_db:
        with open(vec_db, 'rb') as f:
            vector_db = pickle.load(f)
    else:
        vector_db = text_embedder.create_vector_db(examples)
    del embedding_model
    torch.cuda.empty_cache()
    print("Vector database created successfully.")
    print("")

    print("Initializing LLM...")
    if large:
        model_name = "gradientai/Llama-3-70B-Instruct-Gradient-262k"
        llm = LLM(model=model_name, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.95)
    else:
        model_name = "gradientai/Llama-3-8B-Instruct-262k"
        llm = LLM(model=model_name, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.8) # default is to load smaller model
    print(f"Loaded LLM model: {model_name}")
    print("")

    sampling_params = SamplingParams(temperature=0, max_tokens=2, logprobs=10) # set temperature to 0 for repeatable results
    model_evaluator = ModelEvaluator(vector_db, examples, test_samples, llm, sampling_params, labels, zeroshot, summary, text_embedder, cancer_type)

    print("Starting inference...")
    model_evaluator.generate_output()
    print("Inference complete.")

    print("Computing evaluation metrics...")
    acc, prec, rec, f1, auc = model_evaluator.compute_metrics(y_test)
    print("Evaluation complete. Metrics:")
    print(f"  - Accuracy: {acc:.4f}")
    print(f"  - Precision: {prec:.4f}")
    print(f"  - Recall: {rec:.4f}")
    print(f"  - F1 Score: {f1:.4f}")
    print(f"  - AUC: {auc:.4f}")
    print("")

    file_path = "metrics.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            metrics_data = json.load(f)
    else:
        metrics_data = {}

    if exp_type not in metrics_data:
        metrics_data[exp_type] = {}

    # Store metrics under the fold_number key
    metrics_data[exp_type][fold_number] = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "AUC": round(auc, 4)
    }

    with open(file_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Metrics saved to {file_path}")
    print("Script execution finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', type=str, required=True, help='zero_shot_summary/zero_shot_fn/dynamic_summary/dynamic_fn')
    parser.add_argument('--fold_number', type=int, default=0, help='CV Folds')
    parser.add_argument('--large', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use large model if set to True, else use small model')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use for tensor parallelism')
    parser.add_argument('--zero_shot', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use zero-shot prompts, default is dynamic prompting')
    parser.add_argument('--examples', required=True, help="CSV file containing a label column and either note or summary column")
    parser.add_argument('--test_data', required=True, help="CSV file containing a label column and either note or summary column")
    parser.add_argument('--summary', type=lambda x: (str(x).lower() == 'true'), default=True, help="Using summarized note text.")
    parser.add_argument('--vector_db', default=None, help="Pickle file containing list of patient level embeddings")
    parser.add_argument('--cancer_type', type=str, required=True, help="String representing cancer type, current options are ['glioma', 'breast cancer']")

    args = parser.parse_args()
    main(args.exp_type, args.fold_number, args.large, args.num_gpus, args.zero_shot, args.examples, args.test_data, args.summary, args.vector_db, args.cancer_type)