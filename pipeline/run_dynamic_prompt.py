import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
import gc
import re
import json
import torch
import pickle
import string
import argparse
import numpy as np
import pandas as pd
from scipy.special import softmax
from collections import OrderedDict
from vllm import LLM, SamplingParams
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


import warnings
warnings.filterwarnings("ignore")


class TextEmbedder:
    """Create a text embedder object.
    Attributes: Embedding Model
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def chunk_text(self, text, max_tokens):
        """Split input text into a list of text chunks."""
        chunks = []
        words = text.split()
        for i in range(0, len(words), max_tokens):
            chunk = ' '.join(words[i:i+max_tokens])
            chunks.append(chunk)
        return chunks

    def average_embeddings(self, embeddings):
        """Average a list of embeddings."""
        if embeddings:
            return sum(embeddings) / len(embeddings)
        else:
            return None

    def generate_embeddings(self, sample, max_tokens=8000):
        """Takes in a large text, and outputs an embedding representing the entire text."""
        chunks = self.chunk_text(sample, max_tokens)
        chunk_embeddings = []
        for chunk in chunks:
            chunk_embedding = self.embedding_model.encode(chunk)
            chunk_embeddings.append(chunk_embedding)
        return self.average_embeddings(chunk_embeddings)

    def create_vector_db(self, examples):
        """From a list of test, create a vector database of embeddings."""
        db = [self.generate_embeddings(example) for example in examples]
        return db


class ModelEvaluator:
    """"""
    def __init__(self, vector_db, examples, test_samples, llm, sampling_params, labels, zeroshot, summary):
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

    def create_prompt(self, sample, context=True):
        """"""
        if self.summary:
            resp = 'POSITIVE' if self.labels[sample] == 1 else 'NEGATIVE'
            text_prompt = '''<|start_header_id|>user<|end_header_id|>You are an oncologist specializing in glioma at a major cancer hospital. Your task is to predict the 14-month survival outlook for a glioma patient
                based on the following clinical note summary, which represents the patient's status at 0.5 years (6 months) post-diagnosis.\n\nHere is the clinical summary: '''
            if context:
                text_prompt += str(self.examples[sample])
            else:
                text_prompt += str(self.test_samples[sample])
            text_prompt += '''\nPlease analyze this clinical note summary carefully. Based on this analysis as well as the knowledge from the examples, classify the patient's 14 month survival outlook. Respond with 1 word, either 'POSITIVE' (if the patient is likely to survive beyond 14 months) or 'NEGATIVE' (if the patient is unlikely to survive beyond 14 months).'''
            text_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>ANSWER: "
            if context:
                text_prompt += resp + ".<|eot_id|>\n"
            return text_prompt
        else:
            resp = 'POSITIVE' if self.labels[sample] == 1 else 'NEGATIVE'
            text_prompt = '''<|start_header_id|>user<|end_header_id|>You are an oncologist specializing in glioma at a major cancer hospital. Your task is to predict the 14-month survival outlook for a glioma patient
                based on the following clinical notes, which represent the patient's status at 0.5 years (6 months) post-diagnosis.\n\nHere are the clinical notes: '''
            if context:
                text_prompt += str(self.examples[sample])
            else:
                text_prompt += str(self.test_samples[sample])
            text_prompt += '''\nPlease analyze these clinical notes carefully. Based on this analysis as well as the knowledge from the examples, classify the patient's 14 month survival outlook. Respond with 1 word, either 'POSITIVE' (if the patient is likely to survive beyond 14 months) or 'NEGATIVE' (if the patient is unlikely to survive beyond 14 months).'''
            text_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>ANSWER: "
            if context:
                text_prompt += resp + ".<|eot_id|>\n"
            return text_prompt

    def generate_output(self):
        """
        Run the LLM and extract the binary output. If dynamic prompting is enabled, extract the most similar examples
        from the vector database and add those as context to the prompt.
        """
        for i, note in enumerate(self.test_samples):
            if not self.zeroshot:
                embedder = TextEmbedder()
                embedding = embedder.generate_embeddings(note)
                top = self.knn.kneighbors(embedding.reshape(1, -1))
                sample1 = top[1][0][1]
                sample0 = top[1][0][0]

                text_prompt = self.create_prompt(sample0)
                text_prompt += self.create_prompt(sample1)
                if self.summary:
                    sample4 = top[1][0][4]
                    sample3 = top[1][0][3]
                    sample2 = top[1][0][2]
                    text_prompt += self.create_prompt(sample2)
                    text_prompt += self.create_prompt(sample3)
                    text_prompt += self.create_prompt(sample4)
            text_prompt += self.create_prompt(i, context=False)

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
        """Retrieve classification metrics for generated responses."""
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


def ingest_data(example_file, test_file, summary):
    """Reads csv files and outputs 4 lists representing the samples and labels for training and testing sets respectively."""
    df = pd.read_csv(example_file)
    df2 = pd.read_csv(test_file)
    if summary:
        examples = df['summary']
        test_samples = df2['summary']
    else:
        examples = df['note']
        test_samples = df2['note']
    labels = df['label']
    y_test = df2['label']
    examples = [glioma_preprocess(x) for x in examples]
    test_samples = [glioma_preprocess(x) for x in test_samples]
    return examples, labels, test_samples, y_test

def glioma_preprocess(text):
    """Preprocessing function built specifically for UCSF glioma dataset.
    Takes in a string and outputs a preprocessed string."""
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

def main(large, num_gpus, zeroshot, example_file, test_file, summary):
    examples, labels, test_samples, y_test = ingest_data(example_file, test_file, summary)
    embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # initialize embedding model from HuggingFace
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    embedding_model = embedding_model.to(device)
    text_embedder = TextEmbedder(embedding_model)
    vector_db = text_embedder.create_vector_db(examples)
    # tensor parallel to deal with longer context VRAM issues
    if large:
        llm = LLM(model = "gradientai/Llama-3-70B-Instruct-Gradient-262k", tensor_parallel_size=num_gpus, gpu_memory_utilization=0.95)
    else:
        llm = LLM("gradientai/Llama-3-8B-Instruct-262k", tensor_parallel_size=num_gpus, gpu_memory_utilization=0.95) # default is to load smaller model
    sampling_params = SamplingParams(temperature=0, max_tokens=2, logprobs=10) # set temperature to 0 for repeatable results
    model_evaluator = ModelEvaluator(vector_db, examples, test_samples, llm, sampling_params, labels, zeroshot, summary)
    model_evaluator.evaluate()
    acc, prec, rec, f1, auc = model_evaluator.compute_metrics(y_test) # retrieve classification metrics
    print(f"Accuracy: {acc}; Precision: {prec}, Recall: {rec}; F1 score: {f1}; AUC: {auc}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--large', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use large model if set to True, else use small model')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use for tensor parallelism')
    parser.add_argument('--zero_shot', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use zero-shot prompts, default is dynamic prompting')
    parser.add_argument('--examples', required=True, help="CSV file containing a label column and either note or summary column")
    parser.add_argument('--test_data', required=True, help="CSV file containing a label column and either note or summary column")
    parser.add_argument('--summary', type=lambda x: (str(x).lower() == 'true'), default=True, help="Using summarized note text.")
    args = parser.parse_args()
    main(args.large, args.num_gpus, args.zero_shot, args.examples, args.test_data, args.summary)