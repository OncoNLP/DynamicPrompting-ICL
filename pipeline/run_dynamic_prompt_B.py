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
import torch.multiprocessing as mp
from collections import OrderedDict
from vllm import LLM, SamplingParams
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


import warnings
warnings.filterwarnings("ignore")

mp.set_start_method('spawn', force=True)


class TextEmbedder:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def chunk_text(self, text, max_tokens):
        chunks = []
        words = text.split()
        for i in range(0, len(words), max_tokens):
            chunk = ' '.join(words[i:i+max_tokens])
            chunks.append(chunk)
        return chunks

    def average_embeddings(self, embeddings):
        if embeddings:
            return sum(embeddings) / len(embeddings)
        else:
            return None

    def generate_embeddings(self, sample, max_tokens=8000):
        chunks = self.chunk_text(sample, max_tokens)
        chunk_embeddings = []
        for chunk in chunks:
            chunk_embedding = self.embedding_model.encode(chunk)
            chunk_embeddings.append(chunk_embedding)
        return self.average_embeddings(chunk_embeddings)

    def create_vector_db(self, examples):
        db = [self.generate_embeddings(example) for example in tqdm(examples, desc="Generating embeddings")]
        return db


class ModelEvaluator:
    def __init__(self, vector_db, examples, test_samples, llm, sampling_params, labels, y_test, zeroshot, summary, text_embedder):
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
        self.y_test = y_test
        self.zeroshot = zeroshot
        self.summary = summary
        self.text_embedder = text_embedder

    # def create_prompt_shots(self, sample, num):
    #     ans = 'POSITIVE' if self.labels[sample] == 1 else 'NEGATIVE'
    #     text_prompt = f'''EXAMPLE {num}: \n'''
    #     text_prompt += str(self.examples[sample])
    #     text_prompt += f"\nANSWER: {ans}.\n"
    #     return text_prompt

    def create_prompt_shots(self, sample, num):
        ans = 'POSITIVE' if self.labels[sample] == 1 else 'NEGATIVE'
        text_prompt = '''<|start_header_id|>user<|end_header_id|>You are an oncologist specializing in breast cancer at a major cancer hospital. Your task is to predict the 5-years survival outlook for a breast cancer patient
            based on the following clinical notes, which represents the patient's status at 0.5 years (6 months) post-diagnosis.\n\n''' 
        text_prompt += '''Here is a clinical notes summary example: \n'''
        text_prompt += str(self.examples[sample])
        text_prompt += f"\nThe correct answer is {ans}.\n"
        return text_prompt

    def create_prompt(self, sample, context=True):
        if self.summary:
            if context:
                text_prompt = '''<|start_header_id|>user<|end_header_id|>You are an oncologist specializing in breast cancer at a major cancer hospital. Your task is to predict the 5-years survival outlook for a breast cancer patient
                    based on a clinical notes, which represents the patient's status at 0.5 years (6 months) post-diagnosis. I am going to provide you previous patient examples to help guide the decision making.\n'''
                text_prompt += "Here is the summary:\n"
                text_prompt += str(self.test_samples[sample])
                text_prompt += '''\nPlease analyze this clinical notes carefully, considering factors such as tumor progression, treatment response, symptoms, and any relevant biomarkers. Based on this analysis as well as the knowledge from the previous examples, classify the patient's 5-years survival outlook. Respond with either 'POSITIVE' (if the patient is likely to survive beyond 5 years) or 'NEGATIVE' (if the patient is unlikely to survive beyond 5 years).'''
                text_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>ANSWER: "

            else:
                text_prompt = '''<|start_header_id|>user<|end_header_id|>You are an oncologist specializing in glioma at a major cancer hospital. Your task is to predict the 14-month survival outlook for a glioma patient
                   based on the following clinical note summary at 0.5 years. Here is the summary: '''
                text_prompt += str(self.test_samples[sample])
                text_prompt += '''\nBased on your understanding of the clinical note summary for a glioma patient at 0.5 years, classify the 14-month survival outlook for this patient.
                   Please respond with either POSITIVE or NEGATIVE'''
                text_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>The correct answer is "

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

    def evaluate(self):
        for i, note in enumerate(self.test_samples):
            if not self.zeroshot:
                embedder = self.text_embedder
                embedding = embedder.generate_embeddings(note)
                top = self.knn.kneighbors(embedding.reshape(1, -1))
                sample1 = top[1][0][1]
                sample0 = top[1][0][0]

                text_prompt = self.create_prompt_shots(sample0, 1)
                text_prompt += self.create_prompt_shots(sample1, 2)
                if self.summary:
                    sample4 = top[1][0][4]
                    sample3 = top[1][0][3]
                    sample2 = top[1][0][2]
                    text_prompt += self.create_prompt_shots(sample2, 3)
                    text_prompt += self.create_prompt_shots(sample3, 4)
                    text_prompt += self.create_prompt_shots(sample4, 5)

                text_prompt += self.create_prompt(i, context=False)
            else:
                text_prompt = self.create_prompt(i, context=False)

            print(text_prompt)

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


def process_data(example_file, test_file, summary):
    df = pd.read_csv(example_file)
    df2 = pd.read_csv(test_file)
    examples = df['note']
    test_samples = df2['note']
    labels = df['label']
    y_test = df2['label']
    return examples, labels, test_samples, y_test


def visualize_metrics():
    pass


def main(exp_type, large, num_gpus, zeroshot, example_file, test_file, summary):
    print("Starting script execution...")
    print("")

    print("Processing data...")
    examples, labels, test_samples, y_test = process_data(example_file, test_file, summary)
    print(f"Loaded {len(examples)} examples and {len(test_samples)} test samples.")
    print("")

    print("Loading embedding model...")
    embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # embedding_model = embedding_model.to(device)
    # print(f"Embedding model loaded on {device}.")
    print("")

    print("Creating vector database...")
    text_embedder = TextEmbedder(embedding_model)
    vector_db = text_embedder.create_vector_db(examples)
    print("Vector database created successfully.")
    print("")

    print("Initializing LLM model...")
    if large:
        model_name = "gradientai/Llama-3-70B-Instruct-Gradient-262k"
        llm = LLM(model=model_name, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.95)
    else:
        model_name = "gradientai/Llama-3-8B-Instruct-262k"
        llm = LLM(model=model_name, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.75)
    print(f"Loaded LLM model: {model_name}")
    print("")

    sampling_params = SamplingParams(temperature=0, max_tokens=2, logprobs=10)

    print("Starting model evaluation...")
    model_evaluator = ModelEvaluator(vector_db, examples, test_samples, llm, sampling_params, labels, y_test, zeroshot, summary, text_embedder)
    model_evaluator.evaluate()

    print("Computing evaluation metrics...")
    acc, prec, rec, f1, auc = model_evaluator.compute_metrics(y_test)
    print("Evaluation complete. Metrics:")
    print(f"  - Accuracy: {acc:.4f}")
    print(f"  - Precision: {prec:.4f}")
    print(f"  - Recall: {rec:.4f}")
    print(f"  - F1 Score: {f1:.4f}")
    print(f"  - AUC: {auc:.4f}")
    print("")

    file_path = f"evaluation_metrics_{exp_type}.txt"

    # Open the file in write mode
    with open(file_path, "w") as f:
        # Write the evaluation metrics to the file
        f.write("Evaluation complete. Metrics:\n")
        f.write(f"  - Accuracy: {acc:.4f}\n")
        f.write(f"  - Precision: {prec:.4f}\n")
        f.write(f"  - Recall: {rec:.4f}\n")
        f.write(f"  - F1 Score: {f1:.4f}\n")
        f.write(f"  - AUC: {auc:.4f}\n")
        f.write("\n")

    print(f"Metrics saved to {file_path}")

    print("Script execution finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', type=str, required=True, help='zero_shot_summary/zero_shot_fn/dynamic_summary/dynamic_fn')
    parser.add_argument('--large', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use large model if set to True, else use small model')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use for tensor parallelism')
    parser.add_argument('--zero_shot', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use zero-shot prompts, default is dynamic prompting')
    parser.add_argument('--examples', required=True, help="CSV file containing a label column and either note or summary column")
    parser.add_argument('--test_data', required=True, help="CSV file containing a label column and either note or summary column")
    parser.add_argument('--summary', type=lambda x: (str(x).lower() == 'true'), default=True, help="Using summarized note text.")
    args = parser.parse_args()
    main(args.exp_type, args.large, args.num_gpus, args.zero_shot, args.examples, args.test_data, args.summary)