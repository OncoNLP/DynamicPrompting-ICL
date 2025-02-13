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
        db = [self.generate_embeddings(example) for example in examples]
        return db


class ModelEvaluator:
    def __init__(self, vector_db, examples, test_samples, llm, sampling_params, labels, zeroshot):
        self.knn = NearestNeighbors(n_neighbors=2, metric='cosine')
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
    
    def create_prompt(self, sample, context=True):
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

    def evaluate(self):
        for i, note in enumerate(self.test_samples):
            if not self.zeroshot:
                embedder = TextEmbedder()
                embedding = embedder.generate_embeddings(note)
                top = self.knn.kneighbors(embedding.reshape(1, -1))
                sample1 = top[1][0][1]
                sample0 = top[1][0][0]

                text_prompt = self.create_prompt(sample0)
                text_prompt += self.create_prompt(sample1)
            text_prompt += self.create_prompt(i)

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

def process_data():
    pass

def visualize_metrics():
    pass

def main(large, num_gpus, zeroshot):
    #os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7,0'

    examples, labels, test_samples, y_test = process_data()
    embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    emb_model = emb_model.to(device)
    text_embedder = TextEmbedder(embedding_model)
    vector_db = text_embedder.create_vector_db(examples)
    if large:
        llm = LLM(model = "gradientai/Llama-3-70B-Instruct-Gradient-262k", tensor_parallel_size=num_gpus, gpu_memory_utilization=0.95)
    else:
        llm = LLM("gradientai/Llama-3-8B-Instruct-262k", tensor_parallel_size=num_gpus)
    sampling_params = SamplingParams(temperature=0, max_tokens=2, logprobs=10)
    model_evaluator = ModelEvaluator(vector_db, examples, test_samples, llm, sampling_params, labels, zeroshot)
    model_evaluator.evaluate()
    model_evaluator.compute_metrics(y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--large', type=bool, default=False, help='Use large model if set to True, else use small model')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use for tensor parallelism')
    parser.add_argument('--zero-shot', type=bool, default=False, help='Use zero-shot prompts, default is dynamic prompting')
    args = parser.parse_args()
    main(args.large, args.num_gpus)