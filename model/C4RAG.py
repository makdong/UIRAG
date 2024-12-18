import argparse as ap
import json
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import scipy
import torch
import yaml
from lib import write_jsonl
from retriever.retriever import Retriever, HybridRetriever, TriRetriever
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset

CLASSIFIER_PROMPT = {
    "You are an classifier to classify the query into four levels.\n"
    "There are four levels: Full, Partial, Empty, None.\n"
    "Full level means that the query is highly related to the document.\n"
    "Partial level means that the query is partially related to the document.\n"
    "Empty level means that the query is not related to the document.\n"
    "None level means that this query does not need the document.\n"
    "Refer to the query and document below and classify the query into for level. \n"
    "Make sure to answer only one word in: 'Full', 'Partial', 'Empty' and 'None'. \n\n"
    "### Question\n"
    "{question}\n"
    "### Document\n"
    "{references}\n"
    "### Level\n"
}

POPQA_PROMPT = {
    "Refer to the following documents, follow the instruction and answer the question.\n\n"
    "Documents: {references} \n\n"
    "Instruction: Answer the question: {question}"
}

PUBMED_PROMPT = {
    "Read the documents and answer the question: Is the following statement correct or not? \n\n"
    "Documents: {references}\n\n"
    "Statement: {question}\n\n"
    "Only say true if the statement is true; otherwise say false."
}

ARC_PROMPT = {
    "Refer to the following documents, follow the instruction and answer the question.\n\n"
    "Documents: {references}\n"
    "Question: {question}\n\n"
    "Instruction: Given four answer candidates, A, B, C and D, choose the best answer choice.\n"
    "Choices: {choices}" 
}

def postprocess_answer_option_conditioned(answer):
    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")

    return answer

p = ap.ArgumentParser()
p.add_argument('--yaml_filepath', type=str, default="config/en_config.yaml")
p.add_argument('--dataset', type=str, default='popqa')
p.add_argument('--output_directory', type=str)
p.add_argument('--device', type=str, default='cuda')
p.add_argument('--retriever_name', type=str, default='retriever')
args = p.parse_args()

def main():
    retriever_name = args.retriever_name

    if retriever_name.lower() == "retriever":
        retriever = Retriever(args.yaml_filepath)
    elif retriever_name.lower() in ["hybrid_retriever", "hybrid", "hybridretriever"]:
        retriever = HybridRetriever(args.yaml_filepath)
    else:
        raise ValueError(f"Invalid retriever type: {retriever_name}")
    
    tri_retriever = TriRetriever(args.yaml_filepath)
    
    if args.dataset == 'popqa':
        dataset = load_dataset("akariasai/PopQA")
    elif args.dataset == 'pubmed':
        dataset = load_dataset("ncbi/pubmed", trust_remote_code=True)
    elif args.dataset == 'arc':
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset}")
    
    model_name = "meta-llama/Llama-2-7b-hf"
    llm = LLM(model=model_name, tokenizer=model_name)

    data = dataset['test']

    cnt = 0
    ans_cnt = 0
    json_list = []

    for datum in tqdm(data):
        question = datum['question']
        answer = None
        context = None
        
        if args.dataset == 'popqa':
            answer = datum['possible_answers']
        elif args.dataset == 'pubmed':
            answer = datum['answer']
        elif args.dataset == 'arc':
            answer = datum['answerKey']
            context = datum['choices']
        else:
            raise ValueError(f"Invalid dataset type: {args.dataset}")
        
        # initially retrieve the k documents
        documents, passage_list = retriever.retrieve_with_docs(question)
        processed_documents = []
        classifier_sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100)
        
        # classifier to judge the relevance of the documents
        for document in documents:
            input = CLASSIFIER_PROMPT.format(references=document, question=question)
            output = llm.generate(input, classifier_sampling_params)[0]["text"].lower()

            if "full" in output:
                processed_documents.append(document)
            elif "partial" in output:
                modified_document = tri_retriever.retrieve(question, irrelevant_document=document, passage_list=passage_list)
                processed_documents.append(document + modified_document)
            elif "empty" in output:
                modified_document = tri_retriever.retrieve(question, irrelevant_document=document, passage_list=passage_list)
                processed_documents.append(modified_document)
            elif "none" in output:
                processed_documents.append(" ")
            else: # full로 간주 
                processed_documents.append(document)
        
        # generate the answer logit per document, and select the highest logit as the answer.
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200)
        final_document = " ".join(processed_documents)

        if args.dataset == ('popqa'):
            input = POPQA_PROMPT.format(references=final_document, question=question)
        elif args.dataset == ('pubmed'):
            input = PUBMED_PROMPT.format(references=final_document, question=question)
        elif args.dataset == ('arc'):
            input = ARC_PROMPT.format(references=final_document, question=question, choices=context)
        else:
            raise ValueError(f"Invalid dataset type: {args.dataset}")

        output = postprocess_answer_option_conditioned(llm.generate(input, sampling_params)[0]["text"].lower())
        
        json_output = {
                "question": question,
                "answer": answer,
                "output": output
            }
        if context is not None:
            data["choices"] = context

        json_list.append(json_output)

        # eval
        if answer in output:            
            ans_cnt = ans_cnt + 1
        
        cnt = cnt + 1

    output_path = args.output_directory + '_' + args.dataset + '.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_list, f, ensure_ascii=False, indent=4)

    print(f"The accuracy is : {ans_cnt / cnt}, Count : {cnt}, Answer Count : {ans_cnt}.\n")
            
"""
To-Do
1. load_dataset 결과 양식에 맞게 questions, answers, contexts 뽑아내는거 수정하기 -> pubmed 제외하고 clear
2. 일단 contexts 아니고 choices고, arc에만 choices 있으니까 해당 문제 해결하기 -> clear
3. query, golden answer, output 다 저장하기 -> clear
4. TriRetriever 구현하기
5. eval 자연스럽게 연결하기

a. answer가 possible_answers인, 즉 list인 경우는?
b. context는 dictionary 인데 저렇게 되나?
"""
    
"""
1. Dataset loading
1-1. document dataset : wikipedia(동건이가 준거)
2. load pretrained LLM for generation (Llama2-7B)
3. load Retriever : 원하는건 Contriever, 안되면 BM25
4. load Evaluator : Llama2-3B
5. Evaluator를 위한 prompt 준비

classifier 학습
1. 우선은 Llama2-7B 활용
2. 나중에는 distillation 해보자고

loop 돌면
1. query 들어옴
2. retrieve함
3. evaluator + prompt로 classify함
4-1. complete면 document 그대로
4-2. conditional이면 아래 알고리즘 적용해서 document 추가
4-3. crude면 아래 알고리즘 적용해서 document 변경
4-4. cumberesome 이면 document 폐기
5. llama에 먹임
6. output generate
7. loss 계산은 안해도 됨!!
8. accuracy 계산
9. 끝

Algorithm
1. Document Decompose
2. Query와의 similarity 계산
3. 낮은 점수 chunk → Irrelevant information of the query
4. 높은 점수 chunk → Core information of the query
5. a(sim(d_{highchunk},d)) - b(sim(d_{lowchunk},d))$ 를 simlarity로 재탐색
"""