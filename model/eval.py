#!/usr/bin/python
# -*- coding: UTF-8 -*-

import jsonlines
import numpy as np
from tqdm import tqdm
import json
import argparse
from tqdm import tqdm


TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
             "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
             "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
             "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."}

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def accuracy(preds, labels):
    match_count = 0
    for pred, label in zip(preds, labels):
        target = label[0]
        if pred == target:
            match_count += 1

    return 100 * (match_count / len(preds))

def match(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0


def preprocess_input_data(dataset, task=None):
    """
    Change format as below.
    item["instruction"] : input
    item["answers"] : expected output
    """
    new_data = []
    
    if task in TASK_INST:
        instruction = TASK_INST[task]
    else:
        instruction = None

    for item in dataset:
        if "arc" in task:
            choices = item["choices"]
            answer_labels = {}

            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]

                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            
            item["instruction"] = instruction + "\n\n### Input:\n" + item["question"] + choices
            item["answers"] = [item["answerKey"]]
        else:
            if instruction is None:
                prompt = item["question"]
            else:
                prompt = instruction + "\n\n## Input:\n\n" + item["question"]
            
            item["instruction"] = prompt

        new_data.append(item)

    return new_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str, default=None)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--metric', type=str, help="metric to be used during evaluation")
    args = parser.parse_args()

    input_path = args.input_file
    if input_path.endswith(".json"):
        input_data = json.load(open(input_path))
    else:
        input_data = load_jsonlines(input_path)

    input_data = preprocess_input_data(input_data, task=args.task)
    eval_file = args.eval_file

    with open(eval_file, 'r') as f:
        resps = [l.strip() for l in f.readlines()]

    preds = []
    metric_results = []

    for pred, row in tqdm(zip(resps, input_data)):
        pred = pred.strip()
        preds.append(pred)

        if "answers" not in row and "answer" in row:
            row["answers"] = [row["answer"]] if type(row["answer"]) is str else row["answer"]
        
        if args.metric == "accuracy":
            metric_result = accuracy(pred, row["output"])
        elif args.metric == "match":
            if "SUPPORTS" in pred:
                pred = "true"
            elif "REFUTES" in pred:
                pred = "false"
            metric_result = match(pred, row["answers"])
        else:
            raise NotImplementedError

        metric_results.append(metric_result)

    print("Final result: {0}".format(np.mean(metric_results)))

if __name__ == "__main__":
    main()

"""
input file : 데이터셋
eval file : 내 model output을 의미하는거였습니다

input_data의 row 하나는 instruction, answer(answers), output 이렇게 세 개가 있을 수 있는데
answer랑 output이랑 차이가 뭐지? 나는 accuracy metric인데, 이거 output으로 setting하면 안될 것 같은데... 데이터 형태 뜯어보자
huggingface에서 load해서 바로 사용할 수 있게 코드를 바꿔야겠는걸 ~ temp.ipynb 결과 뜯어보면서 바꿔야함
근데 huggingface에서 pubmed download 속도가 너무 느려서 다른 데이터셋으로 실험돌려봐야 할 듯

main에서 preprocess해서 input 먹이고 output file에 저장하고
여기서 preprocess해서 output이랑 내 model output이랑 비교하고
"""