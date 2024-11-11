from gemma import infer, parse_response_cot, make_input
import os
import json
import tqdm
import re
from datasets import load_dataset

CHUNK_SIZE=10
COT=True

def test_infer():
    os.makedirs("output", exist_ok=True)
    req = []
    res = []

    ds = load_dataset("openai/gsm8k", "main", split="test")
    for i in ds:
        req.append({"question": i["question"], "answer": i["answer"]})
    
    req = req[:100]
    corr = 0
    for i in tqdm.trange(0, len(req), CHUNK_SIZE):
        chunk = req[i:i + CHUNK_SIZE]
        if COT:
            inputs = make_input([q["question"] for q in chunk])
        else:
            inputs = make_input([q["question"] for q in chunk], cot_prompt="Answer the following question in just one word: ")
        ans = parse_response_cot(infer(inputs))
        for j, question in enumerate(chunk):
            if not ans[j]:
                continue
            
            correct = ans[j]["answer"].strip() in question["answer"].strip('####')[1].strip()
            res.append({"correct": correct, "question": question["question"], "answer": question["answer"], "response": ans[j]})
            if correct:
                corr += 1


    with open("output/gsm8k.json", "w") as f:
        json.dump({"corr": corr, "total": len(res), "list": res}, f)

if __name__ == "__main__":
    test_infer()