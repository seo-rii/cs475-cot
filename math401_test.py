from gemma import infer, parse_response_cot, make_input
import os
import json
import tqdm
import re

CHUNK_SIZE=10
COT=True

def parse_number(st):
    li = re.findall(r"\d+", st)
    if len(li) == 0:
        return None
    return int(li[0])

def test_infer(filename="math401-llm.json"):
    os.makedirs("output", exist_ok=True)
    req = []
    res = []

    with open(filename, "r") as f:
        data = json.load(f)
        for row in data:
            question = row["query"]
            answer = row["response"]
            req.append({"question": question, "answer": answer})
    
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
            
            correct = parse_number(question["answer"]) == parse_number(ans[j]["answer"])
            res.append({"correct": correct, "question": question["question"], "answer": question["answer"], "response": ans[j]})
            if correct:
                corr += 1


    with open("output/math401.json", "w") as f:
        json.dump({"corr": corr, "total": len(res), "list": res}, f)

if __name__ == "__main__":
    test_infer()