from gemma import infer, parse_response_cot, make_input
import os
import json
import tqdm

CHUNK_SIZE=10
COT=True

def test_infer(filename="prontoqa/1hop.json"):
    os.makedirs("output", exist_ok=True)
    req = []
    res = []

    with open(filename, "r") as f:
        data = json.load(f)
        for question_name in data:
            for example in data[question_name]:
                question = data[question_name][example]["question"] + " " + data[question_name][example]["query"]
                answer = data[question_name][example]["answer"]
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
            correct = question["answer"].lower() in ans[j]["answer"].lower()
            res.append({"correct": correct, "question": question["question"], "answer": question["answer"], "response": ans[j]})
            if correct:
                corr += 1


    with open("output/1hop.json", "w") as f:
        json.dump({"corr": corr, "total": len(res), "list": res}, f)

if __name__ == "__main__":
    test_infer()