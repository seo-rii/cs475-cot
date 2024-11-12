import os
import json
import tqdm
from tot import tree_of_thought

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
        for question_name in data:
            for example in data[question_name]:
                question = data[question_name][example]["question"] + " " + data[question_name][example]["query"]
                answer = data[question_name][example]["answer"]
                req.append({"question": question, "answer": answer})
    
    req = req[:5]
    corr = 0
    for i in tqdm.trange(0, len(req)):
        ret = tree_of_thought(req[i]["question"])
        correct = parse_number(req[i]["answer"]) == parse_number(ret["answer"])
        res.append({"correct": correct, "question": req[i]["question"], "answer": req[i]["answer"], "response": ret})


    with open("output/1hop.json", "w") as f:
        json.dump({"corr": corr, "total": len(res), "list": res}, f)

if __name__ == "__main__":
    test_infer()