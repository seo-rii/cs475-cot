import json
import tqdm
import datetime
import os
from torch.nn.utils.rnn import pad_sequence
from util import get_input, parse_response_cot, COT_PROMPT
import torch
import transformers

QUESTION = "What is 6 times 3?"
BATCH_SIZE = 2
NUM_ITERATIONS = 10

model_id = "meta-llama/Llama-3.2-3B"
pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

def infer(inputs, max_new_tokens = 4096):
    generated_ids = pipeline(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True
        )
    decoded_texts = pipeline.tokenizer.batch_decode(generated_ids)
    return decoded_texts

def make_input(questions, use_cot=True):
    messages = [get_input(question, use_cot=use_cot) for question in questions]
    model_inputs = [pipeline.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]

    return {
        'input_ids': pad_sequence([model_input['input_ids'].reshape(-1) for model_input in model_inputs], batch_first=True, padding_value=0),
        'attention_mask': pad_sequence([model_input['attention_mask'].reshape(-1) for model_input in model_inputs], batch_first=True, padding_value=0),
    }

def main():
    os.makedirs("output", exist_ok=True)
    filename = f"output/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    res = []

    batch_inputs= make_input([QUESTION] * BATCH_SIZE)

    for _ in tqdm.trange(0, NUM_ITERATIONS * BATCH_SIZE, BATCH_SIZE):
        res += parse_response_cot(infer(batch_inputs))

    with open(filename, "w") as f:
        json.dump({'question': QUESTION, 'prompt': COT_PROMPT, 'size': len(res), 'list': res}, f)

if __name__ == "__main__":
    main()