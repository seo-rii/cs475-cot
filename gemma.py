from local_gemma import LocalGemma2ForCausalLM
from transformers import AutoTokenizer
import json
import tqdm
import datetime
import os
from torch.nn.utils.rnn import pad_sequence
from torch import bfloat16
from util import get_input, parse_response_cot, COT_PROMPT

QUESTION = "What is 6 times 3?"
BATCH_SIZE = 2
NUM_ITERATIONS = 10

model = LocalGemma2ForCausalLM.from_pretrained("google/gemma-2-2b-it", preset="auto", torch_dtype=bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

def infer(inputs, max_new_tokens = 1024):
    generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True
        )
    decoded_texts = tokenizer.batch_decode(generated_ids)
    return decoded_texts

def make_input(questions, cot_prompt=COT_PROMPT, use_cot=True):
    messages = [get_input(question, use_cot=use_cot, cot_prompt=cot_prompt) for question in questions]
    model_inputs = [tokenizer.apply_chat_template(message, return_tensors="pt", return_dict=True).to(model.device) for message in messages]

    return {
        'input_ids': pad_sequence([model_input['input_ids'].reshape(-1) for model_input in model_inputs], batch_first=True, padding_side='left'),
        'attention_mask': pad_sequence([model_input['attention_mask'].reshape(-1) for model_input in model_inputs], batch_first=True, padding_side='left'),
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