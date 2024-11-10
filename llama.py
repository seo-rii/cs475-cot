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
DEFAULT_CHAT_TEMPLATE = """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '
' + message['content'] | trim + '<end_of_turn>
' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model
'}}{% endif %}"""

model_id = "meta-llama/Llama-3.2-3B"
pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
pipeline.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

def infer(inputs, max_new_tokens = 4096):
    generated_ids = pipeline.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True
        )
    decoded_texts = pipeline.tokenizer.batch_decode(generated_ids)
    return decoded_texts

def make_input(questions, use_cot=True):
    messages = [get_input(question, use_cot=use_cot) for question in questions]
    model_inputs = [pipeline.tokenizer.apply_chat_template(message, return_tensors="pt", return_dict=True).to(pipeline.model.device) for message in messages]
    
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