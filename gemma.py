from local_gemma import LocalGemma2ForCausalLM
from transformers import AutoTokenizer
import json
import tqdm
import torch
import re
import datetime
import os

QUESTION = "What is 6 times 3?"
BATCH_SIZE = 16
NUM_ITERATIONS = 10
COT_PROMPT = """You are an AI assistant using a chain of thought (CoT) approach with reflection to answer queries. Follow these steps:
1. Think about the problem step by step within the <Thinking> tags.
2. Reflect on your thinking to check for errors or improvements within the <reflection> tags.
3. Make necessary adjustments based on your reflection.
4. Provide your final, concise answer within the <Output> tags.
The actual answer to the query must be contained entirely within the <Output> tags.
Use the following format for your answer:
<Thinking>
[Your step-by-step reasoning goes here. This is your internal thought process, not the final answer.]
<Reflection>
[Your reflection on your reasoning, error checking, or improvements]
</Reflection>
[Any adjustments to your thinking based on your reflection]
</Thinking>
<Output>
[Your final, concise answer to the query. This is the only part that will be shown to the user.]
</Output>"""

model = LocalGemma2ForCausalLM.from_pretrained("google/gemma-2-2b-it", preset="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

messages = [
    {"role": "user", "content": COT_PROMPT},
    {"role": "assistant", "content": "Ok."},
    {"role": "user", "content": QUESTION},
]

def split_text_cot(text):
    parts = re.split(r'(?<=[.!?])\s|\n+|(<[^>]+>)', text)
    return [part.strip() for part in parts if part and part.strip()]

def parse_response_cot(response):
    answer = response.split('<Output>')[-1].split('</Output>')[0].strip()
    if response != answer:
        return {"answer": answer, "parts": split_text_cot(response)}
    
    answer = response.split('<output>')[-1].split('</output>')[0].strip()
    if response != answer:
        return {"answer": answer, "parts": split_text_cot(response)}
    
    return None

def get_input(message, batch = 1):
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to(model.device)
    batch_inputs = {
        k: v.repeat(batch, 1) if isinstance(v, torch.Tensor) else v 
        for k, v in model_inputs.items()
    }
    return batch_inputs

def main():
    os.makedirs("output", exist_ok=True)
    filename = f"output/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    res = []
    batch_inputs = get_input(messages, BATCH_SIZE)
    for _ in tqdm.trange(0, NUM_ITERATIONS * BATCH_SIZE, BATCH_SIZE):
        generated_ids = model.generate(
            **batch_inputs,
            max_new_tokens=4096,
            do_sample=True
        )

        decoded_texts = tokenizer.batch_decode(generated_ids)
        for decoded_text in decoded_texts:
            response = decoded_text.split('<end_of_turn>')[-2].strip()
            parsed_response = parse_response_cot(response)
            if parsed_response is not None:
                res.append(parsed_response)

    with open(filename, "w") as f:
        json.dump({'question': QUESTION, 'prompt': COT_PROMPT, 'size': len(res), 'list': res}, f)

if __name__ == "__main__":
    main()