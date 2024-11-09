import re

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

def split_text_cot(text):
    parts = re.split(r'(?<=[.!?])\s|\n+|(<[^>]+>)', text)
    return [part.strip() for part in parts if part and part.strip()]

def parse_response_cot_one(response):
    response = response.split('<end_of_turn>')[-2].strip()
    answer = response.split('<Output>')[-1].split('</Output>')[0].strip()
    if response != answer:
        return {"answer": answer, "parts": split_text_cot(response)}
    
    answer = response.split('<output>')[-1].split('</output>')[0].strip()
    if response != answer:
        return {"answer": answer, "parts": split_text_cot(response)}
    
    return None

def parse_response_cot(decoded_texts):
    res = []
    for decoded_text in decoded_texts:
        parsed_response = parse_response_cot_one(decoded_text)
        res.append(parsed_response)

    return res

def get_input(question, cot_prompt = COT_PROMPT, use_cot = True):
    return [
        {"role": "user", "content": cot_prompt},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": question},
    ] if use_cot else [
        {"role": "user", "content": question},
    ]
