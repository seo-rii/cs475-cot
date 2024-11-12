import random
from typing import List, Tuple, Any
from torch.nn.utils.rnn import pad_sequence
from util import get_input
import os
import datetime
import json

from gemma import infer, tokenizer, model

QUESTION = "19+2-17=?"

TOT_PROMPTS = {
    1: """You are an AI assistant using a chain of thought (CoT) approach with reflection to answer queries.
You should help solve this problem step by step.
You will have three stages to follow: Initial Analysis, Solution Process, and Final Solution.
You must follow these steps, and provide your responses in the format provided. You must keep XML tags in your responses.
Problem: {question}

Stage 1 - Initial Analysis: What are the key elements to consider?

You must provide your responses in the format provided.
<stage>Your analysis</stage>
<next_options>Next possible approaches</next_options>""",
    
    2: """Stage 2 - Solution Process: How should we solve this based on our analysis?

You must provide your responses in the format provided.
<stage>Your solution process</stage>
<next_options>Next possible steps</next_options>""",
    
    3: """Stage 3 - Final Solution: Let's conclude based on our reasoning. Think one more time and provide the final answer.

You must provide your responses in the format provided.
<stage>Your conclusion</stage>
<answer>Complete sentence answer</answer>
<short_answer>Your final, concise answer to the query. Answer in one word if available.</short_answer>"""
}


def evaluate_thoughts(thoughts: List[tuple[str, Any]], k: int = 2) -> List[Tuple[str, List[str], float]]:
    scored_thoughts = [(thought, history, random.random()) for thought, history in thoughts]
    return sorted(scored_thoughts, key=lambda x: x[2], reverse=True)[:k]

def make_input(prev_conversations: List[List[str]], question: str) -> dict:
    messages = []
    for prev_conversation in prev_conversations:
        conversation = []
        
        for i in range(len(prev_conversation)):
            conversation.append({"role": "user", "content": TOT_PROMPTS[i + 1].format(question=question)})
            conversation.append({"role": "assistant", "content": prev_conversation[i]})
            
        conversation.append({"role": "user", "content": TOT_PROMPTS[len(prev_conversation) + 1].format(question=question)})
        messages.append(conversation)
    
    model_inputs = [
        tokenizer.apply_chat_template(message, return_tensors="pt", return_dict=True).to(model.device) 
        for message in messages
    ]
    
    return {
        'input_ids': pad_sequence([mi['input_ids'].reshape(-1) for mi in model_inputs], batch_first=True, padding_side='left'),
        'attention_mask': pad_sequence([mi['attention_mask'].reshape(-1) for mi in model_inputs], batch_first=True, padding_side='left')
    }

def generate_thoughts(prev_conversations: List[List[str]], question: str, n_samples: int = 5) -> List[str]:
    r = infer(make_input(prev_conversations * n_samples, question))
    li = []
    for i in range(len(prev_conversations)):
        for j in range(n_samples):
            response = r[i * n_samples + j]
            turns = response.replace('<pad>', '').split('<end_of_turn>')
            if len(turns) < 2:
                response = ""
            else:
                response = turns[len(turns) // 2 * 2 - 1]
            li.append((response, [*prev_conversations[i], response]))

    return li

def parse_tot_response(response: str, history) -> dict:
    def extract_tag_content(text: str, tag: str) -> str:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        try:
            start = text.index(start_tag) + len(start_tag)
            end = text.index(end_tag)
            return text[start:end].strip()
        except ValueError:
            return ""

    result = {
        'answer': extract_tag_content(response, 'answer'),
        'short_answer': extract_tag_content(response, 'short_answer'),
        'thinking': history,
    }
    
    return result

def tree_of_thought(question: str) -> str:
    thoughts_1 = generate_thoughts([[]], question)
    top_thoughts_1 = evaluate_thoughts(thoughts_1)

    thoughts_2 = generate_thoughts([his for _, his, _ in top_thoughts_1], question)
    top_thoughts_2 = evaluate_thoughts(thoughts_2)

    thoughts_3 = generate_thoughts([his for _, his, _ in top_thoughts_2], question)
    top_thoughts_3 = evaluate_thoughts(thoughts_3)
    
    final_thought = top_thoughts_3[0][0]

    return parse_tot_response(final_thought, top_thoughts_3[0])

def main():
    os.makedirs("output", exist_ok=True)
    filename = f"output/tot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    final_answer = tree_of_thought(QUESTION)
    
    with open(filename, "w") as f:
        json.dump({
            'question': QUESTION,
            'answer': final_answer['short_answer'],
            'response': final_answer,
        }, f)

if __name__ == "__main__":
    main()