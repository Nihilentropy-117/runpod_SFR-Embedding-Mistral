import runpod
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import os

cache = '/runpod-volume'
os.environ['TRANSFORMERS_CACHE'] = cache
os.environ['HF_HOME'] = cache

device = torch.device('cuda')

tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral')
max_length = 32768
model.to(device)

def get_detailed_instruct(task: str, query: str) -> str:
    return f'Instruct: {task}\nQuery: {query}'


def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def query_embeddings(text, task):
    with torch.no_grad():
        queries = [get_detailed_instruct(task, text)]
        batch_dict = tokenizer(queries, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.squeeze().numpy().tolist()


def passage_embeddings(text):
    with torch.no_grad():
        passages = [text]
        batch_dict = tokenizer(passages, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**batch_dict)
        embeddings1 = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings1, p=2, dim=1)
        return embeddings.squeeze().numpy().tolist()


def handler(job):
    response = {}
    job_input = job["input"]["data"]
    if isinstance(job_input, str):
        job_input = json.loads(job_input)

    task = job_input.get('task')
    text = job_input.get('text')

    if task == 'query':
        task_description = job_input.get('task_description')
        response["embeddings"] = query_embeddings(task_description, text)
    else:
        response["embeddings"] = passage_embeddings(text)

    return json.dumps(response)

# Start the serverless function
runpod.serverless.start({"handler": handler})

