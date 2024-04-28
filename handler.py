import runpod
import json
import torch
from sentence_transformers import SentenceTransformer
import os

cache = '/runpod-volume'
os.environ['TRANSFORMERS_CACHE'] = cache
os.environ['HF_HOME'] = cache

device = torch.device("cuda")

model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")

model.to(device)

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def make_embeddings(text, task_description=None):
    with torch.no_grad():

        if task_description:
            embeddings = model.encode(get_detailed_instruct(task_description, text))
        else:
            embeddings = model.encode(text)

    embeddings = embeddings.squeeze().tolist()
    return embeddings



def handler(job):
    response = {}
    job_input = job["input"]["data"]
    if isinstance(job_input, str):
        job_input = json.loads(job_input)

    task = job_input.get('task')
    text = job_input.get('text')

    if task == 'query':
        task_description = job_input.get('task_description')
        response["embeddings"] = make_embeddings(text=text, task_description=task_description)
    else:
        response["embeddings"] = make_embeddings(text=text)

    return json.dumps(response)

# Start the serverless function
runpod.serverless.start({"handler": handler})

