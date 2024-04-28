import json

import runpod
import keys
import numpy as np

runpod.api_key = keys.RUNPOD_API_KEY
endpoint = runpod.Endpoint(keys.YOUR_ENDPOINT_ID)


def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm_a = np.linalg.norm(embedding1)
    norm_b = np.linalg.norm(embedding2)
    return dot_product / (norm_a * norm_b)


task_description = "Given a web search query"
query_text = "How to bake a chocolate cake"
passage = "Step-by-step guide to baking a chocolate cake"


query_request = {
    "input": {
        "data": {"task": "query", "task_description": task_description, "text": query_text}
    }
}

passage_request = {
            "input": {
                "data":
                    {"task": "passage",
                     "text": passage}
            }
        }

endpoint.purge_queue()
try:
    qE = endpoint.run_sync(
        query_request,
        timeout=120,  # Timeout in seconds.
    )

    print(qE)
except TimeoutError:
    print("Job timed out.")

try:
    pE = endpoint.run_sync(
        passage_request,
        timeout=120,  # Timeout in seconds.
    )

    print(pE)
except TimeoutError:
    print("Job timed out.")

qE = json.loads(qE)["embeddings"]
pE = json.loads(pE)["embeddings"]

similarity = cosine_similarity(qE, pE)
print("Cosine Similarity:", similarity)

