import json
import numpy as np
import runpod
import keys

runpod.api_key = keys.RUNPOD_API_KEY
endpoint = runpod.Endpoint(keys.YOUR_ENDPOINT_ID)


def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm_a = np.linalg.norm(embedding1)
    norm_b = np.linalg.norm(embedding2)
    return dot_product / (norm_a * norm_b)


def process(task_description, query_text, passage):
    query_request = {
        "input": {
            "data": {
                "task": "query",
                "task_description": task_description,
                "text": query_text
            }
        }
    }

    passage_request = {
        "input": {
            "data": {
                "task": "passage",
                "text": passage
            }
        }
    }

    # Purge runpod queue
    # endpoint.purge_queue()

    try:
        qE = endpoint.run_sync(query_request, timeout=120)
        qE = json.loads(qE)["embeddings"]
        print("Query Embedding:", qE)
    except TimeoutError:
        print("Query job timed out.")

    try:
        pE = endpoint.run_sync(passage_request, timeout=120)
        pE = json.loads(pE)["embeddings"]
        print("Passage Embedding:", pE)
    except TimeoutError:
        print("Passage job timed out.")

    similarity = cosine_similarity(qE, pE)
    return similarity


if __name__ == "__main__":
    task_description = "Given a web search query"
    query_text = "How to bake a chocolate cake"
    passage = "Step-by-step guide to baking a chocolate cake"

    print("Cosine Similarity:",
          process(task_description, query_text, passage
                  )
          )
