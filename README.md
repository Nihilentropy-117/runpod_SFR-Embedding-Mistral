# RunPod Serverless Embeddings with Salesforce Mistral Model

This repository houses the source code and Docker configuration necessary for deploying a serverless function on RunPod, leveraging the `Salesforce/SFR-Embedding-Mistral` model to generate text embeddings. This function is ideally suited for applications involving semantic search, document retrieval, and text analysis, capable of processing both queries and passages efficiently.

## Features

Generates embeddings for both queries and passages

## RunPod Deployment Template

To deploy this model on RunPod, you can use the template directly:
- **Template Name**: `nihilentropy_sfr-embedding-mistral`

Alternatively, configure a serverless instance manually:
- **Minimum Requirements**: At least 24GB VRAM and a network-attached volume of at least 30GB.
- **Estimated Cost**: Approximately $2.25/month for storage cache, plus usage time
- **Environment Setup**:
  - Set the environmental variable: `HF_HOME=/runpod-volume`.
  - Load the container image: `nihilentropy/runpod_sfr-embedding-mistral:latest`.

## JSON Schema

The serverless function expects the following JSON payloads for processing:

### Query Request
```json
{
    "input": {
        "data": {"task": "query", "task_description": "Your task description", "text": "Your query text"}
    }
}
```

### Passage Request
```json
{
    "input": {
        "data": {"task": "passage", "text": "Your passage text"}
    }
}
```

### Response Format
The function returns embeddings as an array of floats:
```json
{"embeddings": [0.0004922622465528548, ..., 0.0035579828545451164]}
```

## Authors

- **Gray Lott** - *Initial Development* - [nihilentropy](https://github.com/nihilentropy)
- **GPT-4** - *Documentation Assistance* - [OpenAI GPT-4](https://chat.openai.com)
