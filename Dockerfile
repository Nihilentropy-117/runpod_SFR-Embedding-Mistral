# Use a lightweight Python image
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the model to the image, saves having to use the network drive, but is less performant for no real monetary gain
# RUN python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral')"


# Copy the handler script
ADD handler.py .

# Command to run the handler script
CMD ["python", "-u", "/app/handler.py"]
