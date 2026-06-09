FROM python:3.9-slim

WORKDIR /app

# Install dependencies before copying code to leverage layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files into the working context
COPY main.py .

RUN mkdir -p /mnt/model

CMD ["python3", "main.py"]