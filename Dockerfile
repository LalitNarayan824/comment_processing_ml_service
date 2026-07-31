FROM python:3.10-slim
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .
# Pre-download all 4 public models during the build phase
RUN python -c "from transformers import pipeline; \
pipeline('text-classification', model='lalit-narayan/youtube-comment-intent-classifier'); \
pipeline('text-classification', model='martin-ha/toxic-comment-model'); \
pipeline('sentiment-analysis', model='AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual'); \
pipeline('text-classification', model='valurank/distilroberta-spam-comments-detection')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
