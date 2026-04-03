FROM python:3.10-slim
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .
# Pre-download models to bake them into the image (optional but recommended)
RUN python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'); pipeline('text-classification', model='unitary/multilingual-toxic-xlm-roberta'); pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]