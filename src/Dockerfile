FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user api_vision.py .
COPY --chown=user modelo_vision.pth .

EXPOSE 7860

CMD ["uvicorn", "api_vision:app", "--host", "0.0.0.0", "--port", "7860"]
