FROM python:3.10-slim

ARG RUN_ID
ENV RUN_ID=${RUN_ID}

WORKDIR /app

CMD echo "Downloading model for Run ID: ${RUN_ID}" && \
    echo "Model download simulated successfully."
