FROM python:3.10.0-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
    
RUN pipenv install --system --deploy

COPY [ "model.bin", "./" ]
