FROM python:3.9-slim
WORKDIR .
RUN mkdir backend
COPY . backend
RUN apt-get update
RUN apt-get install build-essential libcairo2-dev pkg-config python3-dev ffmpeg -y
RUN python -m pip install --upgrade pip
RUN pip3 install -r backend/requirements.txt --no-cache-dir