FROM pytorch/pytorch
COPY . /app
WORKDIR /app
RUN pip install -r /app/requirements.txt
CMD ["python", "/app/src/train.py"]

