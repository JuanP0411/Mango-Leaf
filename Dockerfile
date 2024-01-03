FROM tensorflow/tensorflow:2.15.0

WORKDIR /code

COPY . .

RUN pip install -r /code/requirements.txt
RUN pip install pip install "fastapi[all]"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]