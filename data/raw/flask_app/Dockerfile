FROM python:3.12.2

COPY . /app

RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

ENTRYPOINT ["python"]

CMD ["app.py"]