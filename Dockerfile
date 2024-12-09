

FROM python:3.11-slim-buster

WORKDIR /RegresiLele

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

COPY . .

ENV FLASK_APP=app.py

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8080" ]