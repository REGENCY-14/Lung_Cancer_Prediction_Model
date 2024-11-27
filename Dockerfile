FROM python:3.11-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy


COPY ["predict.py","best_rfmodel","app.py", "./"]

EXPOSE 9696

ENTRYPOINT ["pipenv", "run", "waitress-serve", "--host=0.0.0.0", "--port=5050", "app:app"]