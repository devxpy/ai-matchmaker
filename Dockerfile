# base python image
FROM library/python:3.10-slim

# set the working directory in the container
ENV WORKDIR /app
WORKDIR $WORKDIR

# copy poetry files
COPY ./pyproject.toml ./poetry.lock ./
# install python dependencies
RUN pip install -U poetry pip && poetry install --only main --no-interaction
RUN poetry run pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu

# cache the embeddings model
RUN poetry run python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")'

# copy the code into the container
COPY . .

ENV FORWARDED_ALLOW_IPS='*'
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD poetry run gunicorn app:server --threads ${THREADS:-1} --bind 0.0.0.0:8000
