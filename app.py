import hashlib
import io
import os
import pickle
import re
import typing
from functools import lru_cache, wraps
from threading import Thread
from time import sleep
from traceback import print_exc

import dash
import numpy as np
import openai
import pandas as pd
import plotly.graph_objects as go
import redis as redis
import umap
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from decouple import config, UndefinedValueError
from furl import furl
from googleapiclient import discovery
from googleapiclient.http import MediaIoBaseDownload
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

service_account_key_path = "serviceAccountKey.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path
# save json file from env var if available
try:
    _json = config("GOOGLE_APPLICATION_CREDENTIALS_JSON")
except UndefinedValueError:
    pass
else:
    with open(service_account_key_path, "w") as f:
        f.write(_json)


SHEETS_URL = config("SHEETS_URL")
openai.api_key = config("OPENAI_API_KEY")
r = redis.Redis.from_url(config("REDIS_CACHE_URL", "redis://localhost:6379"))

batch_size = 100


app = dash.Dash(__name__)
server = app.server
app.layout = html.Div(
    html.Div(
        [
            html.H1("Top Matches"),
            dash_table.DataTable(
                id="live-update-matches",
                fill_width=False,
                style_header={
                    "font-weight": "bold",
                },
                style_cell={
                    "padding": "5px",
                },
            ),
            html.H1("UMAP visualization"),
            dcc.Graph(id="live-update-graph", style={"height": "80vh"}),
            # html.Div(id="live-update-text"),
            dcc.Interval(
                id="interval-component",
                interval=1000,  # in milliseconds
                n_intervals=0,
            ),
        ],
        style={"fontFamily": "system-ui"},
    )
)


@lru_cache
def get_shared():
    shared = {}
    Thread(target=bg_thread, args=[shared]).start()
    return shared


def bg_thread(shared):
    prev_bytes = None
    while True:
        csv_bytes = download_sheet(furl(SHEETS_URL))
        if prev_bytes != csv_bytes:
            prev_bytes = csv_bytes
            df = pd.read_csv(io.BytesIO(csv_bytes))
            try:
                _on_change(df, shared)
            except:
                print_exc()
        sleep(1)


def _on_change(df, shared):
    df = df.rename(
        columns=lambda col: col.strip().lower().replace(" ", "_")
    )

    keys = df.name.dropna().tolist()
    docs = df.interests.dropna().tolist()

    embeds = np.array([
        embed
        for i in range(0, len(docs), batch_size)
        for embed in get_embeddings(docs[i: i + batch_size])
    ])

    data = umap.UMAP(random_state=42, n_neighbors=len(keys) // 2).fit_transform(
        embeds
    )
    data = StandardScaler().fit_transform(data)
    shared["plot"] = go.Scatter(
        x=[row[0] for row in data],
        y=[row[1] for row in data],
        text=keys,
        customdata=docs,
        hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
        mode="markers+text",
        textposition="top center",
    )

    sims = embeds @ embeds.T
    sorted_sims = np.dstack(
        np.unravel_index(np.argsort(sims, axis=None), sims.shape)
    )[0][::-1]
    visited = set()
    matches = []
    for x, y in sorted_sims:
        if x == y:
            continue
        if x in visited or y in visited:
            continue
        visited.add(x)
        visited.add(y)
        matches.append(
            {
                "Name 1": keys[x],
                "Name 2": keys[y],
                "Similarity": round(sims[x, y], 4),
            }
        )
    shared["top_matches"] = pd.DataFrame.from_records(matches)


whitespace_re = re.compile("\s+")


def get_embeddings(texts: list[str], engine: str = "text-embedding-ada-002"):
    texts = [whitespace_re.sub(" ", text) for text in texts]
    res = _openai_embedding_create(input=tuple(texts), engine=engine)
    return [record["embedding"] for record in res["data"]]




def download_sheet(f: furl, mime_type: str = "text/csv") -> bytes:
    # get drive file id
    file_id = url_to_gdrive_file_id(f)
    # get metadata
    service = discovery.build("drive", "v3")
    # get files in drive directly
    request = service.files().export_media(fileId=file_id, mimeType=mime_type)
    # download
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    # to dataframe
    return file.getvalue()


def url_to_gdrive_file_id(f: furl) -> str:
    # extract google drive file ID
    try:
        # https://drive.google.com/u/0/uc?id=FILE_ID&...
        file_id = f.query.params["id"]
    except KeyError:
        # https://drive.google.com/file/d/FILE_ID/...
        # https://docs.google.com/document/d/FILE_ID/...
        try:
            file_id = f.path.segments[f.path.segments.index("d") + 1]
        except (IndexError, ValueError):
            raise ValueError(f"Bad google drive link: {str(f)!r}")
    return file_id


@app.callback(
    Output("live-update-graph", "figure"), Input("interval-component", "n_intervals"),
)
def update_graph_live(n):
    fig = make_subplots()
    data = get_shared()
    if "plot" in data:
        fig.add_trace(data["plot"])
    return fig


@app.callback(
    Output("live-update-matches", "data"),
    Input("interval-component", "n_intervals"),
)
def update_matches(n):
    data = get_shared()
    if "top_matches" not in data:
        return
    df = data["top_matches"]
    return df.to_dict("records")


F = typing.TypeVar("F", bound=typing.Callable[..., typing.Any])
def redis_cache_decorator(fn: F) -> F:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # hash the args and kwargs so they are not too long
        args_hash = hashlib.sha256(f"{args}{kwargs}".encode()).hexdigest()
        # create a readable cache key
        cache_key = f"ai-matchmaker/redis-cache-decorator/v1/{fn.__name__}/{args_hash}"
        cache_val = r.get(cache_key)
        # if the cache exists, return it
        if cache_val:
            return pickle.loads(cache_val)
        # otherwise, run the function and cache the result
        else:
            result = fn(*args, **kwargs)
            cache_val = pickle.dumps(result)
            r.set(cache_key, cache_val)
            return result

    return wrapper


@redis_cache_decorator
def _openai_embedding_create(*args, **kwargs):
    return openai.Embedding.create(*args, **kwargs)


# @app.callback(
#     Output("live-update-text", "children"), Input("interval-component", "n_intervals")
# )
# def update_metrics(n):
#     # lon, lat, alt = satellite.get_lonlatalt(datetime.datetime.now())
#     lon, lat, alt = random.random() * 360, random.random() * 180, random.random() * 1000
#     style = {"padding": "5px", "fontSize": "16px"}
#     return [
#         html.DataTa,
#         html.Span("Latitude: {0:.2f}".format(lat), style=style),
#         html.Span("Altitude: {0:0.2f}".format(alt), style=style),
#     ]


if __name__ == "__main__":
    app.run_server(debug=True)
