import hashlib
import io
import os
import pickle
import re
import string
import textwrap
import typing
from functools import lru_cache, wraps
from threading import Thread
from time import sleep
from traceback import print_exc

import dash
import networkx as nx
import numpy as np
# import openai
import pandas as pd
import plotly.graph_objects as go
import redis as redis
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from decouple import config, UndefinedValueError
from furl import furl
from googleapiclient import discovery
from googleapiclient.http import MediaIoBaseDownload
from sentence_transformers import SentenceTransformer

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
# openai.api_key = config("OPENAI_API_KEY")
r = redis.Redis.from_url(config("REDIS_CACHE_URL", "redis://localhost:6379"))

batch_size = 100

app = dash.Dash(__name__)
server = app.server

graph_config = dict(
    displayModeBar=True,
    scrollZoom=True,
    displaylogo=False,
    modeBarButtonsToRemove=["lasso2d", "autoScale2d", "zoom", "select"],
)

fig_layout = go.Layout(
    plot_bgcolor="rgb(250,250,250)",
    dragmode="pan",
    margin=dict(l=20, r=20, t=20, b=20),
    uirevision="some-constant-figure",
    xaxis=dict(
        automargin=True,
        showticklabels=False,
        zeroline=False,
        showgrid=True,
        scaleanchor="x",
        scaleratio=1,
    ),
    yaxis=dict(
        automargin=True,
        showticklabels=False,
        zeroline=False,
        showgrid=True,
        scaleanchor="x",
        scaleratio=1,
    ),
)

heatmap_layout = go.Layout(
    plot_bgcolor="rgba(0,0,0,0)",
    dragmode="pan",
    margin=dict(l=120, b=120, r=40, t=40),
    uirevision="some-constant-heatmap",
    xaxis=dict(
        showgrid=False,
        tickangle=-45,
    ),
    yaxis=dict(
        showgrid=False,
        tickangle=-45,
    ),
)


@lru_cache
def get_shared():
    shared = {
        "fig": go.Figure(layout=fig_layout),
        "heatmap": go.Figure(layout=heatmap_layout),
        "submissions": None,
    }
    Thread(target=bg_thread, args=[shared], daemon=True).start()
    return shared


def serve_layout():
    data = get_shared()

    return html.Div(
        html.Div(
            [
                dcc.Markdown(
                    """
    # Who's interests are _nearest_ to mine?

    We calculate the distance matrix from the embeddings of your interests.   
    Then we plot a minimum spanning tree of the graph using a spring layout algorithm

    Hover to see the interests of the person.            
                    """
                ),
                dcc.Graph(
                    id="live-update-graph",
                    style={"height": "90vh"},
                    figure=data["fig"],
                    config=graph_config,
                ),
                dcc.Markdown(
                    """
    ## Heatmap

    The distance matrix plotted as a heatmap.

    The darker the cell, the closer your interests are to them.   
    Note that this doesn't mean that the converse is also true. They might have better matches than you. 
                    """
                ),
                dcc.Graph(
                    id="live-update-heatmap",
                    style={"height": "90vh"},
                    figure=data["heatmap"],
                    config=graph_config,
                ),
                dcc.Markdown(
                    """
    ## All submissions
    The list of all submissions.
    Note that if someone is in your nearest list, that doesn't mean you are in *their* nearest.                 
                    """
                ),
                dash_table.DataTable(
                    id="live-update-submissions",
                    data=data["submissions"],
                    fill_width=True,
                    style_table={
                        "max-width": "100%",
                        "overflow-x": "scroll",
                    },
                    style_header={
                        "font-weight": "bold",
                    },
                    style_cell={
                        "max-width": "100%",
                        "padding": "5px",
                        "text-align": "left",
                    },
                ),
                dcc.Interval(
                    id="interval-component",
                    interval=2000,
                ),
            ],
            style={"fontFamily": "system-ui"},
        )
    )


@app.callback(
    Output("live-update-graph", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_graph_live(n):
    return get_shared()["fig"]


@app.callback(
    Output("live-update-heatmap", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_heatmap_live(n):
    return get_shared()["heatmap"]


@app.callback(
    Output("live-update-submissions", "data"),
    Input("interval-component", "n_intervals"),
)
def update_submisions(n):
    return get_shared()["submissions"]


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
        sleep(5)


def _on_change(df: pd.DataFrame, shared):
    df = df.dropna(axis=1).dropna(axis=0)
    n = len(df)

    names = df["Name"].tolist()
    interests = df["Interests"].tolist()

    embeds = np.array(
        [
            embed
            for i in range(0, n, batch_size)
            for embed in get_embeddings(interests[i : i + batch_size])
        ]
    )
    sims = embeds @ embeds.T

    interests = list(map(wrap_text, interests))

    G = nx.Graph()
    for idx, name in enumerate(names):
        G.add_node(idx, label=name)
    for x in range(n):
        for y in range(n):
            if x == y:
                continue
            G.add_edge(x, y, weight=1 - sims[x, y])
    G = nx.minimum_spanning_tree(G)
    pos = nx.spring_layout(G, seed=42)

    shared["fig"] = dict(
        data=[
            go.Scatter(
                x=[pos[k][0] for k in pos],
                y=[pos[k][1] for k in pos],
                mode="markers+text",
                text=[G.nodes[k]["label"] for k in pos],
                textposition="top center",
                customdata=interests,
                hovertemplate="""
<b>%{text}</b>
<br>
%{customdata}
<extra></extra>
            """,
                marker=dict(
                    color=[sum(sims[k]) for k in pos],
                    colorscale="Oryel",
                    size=10,
                ),
            )
        ],
        layout=fig_layout,
    )

    heatmap = sims.copy()
    heatmap[np.diag_indices_from(heatmap)] = np.nan

    shared["heatmap"] = dict(
        data=[
            go.Heatmap(
                z=heatmap,
                x=names,
                y=names,
                customdata=[
                    [f"{interests[x]}<br><br>{interests[y]}" for x in range(n)]
                    for y in range(n)
                ],
                hovertemplate="""
%{x} & %{y} (%{z:.4f})
<br><br>
%{customdata}
<extra></extra>
                """,
                colorscale="YlGn",
            )
        ],
        layout=heatmap_layout,
    )

    sims[np.diag_indices_from(sims)] = 0
    top_k = np.argsort(sims, axis=1)[:, -3:]
    nearests = [
        [f"{names[y]} ({sims[x, y]:.3f})" for x, y in enumerate(top_k[:, k])]
        for k in range(top_k.shape[1])
    ]
    pos = len(df.columns) - 1
    for k in range(len(nearests)):
        tag = string.ascii_uppercase[len(nearests) - k - 1]
        df.insert(pos, f"Nearest {tag}", nearests[k])

    shared["submissions"] = df.iloc[::-1].to_dict("records")


whitespace_re = re.compile("\s+")


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


def get_embeddings(texts: list[str]):
    texts = [whitespace_re.sub(" ", text) for text in texts]
    return _sentence_embedding_create(input=tuple(texts))


# @redis_cache_decorator
# def _openai_embedding_create(input, engine: str = "text-embedding-ada-002"):
#     res = openai.Embedding.create(input=input, engine=engine)
#     return [record["embedding"] for record in res["data"]]


@redis_cache_decorator
def _sentence_embedding_create(input, engine="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(engine)
    embeddings = model.encode(input)
    return embeddings


def wrap_text(text: str, maxlen: int = 40, sep: str = " …") -> str:
    return "<br>".join(textwrap.wrap(text.strip(), width=maxlen))
    # if len(text) <= maxlen:
    #     return text
    # assert len(sep) <= maxlen
    # match = re.match(r"^(.{0,%d}\S)(\s)" % (maxlen - len(sep) - 1), text, flags=re.S)
    # if match:
    #     trunc = match.group(1)
    # else:
    #     trunc = text[: maxlen - len(sep)]
    # return trunc + sep


app.title = "AI Matchmaker"
app.layout = serve_layout


if __name__ == "__main__":
    app.run_server(debug=True)
