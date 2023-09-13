from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import product


def output_gif(args, maps, filename):
    images = []
    for height_map in maps:
        im = Image.fromarray((np.flipud(height_map.T) * 255).astype(np.uint8)).resize(
            (args["output_resolution_x"], args["output_resolution_y"]), resample=Image.BOX)
        images.append(im)
    images[0].save(filename, append_images=images[1:], save_all=True, duration=100, loop=0)


def output_hdf5(maps):
    # todo
    raise Exception("not yet implemented.")


def visualize_auto_corr(acorrs):
    # Line Plot:

    # x_size, y_size, lags = acorrs.shape
    # x_axis = [i for i in range(lags)]
    # fig = go.Figure()
    # for x, y in product(range(x_size), range(y_size)):
    #     fig.add_trace(go.Scatter(x=x_axis, y=acorrs[x, y, :], mode='lines'))
    # fig.show()

    # Box Plot:
    fig = go.Figure()
    fig.add_trace(go.Box(y=acorrs[:, :, 1].flatten(), name="Lag=1"))
    fig.add_trace(go.Box(y=acorrs[:, :, 2].flatten(), name="Lag=2"))
    fig.add_trace(go.Box(y=acorrs[:, :, 3].flatten(), name="Lag=3"))
    fig.add_trace(go.Box(y=acorrs[:, :, 4].flatten(), name="Lag=4"))
    fig.add_trace(go.Box(y=acorrs[:, :, 5].flatten(), name="Lag=5"))
    # fig.update_traces(boxpoints='all', jitter=0.3)
    fig.update_layout(showlegend=False, yaxis=dict(title='Auto Correlation'))
    fig.show()


def visualize_taus(taus):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=taus))
    fig.layout.height = 500
    fig.layout.width = 500
    fig.show()


def make_bw_legend(height):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=[[-int(height / 2), 0, int(height / 2)]],
                             colorscale=[
                                 [0, "rgb(0, 0, 0)"],
                                 [0.5, "rgb(127, 127, 127)"],
                                 [1, "rgb(255, 255, 255)"]]
                             ))
    fig.show()
