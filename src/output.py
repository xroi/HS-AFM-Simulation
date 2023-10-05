from PIL import Image
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def output_gif(args, maps, filename, z_center, min_z, max_z):
    """z_center is the real center"""
    images = []
    for height_map in maps:
        scaled_map = (height_map - min_z) / (max_z - 1 - min_z)
        cm = plt.get_cmap(args["output_gif_color_map"])
        data = cm(scaled_map)
        for i in range(3):
            data[:, :, i] = np.flipud(data[:, :, i].T)
        im = Image.fromarray((data[:, :, :3] * 255).astype(np.uint8), 'RGB')
        im = im.resize((args["output_resolution_x"], args["output_resolution_y"]), resample=Image.BOX)
        images.append(im)
    images[0].save(filename, append_images=images[1:], save_all=True, duration=150, loop=0)


def save_pickle(real_time_maps, needle_maps, args, file_name):
    save_dict = {'real_time_maps': real_time_maps, 'needle_maps': needle_maps, 'args': args}
    with open(file_name, 'wb') as f:
        pickle.dump(save_dict, f)


def load_pickle(file_name):
    """returns a dictionary with the following keys: real_time_maps, needle_maps, args"""
    with open(file_name, 'rb') as f:
        return pickle.load(f)


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


def visualize_taus(taus, voxel_size, min_x, max_x, min_y, max_y, center_x, center_y, dtick, file_path):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=np.fliplr(np.flipud(taus)), colorbar={"title": 'Tau'}))
    fig.layout.height = 500
    fig.layout.width = 500
    fig.update_layout(xaxis={
        "tickmode": 'array',
        "tickvals": [i for i in range(int(taus.shape[0]))],
        "ticktext": [taus_tick_val(i, voxel_size, center_x, dtick) for i in
                     range(int(min_x), int(max_x))]
    }, yaxis={
        "tickmode": 'array',
        "tickvals": [i for i in range(int(taus.shape[1]))],
        "ticktext": [taus_tick_val(i, voxel_size, center_y, dtick) for i in
                     range(int(min_y), int(max_y))]
    })
    fig.update_layout(title="",
                      yaxis={"title": 'Distance from center (A)'},
                      xaxis={"title"    : 'Distance from center (A)',
                             "tickangle": 0}, )
    fig.write_html(file_path)


def taus_tick_val(i, voxel_size, center, dtick):
    if i % dtick != 0:
        return ""
    return str(int((i - center) * voxel_size))


def make_3_color_legend(height, bottom_color, center_color, top_color):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=[[-int(height / 2), 0, int(height / 2)]],
                             colorscale=[
                                 [0, bottom_color],
                                 [0.5, center_color],
                                 [1, top_color]]
                             ))
    fig.show()


def make_bw_legend(height):
    make_3_color_legend(height, "rgb(0, 0, 0)", "rgb(127, 127, 127)", "rgb(255, 255, 255)")


def make_matplot_legend(height, color_map):
    ax = plt.subplot()
    im = ax.imshow(np.arange(-int(height / 2), int(height / 2), 10).reshape(int(height / 10), 1), cmap=color_map)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="200%", pad=1)
    plt.colorbar(im, cax=cax)
    plt.show()
