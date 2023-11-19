from PIL import Image
import numpy as np
import pickle
import plotly.express as px
import auto_corr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def output_gif(args, maps, filename, z_center, min_z, max_z):
    """z_center is the real center"""
    images = []
    for height_map in maps:
        scaled_map = (height_map - min_z) / (max_z - 1 - min_z)
        cm = plt.get_cmap(args["output_gif_color_map"])
        data = cm(scaled_map)
        im = Image.fromarray((data[:, :, :3] * 255).astype(np.uint8), 'RGB')
        im = im.resize((args["output_resolution_y"], args["output_resolution_x"]), resample=Image.BOX).rotate(angle=90,
                                                                                                              expand=1)
        images.append(im)
    images[0].save(filename, append_images=images[1:], save_all=True, duration=150, loop=0)


def save_pickle(real_time_maps, needle_maps, args, file_name):
    save_dict = {'real_time_maps': real_time_maps, 'rasterized_maps': needle_maps, 'args': args}
    with open(file_name, 'wb') as f:
        pickle.dump(save_dict, f)


def load_pickle(file_name):
    """returns a dictionary with the following keys: real_time_maps, rasterized_maps, args"""
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def visualize_taus(taus, voxel_size, min_x, max_x, min_y, max_y, center_x, center_y, dtick, file_path):
    voxel_size = voxel_size / 10
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=np.fliplr(np.flipud(taus)), colorbar={"title": 'Tau (μs)'}))
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
                      yaxis={"title": 'Distance from center (nm)'},
                      xaxis={"title"    : 'Distance from center (nm)',
                             "tickangle": 0},
                      font=dict(size=20))
    fig.write_image(file_path, width=500, height=500)


def visualize_height_by_radial_distance(ring_means, envelope_heights, file_path, sym=False, yrange=None):
    """ring means is a 2d array where axis1 are the means over time, and axis2 are different runs
    todo allow this and show on lower opacity"""
    max_r = len(ring_means)
    if sym:
        x = [i for i in range(-max_r + 1, max_r)]
        y = np.concatenate((np.flip(ring_means)[:-1], ring_means))
        envelope_heights = np.concatenate((np.flip(envelope_heights)[:-1], envelope_heights))
    else:
        x = [i for i in range(max_r)]
        y = ring_means
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', line=dict(width=6), marker=dict(size=8),
                             name="Mean AFM Simulation Height"))
    fig.add_trace(go.Scatter(x=x, y=envelope_heights, mode='lines', line=dict(width=12, color="#2e2f30"),
                             name="Nuclear Envelope", fill='tozeroy', fillcolor='#2e2f30'))
    fig.data[0].line.color = "#0c23f5"
    fig.update_layout(xaxis_title="Distance from center (nm)",
                      yaxis_title="Height (nm)",
                      font=dict(size=40),
                      template="plotly_white",
                      xaxis=dict(dtick=2.5),
                      yaxis=dict(dtick=2.5),
                      showlegend=False)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#9e9d99', zerolinecolor='#9e9d99')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#9e9d99')
    fig.add_hline(y=7.5, line_width=0, line_dash="dash", line_color="Black", annotation_text="Nuclear Envelope",
                  annotation_position="top left")
    fig.update_yaxes(scaleratio=1)
    if yrange:
        fig.update_layout(yaxis_range=yrange, xaxis_range=[-25, 25])
    fig.write_image(file_path, width=3000, height=700)


def visualize_tcf_samples(acorrs, taus, dist_px, amount, file_path):
    nlags = acorrs.shape[2]
    fig = make_subplots(rows=1,
                        cols=amount,
                        subplot_titles=[
                            f"({int(acorrs.shape[0] / 2) + i * dist_px},{int(acorrs.shape[1] / 2)}) tau="
                            f"{taus[int(acorrs.shape[0] / 2) + i * dist_px, int(acorrs.shape[1] / 2)]}μs"
                            for i
                            in range(nlags)],
                        shared_yaxes='all')
    x = [i for i in range(nlags)]
    for i in range(amount):
        x_coord = int(acorrs.shape[0] / 2) + i * dist_px
        y_coord = int(acorrs.shape[1] / 2)
        fig.add_trace(
            go.Scatter(x=x, y=acorrs[x_coord, y_coord, :], mode='markers', line_color='#e00000'),
            row=1,
            col=i + 1
        )
        fig.add_trace(
            go.Scatter(x=x, y=[auto_corr.model_func(j, taus[x_coord, y_coord]) for j in range(nlags)],
                       line_color="#000be0"),
            row=1, col=i + 1
        )
    fig.update_layout(font=dict(size=20),
                      template="plotly_white",
                      width=1000 * amount,
                      height=1000,
                      autosize=False,
                      showlegend=False,
                      yaxis_title="Correlation",
                      xaxis_title="Time Lag (μs)")
    fig.write_image(file_path, width=800 * amount, height=800)


def visualize_tau_by_radial_distance(ring_means, file_path, sym=False, yrange=False):
    max_r = len(ring_means)
    if sym:
        x = [i for i in range(-max_r + 1, max_r)]
        y = np.concatenate((np.flip(ring_means), ring_means))
    else:
        x = [i for i in range(max_r)]
        y = ring_means
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'))
    fig.update_layout(xaxis_title="Distance from center (nm)",
                      yaxis_title="Mean tau (μs)",
                      font=dict(size=20),
                      template="plotly_white",
                      xaxis=dict(dtick=10))
    if yrange:
        fig.update_layout(yaxis_range=yrange)
    fig.write_image(file_path, width=500, height=500)


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


def make_matplot_legend(min, max, color_map):
    ax = plt.subplot()
    im = ax.imshow(np.arange(min, max, 10).reshape(int((max - min) / 10), 1), cmap=color_map)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="30%", pad=1)
    plt.colorbar(im, cax=cax, label="Height from center plane (nm)")
    plt.show()


def visualize_energy_plot(y, file_path):
    x = [i for i in range(len(y))]
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'))
    fig.update_layout(xaxis_title="Time (µs)",
                      yaxis_title="Energy(kcal/mol)",
                      font=dict(size=20),
                      template="plotly_white")
    fig.write_image(file_path)
