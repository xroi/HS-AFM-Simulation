import pickle

import matplotlib.pyplot as plt
import moviepy.editor as mp
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image, ImageDraw, ImageFont
from plotly.subplots import make_subplots

import auto_corr


def output_gif(args, maps, filename, z_center, min_z, max_z, timestamp_step=-1, max_frames=240, add_legend=False,
               crop_from_sides_px=0, draw_inner_circle_r=-1, draw_outer_circle_r=-1, as_images=False, as_mp4=False,
               frame_duration=42, add_scale=False):
    """z_center is the real center"""
    images = []
    if timestamp_step != -1:
        timestamp_font = ImageFont.truetype('arial.ttf', 60)
    if add_scale is True:
        scale_font = ImageFont.truetype('arial.ttf', 30)
    if add_legend:
        legend_fig = make_matplot_colorbar(0, max_z - min_z + 1, args["output_gif_color_map"])
        legend_im = fig2img(legend_fig)
        hpercent = (args["output_resolution_y"] / float(legend_im.size[1]))
        wsize = int((float(legend_im.size[0]) * float(hpercent)))
        legend_im = legend_im.resize((wsize, args["output_resolution_y"]), Image.Resampling.LANCZOS)
        legend_im = legend_im.crop((int((legend_im.size[0] / 2) + 80), 0, legend_im.size[0] - 250, legend_im.size[1]))
    for i, height_map in enumerate(maps):
        if crop_from_sides_px > 0:
            height_map = height_map[crop_from_sides_px:-crop_from_sides_px, crop_from_sides_px:-crop_from_sides_px]
        scaled_map = (height_map - min_z) / (max_z - 1 - min_z)
        cm = plt.get_cmap(args["output_gif_color_map"])
        data = cm(scaled_map)
        im = Image.fromarray((data[:, :, :3] * 255).astype(np.uint8), 'RGB')
        im = im.resize((args["output_resolution_y"], args["output_resolution_x"]), resample=Image.BOX).rotate(angle=90,
                                                                                                              expand=1)
        id = ImageDraw.Draw(im, "RGBA")
        pixel_size = args["output_resolution_x"] / (maps[0].shape[0] - crop_from_sides_px * 2)
        if timestamp_step != -1:
            id.text((30, 30), f"{(i * timestamp_step):.3f} μs", fill=(0, 0, 0, 255), font=timestamp_font)
        if draw_inner_circle_r != -1:
            r = draw_inner_circle_r * pixel_size
            id.ellipse([(im.size[0] / 2 - r),
                        (im.size[1] / 2 - r),
                        (im.size[0] / 2 + r),
                        (im.size[1] / 2 + r)],
                       outline=(0, 0, 0, 125), width=5)
        if draw_outer_circle_r != -1:
            r = draw_outer_circle_r * pixel_size
            id.ellipse([(im.size[0] / 2 - r), (im.size[1] / 2 - r),
                        (im.size[0] / 2 + r), (im.size[1] / 2 + r)],
                       outline=(0, 0, 0, 125), width=5)
        if add_scale is True:
            scale_text_coords = (im.size[0] - 7 * pixel_size, im.size[1] - 5 * pixel_size)
            scale_coords = [im.size[0] - 7 * pixel_size, im.size[1] - 3 * pixel_size,
                            im.size[0] - 2 * pixel_size, im.size[1] - 2 * pixel_size]
            id.text(scale_text_coords, f"5 nm", fill=(0, 0, 0, 255), font=scale_font)
            id.rectangle(scale_coords, fill="#000000")
        if add_legend:
            new_im = Image.new('RGB', (im.size[0] + legend_im.size[0], im.size[1]), (250, 250, 250))
            new_im.paste(im, (0, 0))
            new_im.paste(legend_im, (im.size[0], 0))
            im = new_im
        images.append(im)
        if i == max_frames:
            break
    if as_images:
        for i, im in enumerate(images):
            im.save(f"{filename}_{i}.png")
    else:
        # As gif
        images[0].save(filename, append_images=images[1:], save_all=True, duration=frame_duration, loop=0)
        if as_mp4:
            clip = mp.VideoFileClip(filename)
            clip.write_videofile(f"{filename}.mp4")


def save_pickle(non_rasterized_maps, needle_maps, args, file_name):
    save_dict = {'non_rasterized_maps': non_rasterized_maps, 'rasterized_maps': needle_maps, 'args': args}
    with open(file_name, 'wb') as f:
        pickle.dump(save_dict, f)


def load_pickle(file_name):
    """returns a dictionary with the following keys: non_rasterized_maps, rasterized_maps, args"""
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def visualize_taus(taus, voxel_size, min_x, max_x, min_y, max_y, center_x, center_y, dtick, file_path):
    cm = plt.get_cmap("jet")
    # cm = matplotlib_to_plotly(cm, 255, zero_color="(255, 255, 255)")
    cm = matplotlib_to_plotly(cm, 255, zero_color=None)
    voxel_size = voxel_size / 10
    fig = go.Figure()
    # fig.add_trace(
    #     go.Heatmap(z=np.fliplr(np.flipud(taus)), colorbar={"title": 'Tau (μs)'}, zmin=0, zmax=0.25, colorscale=cm))
    fig.add_trace(
        go.Heatmap(z=np.fliplr(np.flipud(taus)), colorbar={"title": 'Tau'}, zmin=0, zmax=0.4,
                   colorscale=cm))
    fig.layout.height = 500
    fig.layout.width = 500
    fig.update_layout(xaxis={
        "tickmode": 'array',
        "tickvals": [i - 0.5 for i in range(0, 41, 5)],
        "ticktext": [i for i in range(-20, 21, 5)]
    }, yaxis={
        "tickmode": 'array',
        "tickvals": [i - 0.5 for i in range(0, 41, 5)],
        "ticktext": [i for i in range(-20, 21, 5)]
    })
    fig.update_layout(title="",
                      yaxis={"title": 'Distance (nm)'},
                      xaxis={"title": 'Distance (nm)',
                             "tickangle": 0},
                      font=dict(size=20))
    fig.add_shape(
        {'type': "circle", 'x0': 8.5 - 0.5, 'y0': 8.5 - 0.5, 'x1': 31.5 + 0.5, 'y1': 31.5 + 0.5,
         'xref': f'x',
         'yref': f'y', "line": dict(width=2, color="Black"), 'opacity': 0.7}, )
    fig.add_shape(
        {'type': "circle", 'x0': 1 - 0.5, 'y0': 1 - 0.5, 'x1': 39 + 0.5, 'y1': 39 + 0.5,
         'xref': 'x',
         'yref': 'y', "line": dict(width=2, color="Black"), 'opacity': 0.7}, )
    fig.update_layout(yaxis=dict(scaleanchor='x'))
    fig.write_image(file_path, width=550, height=500)


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


def visualize_tcf_samples(acorrs, taus, dist_px, amount, file_path, show_model=True):
    factor = 0.01
    nlags = acorrs.shape[2]
    fig = make_subplots(rows=1,
                        cols=amount,
                        subplot_titles=[
                            f"({int(acorrs.shape[0] / 2) + i * dist_px},{int(acorrs.shape[1] / 2)}) tau="
                            f"{'%.3f' % (taus[int(acorrs.shape[0] / 2) + i * dist_px, int(acorrs.shape[1] / 2)] * factor)}μs"
                            for i
                            in range(amount)],
                        shared_yaxes='all')
    x = [i * factor for i in range(nlags)]
    for i in range(amount):
        x_coord = int(acorrs.shape[0] / 2) + i * dist_px
        y_coord = int(acorrs.shape[1] / 2)
        fig.add_trace(
            go.Scatter(x=x, y=acorrs[x_coord, y_coord, :], mode='markers', line_color='#e00000'),
            row=1,
            col=i + 1
        )
        if show_model:
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
    fig.update_yaxes(type="log")
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


def make_matplot_colorbar(min, max, color_map):
    ax = plt.subplot()
    im = ax.imshow(np.arange(min, max, 5).reshape(int((max - min) / 5) + 1, 1), cmap=color_map)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="30%", pad=1)
    plt.colorbar(im, cax=cax, label="Height (nm)")
    # plt.show()
    return plt


def make_plotly_colorbar(min, max, color_map_name):
    # Create a heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[[min, max]],
        colorscale=matplotlib_to_plotly(plt.get_cmap(color_map_name), 255),
        showscale=True  # This shows the colorbar
    ))
    # Hide the heatmap
    # fig.data[0].update(z=[[None, None]], showscale=True)
    fig.show()


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def visualize_energy_plot(y, file_path):
    x = [i for i in range(len(y))]
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'))
    fig.update_layout(xaxis_title="Time (µs)",
                      yaxis_title="Energy(kcal/mol)",
                      font=dict(size=20),
                      template="plotly_white")
    fig.write_image(file_path)


def matplotlib_to_plotly(cmap, pl_entries, zero_color=None):
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        if k == 0 and zero_color != None:
            pl_colorscale.append([k * h, 'rgb' + zero_color])
        else:
            C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
            pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

    return pl_colorscale
