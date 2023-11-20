import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import output
import utils
from functools import partial
import raster


def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

    return pl_colorscale


def taus_tick_val(i, voxel_size, center, dtick):
    if i % dtick != 0:
        return ""
    return str(int((i - center) * voxel_size))


def visualize_creation(real_time_maps, rasterized_maps):
    times = raster.get_times_array(80, 80, 800, 10)
    # def get_title(x, y):
    #     return f"{(int(times[x, y]) / 1000):.3f}μs" if x != -1 else ""
    # mat2 = [(35, 44), (36, 44), (-1, -1), (43, 44), (44, 44),
    #         (35, 43), (36, 43), (-1, -1), (43, 43), (44, 43),
    #         (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
    #         (35, 36), (36, 36), (-1, -1), (43, 36), (44, 36),
    #         (35, 35), (36, 35), (-1, -1), (43, 35), (44, 35)]

    mat3 = [(35, 44), (36, 44), (37, 44), (-1, -1), (42, 44), (43, 44), (44, 44),
            (35, 43), (36, 43), (37, 43), (-1, -1), (42, 43), (43, 43), (44, 43),
            (35, 42), (36, 42), (37, 42), (-1, -1), (42, 42), (43, 42), (44, 42),
            (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
            (35, 37), (36, 37), (37, 37), (-1, -1), (42, 37), (43, 37), (44, 37),
            (35, 36), (36, 36), (37, 36), (-1, -1), (42, 36), (43, 36), (44, 36),
            (35, 35), (36, 35), (37, 35), (-1, -1), (42, 35), (43, 35), (44, 35)]
    times3 = [
        1810, 1820, 1830, -1, 1880, 1890, 1900, -1, -2, -1, -1, -1, -1,
        1610, 1620, 1630, -1, 1680, 1690, 1700, -1, -1, -1, -1, -1, -1,
        1410, 1420, 1430, -1, 1480, 1490, 1500, -1, -1, -1, -1, -1, -1,
        - 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        410, 420, 430, -1, 480, 490, 500, -1, -1, -1, -1, -1, -1,
        210, 220, 230, -1, 280, 290, 300, -1, -1, -1, -1, -1, -1,
        10, 20, 30, -1, 80, 90, 101 - 1, -1, -1, -1, -1, -1,
    ]
    squares = [(0, 9), (1, 9), (2, 9), (-1, -1), (7, 9), (8, 9), (9, 9),
               (0, 8), (1, 8), (2, 8), (-1, -1), (7, 8), (8, 8), (9, 8),
               (0, 7), (1, 7), (2, 7), (-1, -1), (7, 7), (8, 7), (9, 7),
               (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
               (0, 2), (1, 2), (2, 2), (-1, -1), (7, 2), (8, 2), (9, 2),
               (0, 1), (1, 1), (2, 1), (-1, -1), (7, 1), (8, 1), (9, 1),
               (0, 0), (1, 0), (2, 0), (-1, -1), (7, 0), (8, 0), (9, 0)]
    specs = [[{}, {}, {}, {}, {}, {}, {}, {}, {"colspan": 4, "rowspan": 3}, {}, {}, {}, {}],
             [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
             [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
             [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
             [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
             [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
             [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]]

    def get_subplot_title(t):
        if t == -1:
            return ""
        if t == -2:
            return 'Simulated AFM Image'
        else:
            return f"{(t / 1000):.3f} μs"

    fig = make_subplots(rows=7,
                        cols=13,
                        subplot_titles=[get_subplot_title(t) for t in times3],
                        column_widths=[1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13,
                                       1 / 13, 1 / 13, 1 / 13],
                        row_heights=[1 / 7, 1 / 7, 1 / 7, 0, 1 / 7, 1 / 7, 1 / 7],
                        specs=specs
                        # shared_yaxes='all'
                        )
    cm = plt.get_cmap("jet")
    cm = matplotlib_to_plotly(cm, 255)

    fig.add_trace(
        go.Heatmap(z=np.swapaxes(rasterized_maps[0], 0, 1)[35:45, 35:45] - 40,
                   colorbar={"title": 'Height (nm)', "len": 0.5, "thickness": 50},
                   colorscale=cm, zmax=21, zmin=0, xgap=2, ygap=2),
        row=1, col=9)

    def get_map_of_index(x, y):
        return np.swapaxes(real_time_maps[int(times[x, y] / 10)], 0, 1)[35:45, 35:45] - 40

    def add_to_fig(x, y, row, col):
        fig.add_trace(
            go.Heatmap(z=get_map_of_index(x, y),
                       colorbar={"title": 'Height (nm)', "len": 0.5, "thickness": 50},
                       colorscale=cm, zmax=21,
                       zmin=0, xgap=1, ygap=1),
            row=row, col=col)

    shapes = []

    def add_to_shapes(x, y, row, col):
        shapes.append(
            {'type': "rect", 'x0': x - 0.5, 'y0': y - 0.5, 'x1': x + 0.5, 'y1': y + 0.5,
             'xref': f'x{(row - 1) * 13 + col}',
             'yref': f'y{(row - 1) * 13 + col}', "line": dict(width=9, color="Magenta")})

    for i, (x, y) in enumerate(mat3):
        row = int(i / 7) + 1
        col = i % 7 + 1
        if x != -1:
            add_to_fig(x, y, row, col)
            add_to_shapes(squares[i][0], squares[i][1], row, col)

    fig['layout'].update(shapes=shapes)
    fig.update_layout(font=dict(size=60))
    fig.update_annotations(font_size=65)

    fig.update_yaxes(tickmode="array",
                     tickvals=[i - 0.5 for i in range(1, 10, 2)],
                     ticktext=[i for i in range(-4, 6, 2)],
                     tick0=5,
                     scaleanchor="x",
                     tickfont=dict(size=46))
    fig.update_xaxes(tickmode="array",
                     tickvals=[i - 0.5 for i in range(1, 10, 2)],
                     ticktext=[i for i in range(-4, 6, 2)],
                     tickangle=0,
                     tickfont=dict(size=46))
    fig.update_yaxes(tickmode="array",
                     tickvals=[i - 0.5 for i in range(0, 11, 1)],
                     ticktext=[i for i in range(-5, 6, 1)],
                     tickangle=0,
                     row=1, col=9, scaleanchor=None,
                     tickfont=dict(size=60))
    fig.update_xaxes(tickmode="array",
                     tickvals=[i - 0.5 for i in range(0, 11, 1)],
                     ticktext=[i for i in range(-5, 6, 1)],
                     tickangle=0,
                     row=1, col=9,
                     tickfont=dict(size=60))

    # fig.update_layout(legend={"xanchor": "right", "x": 1.00})
    # fig.update_layout(yaxis=dict(scaleanchor='x'))

    fig.write_image("creation_fig.png", width=5500, height=3350)
