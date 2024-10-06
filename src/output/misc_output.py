import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import raster
from output import output


def taus_tick_val(i, voxel_size, center, dtick):
    if i % dtick != 0:
        return ""
    return str(int((i - center) * voxel_size))


def visualize_creation(real_time_maps, rasterized_maps):
    color_scale = (0, 13)
    raster_i = 6
    times = raster.get_times_array(40, 40, 400, 10, 0, raster_i)
    shapes = []
    # def get_title(x, y):
    #     return f"{(int(times[x, y]) / 1000):.3f}μs" if x != -1 else ""
    # mat2 = [(35, 44), (36, 44), (-1, -1), (43, 44), (44, 44),
    #         (35, 43), (36, 43), (-1, -1), (43, 43), (44, 43),
    #         (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
    #         (35, 36), (36, 36), (-1, -1), (43, 36), (44, 36),
    #         (35, 35), (36, 35), (-1, -1), (43, 35), (44, 35)]

    # for 80x80
    # mat3 = [(35, 44), (36, 44), (37, 44), (-1, -1), (42, 44), (43, 44), (44, 44),
    #         (35, 43), (36, 43), (37, 43), (-1, -1), (42, 43), (43, 43), (44, 43),
    #         (35, 42), (36, 42), (37, 42), (-1, -1), (42, 42), (43, 42), (44, 42),
    #         (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
    #         (35, 37), (36, 37), (37, 37), (-1, -1), (42, 37), (43, 37), (44, 37),
    #         (35, 36), (36, 36), (37, 36), (-1, -1), (42, 36), (43, 36), (44, 36),
    #         (35, 35), (36, 35), (37, 35), (-1, -1), (42, 35), (43, 35), (44, 35)]

    # for40x40
    mat3 = [(15, 24), (16, 24), (17, 24), (-1, -1), (22, 24), (23, 24), (24, 24),
            (15, 23), (16, 23), (17, 23), (-1, -1), (22, 23), (23, 23), (24, 23),
            (15, 22), (16, 22), (17, 22), (-1, -1), (22, 22), (23, 22), (24, 22),
            (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
            (15, 17), (16, 17), (17, 17), (-1, -1), (22, 17), (23, 17), (24, 17),
            (15, 16), (16, 16), (17, 16), (-1, -1), (22, 16), (23, 16), (24, 16),
            (15, 15), (16, 15), (17, 15), (-1, -1), (22, 15), (23, 15), (24, 15)]

    # times3 = [
    #     1810, 1820, 1830, -1, 1880, 1890, 1900, -1, -2, -1, -1, -1, -1,
    #     1610, 1620, 1630, -1, 1680, 1690, 1700, -1, -1, -1, -1, -1, -1,
    #     1410, 1420, 1430, -1, 1480, 1490, 1500, -1, -1, -1, -1, -1, -1,
    #     - 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    #     410, 420, 430, -1, 480, 490, 500, -1, -1, -1, -1, -1, -1,
    #     210, 220, 230, -1, 280, 290, 300, -1, -1, -1, -1, -1, -1,
    #     10, 20, 30, -1, 80, 90, 101 - 1, -1, -1, -1, -1, -1,
    # ]
    # new_times = raster.get_times_array(40, 40, 400, 10, 0, 0)
    # times3 = [new_times[x, y] if x != -1 else -1 for (x, y) in mat3]
    times3 = [19360.0, 19370.0, 19380.0, -1, 19430.0, 19440.0, 19450.0, -1, -2, -1, -1, -1, -1,
              18560.0, 18570.0, 18580.0, -1, 18630.0, 18640.0, 18650.0, -1, -1, -1, -1, -1, -1,
              17760.0, 17770.0, 17780.0, -1, 17830.0, 17840.0, 17850.0, -1, -1, -1, -1, -1, -1,
              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
              13760.0, 13770.0, 13780.0, -1, 13830.0, 13840.0, 13850.0, -1, -1, -1, -1, -1, -1,
              12960.0, 12970.0, 12980.0, -1, 13030.0, 13040.0, 13050.0, -1, -1, -1, -1, -1, -1,
              12160.0, 12170.0, 12180.0, -1, 12230.0, 12240.0, 12250.0 - 1, -1, -1, -1, -1, -1, ]

    squares = [(0, 9), (1, 9), (2, 9), (-1, -1), (7, 9), (8, 9), (9, 9),
               (0, 8), (1, 8), (2, 8), (-1, -1), (7, 8), (8, 8), (9, 8),
               (0, 7), (1, 7), (2, 7), (-1, -1), (7, 7), (8, 7), (9, 7),
               (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1),
               (0, 2), (1, 2), (2, 2), (-1, -1), (7, 2), (8, 2), (9, 2),
               (0, 1), (1, 1), (2, 1), (-1, -1), (7, 1), (8, 1), (9, 1),
               (0, 0), (1, 0), (2, 0), (-1, -1), (7, 0), (8, 0), (9, 0)]
    r = 0.1
    specs = [[{}, {}, {}, {}, {}, {}, {}, {}, {"colspan": 4, "rowspan": 3}, {}, {}, {}, {}],
             [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
             [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
             [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
             [{}, {}, {}, {}, {}, {}, {}, {}, {"colspan": 4, "rowspan": 3}, {}, {}, {}, {}],
             [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
             [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]]

    def get_subplot_title(t):
        if t == -1:
            return ""
        if t == -2:
            return 'Simulated Rasterized AFM Image'
        else:
            return f"{int(t)} ns"

    fig = make_subplots(rows=7,
                        cols=13,
                        subplot_titles=[get_subplot_title(t) for t in times3],
                        column_widths=[1 / 13, 1 / 13, 1 / 13, 1 / 20, 1 / 13, 1 / 13, 1 / 13, 1 / 30, 1 / 13, 1 / 13,
                                       1 / 13, 1 / 13, 1 / 13],
                        row_heights=[1 / 7, 1 / 7, 1 / 7, 0, 1 / 7, 1 / 7, 1 / 7],
                        specs=specs,
                        # shared_yaxes='all'
                        horizontal_spacing=0.02
                        )
    cm = plt.get_cmap("jet")
    cm = output.matplotlib_to_plotly(cm, 255)

    fig.add_trace(
        go.Heatmap(z=np.swapaxes(rasterized_maps[raster_i], 0, 1)[15:25, 15:25] - 40,
                   colorbar={"title": 'Height (nm)', "len": 0.5, "thickness": 50},
                   colorscale=cm, zmax=color_scale[1], zmin=color_scale[0], xgap=2, ygap=2),
        row=1, col=9)

    fig.add_trace(
        go.Heatmap(z=np.swapaxes(rasterized_maps[raster_i], 0, 1) - 40,
                   colorbar={"title": 'Height (nm)', "len": 0.5, "thickness": 50},
                   colorscale=cm, zmax=color_scale[1], zmin=color_scale[0], xgap=2, ygap=2),
        row=5, col=9)
    shapes.append(
        {'type': "rect", 'x0': 15 - 0.5, 'y0': 15 - 0.5, 'x1': 24 + 0.5, 'y1': 24 + 0.5,
         'xref': f'x{(5 - 1) * 13 + 9}',
         'yref': f'y{(5 - 1) * 13 + 9}', "line": dict(width=20, color="Magenta"), 'opacity': 0.8})
    shapes.append(
        {'type': "circle", 'x0': 8.5 - 0.5, 'y0': 8.5 - 0.5, 'x1': 31.5 + 0.5, 'y1': 31.5 + 0.5,
         'xref': f'x{(5 - 1) * 13 + 9}',
         'yref': f'y{(5 - 1) * 13 + 9}', "line": dict(width=7, color="Black"), 'opacity': 0.8}, )
    shapes.append(
        {'type': "circle", 'x0': 1 - 0.5, 'y0': 1 - 0.5, 'x1': 39 + 0.5, 'y1': 39 + 0.5,
         'xref': f'x{(5 - 1) * 13 + 9}',
         'yref': f'y{(5 - 1) * 13 + 9}', "line": dict(width=7, color="Black"), 'opacity': 0.8}, )

    def get_map_of_index(x, y):
        return np.swapaxes(real_time_maps[int(times[x, y] / 10)], 0, 1)[15:25, 15:25] - 40

    def add_to_fig(x, y, row, col):
        fig.add_trace(
            go.Heatmap(z=get_map_of_index(x, y),
                       colorbar={"title": 'Height (nm)', "len": 0.5, "thickness": 50},
                       colorscale=cm, zmax=color_scale[1],
                       zmin=color_scale[0], xgap=1, ygap=1),
            row=row, col=col)

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
    fig.update_layout(font=dict(size=68))
    fig.update_annotations(font_size=68)

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
                     tickfont=dict(size=60)
                     )
    fig.update_yaxes(tickmode="array",
                     tickvals=[i - 0.5 for i in range(0, 41, 5)],
                     ticktext=[i for i in range(-20, 21, 5)],
                     tickangle=0,
                     row=5, col=9, scaleanchor=None,
                     tickfont=dict(size=60))
    fig.update_xaxes(tickmode="array",
                     tickvals=[i - 0.5 for i in range(0, 41, 5)],
                     ticktext=[i for i in range(-20, 21, 5)],
                     tickangle=0,
                     row=5, col=9,
                     tickfont=dict(size=60))

    # fig.update_layout(legend={"xanchor": "right", "x": 1.00})
    # fig.update_layout(yaxis=dict(scaleanchor='x'))

    fig.write_image("creation_fig.png", width=5500, height=3350)
