import numpy as np
import plotly.graph_objects as go
import output
import utils
import height_funcs


def do_multi_analysis(paths, output_prefix):
    real_time_maps = []
    ring_means = []
    args = output.load_pickle(paths[0])["args"]
    args2 = args
    args2["tip_radius_px"] = 0
    for path in paths:
        real_time_maps.append(output.load_pickle(path)["real_time_maps"])
    for i in range(len(paths)):
        ring_means.append(utils.get_ring_means_array(np.mean(np.dstack(real_time_maps[i]), axis=2),
                                                     real_time_maps[i][0].shape[0] / 2,
                                                     real_time_maps[i][0].shape[1] / 2) - 40)
    envelope_heights = np.array(
        [height_funcs.get_slab_top_z(x, 20, (20, 20, 40), args2) - 40
         for x in range(20, 20 + 26)])
    # ring_means = [ring_means[0], ring_means[-1]]
    visualize_height_by_radial_distance(ring_means, envelope_heights,
                                        ["0μM",
                                         "10μM", "20μM", "50μM", "100μM", "200μM", "Multi Passive",
                                         "Multi NTR"
                                         ],
                                        ["#CCCCFF",
                                         "#8888FF", "#4644FF", "#0B00FF", "#0500D5", " #000080", "#CCCCFF",
                                         "#fa0004"
                                         ],
                                        f"outputs/25-11-2023-long/multi_height_radial_with_multi_ntr.png",
                                        sym=True,
                                        yrange=[0, 20])
    # visualize_mean_0_height_by_concentration([0,
    #                                           10, 20, 50, 100, 200
    #                                           ],
    #                                          [arr[0] for arr in ring_means[:-1]],
    #                                          ring_means[-2][0],
    #                                          f"outputs/25-11-2023-long/{output_prefix}_mean_zero_heights.png")


def visualize_mean_0_height_by_concentration(x, y, multi_passive_y, file_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers'))
    fig.update_layout(xaxis_title="NTR Concentration (μM)",
                      yaxis_title="Height (nm)",
                      font=dict(size=20),
                      template="plotly_white")
    fig.add_hline(y=7.5, line_width=1, line_dash="dash", line_color="Black", annotation_text="Nuclear Envelope",
                  annotation_position="top left")
    fig.add_hline(y=multi_passive_y, line_width=1, line_dash="dash", line_color="#8888FF",
                  annotation_text="Multi Passive",
                  annotation_position="top left")
    # fig.add_hline(y=multi_ntr_y, line_width=1, line_dash="dash", line_color="Red",
    #               annotation_text="Multi NTR mean height at (0,0)",
    #               annotation_position="top left")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#b5b4b1', zerolinecolor='#b5b4b1')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#b5b4b1', linecolor='#b5b4b1', zerolinecolor='#b5b4b1')
    fig.update_layout(xaxis_range=[0, 210], yaxis_range=[2.9, 8.1])
    fig.write_image(file_path, width=500, height=500)


def visualize_height_by_radial_distance(ring_means, envelope_heights, names, colors, file_path, sym=False, yrange=None):
    """ring means is an array of 2d arrays"""
    fig = go.Figure()
    max_r = len(ring_means[0])
    if sym:
        x = [i for i in range(-max_r + 1, max_r)]
        y = [np.concatenate((np.flip(ring_mean)[:-1], ring_mean)) for ring_mean in ring_means]
        envelope_heights = np.concatenate((np.flip(envelope_heights)[:-1], envelope_heights))
    else:
        x = [i for i in range(max_r)]
        y = ring_means
    for i in range(len(names)):
        fig.add_trace(go.Scatter(x=x, y=y[i], mode='lines+markers', opacity=1, name=names[i],
                                 line=dict(width=6, dash='dash' if i == 6 else None),
                                 marker=dict(size=8, color=colors[i])))
    fig.update_layout(xaxis_title="Distance from center (nm)",
                      yaxis_title="Height (nm)",
                      font=dict(size=40),
                      template="plotly_white",
                      xaxis=dict(dtick=2.5),
                      yaxis=dict(dtick=2.5))
    fig.add_trace(
        go.Scatter(x=list(range(-25, 26)), y=envelope_heights, mode='lines', line=dict(width=12, color="#2e2f30"),
                   name="Nuclear Envelope", fill='tozeroy', fillcolor='#2e2f30'))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#9e9d99', zerolinecolor='#9e9d99')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#9e9d99')
    fig.add_hline(y=7.5, line_width=0, line_dash="dash", line_color="Black", annotation_text="Nuclear Envelope",
                  annotation_position="top left")
    fig.update_yaxes(scaleratio=1)
    if yrange:
        fig.update_layout(yaxis_range=yrange, xaxis_range=[-25, 25])
    # 700 pre
    fig.write_image(file_path, width=3000, height=1100)


if __name__ == "__main__":
    do_multi_analysis(["outputs/25-11-2023-long/0uM.pickle",
                       "outputs/25-11-2023-long/10uM.pickle",
                       "outputs/25-11-2023-long/20uM.pickle",
                       "outputs/25-11-2023-long/50uM.pickle",
                       "outputs/25-11-2023-long/100uM.pickle",
                       "outputs/25-11-2023-long/200uM.pickle",
                       "outputs/25-11-2023-long/multi-passive.pickle",
                       "outputs/25-11-2023-long/multi-ntr.pickle"
                       ], "multi-analysis")
