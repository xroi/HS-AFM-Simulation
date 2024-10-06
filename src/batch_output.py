import numpy as np
import plotly.graph_objects as go
from scipy.stats import sem

import auto_corr
import utils
from output import output



def visualize_height_by_radial_distance(ring_means_1, ring_means_2, envelope_heights, file_path, sym=False,
                                        yrange=None, type="std"):
    """ring means is an array of 2d arrays"""
    fig = go.Figure()
    max_r = len(ring_means_1[0])
    if sym:
        x = [i for i in range(-max_r + 1, max_r)]
        y_1 = [np.concatenate((np.flip(ring_mean)[:-1], ring_mean)) for ring_mean in ring_means_1]
        y_2 = [np.concatenate((np.flip(ring_mean)[:-1], ring_mean)) for ring_mean in ring_means_2]
        envelope_heights = np.concatenate((np.flip(envelope_heights)[:-1], envelope_heights))
    else:
        x = [i for i in range(max_r)]
        y_1 = ring_means_1
        y_2 = ring_means_2
    # for i in range(len(ring_means_1)):
    #     fig.add_trace(go.Scatter(x=x, y=y_1[i], mode='lines', opacity=1,
    #                              line=dict(width=2), marker=dict(size=2, color='#342feb'), showlegend=False))
    # for i in range(len(ring_means_2)):
    #     fig.add_trace(go.Scatter(x=x, y=y_2[i], mode='lines', opacity=1,
    #                              line=dict(width=2), marker=dict(size=2, color='#342feb'), showlegend=False))
    mean_y_1 = np.mean(np.stack(y_1, axis=0), axis=0)
    mean_y_2 = np.mean(np.stack(y_2, axis=0), axis=0)
    if type == "std":
        y_1_upper = mean_y_1 + np.std(np.stack(y_1, axis=0), axis=0)
        y_1_lower = mean_y_1 - np.std(np.stack(y_1, axis=0), axis=0)
        y_2_upper = mean_y_2 + np.std(np.stack(y_2, axis=0), axis=0)
        y_2_lower = mean_y_2 - np.std(np.stack(y_2, axis=0), axis=0)
    elif type == "sem":
        y_1_upper = mean_y_1 + sem(np.stack(y_1, axis=0), axis=0)
        y_1_lower = mean_y_1 - sem(np.stack(y_1, axis=0), axis=0)
        y_2_upper = mean_y_2 + sem(np.stack(y_2, axis=0), axis=0)
        y_2_lower = mean_y_2 - sem(np.stack(y_2, axis=0), axis=0)

    y_1_color = '#00d2eb'
    y_1_band_color = "rgba(0,0,0,0.5)"
    y_2_color = '#1100EB'
    y_2_band_color = "rgba(0,0,0,0.5)"
    fig.add_trace(go.Scatter(x=x, y=y_1_upper,
                             line=dict(width=0, color=y_1_color),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=mean_y_1, mode='lines', opacity=1,
                             line=dict(width=6, color=y_1_color), fill='tonexty', fillcolor=y_1_band_color,
                             name="0μM (n=50)"))
    fig.add_trace(go.Scatter(x=x, y=y_1_lower,
                             line=dict(width=0, color=y_1_color),
                             fillcolor=y_1_band_color,
                             fill='tonexty',
                             showlegend=False))

    fig.add_trace(go.Scatter(x=x, y=y_2_upper,
                             line=dict(width=0, color=y_2_color),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=mean_y_2, mode='lines', opacity=1,
                             line=dict(width=6, color=y_2_color), fill='tonexty', fillcolor=y_2_band_color,
                             name="200μM (n=50)"))
    fig.add_trace(go.Scatter(x=x, y=y_2_lower,
                             line=dict(width=0, color=y_2_color),
                             fillcolor=y_2_band_color,
                             fill='tonexty',
                             showlegend=False))

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
    fig.write_image(file_path, width=3000, height=700)


def visualize_tau_by_radial_distance(ring_means, file_path, sym=False, yrange=False, outer_radius_px=None,
                                     inner_radius_px=None):
    fig = go.Figure()
    max_r = len(ring_means[0])
    if sym:
        x = [i for i in range(-max_r + 1, max_r)]
        y = [np.concatenate((np.flip(ring_mean), ring_mean)) for ring_mean in ring_means]
    else:
        x = [i for i in range(max_r)]
        y = ring_means
    for i in range(len(ring_means)):
        fig.add_trace(go.Scatter(x=x, y=y[i], mode='lines', opacity=0.5, line_color='#2e15e8'))
    fig.add_trace(go.Scatter(x=x, y=np.mean(np.stack(y, axis=0), axis=0), mode='lines', opacity=1,
                             line_color='#eb0514'))
    if outer_radius_px and inner_radius_px:
        for val in [(inner_radius_px, "inner"), (outer_radius_px, "outer")]:
            fig.add_annotation(
                x=val[0],
                y=yrange[0],
                xref="x",
                yref="y",
                # text=val[1],
                showarrow=True,
                font=dict(
                    family="Courier New, monospace",
                    size=16,
                    color="#000000"
                ),
                align="center",
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#636363",
                # bordercolor="#000000",
                ax=0,
                ay=-30,
                opacity=0.8
            )
    fig.update_layout(xaxis_title="Distance from center (nm)",
                      yaxis_title="Mean tau (μs)",
                      font=dict(size=20),
                      template="plotly_white",
                      xaxis=dict(dtick=10),
                      showlegend=False)
    if yrange:
        fig.update_layout(yaxis_range=yrange)
    fig.write_image(file_path)


def do_batch_analysis(input_path, output_path, amount):
    real_time_maps = []
    ring_means = []
    args = output.load_pickle(f"{input_path}/0.pickle")["args"]
    for i in range(amount):
        real_time_maps.append(output.load_pickle(f"{input_path}/{i}.pickle")["real_time_maps"])
    for i in range(amount):
        ring_means.append(utils.get_ring_means_array(np.mean(np.dstack(real_time_maps[i]), axis=2),
                                                     real_time_maps[i][0].shape[0] / 2,
                                                     real_time_maps[i][0].shape[1] / 2) - 80)
    outer_radius_px = int(args["tunnel_radius_a"] / args["voxel_size_a"])
    inner_radius_px = int((args["tunnel_radius_a"] - (args["slab_thickness_a"] / 2)) / args["voxel_size_a"])
    visualize_height_by_radial_distance(ring_means, f"{output_path}_batch_height_radial_real_time.png", sym=True,
                                        yrange=[5, 20],
                                        outer_radius_px=outer_radius_px, inner_radius_px=inner_radius_px)

    real_time_acorrs = []
    taus = []
    tau_ring_means = []
    for i in range(amount):
        real_time_acorrs.append(auto_corr.temporal_auto_correlate(real_time_maps[i], 1))
        taus.append(auto_corr.calculate_taus(real_time_acorrs[i]))
        tau_ring_means.append(
            utils.get_ring_means_array(taus[i], real_time_maps[i][0].shape[0] / 2, real_time_maps[i][0].shape[1] / 2))
    visualize_tau_by_radial_distance(tau_ring_means, f"{output_path}_batch_tau_radial_real_time.png", sym=True,
                                     yrange=[-0.5, 1.5],
                                     outer_radius_px=outer_radius_px, inner_radius_px=inner_radius_px)


if __name__ == "__main__":
    do_batch_analysis("Outputs/12-10-2023-NTR-BATCH", "ntr", 10)
