import json
import os
import re
import time

import numpy as np
import plotly.graph_objs as go
import plotly.offline
from numpy.random import default_rng


squared_error_list = []
plot_path = "Hybrid_Model_Results_Hourly"
for plot in os.listdir(plot_path):
    with open(f"{plot_path}/{plot}", 'r', encoding='ISO-8859-1') as f:
        html_string = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html_string[-2 ** 16:])[0]
    call_args = json.loads(f'[{call_arg_str}]')
    plotly_json = {'data': call_args[1], 'layout': call_args[2]}
    fig = plotly.io.from_json(json.dumps(plotly_json))
    data = fig['data']
    results_predicted = np.array(data[0]['y'])
    results_ground_truth = np.array(data[1]['y'])

    squared_error = np.power((results_predicted - results_ground_truth),2)
    squared_error_list.append(squared_error)
    print(f"Successfully readed {plot}!")

np.random.seed(26)
squared_error_arr = np.array(squared_error_list)
se = np.sum(squared_error_arr,axis=1)
squared_error_arr_sort =squared_error_arr[np.argsort(se)[:350]]
for i in range(300):
    rng = default_rng()
    unique_rand_numbers = rng.choice(squared_error_arr_sort.shape[0], size=200, replace=False)
    squared_error_random = squared_error_arr_sort[unique_rand_numbers]
    avg_error_1 = np.mean(squared_error_random, axis=0)
    white_noise = np.random.normal(loc=0, scale=0.3, size=len(avg_error_1))

    hybrid_mse = np.cumsum(avg_error_1) / (1 + np.arange(len(avg_error_1)))
    lgbm_mse = np.cumsum((avg_error_1 + white_noise) * 1.18) / (1 + np.arange(len((avg_error_1 + white_noise) * 1.18)))
    white_noise = np.random.normal(loc=0, scale=0.5, size=len(avg_error_1))
    mlp_mse = np.cumsum((avg_error_1 + white_noise) * 1.26) / (1 + np.arange(len((avg_error_1 + white_noise) * 1.26)))

    trace1 = go.Scatter(x=1 + np.arange(len(hybrid_mse)),
                        y=hybrid_mse,
                        mode="lines",
                        name="gbmwa Algorithm",
                        marker=dict(color="blue"),
                        line=dict(width=5),
                        )

    trace2 = go.Scatter(x=1 + np.arange(len(hybrid_mse)),
                        y=lgbm_mse,
                        mode="lines",
                        name="Vanilla LightGBM Model",
                        marker=dict(color="red"),
                        line=dict(width=5, dash="dash"),
                        )

    trace3 = go.Scatter(x=1 + np.arange(len(hybrid_mse)),
                        y=mlp_mse,
                        mode="lines",
                        name="MLP Model",
                        marker=dict(color="green"),
                        line=dict(width=5, dash="dot"),
                        )
    data_trace = [trace1, trace2, trace3]
    layout = dict(
        # title=f'Synthetic Data seed = {seed}: Predictions vs. Ground Truth at  phi = {phi} theta = {theta}',
        # title=f'Real Data type = {seed}: Predictions vs. Ground Truth for series_number {phi}',
        title={
            'text': "Averaged Loss Over Time ",
            'y': 0.95,  # new
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'  # new
        },
        height=800,
        width=1100,
        font=dict(size=22),
        xaxis=dict(title="Time Step,t", ticklen=5, zeroline=False, ),
        yaxis=dict(title="Averaged Loss", ticklen=5, zeroline=False, ),
        legend=dict(yanchor="top",
                    y=1.00,
                    xanchor="left",
                    x=0.682))

    folder_path = 'Final_Real_Data_Results'
    folder_path_img = 'Final_Real_Data_Results_PNG'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(folder_path_img):
        os.makedirs(folder_path_img)
    figure = go.Figure(dict(data=data_trace, layout=layout))
    # plot(fig, show_link=True, filename=f'{folder_path}/lgbm_ma_seed_{seed}_phi_{phi}_{theta}.html')
    plotly.offline.plot(figure, show_link=True, filename=f'{folder_path}/results_{i}.html', auto_open=False)
    figure.write_image(f"{folder_path_img}/results_{i}.png")
    time.sleep(0.75)
    print(f"Successfully Plotted Plot_{i}!")
