import os
import pickle
import warnings

import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse

warnings.filterwarnings(action='ignore', category=UserWarning)
from Hybrid_Model import HybridModel
from Single_LightGBM import Single_LightGBM


def Hybrid_Model_Pipeline(seed, phi, theta, y_train_new, iterations, train_params, early_stop_tolerance, y_test_new,
                          mape_list,
                          mae_list, mse_list):
    model_LGBM_MA = HybridModel(data=y_train_new, lags=[1, 2, 3], lr=0.1, learner="lgbm", val_ratio=0.2)
    model_LGBM_MA.train(iterations=iterations, train_params=train_params, early_stop_tolerance=early_stop_tolerance)
    predicted_output = model_LGBM_MA.forecast(len(y_test_new), test_values=y_test_new)

    print("Test results (w/ ground truth):  MAPE =", mape(y_test_new, predicted_output), ", MAE =",
          mae(y_test_new, predicted_output), ", MSE =", mse(y_test_new, predicted_output))
    mape_list.append(mape(y_test_new, predicted_output))
    mae_list.append(mae(y_test_new, predicted_output))
    mse_list.append(mse(y_test_new, predicted_output))

    folder_path = 'Hybrid_Model_Loss_Results_M4_Hourly'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(f'{folder_path}/mape_list.pkl', 'wb') as f:
        pickle.dump(mape_list, f)

    with open(f'{folder_path}/mae_list.pkl', 'wb') as f:
        pickle.dump(mae_list, f)

    with open(f'{folder_path}/mse_list.pkl', 'wb') as f:
        pickle.dump(mse_list, f)

    trace1 = go.Scatter(x=np.arange(len(predicted_output)),
                        y=predicted_output,
                        mode="lines",
                        name="Predicted(w/Ground Truth)",
                        marker=dict(color="blue"),
                        text=f"Test results (w/ ground truth):  MAPE ={mape(y_test_new, predicted_output)}, MAE = {mae(y_test_new, predicted_output)},MSE = {mse(y_test_new, predicted_output)}",
                        )

    trace2 = go.Scatter(x=np.arange(len(y_test_new)),
                        y=y_test_new.squeeze(),
                        mode="lines",
                        name="Ground Truth",
                        marker=dict(color="orange")
                        )
    data_trace = [trace1, trace2]
    layout = dict(
        # title=f'Synthetic Data seed = {seed}: Predictions vs. Ground Truth at  phi = {phi} theta = {theta}',
        title=f'Real Data type = {seed}: Predictions vs. Ground Truth for series_number {phi}',
        xaxis=dict(title="Data Number", ticklen=5, zeroline=False),
        yaxis=dict(title="Values", ticklen=5, zeroline=False))

    folder_path = 'Hybrid_Model_Results_Hourly'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig = dict(data=data_trace, layout=layout)
    # plot(fig, show_link=True, filename=f'{folder_path}/lgbm_ma_seed_{seed}_phi_{phi}_{theta}.html')
    plot(fig, show_link=True, filename=f'{folder_path}/lgbm_ma_type_{seed}_number_{phi}.html')


def Single_LightGBM_Pipeline(seed, phi, theta, y_train_new, iterations, train_params, early_stop_tolerance, y_test_new,
                             mape_list,
                             mae_list, mse_list):
    model_LGBM_Single = Single_LightGBM(data=y_train_new, lags=[1, 2, 3], val_ratio=0.2)
    model_LGBM_Single.train(iterations=iterations, train_params=train_params, early_stop_tolerance=early_stop_tolerance)
    predicted_output = Single_LightGBM.forecast(num_periods=len(y_test_new), X=y_test_new)

    print("Test results (w/ ground truth):  MAPE =", mape(y_test_new, predicted_output), ", MAE =",
          mae(y_test_new, predicted_output), ", MSE =", mse(y_test_new, predicted_output))
    mape_list.append(mape(y_test_new, predicted_output))
    mae_list.append(mae(y_test_new, predicted_output))
    mse_list.append(mse(y_test_new, predicted_output))

    folder_path = 'Single_LGBM_Results'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(f'{folder_path}/mape_list.pkl', 'wb') as f:
        pickle.dump(mape_list, f)

    with open(f'{folder_path}/mae_list.pkl', 'wb') as f:
        pickle.dump(mae_list, f)

    with open(f'{folder_path}/mse_list.pkl', 'wb') as f:
        pickle.dump(mse_list, f)

    trace1 = go.Scatter(x=np.arange(len(predicted_output)),
                        y=predicted_output,
                        mode="lines",
                        name="Predicted(w/Ground Truth)",
                        marker=dict(color="blue"),
                        text=f"Test results (w/ ground truth):  MAPE ={mape(y_test_new, predicted_output)}, MAE = {mae(y_test_new, predicted_output)},MSE = {mse(y_test_new, predicted_output)}",
                        )

    trace2 = go.Scatter(x=np.arange(len(y_test_new)),
                        y=y_test_new.squeeze(),
                        mode="lines",
                        name="Ground Truth",
                        marker=dict(color="orange")
                        )
    data_trace = [trace1, trace2]
    layout = dict(
        title=f'Synthetic Data seed = {seed}: Predictions vs. Ground Truth at  phi = {phi} theta = {theta}',
        xaxis=dict(title="Data Number", ticklen=5, zeroline=False),
        yaxis=dict(title="Values", ticklen=5, zeroline=False))

    fig = dict(data=data_trace, layout=layout)
    plot(fig, show_link=True, filename=f'{folder_path}/lgbm_ma_seed_{seed}_phi_{phi}_{theta}.html')
