import multiprocessing as mp
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings(action='ignore', category=UserWarning)
from Pipelines import Hybrid_Model_Pipeline

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=UserWarning)
    np.random.seed(12)
    seeds = np.arange(start=300, stop=330)
    lists_exists = os.path.isfile('mse_list.pkl')
    if lists_exists:
        with open('mape_list.pkl', 'rb') as f:
            mape_list = pickle.load(f)

        with open('mae_list.pkl', 'rb') as f:
            mae_list = pickle.load(f)

        with open('mse_list.pkl', 'rb') as f:
            mse_list = pickle.load(f)
    else:

        mse_list = []
        mape_list = []
        mae_list = []

        with open('mape_list.pkl', 'wb') as f:
            pickle.dump(mape_list, f)

        with open('mae_list.pkl', 'wb') as f:
            pickle.dump(mape_list, f)

        with open('mse_list.pkl', 'wb') as f:
            pickle.dump(mape_list, f)

    check_point_exists = os.path.isfile('seeds.pkl')

    if check_point_exists:
        with open('seeds.pkl', 'rb') as f:
            seeds = pickle.load(f)
    else:
        with open('seeds.pkl', 'wb') as f:
            pickle.dump(seeds, f)

    # for seed in seeds:
    data_path = "m4_hourly_selections"
    for filename in os.listdir(data_path):
        # if int(filename.split("_")[0])<212:
        #     continue
        # else:
            filepath = os.path.join(data_path, filename)
            print(f"Training with series: {filepath}")

            folder_path = 'Hybrid_Model_Results'
            with open('mape_list.pkl', 'rb') as f:
                mape_list = pickle.load(f)

            with open('mae_list.pkl', 'rb') as f:
                mae_list = pickle.load(f)

            with open('mse_list.pkl', 'rb') as f:
                mse_list = pickle.load(f)

                # Use Real Data

            scaler = MinMaxScaler()
            y = scaler.fit_transform(pd.read_csv(filepath, usecols=["y"])).squeeze()
            # y = pd.read_csv(filepath, usecols=["y"]).values.squeeze()
            test_size = 100
            # Train Test Split
            y_train_new = y[:-test_size]
            y_test_new = y[-test_size:]

            # model = LGBModel(y_train_new, lags=[1, 2, 3], iterations=1000)
            # model.train(learning_rate=0.005, trees_initial_model=2, max_depth=3, num_leaves=5, early_stop_tolerance=30)

            params_mlp = {
                "hidden_layer_sizes": (3,),
                "activation": "relu",
                "alpha": 0.0001,
                "learning_rate_init": 0.1,
                # "random_state": 1,
                "max_iter": 100
            }
            params_lgbm = {'boosting': 'gbdt',
                           'metric': 'l2',
                           'learning_rate': 0.1,
                           "bagging_fraction": 1,
                           "bagging_freq": 0,
                           'max_depth': 4,
                           'num_leaves': 15,
                           'verbose': -1,
                           # "min_data_in_bin": 1,
                           "min_data_in_leaf": 1,
                           "lambda_l1": 0.0,
                           "lambda_l2": 0.2,
                           "boost_from_average": False,
                           "num_boost_round": 5
                           }

            # model_LGBM_MA = HybridModel(data=y_train_new, lags=[1, 2, 3], lr=0.1, learner="lgbm", val_ratio=0.2)
            process_lgbm = mp.Process(target=Hybrid_Model_Pipeline,
                                      args=(
                                          "m4", filename.split('_')[0], "m4", y_train_new, 500, params_lgbm, 10, y_test_new,
                                          mape_list,
                                          mae_list, mse_list))

            process_lgbm.start()
            process_lgbm.join(timeout=60 * 45)

            if process_lgbm.is_alive():
                print("Timeout! Moving on to the next iteration...")
                process_lgbm.kill()
                continue

            # config = MLP_Config(**{
            #     'task_name': 'reg_weights',
            #     'learning_rate': 0.1,
            #     'regularizer': 'L2',
            #     'momentum': 0.9,
            #     'epochs': 1000,
            #     'early_stopping': False,
            #     'hidden_params': [(5, 4), (4, 5)],
            #     'lambda':[0.01],
            #     'sigma':[0.05]
            # })
            #
            # train_loader, val_loader, test_loader = create_pytorch_data()
            # checkpoint_cb = ModelCheckpoint(dirpath="ckpt", monitor='val_loss', mode='min')
            # callbacks = [checkpoint_cb]
            # if config.early_stopping:
            #     callbacks.append(EarlyStopping(monitor='val_loss', patience=10))

            # model_MLP = MLP(hidden_params=config.hidden_params, configs=config)
            #
            # trainer = pl.Trainer(max_epochs=config.epochs, callbacks=callbacks)
            # trainer.fit(model_MLP, train_dataloader=train_loader, val_dataloaders=val_loader)

            # plt.plot(predicted_output)
            # plt.plot(y_test_new)
            # plt.legend(["Predicted (w/ Ground truth)", "Ground truth"])
            # plt.show

print()
# plt.plot(model.mse)
# plt.plot(model.mse_val)
# plt.show()

# Multiple Simulations with Different Seeds
# arima_params = {"phi": np.array([0.8, -0.6]),
#                 "theta": np.array([0.7, -0.4, 0.3]),
#                 "mu": 0,
#                 "sigma": 1,
#                 "t": 0,
#                 "n": 700,
#                 "test_size": 100}
# mape_values, mae_values, mse_values = run_simulation(50, 300, [1, 2, 3], arima_params, 0.1, params_mlp)
#
# cum_mse = np.cumsum(mse_values) / (1 + np.arange(len(mse_values)))
# plt.plot(cum_mse)
