import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings(action='ignore', category=UserWarning)


class Single_LightGBM:
    def __init__(self, data, val_ratio, params=None, lags=[1, 2, 3]):
        if params is None:
            params = {}
        self.params = params
        self.model = None

        self.data = data
        self.lags = lags

        # Put the time series data in dataframe
        self.df = pd.DataFrame(data, columns=["y"])
        # Add lagged values as features
        self.df = self.add_lagged_values(self.df, "y", lags=self.lags)
        self.val_size = int(len(self.data) * val_ratio)

        # Record MSE values
        self.mse = []
        self.mse_val = []

    @staticmethod
    def add_lagged_values(df, col_name, lags):
        for lag in lags:
            df[col_name + '_lag_' + str(lag)] = df[col_name].transform(lambda x: x.shift(lag, fill_value=0))
        return df

    def train(self, batch_size=0, num_boost_round=2, early_stopping_rounds=None):
        if batch_size > 0:
            batch_indexes = np.random.choice(np.arange(len(self.data) - self.val_size), (64,))
        else:
            batch_indexes = np.arange(len(self.data) - self.val_size)

        x_train = self.df.drop("y", axis=1).values[:-self.val_size, :][batch_indexes, :]
        y_train = self.df["y"].values[:-self.val_size, :][batch_indexes, :]

        # Fit a regression tree to data
        train_dataset = lgb.Dataset(x_train, label=y_train)
        with tqdm(total=num_boost_round) as pbar:
            self.model = lgb.train(self.params, train_set=train_dataset,
                                   valid_sets=[train_dataset, ],
                                   verbose_eval=0, keep_training_booster=True, num_boost_round=num_boost_round,
                                   early_stopping_rounds=early_stopping_rounds,
                                   callbacks=[lgb.Callback().on_iteration(lambda i, _: pbar.update(1))])

    def predict(self, X):
        return self.model.predict(X)

    def forecast(self, X, num_periods=1, return_conf_int=False):
        forecast = self.model.predict(X, num_periods=num_periods)
        if return_conf_int:
            forecast_ci = self.model.predict(X, num_periods=num_periods, return_std=True)
            return forecast, forecast_ci
        else:
            return forecast
