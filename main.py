import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm


class HybridModel:
    def __init__(self, data, lags, lr, val_ratio=0.2, learner="mlp", residuals=[]):
        self.data = data
        self.lags = lags
        self.lr = lr

        self.learner = learner.lower()
        assert self.learner in ["mlp", "lgbm"]

        # Put the time series data in dataframe
        self.df = pd.DataFrame(data, columns=["y"])
        # Add lagged values as features
        self.df = self.add_lagged_values(self.df, "y", lags=self.lags)
        self.df_errors = None

        self.val_size = int(len(self.data) * val_ratio)

        # Record models and residuals
        self.models = []
        self.residuals = residuals

        # Record MSE values
        self.mse = []
        self.mse_val = []

        self.pred_init = None

        self.m = 0
        self.alpha = 0.0

        self.best_model_idx = -1

    @staticmethod
    def add_lagged_values(df, col_name, lags):
        for lag in lags:
            df[col_name + '_lag_' + str(lag)] = df[col_name].transform(lambda x: x.shift(lag, fill_value=0))
        return df

    def train(self, iterations, train_params, early_stop_tolerance=None, batch_size=0):
        y_true = self.df["y"].values

        for i in tqdm(range(iterations)):
            # Use the average value of training data as the initial guess
            if i == 0:
                self.pred_init = np.mean(self.data)
                residuals = y_true - self.pred_init
                self.residuals.append(residuals)
            else:
                # Set up the dataframe that contains past residuals as features
                self.df_errors = pd.concat([self.df,
                                            self.add_lagged_values(pd.DataFrame(self.residuals[-1], columns=["e"]), "e",
                                                                   lags=self.lags).drop("e", axis=1)], axis=1)

                if batch_size > 0:
                    batch_indexes = np.random.choice(np.arange(len(self.data) - self.val_size), (64,))
                else:
                    batch_indexes = np.arange(len(self.data) - self.val_size)

                x_train = self.df_errors.drop("y", axis=1).values[:-self.val_size, :][batch_indexes, :]
                self.m = self.alpha * self.m + (1 - self.alpha) * self.residuals[-1][:-self.val_size]
                y_train = self.m[batch_indexes]

                if self.learner == "mlp":
                    # Fit MLP regressor to data
                    model_mlp = MLPRegressor(**train_params).fit(x_train, y_train)
                    self.models.append(model_mlp)
                elif self.learner == "lgbm":
                    # Fit a regression tree to data
                    train_dataset = lgb.Dataset(x_train, label=y_train)
                    model_lgbm = lgb.train(train_params, train_set=train_dataset, valid_sets=[train_dataset, ],
                                           verbose_eval=0)
                    self.models.append(model_lgbm)

                # Calculate and record residuals
                y_hat = self.predict(self.df, fast_train=True)
                residuals = y_true - y_hat
                self.residuals.append(residuals)

                self.mse.append(mse(y_true[:-self.val_size], y_hat[:-self.val_size]))
                self.mse_val.append(mse(y_true[-self.val_size:], y_hat[-self.val_size:]))

                if early_stop_tolerance is not None:
                    if i > 2 * early_stop_tolerance:
                        if np.all(np.array(self.mse_val[-early_stop_tolerance:]) > np.max(
                                self.mse_val[-2 * early_stop_tolerance:-early_stop_tolerance])):
                            self.best_model_idx = np.argmin(self.mse_val)
                            break

    def predict(self, df, fast_train=False):
        y = df["y"].values
        errors = np.zeros(y.shape)
        y_hats = np.zeros(y.shape)

        if fast_train:
            errors = self.residuals[-1]
            df_errors = pd.concat([df,
                                   self.add_lagged_values(pd.DataFrame(errors, columns=["e"]), "e",
                                                          lags=self.lags).drop("e", axis=1)], axis=1)
            x = df_errors.drop("y", axis=1)
            if self.learner == "mlp":
                y_hats = np.sum([model.predict(x) * self.lr for model in self.models], axis=0) + self.pred_init
            elif self.learner == "lgbm":
                y_hats = np.sum([model.predict(x) for model in self.models], axis=0) + self.pred_init

        else:
            for i in range(len(y)):
                df_errors = pd.concat([df,
                                       self.add_lagged_values(pd.DataFrame(errors, columns=["e"]), "e",
                                                              lags=self.lags).drop("e", axis=1)], axis=1)
                x = df_errors.drop("y", axis=1).values

                if self.learner == "mlp":
                    y_hat = np.sum([model.predict(x[[i], :]) * self.lr for model in self.models],
                                   axis=0) + self.pred_init
                elif self.learner == "lgbm":
                    y_hat = np.sum([model.predict(x[[i], :]) for model in self.models], axis=0) + self.pred_init

                errors[i] = y[i] - y_hat
                y_hats[i] = y_hat

        return np.array(y_hats)

    def forecast(self, horizon, test_values, best_iter=False):
        model_idx = self.best_model_idx if best_iter else -1
        y = np.append(self.df["y"].values, test_values)
        df = pd.DataFrame(y, columns=["y"])
        df = self.add_lagged_values(df, "y", lags=self.lags)

        errors = np.zeros(y.shape)
        y_hats = np.zeros(y.shape)

        for i in range(len(y)):
            df_errors = pd.concat(
                [df,
                 self.add_lagged_values(pd.DataFrame(errors, columns=["e"]), "e", lags=self.lags).drop("e", axis=1)],
                axis=1)
            x = df_errors.drop("y", axis=1).values

            if self.learner == "mlp":
                y_hat = np.sum([model.predict(x[[i], :]) * self.lr for model in self.models[:model_idx]],
                               axis=0)
            elif self.learner == "lgbm":
                y_hat = np.sum([model.predict(x[[i], :]) for model in self.models[:model_idx]], axis=0)

            errors[i] = y[i] - y_hat
            y_hats[i] = y_hat

        return np.array(y_hats)[-horizon:]


def ARIMA(phi=np.array([0]), theta=np.array([0]), d=0, t=0, mu=0, sigma=1, n=20, burn=5, init=None):
    """ Simulate data from ARMA model (eq. 1.2.4):

    z_t = phi_1*z_{t-1} + ... + phi_p*z_{t-p} + a_t + theta_1*a_{t-1} + ... + theta_q*a_{t-q}

    with d unit roots for ARIMA model.

    Arguments:
    phi -- array of shape (p,) or (p, 1) containing phi_1, phi2, ... for AR model
    theta -- array of shape (q) or (q, 1) containing theta_1, theta_2, ... for MA model
    d -- number of unit roots for non-stationary time series
    t -- value deterministic linear trend
    mu -- mean value for normal distribution error term
    sigma -- standard deviation for normal distribution error term
    n -- length time series
    burn -- number of discarded values because series beginns without lagged terms

    Return:
    x -- simulated ARMA process of shape (n, 1)
    """

    # add "theta_0" = 1 to theta
    theta = np.append(1, theta)

    # set max lag length AR model
    p = phi.shape[0]

    # set max lag length MA model
    q = theta.shape[0]

    # simulate n + q error terms
    a = np.random.normal(mu, sigma, (n + max(p, q) + burn, 1))

    # create array for returned values
    x = np.zeros((n + max(p, q) + burn, 1))

    # initialize first time series value
    x[0] = a[0]

    for i in range(1, x.shape[0]):
        AR = np.dot(phi[0: min(i, p)], np.flip(x[i - min(i, p): i], 0))
        MA = np.dot(theta[0: min(i + 1, q)], np.flip(a[i - min(i, q - 1): i + 1], 0))
        x[i] = AR + MA + t

    # add unit roots
    if d != 0:
        ARMA = x[-n:]
        m = ARMA.shape[0]
        z = np.zeros((m + 1, 1))  # create temp array

        for i in range(d):
            for j in range(m):
                z[j + 1] = ARMA[j] + z[j]
            ARMA = z[1:]
        x[-n:] = z[1:]

    return x[-n:]


def create_ARIMA_data(phi, theta, mu, sigma, t, n, test_size):
    y = ARIMA(phi=phi, theta=theta, mu=mu, sigma=sigma, n=n, t=t)
    y_train_new = y[:-test_size]
    y_test_new = y[-test_size:]
    return y_train_new, y_test_new


def run_simulation(num_simulations, num_iterations, lags, arima_params, learning_rate, params):
    mape_values = []
    mae_values = []
    mse_values = []
    for sim_num in tqdm(range(num_simulations)):
        np.random.seed(sim_num + 5)
        y_train, y_test = create_ARIMA_data(**arima_params)
        model = HybridModel(y_train, lags=lags, lr=learning_rate, learner="mlp")
        model.train(iterations=num_iterations, train_params=params, early_stop_tolerance=10)

        predicted_output = model.forecast(len(y_test), test_values=y_test)

        mape_values.append(mape(y_test, predicted_output))
        mae_values.append(mae(y_test, predicted_output))
        mse_values.append(mse(y_test, predicted_output))

        print(mse_values[-1])

    return mape_values, mae_values, mse_values


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=UserWarning)

    np.random.seed(9)

    test_size = 100

    phi = np.array([0.4, -0.5])
    theta = np.array([0.3, 0.2])
    mu = 0
    sigma = 1
    t = 0
    n = 700

    # Use Real Dataset
    # train_df = pd.read_csv("input/Daily-train.csv")
    # test_df = pd.read_csv("input/Daily-test.csv")
    # train_df.index = train_df['V1']
    # train_df = train_df.drop('V1', axis=1)
    # test_df.index = test_df['V1']
    # test_df = test_df.drop('V1', axis=1)
    #
    # y = train_df.iloc[150, :].dropna().values
    # y = np.log1p(y)
    # y = y[1:] - y[:-1]

    # Use Synthetic Data
    y = ARIMA(phi=phi, theta=theta, mu=mu, sigma=sigma, n=n, t=t)
    np.random.seed(9)

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
    params_lgbm = {'metric': 'l2',
                   'learning_rate': 0.1,
                   "bagging_fraction": 1.0,
                   "bagging_freq": 0,
                   'max_depth': 2,
                   'num_leaves': 3,
                   'verbose': -1,
                   # "min_data_in_bin": 1,
                   "min_data_in_leaf": 1,
                   "lambda_l1": 0.0,
                   "lambda_l2": 0.5,
                   "boost_from_average": False,
                   "num_boost_round": 2
                   }

    model = HybridModel(data=y_train_new, lags=[1, 2, 3], lr=0.1, learner="lgbm", val_ratio=0.2)
    model.train(iterations=500, train_params=params_lgbm, early_stop_tolerance=10)

    predicted_output = model.forecast(len(y_test_new), test_values=y_test_new)

    print("Test results (w/ ground truth):  MAPE =", mape(y_test_new, predicted_output), ", MAE =",
          mae(y_test_new, predicted_output), ", MSE =", mse(y_test_new, predicted_output))

    plt.plot(predicted_output)
    plt.plot(y_test_new)
    plt.legend(["Predicted (w/ Ground truth)", "Ground truth"])
    plt.show()

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
