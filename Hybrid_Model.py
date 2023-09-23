import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm


class HybridModel:
    def __init__(self, data, lags, lr, val_ratio=0.2, learner="mlp", residuals=[]):
        self.data = data
        self.lags = lags
        self.lr = lr
        self.epsilon = np.logspace(-5, -1, 10)
        self.error_check = True
        self.learner = learner.lower()
        assert self.learner in ["mlp", "lgbm"]

        # Put the time series data in dataframe
        self.df = pd.DataFrame(data, columns=["y"])
        # Add lagged values as features
        self.df = self.add_lagged_values(self.df, "y", lags=self.lags)
        self.df_errors = None

        self.val_size = int(len(self.data) * val_ratio)

        # Record Models and Residuals
        self.models = []
        self.residuals = residuals

        # Record MSE values
        self.mse = []
        self.mse_val = []

        self.pred_init = None
        self.grads = []
        self.cache = []
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
                self.pred_init = np.mean(self.data[:-self.val_size])
                residuals = y_true - self.pred_init
                grads = residuals
                self.residuals.append(residuals)
                self.grads.append(grads)
            else:
                self.models.append(None)
                self.grads.append(self.residuals[-1])
                self.residuals.append(self.residuals[-1])
                break_cond = False

                while self.error_check:
                    for epsilon in self.epsilon:
                        if break_cond:
                            break_cond = False
                            break
                        for iter in range(25):
                            # Set up the dataframe that contains past residuals as features
                            self.df_errors = pd.concat([self.df,
                                                        self.add_lagged_values(
                                                            pd.DataFrame(self.residuals[-1], columns=["e"]),
                                                            "e",
                                                            lags=self.lags).drop("e", axis=1)], axis=1)

                            if batch_size > 0:
                                batch_indexes = np.random.choice(np.arange(len(self.data) - self.val_size), (64,))
                            else:
                                batch_indexes = np.arange(len(self.data) - self.val_size)

                            x_train = self.df_errors.drop("y", axis=1).values[:-self.val_size, :][batch_indexes, :]
                            # self.m = self.alpha * self.m + (1 - self.alpha) * self.residuals[-1][:-self.val_size]
                            y_train = self.grads[-1][:-self.val_size]

                            if self.learner == "mlp":
                                # Fit MLP regressor to data
                                model_mlp = MLPRegressor(**train_params).fit(x_train, y_train)
                                self.models[-1] = model_mlp

                            elif self.learner == "lgbm":
                                # Fit a regression tree to data
                                train_dataset = lgb.Dataset(x_train, label=y_train)
                                model_lgbm = lgb.train(train_params, train_set=train_dataset,
                                                       valid_sets=[train_dataset, ],
                                                       verbose_eval=0, keep_training_booster=True)
                                self.models[-1] = model_lgbm

                            # predict grad,and residualsFound `num_boost_round` in params. Will use it instead of argument
                            y_hat, y_hat_no_last_tree = self.predict(self.df, fast_train=True)
                            residuals = y_true - y_hat
                            grads = y_true - y_hat_no_last_tree

                            if iter != 0:
                                iter_cond_2 = np.mean(
                                    abs(residuals - np.mean(np.array(self.cache), axis=0))) < epsilon
                            else:
                                iter_cond_2 = abs(residuals - self.residuals[-1]).all() < 1e-7

                            self.cache.append(residuals)

                            # if abs(residuals - self.residuals[-1]).all() < 1e-7 or iter_cond_2:
                            # # if np.mean(abs(residuals - np.mean(np.array(self.cache), axis=0))) < epsilon:
                            # if abs(residuals - self.residuals[-1]).all() == 0:
                            #     continue

                            if iter_cond_2:
                                self.error_check = False
                                self.cache = []
                                self.residuals[-1] = residuals
                                self.grads[-1] = grads
                                print(f"Iteration_{i} is converged with epsilon:{epsilon}")
                                break_cond = True
                                break

                            else:
                                self.residuals[-1] = residuals
                                self.grads[-1] = grads
                                # delete to model for refitting
                                self.models[-1] = None
                                # Reset up the dataframe that contains past residuals as features

                    # if residuals cannot converge to epsilon, continue:
                    self.models[-1] = model_lgbm
                    self.error_check = False
                    self.cache = []
                    self.residuals[-1] = residuals
                    self.grads[-1] = grads

                self.mse.append(mse(y_true[:-self.val_size], y_hat[:-self.val_size]))
                self.mse_val.append(mse(y_true[-self.val_size:], y_hat[-self.val_size:]))
                self.error_check = True
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
        y_hats_no_last_tree = np.zeros(y.shape)
        if fast_train:
            errors = self.residuals[-1]
            df_errors = pd.concat([df,
                                   self.add_lagged_values(pd.DataFrame(errors, columns=["e"]), "e",
                                                          lags=self.lags).drop("e", axis=1)], axis=1)
            x = df_errors.drop("y", axis=1)
            if self.learner == "mlp":
                y_hats = np.sum([model.predict(x) * self.lr for model in self.models], axis=0) + self.pred_init
                y_hats_no_last_tree = np.sum([model.predict(x) * self.lr for model in self.models[:-1]],
                                             axis=0) + self.pred_init
            elif self.learner == "lgbm":
                y_hats = np.sum([model.predict(x) for model in self.models], axis=0) + self.pred_init
                y_hats_no_last_tree = np.sum([model.predict(x) for model in self.models[:-1]],
                                             axis=0) + self.pred_init

        else:
            for i in range(len(y)):
                df_errors = pd.concat([df,
                                       self.add_lagged_values(pd.DataFrame(errors, columns=["e"]), "e",
                                                              lags=self.lags).drop("e", axis=1)], axis=1)
                x = df_errors.drop("y", axis=1).values

                if self.learner == "mlp":
                    y_hat = np.sum([model.predict(x[[i], :]) * self.lr for model in self.models],
                                   axis=0) + self.pred_init
                    y_hats_no_last_tree = np.sum([model.predict(x[[i], :]) * self.lr for model in self.models[:-1]],
                                                 axis=0) + self.pred_init
                elif self.learner == "lgbm":
                    y_hat = np.sum([model.predict(x[[i], :]) for model in self.models], axis=0) + self.pred_init
                    y_hats_no_last_tree = np.sum([model.predict(x[[i], :]) for model in self.models[:-1]],
                                                 axis=0) + self.pred_init
                errors[i] = y[i] - y_hat
                y_hats[i] = y_hat

        return np.array(y_hats), np.array(y_hats_no_last_tree)

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
                 self.add_lagged_values(pd.DataFrame(errors, columns=["e"]), "e", lags=self.lags).drop("e",
                                                                                                       axis=1)],
                axis=1)
            x = df_errors.drop("y", axis=1).values

            if self.learner == "mlp":
                y_hat = np.sum([model.predict(x[[i], :]) * self.lr for model in self.models[:model_idx]],
                               axis=0) + self.pred_init
            elif self.learner == "lgbm":
                y_hat = np.sum([model.predict(x[[i], :]) for model in self.models[:model_idx]], axis=0) + self.pred_init

            errors[i] = y[i] - y_hat
            y_hats[i] = y_hat

        return np.array(y_hats)[-horizon:]
