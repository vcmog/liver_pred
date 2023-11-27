import pandas as pd
import numpy as np
from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.api as sm
from functions import progress
import seaborn as sns
import matplotlib as plt


class PropensityScoreMatcher:
    def __init__(self, case, control, yvar, formula=None, exclude=[]):
        # assign unique indices to test and control
        t, c = [i.copy().reset_index(drop=True) for i in (case, control)]
        t = t.dropna(axis=1, how="all")
        c = c.dropna(axis=1, how="all")
        c.index += len(t)  # shift c index to go after t
        data = pd.concat([t, c], axis=0)
        self.data = pd.get_dummies(data, drop_first=True, dtype=int)
        self.yvar = yvar
        self.exclude = exclude + [self.yvar]

        self.nmodels = 1
        self.models = []
        self.model_accuracy = []
        self.data[yvar] = self.data[yvar].astype(int)  # binary
        self.xvars = [i for i in self.data.columns if i not in self.exclude]
        self.X = self.data[self.xvars]
        self.case = self.data[self.data[yvar] == True]
        self.control = self.data[self.data[yvar] == False]
        self.casen = len(self.case)
        self.controln = len(self.control)
        self.minority, self.majority = [
            i[1] for i in sorted(zip([self.casen, self.controln], [1, 0]))
        ]
        self.forumula = "formula 'n{} ~ {}".format(yvar, "+".join(self.xvars))

    def fit_score(self, balance=True, nmodels=None):
        if len(self.models) > 0:
            self.models = []
        if len(self.model_accuracy) > 0:
            self.model_accuracy = []
        if balance:
            if nmodels is None:
                # fit multiple models based on imbalance severity (rounded up to nearest tenth)
                minor, major = [
                    self.data[self.data[self.yvar] == i]
                    for i in (self.minority, self.majority)
                ]
                nmodels = int(np.ceil((len(major) / len(minor)) / 10) * 10)
            self.nmodels = nmodels
            i = 0
            while i < nmodels:
                # progress(i + 1, nmodels, prestr="Fitting Models on Balanced Samples")

                df = self.balanced_sample()
                X = np.asarray(df.drop(self.yvar, axis=1))
                y = np.asarray(df[self.yvar].to_list(), dtype="float")

                glm = GLM(y, X, family=sm.families.Binomial())
                res = glm.fit()
                self.model_accuracy.append(self._scores_to_accuracy(res, X, y))
                self.models.append(res)
                i = i + 1
            self.average_accuracy = round(np.mean(self.model_accuracy) * 100, 2)

    def predict_scores(self):
        scores = np.zeros(len(self.X))
        for i in range(self.nmodels):
            m = self.models[i]
            scores += m.predict(np.asarray(self.X))
        self.data["scores"] = scores / self.nmodels

    def balanced_sample(self, data=None):
        if not data:
            data = self.data
        minor, major = (
            data[data[self.yvar] == self.minority],
            data[data[self.yvar] == self.majority],
        )
        return pd.concat([major.sample(len(minor)), minor], sort=True)

    def plot_scores(self):
        """
        Plots the distribution of propensity scores before matching between
        our test and control groups
        """
        assert (
            "scores" in self.data.columns
        ), "Propensity scores haven't been calculated, use Matcher.predict_scores()"
        sns.distplot(self.data[self.data[self.yvar] == 0].scores, label="Control")
        sns.distplot(self.data[self.data[self.yvar] == 1].scores, label="Test")
        plt.legend(loc="upper right")
        plt.xlim((0, 1))
        plt.title("Propensity Scores Before Matching")
        plt.ylabel("Percentage (%)")
        plt.xlabel("Scores")

    @staticmethod
    def _scores_to_accuracy(m, X, y):
        preds = [[1.0 if i >= 0.5 else 0.0 for i in m.predict(X)]]
        return (y.T == preds).sum() * 1.0 / len(y)
