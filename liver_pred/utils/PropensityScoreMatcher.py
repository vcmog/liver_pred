import pandas as pd
import numpy as np

# from functions import progress
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


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
        self.case = self.data[self.data[yvar] == 1]
        self.control = self.data[self.data[yvar] == 0]
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
                # fit multiple models based on imbalance severity (rounded up
                # to nearest tenth)
                minor, major = [
                    self.data[self.data[self.yvar] == i]
                    for i in (self.minority, self.majority)
                ]
                nmodels = int(np.ceil((len(major) / len(minor)) / 10) * 10)
            self.nmodels = nmodels
            i = 0
            while i < nmodels:
                # progress(i + 1, nmodels, prestr="Fitting Models on
                #  Balanced Samples")

                df = self.balanced_sample()
                X = df[self.xvars]
                y = df[self.yvar]

                lr = LogisticRegression(max_iter=500, C=1e5)
                res = lr.fit(X, y)
                self.model_accuracy.append(self._scores_to_accuracy(res, X, y))
                self.models.append(res)
                i = i + 1
            self.average_accuracy = round(np.mean(self.model_accuracy) * 100, 2)
            print("\nAverage Accuracy:", "{}%".format(self.average_accuracy))

    def predict_scores(self):
        scores = np.zeros(len(self.X))
        for i in range(self.nmodels):
            m = self.models[i]
            scores += m.predict_proba(self.X)[:, 1]
        self.data["scores"] = scores / self.nmodels

    def balanced_sample(self, data=None):
        if not data:
            data = self.data
        minor, major = (
            data[data[self.yvar] == self.minority],
            data[data[self.yvar] == self.majority],
        )
        return pd.concat([major.sample(len(minor)), minor], sort=True)

    def plot_scores(self, save_fig=False, save_path=None):
        """
        Plots the distribution of propensity scores before matching between
        our test and control groups
        """
        assert (
            "scores" in self.data.columns
        ), "Propensity scores not yet calculated, use Matcher.predict_scores()"
        hist = sns.displot(
            self.data, x="scores", hue="outcome", stat="percent", common_norm=False
        )
        # sns.displot(cases.scores, label="Case")
        # plt.legend(loc="upper right")
        plt.xlim((0, 1))
        plt.title("Propensity Scores Before Matching")
        plt.ylabel("Percentage (%)")
        plt.xlabel("Scores")

        if save_fig:
            assert save_path, "No path provided for figure destination"
            hist.savefig(save_path)

    def match(self, threshold=0.001, nmatches=1, method="min", max_rand=10):
        """
        Finds suitable match(es) for each record in the minority
        dataset, if one exists. Records are exlcuded from the final
        matched dataset if there are no suitable matches.

        self.matched_data contains the matched dataset once this
        method is called

        Parameters
        ----------
        threshold : float
            threshold for fuzzy matching matching
            i.e. |score_x - score_y| >= theshold
        nmatches : int
            How majority profiles should be matched
            (at most) to minority profiles
        method : str
            Strategy for when multiple majority profiles
            are suitable matches for a single minority profile
            "random" - choose randomly (fast, good for testing)
            "min" - choose the profile with the closest score
        max_rand : int
            max number of profiles to consider when using random tie-breaks

        Returns
        -------
        None
        """

        if "scores" not in self.data.columns:
            print("Propensity Scores have not been calculated. Using defaults...")
            self.fit_scores()
            self.predict_scores()
        case_scores = self.data[self.data[self.yvar] == 1][["scores"]]
        ctrl_scores = self.data[self.data[self.yvar] == 0][["scores"]]
        result, match_ids, matched_patients = [], [], set()
        for i in range(len(case_scores)):
            # uf.progress(i+1, len(test_scores), 'Matching Control to Test...')
            score = case_scores.iloc[i]
            if method == "random":
                bool_match = abs(ctrl_scores - score) <= threshold
                matches = ctrl_scores.loc[bool_match[bool_match.scores].index]
            elif method == "min":
                matches = abs(ctrl_scores - score).sort_values("scores").head(nmatches)
            else:
                raise (
                    AssertionError,
                    "Invalid method parameter, use ('random', 'min')",
                )
            if len(matches) == 0:
                continue
            # randomly choose nmatches indices, if len(matches) > nmatches
            selected_matches = []
            for _ in range(min(nmatches, len(matches))):
                chosen = matches.index[0]  # Default to first match
                if method == "random":
                    chosen = np.random.choice(matches.index, 1)[0]
                selected_matches.append(chosen)
                matches = matches.drop(chosen)
            selected_matches = [
                m for m in selected_matches if m not in matched_patients
            ]
            if len(selected_matches) == 0:
                continue
            match_ids.extend([i] * (len(selected_matches) + 1))
            result.extend([case_scores.index[i]] + selected_matches)
            matched_patients.update(selected_matches)

        self.matched_data = self.data.loc[result]
        self.matched_data["match_id"] = match_ids
        self.matched_data["record_id"] = self.matched_data.index

    def tune_threshold(
        self,
        method="random",
        nmatches=1,
        rng=np.arange(0, 0.001, 0.0001),
        save_fig=False,
        save_path=None,
    ):
        """
        Matches data over a grid to optimize threshold value and plots results.

        Parameters
        ----------
        method : str
            Method used for matching (use "random" for this method)
        nmatches : int
            Max number of matches per record. See pymatch.match()
        rng: : list / np.array()
            Grid of threshold values

        Returns
        -------
        None

        """
        results = []
        for i in rng:
            self.match(method=method, nmatches=nmatches, threshold=i)
            results.append(self.prop_retained())
        plt.figure()
        plt.plot(rng, results)
        plt.title("Proportion of Data retained for grid of threshold values")
        plt.ylabel("Proportion Retained")
        plt.xlabel("Threshold")
        plt.xticks(rng)
        if save_fig:
            assert save_path, "No path provided for figure destination"
            plt.savefig(save_path)

    def prop_retained(self):
        """
        Returns the proportion of data retained after matching
        """

        return (
            len(self.matched_data[self.matched_data[self.yvar] == self.minority])
            * 1.0
            / len(self.data[self.data[self.yvar] == self.minority])
        )

    @staticmethod
    def _scores_to_accuracy(m, X, y):
        preds = [1.0 if i >= 0.5 else 0.0 for i in m.predict(X)]
        return (y == preds).sum() * 1.0 / len(y)
