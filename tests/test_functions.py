import sys

sys.path.append("..")  # Adds higher directory to python modules
sys.path.append("../liver_pred/utils")  # Adds current directory to python modules

import pytest
import pandas as pd
import numpy as np

import liver_pred.liver_pred.utils.preprocessing.feature_generation as fg


def test_rnn_feature_eng():

    patient1 = pd.DataFrame(
        {
            "subject_id": [1, 1, 1, 1, 1],
            "charttime": pd.to_datetime(
                ["2022-01-01", "2022-01-01", "2023-12-03", "2024-01-01", "2024-01-03"]
            ),
            "label": ["A", "B", "A", "A", "B"],
            "valuenum": [1.1, 1.2, 1.3, 1.4, 1.5],
            "outcome": [0, 0, 0, 0, 0],
            "index_date": pd.to_datetime(
                ["2024-01-10", "2024-01-10", "2024-01-10", "2024-01-10", "2024-01-10"]
            ),
        }
    )
    t1 = patient1["charttime"][3] - patient1["charttime"][2]
    t2 = patient1["charttime"][4] - patient1["charttime"][3]
    t3 = patient1["charttime"][2] - patient1["charttime"][0]
    lab_output = np.array([[[1.1, 1.2, 0], [1.3, 0.0, t1.days], [1.4, 0.0, t2.days]]])
    outcome = np.array([0])

    assert (
        fg.create_array_for_RNN(patient1, lead_time=8, max_history=None)[0].shape
        == lab_output.shape
    )


pytest.main()

# patient2 = pd.DataFrame({'charttime': pd.to_datetime(['2006-01-01', '2002-03-03']),
#                                'label': ['A', 'A'],
#                                    'valuenum': [1.2, 1.3],
#                                    'outcome' : [0,0],
#                                    'index_date' : pd.to_datetime(['2024-01-10', '2024-01-10'])
#        })
