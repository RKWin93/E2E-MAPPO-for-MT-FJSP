import pytest


@pytest.fixture(scope="function")
def visualisation_dataframe():
    plotly_gantt_chart_data_dict = {
        'Task': {
            0: 'Job 0',
            1: 'Job 0',
            2: 'Job 0',
            3: 'Job 0',
            4: 'Job 1',
            5: 'Job 1',
            6: 'Job 1',
            7: 'Job 1'
        },
        'Start': {
            0: 5,
            1: 16,
            2: 21,
            3: 24,
            4: 0,
            5: 5,
            6: 21,
            7: 36
        },
        'Finish': {
            0: 16,
            1: 19,
            2: 24,
            3: 36,
            4: 5,
            5: 21,
            6: 28,
            7: 40
        },
        'Resource': {
            0: 'Machine 0',
            1: 'Machine 1',
            2: 'Machine 2',
            3: 'Machine 3',
            4: 'Machine 0',
            5: 'Machine 2',
            6: 'Machine 1',
            7: 'Machine 3'
        }
    }
    import pandas as pd
    plotly_gantt_chart_df = pd.DataFrame.from_dict(plotly_gantt_chart_data_dict)
    yield plotly_gantt_chart_df


@pytest.fixture(scope="function")
def custom_jsp_instance():
    import numpy as np
    custom_jsp_instance = np.array([
        [
            [0, 1, 2, 3],  # job 0
            [0, 2, 1, 3]  # job 1
        ],
        [
            [11, 3, 3, 12],  # task durations of job 0
            [5, 16, 7, 4]  # task durations of job 1
        ]

    ])
    yield custom_jsp_instance
