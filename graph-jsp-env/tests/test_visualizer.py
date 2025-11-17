import pytest

from graph_jsp_env.disjunctive_graph_jsp_visualizer import DisjunctiveGraphJspVisualizer


def test_terminal_gantt_vis(visualisation_dataframe):
    import numpy as np
    import matplotlib.pyplot as plt

    c_map = plt.cm.get_cmap("jet")  # select the desired cmap
    arr = np.linspace(0, 1, 4)  # create a list with numbers from 0 to 1 with n items
    colors = {f"Machine {resource}": c_map(val)[:3] for resource, val in enumerate(arr)}

    visualizer = DisjunctiveGraphJspVisualizer()

    visualizer.gantt_chart_console(
        df=visualisation_dataframe,
        colors=colors
    )


# https://stackoverflow.com/questions/73973332/check-if-were-in-a-github-action-tracis-ci-circle-ci-etc-testing-environme
# link above does not seem to work here
@pytest.mark.skip(reason="no idea how to properly test guis with github actions")
def test_window_gantt_vis(visualisation_dataframe):
    import numpy as np
    import matplotlib.pyplot as plt

    c_map = plt.cm.get_cmap("jet")  # select the desired cmap
    arr = np.linspace(0, 1, 4)  # create a list with numbers from 0 to 1 with n items
    colors = {f"Machine {resource}": c_map(val)[:3] for resource, val in enumerate(arr)}

    visualizer = DisjunctiveGraphJspVisualizer()

    visualizer.render_gantt_in_window(
        df=visualisation_dataframe,
        colors=colors
    )
