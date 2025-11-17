import io

import cv2
import signal
import shutil
import itertools

import networkx as nx
import matplotlib as mpl
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from typing import Union

from graph_jsp_env.disjunctive_graph_logger import log


def handler_stop_signals(*_) -> None:
    """
    closes all `cv2`-windows when the process is killed
    """
    cv2.destroyAllWindows()


signal.signal(signal.SIGINT, handler_stop_signals)
signal.signal(signal.SIGTERM, handler_stop_signals)


class DisjunctiveGraphJspVisualizer:
    """
    this class contains the code for all the different rendering options of a `DisjunctiveGraphJssEnv`,
    but it can be also used as a standalone visualizer.
    """

    COLOR_ESCAPE_SEQUENCE = '\33[0m'

    @staticmethod
    def rgb_color_sequence(r: Union[int, float], g: Union[int, float], b: Union[int, float],
                           *, format_type: str = 'foreground') -> str:
        """
        generates a color-codes, that change the color of text in console outputs.

        rgb values must be numbers between 0 and 255 or 0.0 and 1.0.

        :param r:               red value.
        :param g:               green value
        :param b:               blue value

        :param format_type:     specifies weather the foreground-color or the background-color shall be adjusted.
                                valid options: 'foreground','background'
        :return:
        """
        if format_type == 'foreground':
            f = '\033[38;2;{};{};{}m'.format  # font rgb format
        elif format_type == 'background':
            f = '\033[48;2;{};{};{}m'.format  # font background rgb format
        else:
            raise ValueError(f"format {format_type} is not defined. Use 'foreground' or 'background'.")
        rgb = [r, g, b]

        if isinstance(r, int) and isinstance(g, int) and isinstance(b, int):
            if min(rgb) < 0 and max(rgb) > 255:
                raise ValueError("rgb values must be numbers between 0 and 255 or 0.0 and 1.0")
            return f(r, g, b)
        if isinstance(r, float) and isinstance(g, float) and isinstance(b, float):
            if min(rgb) < 0 and max(rgb) > 1:
                raise ValueError("rgb values must be numbers between 0 and 255 or 0.0 and 1.0")
            return f(*[int(n * 255) for n in [r, g, b]])

    @staticmethod
    def wrap_with_color_codes(s: object, /, r: Union[int, float], g: Union[int, float], b: Union[int, float], **kwargs) \
            -> str:
        """
        stringify an object and wrap it with console color codes. It adds the color control sequence in front and one
        at the end that resolves the color again.

        rgb values must be numbers between 0 and 255 or 0.0 and 1.0.

        :param s: the object to stringify and wrap
        :param r: red value.
        :param g: green value.
        :param b: blue value.
        :param kwargs: additional argument for the 'DisjunctiveGraphJspVisualizer.rgb_color_sequence'-method.
        :return:
        """
        return f"{DisjunctiveGraphJspVisualizer.rgb_color_sequence(r, g, b, **kwargs)}" \
               f"{s}" \
               f"{DisjunctiveGraphJspVisualizer.COLOR_ESCAPE_SEQUENCE}"

    @staticmethod
    def gantt_chart_console(df: pd.DataFrame, colors: dict) -> None:
        """
        console version of the `gantt_chart_rgb_array`-method. prints a gant chart to the console.
        the parameters need to follow the plotly specification.
        see: https://plotly.com/python/gantt/ or `gantt_chart_rgb_array`

        :param df:      dataframe according to `plotly` specification (https://plotly.com/python/gantt/).

        :param colors:  a dict that maps resources to color values. see example below.
                        Note: make sure that the key match the resources specified in `:param df:`

        :return:        a `plotly` gantt chart as rgb array.

        color example

            import numpy as np

            c_map = plt.cm.get_cmap("jet")  # select the desired cmap
            arr = np.linspace(0, 1, 10)  # create a list with numbers from 0 to 1 with n items
            colors = {resource: c_map(val) for resource, val in enumerate(arr)}
        """
        # w, h = shutil.get_terminal_size((80, 20))  # enable emulate output in terminal ...
        w, h = shutil.get_terminal_size((140, 30))  # enable emulate output in terminal ... todo 手动修改console里边画图的长和宽

        if len(df) > 0:
            machines = sorted(df['Resource'].unique())
            jobs = df['Task'].unique()
            jobs.sort()
        else:
            jobs, machines = None, None

        len_prefix = 10  # 图表的左边界，不要乱改，会画不全
        len_suffix = 15  # 图标的右边界
        # len_prefix = 5  # 图表的左边界
        # len_suffix = 8  # 图标的右边界

        x_pixels = w - len_prefix - len_suffix
        # x_pixels = (w - len_prefix - len_suffix) * 2 # todo 我把像素点扩大，有效果吗？
        x_max = df['Finish'].max() + 1 if len(df) > 0 else x_pixels
        if x_pixels < 0:
            log.warn("terminal window to small")
            return

        x_axis_tick_small = "╤════"
        x_axis_tick_big = "╦════"
        len_tick = len(x_axis_tick_big)
        num_hole_ticks = x_pixels // len_tick
        len_last_tick = x_pixels % len_tick
        x_axis = "".join([
            f"{'':<{len_prefix - 1}}╚",
            *[x_axis_tick_big if i % 5 == 0 else x_axis_tick_small for i in range(num_hole_ticks)],
            "═" * len_last_tick + "╝"
        ])
        x_chart_frame_top = "".join([
            f"{'':<{len_prefix - 1}}╔",
            "═" * x_pixels,
            "╗"
        ])

        x_interval_increment5 = x_max / num_hole_ticks
        x_interval_increment1 = x_max / x_pixels  # todo （makespan/图表的长度=像素点，合理，太小的花不上去！）动态的，导致有些太小的没有画上，但是是真实已经调度的了，算过makespan的！
        # x_interval_increment1 = x_max / (x_pixels * 2)  # todo 每个像素点的最小值是动态的，取决于最大x，即makespan，不如规定一个最小值？？？？会不会超机密集？直接画不全了

        x_axis_label = "".join([
            f"{'':<{len_prefix}}",
            *[
                f"{f'{i * x_interval_increment5:.1f}':<5}" if i % 5 == 0 else f"{'':<5}"
                for i in range(num_hole_ticks)
            ]
        ])

        rows = []
        if len(df) > 0:
            for j, m in itertools.zip_longest(jobs, machines):
                matching_tasks = df.loc[df['Task'] == j].iterrows()
                chart_str = [i * x_interval_increment1 for i in range(x_pixels)]
                # chart_str = [i * x_interval_increment1 for i in range(x_pixels*10)] # 改了会乱，不搞了，浪费时间！
                for _, (_, start, finish, resource) in matching_tasks:
                    chart_str = [
                        f"{DisjunctiveGraphJspVisualizer.rgb_color_sequence(*colors[resource])}█"
                        # f"{DisjunctiveGraphJspVisualizer.rgb_color_sequence(*colors[resource])}|"  # 表示画图的图标
                        if not isinstance(v, str)
                           and start <= v <= finish
                        else v for v in chart_str
                    ]
                prefix = f"{f'{j}':<{len_prefix - 1}}║" if j else f"{'':<{len_prefix - 1}}║"
                colored_block = DisjunctiveGraphJspVisualizer.wrap_with_color_codes("█", *colors[m]) if m else None
                suffix = f"{f'║ {m}':<{len_suffix - 1}}" + f"{colored_block}" if m else f"{f'║':<{len_suffix}}"

                chart_str = [" " if not isinstance(v, str) else v for v in chart_str]
                chart_str = "".join(chart_str)
                rows.append(f"{prefix}{chart_str}{DisjunctiveGraphJspVisualizer.COLOR_ESCAPE_SEQUENCE}{suffix}")
        else:
            rows = ["".join([f"{f'':<{len_prefix - 1}}║", " " * x_pixels, "║"])]

        gant_str = "\n".join([
            x_chart_frame_top,
            *rows,
            x_axis,
            x_axis_label
        ])
        print(gant_str)

    @staticmethod
    def render_rgb_array(vis: np.ndarray, *,
                         window_title: str = "Job Shop Scheduling", wait: int = 1) -> None:
        """
        renders a rgb-array in an `cv2` window.
        the window will remain open for `:param wait:` ms or till the user presses any key.

        :param vis:             the rgb-array to render.
        :param window_title:    the title of the `cv2`-window
        :param wait:            time in ms that the `cv2`-window is open.
                                if `None`, then the window will remain open till a keyboard occurs.

        :return:
        """
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_title, vis)
        # https://stackoverflow.com/questions/64061721/opencv-to-close-the-window-on-a-specific-key
        k = cv2.waitKey(wait) & 0xFF
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()

    def __init__(self, *,
                 dpi=55, width=15, height=10,
                 scheduled_color="#DAF7A6", not_scheduled_color="#FFC300", color_job_edge="tab:gray",
                 node_drawing_kwargs=None,
                 edge_drawing_kwargs=None,
                 critical_path_drawing_kwargs=None):
        """
        :param dpi:                             parameter for `matplotlib.pyplot.figure`
        :param width:                           parameter for `matplotlib.pyplot.figure`
        :param height:                          parameter for `matplotlib.pyplot.figure`

        :param scheduled_color:                 parameter for `nx.draw_networkx_nodes`
        :param not_scheduled_color:             parameter for `nx.draw_networkx_nodes`
        :param color_job_edge:                  parameter for `nx.draw_networkx_edges`

        :param node_drawing_kwargs:             kwargs for `nx.draw_networkx_nodes`
        :param edge_drawing_kwargs:             kwargs for `nx.draw_networkx_nodes`
        :param critical_path_drawing_kwargs:    kwargs for `nx.draw_networkx_edges`
        """
        self.width = width
        self.height = height
        self.dpi = dpi
        self.color_scheduled = scheduled_color
        self.color_not_scheduled = not_scheduled_color
        self.color_job_edge = color_job_edge
        self.node_drawing_kwargs = {
            "node_size": 800,
            "linewidths": 5
        } if node_drawing_kwargs is None else node_drawing_kwargs
        self.edge_drawing_kwargs = {
            "arrowsize": 30
        } if edge_drawing_kwargs is None else edge_drawing_kwargs
        self.critical_path_drawing_kwargs = {
            "edge_color": 'r',
            "width": 20,
            "alpha": 0.1,
        } if critical_path_drawing_kwargs is None else critical_path_drawing_kwargs

    def graph_rgb_array(self, G: nx.DiGraph) -> np.ndarray:
        """
        Wrapper for `nx` drawing operations.

        :param G:   the `nx.DiGraph` of an `DisjunctiveGraphJssEnv` instance
        :return:    a plot of the provided graph as rgb array.
        """

        plt.figure(dpi=self.dpi)
        plt.axis("off")
        plt.tight_layout()

        pos: dict = nx.get_node_attributes(G, 'pos')  # node positions
        fig = mpl.pyplot.gcf()
        fig.set_size_inches(self.width, self.height)

        # draw nodes
        for task, data in G.nodes(data=True):
            nx.draw_networkx_nodes(G, pos,
                                   nodelist=[task],
                                   edgecolors=data["color"],
                                   node_color=self.color_scheduled if data["scheduled"] else self.color_not_scheduled,
                                   **self.node_drawing_kwargs
                                   )
        # draw node labels
        nx.draw_networkx_labels(G, pos)
        # draw edges
        for from_, to_, data_dict in G.edges(data=True):
            if data_dict["job_edge"]:
                nx.draw_networkx_edges(G, pos,
                                       edgelist=[(from_, to_)],
                                       alpha=0.5,
                                       edge_color=self.color_job_edge,
                                       **self.edge_drawing_kwargs
                                       )
            else:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(from_, to_)],
                    edge_color=G.nodes[from_]["color"],
                    **self.edge_drawing_kwargs
                )

        # draw edge labels
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        # longest path corresponds to makespan of the jsp
        longest_path = nx.dag_longest_path(G)
        longest_path_edges = list(zip(longest_path, longest_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=longest_path_edges, **self.critical_path_drawing_kwargs)

        # convert canvas to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        try:
            # with scientific mode enabled in pycharm this code works
            # no idea why enabling scientific mode in pycharm changes anything at all :o
            # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            w, h = fig.canvas.get_width_height()
            img = img.reshape((h, w, 3))
        except ValueError:
            w, h = fig.canvas.get_width_height()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=self.dpi)
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()
            img = cv2.imdecode(img_arr, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.reshape((h, w, 3))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # clear current frame
        plt.clf()
        plt.close('all')
        return img

    def gantt_chart_rgb_array(self, df: pd.DataFrame, *, colors: dict) -> np.ndarray:
        """

        wrapper for `plotly` gantt chart function. turn a gantt chart into a rgb array.

        see: https://plotly.com/python/gantt/

        :param df:      dataframe according to `plotly` specification (https://plotly.com/python/gantt/).

        :param colors:  a dict that maps resources to color values. see example below.
                        Note: make sure that the key match the resources specified in `:param df:`

        :return:        a `plotly` gantt chart as rgb array.

        color example

            import numpy as np

            c_map = plt.cm.get_cmap("jet")  # select the desired cmap
            arr = np.linspace(0, 1, 10)  # create a list with numbers from 0 to 1 with n items (n = 10 here)
            colors = {resource: c_map(val) for resource, val in enumerate(arr)}
            <<pass `colors` as parameter>>

        """

        plt.figure(dpi=self.dpi)
        plt.axis("off")
        plt.tight_layout()

        fig = mpl.pyplot.gcf()
        fig.set_size_inches(self.width, self.height)

        # Gantt chart
        width, height = fig.canvas.get_width_height()
        if not len(df):
            df = pd.DataFrame([{"Task": "Job 0", "Start": 0, "Finish": 0, "Resource": "Machine 0"}])
        fig = ff.create_gantt(df=df, show_colorbar=True, index_col='Resource', group_tasks=True, colors=colors)
        fig.update_layout(xaxis_type='linear')

        img_str = fig.to_image(format="jpg", width=width, height=height)

        nparr = np.frombuffer(img_str, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # clear current frame
        plt.clf()
        plt.close('all')
        return img

    @staticmethod
    def graph_console(G: nx.DiGraph, /, shape: tuple, colors: dict) -> None:
        """
        console version of the `graph_rgb_array`-method. prints a graph that indicates which tasks are scheduled to
        the console.

        :param G:       the `nx.DiGraph` of an `DisjunctiveGraphJssEnv` instance
        :param shape:   size of the jsp
        :param colors:  a dict that maps resources to color values. see example below.
                        Note: make sure that the key match the resources specified in `:param df:`

        :return:        None

        import numpy as np

            c_map = plt.cm.get_cmap("jet")  # select the desired cmap
            arr = np.linspace(0, 1, 10)  # create a list with numbers from 0 to 1 with n items
            colors = {resource: c_map(val) for resource, val in enumerate(arr)}
        """
        # w, _ = shutil.get_terminal_size((80, 20))
        w, _ = shutil.get_terminal_size((120, 30)) # todo 手动修改console里边画图的长和宽
        n_jobs, n_machines = shape

        len_prefix = 10  # 不要乱改，会画不全
        len_suffix = 15
        # len_prefix = 5  # 图表的左边界
        # len_suffix = 8  # 图标的右边界

        if w < 2 * n_jobs + len_prefix + len_suffix:
            log.warn("terminal window to small")
            # return   # 暂时先注释了，即使报错了，也可以继续画？

        task_id = 0
        machine_strings = [
            f"{m :>{len_suffix - 5}} {DisjunctiveGraphJspVisualizer.wrap_with_color_codes('█', r, g, b)}"
            for m, (r, g, b) in colors.items()
        ]

        for j, m_str in itertools.zip_longest(range(n_jobs), machine_strings):
            row = f"Job {j}" if j is not None else ""
            row = f"{row:<{len_prefix}}"
            for task_in_job in range(n_machines):
                task_id = task_id + 1
                if task_id < len(G.nodes) - 1:
                    node = G.nodes[task_id]
                    node_str = "●" if node["scheduled"] else "◯"
                    r, g, b, *_ = node["color"]
                    node_str = DisjunctiveGraphJspVisualizer.wrap_with_color_codes(node_str, r, g, b)
                    if task_in_job < n_machines - 1:
                        node_str += "-"
                else:
                    node_str = " "
                    if task_in_job < n_machines - 1:
                        node_str += " "
                row += node_str
            print("".join([row, " " * 4, m_str if m_str is not None else ""]))

    def render_graph_in_window(self, G: nx.DiGraph, **render_kwargs: dict) -> None:
        """
        wrapper for the `graph_rgb_array`- and `render_rgb_array`-methods

        :param G:               parameter for `graph_rgb_array`
        :param render_kwargs:   additional parameters for `render_rgb_array`

        :return:                None
        """
        vis = self.graph_rgb_array(G)
        self.render_rgb_array(vis, **render_kwargs)

    def render_gantt_in_window(self, df: pd.DataFrame, *, colors: dict, **render_kwargs: dict) -> None:
        """
        wrapper for the `gantt_chart_rgb_array`- and `render_rgb_array`-methods

        :param df:              parameter for `gantt_chart_rgb_array`
        :param colors:          parameter for `gantt_chart_rgb_array`
        :param render_kwargs:   additional parameters for `render_rgb_array`

        :return:                None
        """
        vis = self.gantt_chart_rgb_array(df=df, colors=colors)
        self.render_rgb_array(vis, **render_kwargs)

    def render_graph_and_gant_in_window(self, *, G: nx.DiGraph, df: pd.DataFrame, colors: dict,
                                        stack: str = "horizontally", **render_kwargs: dict) -> None:
        """
        wrapper for the `graph_rgb_array`-, `gantt_chart_rgb_array`- and `render_rgb_array`-methods.


        :param G:               parameter for `graph_rgb_array`

        :param df:              parameter for `gantt_chart_rgb_array`
        :param colors:          parameter for `gantt_chart_rgb_array`

        :param stack:           specifies how the graph and gantt char shall be stacked.
                                valid options: 'horizontally', 'vertically'

        :param render_kwargs:   additional parameters for `render_rgb_array`

        :return:                None
        """
        graph_vis = self.graph_rgb_array(G)
        gantt_vis = self.gantt_chart_rgb_array(df=df, colors=colors)

        if stack == 'horizontally':
            vis = np.concatenate((graph_vis, gantt_vis), axis=1)
        elif stack == 'vertically':
            vis = np.concatenate((graph_vis, gantt_vis), axis=0)
        else:
            raise ValueError(f"stack={stack} is not a valid argument. only 'horizontally' and 'vertically' "
                             f"are allowed here.")

        self.render_rgb_array(vis, **render_kwargs)

    def gantt_and_graph_vis_as_rgb_array(self, *, G: nx.DiGraph, df: pd.DataFrame, colors: dict,
                                         stack: str = "horizontally") -> np.ndarray:
        """
        wrapper for the `graph_rgb_array`-, `gantt_chart_rgb_array`- and `render_rgb_array`-methods.


        :param G:               parameter for `graph_rgb_array`

        :param df:              parameter for `gantt_chart_rgb_array`
        :param colors:          parameter for `gantt_chart_rgb_array`

        :param stack:           specifies how the graph and gantt char shall be stacked.
                                        valid options: 'horizontally', 'vertically'

        :return:                stacked chart as rgb-array
        """
        graph_vis = self.graph_rgb_array(G)
        gantt_vis = self.gantt_chart_rgb_array(df=df, colors=colors)

        if stack == 'horizontally':
            vis = np.concatenate((graph_vis, gantt_vis), axis=1)
        elif stack == 'vertically':
            vis = np.concatenate((graph_vis, gantt_vis), axis=0)
        else:
            raise ValueError(f"stack={stack} is not a valid argument. only 'horizontally' and 'vertically' "
                             f"are allowed here.")

        return vis
