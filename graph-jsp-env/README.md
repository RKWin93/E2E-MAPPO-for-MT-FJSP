![Tests](https://github.com/mCodingLLC/SlapThatLikeButton-TestingStarterProject/actions/workflows/tests.yml/badge.svg)

<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://cybernetics-lab.de/">
    <img src="https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/logo.png" alt="Logo" height="80">
  </a>

  <h1 align="center">
     Graph Job Shop Problem Gym Environment 
  </h1>

   <a>
    <img src="https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/graph_jsp_tikz.png" alt="Logo" height="180">
  </a>

</div>




# About The Project
This provides an implementation [OpenAi Gym Environment](https://gym.openai.com/) 
of the Job Shop Scheduling Problem (JSP) using the disjunctive graph approach.
The environment offers multiple visualisation options, some of which are shown below 


![](https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/ganttAndGraph.png)
![](https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/console.png)

Github: https://github.com/Alexander-Nasuta/graph-jsp-env

PyPi: https://pypi.org/project/graph-jsp-env/

# Quick Start

### Install the Package 
Install the package with pip:
```
   pip install graph-jsp-env
```
### Minimal Working Example
the code below shows a minimal working example without any reinforcement learning 
```
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
import numpy as np

jsp = np.array([
    [
        [0, 1, 2, 3],  # job 0
        [0, 2, 1, 3]  # job 1
    ],
    [
        [11, 3, 3, 12],  # task durations of job 0
        [5, 16, 7, 4]  # task durations of job 1
    ]

])
env = DisjunctiveGraphJspEnv(jps_instance=jsp)

# loop over all actions
for i in range(env.total_tasks_without_dummies):
    _ = env.step(i)
    env.render()
# schedule is done when every action/node is scheduled
env.render(wait=None)  # with wait=None the window remains open till a button is pressed
```

### Manual Scheduling

I recommend to do the schedule process manually once, before letting reinforcement agents do the work.
To do so first install `inquirer`. This package will handle your input, that you will select in the console.

```
pip install inquirer
```

Then run the following code:
```
import inquirer
import numpy as np

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from graph_jsp_env.disjunctive_graph_logger import log

jsp = np.array([
    [
        [0, 1, 2, 3],  # job 0
        [0, 2, 1, 3],  # job 1
    ],
    [
        [11, 3, 3, 12],  # task durations of job 0
        [5, 16, 7, 4],  # task durations of job 1
    ]

])

env = DisjunctiveGraphJspEnv(
    jps_instance=jsp,
    scaling_divisor=40.0  # makespan of the optimal solution for this instance
)

done = False
log.info("each task/node corresponds to an action")

while not done:
    env.render(
        show=["gantt_console", "gantt_window", "graph_console", "graph_window"],
        # ,stack='vertically'
    )
    questions = [
        inquirer.List(
            "task",
            message="Which task should be scheduled next?",
            choices=[
                (f"Task {task_id}", task_id)
                for task_id, bol in enumerate(env.valid_action_mask(), start=1)
                if bol
            ],
        ),
    ]
    action = inquirer.prompt(questions)["task"] - 1  # note task are index 1 in the viz, but index 0 in action space
    n_state, reward, done, info = env.step(action)
    # note: gantt_window and graph_window use a lot of resources

log.info(f"the JSP is completely scheduled.")
log.info(f"makespan: {info['makespan']}")
log.info("press any key to close the window (while the window is focused).")
# env.render(wait=None)  # wait for keyboard input before closing the render window
env.render(
    wait=None,
    show=["gantt_console", "graph_console", "graph_window"],
    # stack='vertically'
)
```

# Demonstrator (windows executable)

A windows .exe-demonstrator is available on [sciebo](https://rwth-aachen.sciebo.de/s/UqUx4XntTpk2uMQ). 
It needs a while before the first console Outputs appear.
This demonstrator is essentially the manual Scheduling above with the [ft06](http://jobshop.jjvh.nl/instance.php?instance_id=6)
JSP instance.

![](https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/demo_window.png)
![](https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/demo_console.png)


# Project Structure
This project is still in development and will have some significant changes before version 1.0.0.
This project ist structured according to [James Murphy's testing guide](https://www.youtube.com/watch?v=DhUpxWjOhME) and 
this [PyPi-publishing-guide](https://realpython.com/pypi-publish-python-package/).

# Getting Started

If you just want to use the environment, then only the Usage section is relevant for you.
If you want to further develop the environment the follow the instructions in the Development section.

## Usage

Install the package with pip:
```
   pip install graph-jsp-env
```

TODO: present all major features of the env with ray, stb3

## Development 

To run this Project locally on your machine follow the following steps:

1. Clone the repo
   ```sh
   git clone https://github.com/Alexander-Nasuta/graph-jsp-env.git
   ```
2. Install the python requirements_dev packages. `requirements_dev.txt` includes all the packages of
specified `requirements.txt` and some additional development packages like `mypy`, `pytext`, `tox` etc. 
    ```sh
   pip install -r requirements_dev.txt
   ```
3. Install the modules of the project locally. For more info have a look at 
[James Murphy's testing guide](https://www.youtube.com/watch?v=DhUpxWjOhME)
   ```sh
   pip install -e .
   ```

### Testing

For testing make sure that the dev dependencies are installed (`requirements_dev.txt`) and the models of this 
project are set up (i.e. you have run `pip install -e .`).  

Then you should be able to run

```sh
mypy src
```

```sh
flake8 src
```

```sh
pytest
```

or everthing at once using `tox`.

```sh
tox
```

In this Section describes the used Setup and Development tools. 
This only relevant if you plan on further develop

### Hardware

All the code was developed and tested locally on an Apple M1 Max 16" MacBook Pro (16-inch, 2021) with 64 GB Unified Memory.

The **code** should run perfectly fine on other devices and operating Systems (see Github tests). 

### Python Environment Management

#### Mac
On a Mac I recommend using [Miniforge](https://github.com/conda-forge/miniforge) instead of more common virtual
environment solutions like [Anacond](https://www.anaconda.com) or [Conda-Forge](https://conda-forge.org/#page-top).

Accelerate training of machine learning models with TensorFlow on a Mac requires a special installation procedure, 
that can be found [here](https://developer.apple.com/metal/tensorflow-plugin/).
However, this repository provides only the gym environment and no concrete reinforcement learning agents.
Todo: example project with sb3 and rl


Setting up Miniforge can be a bit tricky (especially when Anaconda is already installed).
I found this [guide](https://www.youtube.com/watch?v=w2qlou7n7MA) by Jeff Heaton quite helpful.

#### Windows

On a **Windows** Machine I recommend [Anacond](https://www.anaconda.com), since [Anacond](https://www.anaconda.com) and 
[Pycharm](https://www.jetbrains.com/de-de/pycharm/) are designed to work well with each 
other. 

### IDEA

I recommend to use [Pycharm](https://www.jetbrains.com/de-de/pycharm/).
Of course any code editor can be used instead (like [VS code](https://code.visualstudio.com/) 
or [Vim](https://github.com/vim/vim)).

This section goes over a few recommended step for setting up the Project properly inside [Pycharm](https://www.jetbrains.com/de-de/pycharm/).

#### PyCharm Setup
1. Mark the `src` directory as `Source Root`.
```
   right click on the 'src' -> 'Mark directory as' -> `Source Root`
```

2. Mark the `resources` directory as `Resource Root`.
```
   right click on the 'resources' -> 'Mark directory as' -> `Resource Root`
```

3. Mark the `tests` directory as `Test Source Root`.
```
   right click on the 'tests' -> 'Mark directory as' -> `Test Source Root`
```

afterwards your project folder should be colored in the following way:

<div align="center">
  <a>
    <img src="https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/mark_project_folders.png"  height="320">
  </a>
</div>

4. (optional) When running a script enable `Emulate terminal in output console`
```
Run (drop down) | Edit Configurations... | Configuration | ☑️ Emulate terminal in output console
```

![](https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/colored_logs_settings.png)



# License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- MARKDOWN LINKS & IMAGES todo: add Github, Linked in etc.-->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[screenshot]: resources/readme_images/screenshot.png


