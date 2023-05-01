# Developer Guide

## Table of Contents <!-- omit in toc -->
- [Preparing for Development](#preparing-for-development)
    - [Getting the Development Environment](#getting-the-development-environment)
    - [Getting the Source Code](#getting-the-source-code)
    - [Installing Dependencies](#installing-dependencies)
- [Running the Server for Testing](#running-the-server-for-testing)
- [Linting and Formatting](#linting-and-formatting)
- [Editor Configurations](#editor-configurations)

## Preparing for Development
### Getting the Development Environment
[USCC AIoT workspace](https://github.com/Botang-l/AIot.git) is a docker-based development environment for the projects. Please refer to the [User Guide](https://hackmd.io/_A4_9Lp9SDar0zZVLzISYQ?view) for getting and using the workspace.

### Getting the Source Code
```shell
user-workspace:~$ cd projects/
user-workspace:~/projects$ $ git clone https://github.com/Botang-l/AIoT_code
user-workspace:~/projects$ cd AIoT_code
```

### Installing Dependencies

For the development, it is recommended to use [virtualenv](https://pypi.org/project/virtualenv/), a Python virtual environment, to manage all needed packages. The required packages are listed in [deployment/requirements.txt](../deployment/requirements.txt). Make sure to install Python packages and execute backend server when the shell prefix `(venv)` appears, which indicates that the virtualenv `venv` is enabled.

```shell
$ pip3 install -r deployment/requirements.txt    # install packages
```

## Running the Code for Testing
> pending

## Linting and Formatting
To remain consistent coding style and statically analyze the correctness of our source code, we will need to define the coding style which should be obeyed by all developers. First of all, we use [EditorConfig](https://editorconfig.org/) to set the editor configurations of [coding styles](../.editorconfig) and synchronize between different IDEs. Check whether your favorite IDE is compatible with EditorConfig on its [Pre-installed List](https://editorconfig.org/#pre-installed) and install necessary plugins if your IDE is on the [Editor Plugin List](https://editorconfig.org/#download) before starting development. On the other hand, we also use linters and formatters to help developers. The installation commands are listed below.

For the AI model developed in Python, we choose [Pyright](https://github.com/microsoft/pyright) from Microsoft and [yapf](https://github.com/google/yapf) from Google as the linter and formatter.

```shell
$ user-workspace:~$ cd projects/AIoT_code

user-workspace:~$ pyright .                  # statically type checking
... skip ...
Found 15 source files
pyright 1.1.279
0 errors, 0 warnings, 0 informations
Completed in 1.008sec
... skip ...

user-workspace:~$ $ yapf -i -r -vv .           # coding style formatting
Reformatting ./server.py
Reformatting ./flaskr/utils.py
Reformatting ./flaskr/config.py
... skip ...
```

## Editor Configurations
There are some basic configurations of common editors for this repository. Please note that these settings might need to be customized according to your environment or be integrated with the configurations of the other submodules in the ITH website repository. It is also welcome to provide configurations for more editors or update the existing files with more appropriate settings.

- [VS Code](editors/vscode/): copy to `/path/to/ITH/website/`, remame it as `.vscode/`, and restart VS Code.