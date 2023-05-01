# ITH Website

## Development

-   Enter the [ITH workspace](https://playlab.computing.ncku.edu.tw:4001/ITH/workspace)
-   Clone this repo
    ```shell
    $ git clone https://github.com/Botang-l/AIoT_code
    $ cd AIoT_code/
    ```
-   Install the Python dependencies
    ```shell
    $ pip3 install -r deployment/requirements.txt
    ```
-   Run the server on port 5000
    ```shell
    $ python3 server.py
    ```
- Look at [Developer Guide](docs/DEVELOPER.md) for more details about how to start developing this repository.
- Refer to [contributing guidelines](CONTRIBUTING.md) for the conventions and rules that contributors should follow.

## Project structure

```
AIoT_code/
├── data/
│   ├── raw/                    # raw data
│   ├── processed/              # processed data
│   └── models/                 # saved models weights
├── reinforcement_learning/     # RL related code and parameters
│   ├── data/                   # RL-specific folder for storing data required for model training
│   ├── models/                 # RL-specific folder for storing trained models
│   ├── utils.py                # RL-specific utility functions
│   ├── train.py                # main code for training RL model
│   └── config.json             # parameter settings for RL model
├── time_series_model/          # LSTM related code and parameters
│   ├── data/                   # LSTM-specific folder for storing data required for model training
│   ├── models/                 # LSTM-specific folder for storing trained models
│   ├── utils.py                # LSTM-specific utility functions
│   ├── train.py                # main code for training LSTM model
│   └── config.json             # parameter settings for LSTM model
├── deployment/                 # configuration for deployment
│   └── requirements.txt        # Python package list
├── docs/                       
│   ├── DEVELOPER.md            # developer guide
│   └── editors/                # example configurations for editors
├── config.json                 # parameter settings for the overall project
└── main.py
```
