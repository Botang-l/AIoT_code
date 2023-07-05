# AIoT_code
## Development

-   Clone this repo
    ```shell
    $ git clone https://github.com/Botang-l/AIoT_code
    $ cd AIoT_code/
    ```
-   Install the Python dependencies
    ```shell
    $ pip3 install -r deployment/requirements.txt
    ```
-   Run the code
    ```shell
    $ python3 main.py
    ```
- Look at [Developer Guide](docs/DEVELOPER.md) for more details about how to start developing this repository.
- Refer to [Contributing Guidelines](CONTRIBUTING.md) for the conventions and rules that contributors should follow.

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
├── time_series_model/          # RNN related code and parameters
│   ├── data/                   # RNN-specific folder for storing data required for model training
│   ├── models/                 # RNN-specific folder for storing trained models
│   ├── utils.py                # RNN-specific utility functions
│   ├── train.py                # main code for training LSTM model
│   └── config.json             # parameter settings for LSTM model
├── deployment/                 # configuration for deployment
│   └── requirements.txt        # Python package list
├── docs/                       
│   ├── DEVELOPER.md            # developer guide
│   └── editors/                # example configurations for editors
├── dashboard/                  # dashboard code
│   ├── dashboard.py           
├── config.json                 # parameter settings for the overall project
└── main.py
```
## how to maintain this repo
每次增修內容前請依循下列流程進行：
1. Pull origin/develop 最新版本
    ```shell
    $ git pull origin develop
    ```
2. 在 local 新增 branch 並切換
    ```shell
    $ git checkout -b <NEW_BRANCH_NAME>
    ```
3. 編輯完成後進行 commit
    ```shell
    $ yapf -i -r -vv .
    $ git add .
    $ git commit -m "COMMIT_MSG"
    ```
4. 回到 master 再次獲取 origin/develop 的最新版本、與自己的修正合併並修正出現的 conflict
    ```shell
    $ git checkout develop
    $ git pull
    $ git checkout <NEW_BRANCH_NAME>
    $ git rebase develop
    ```
5. 將新 branch 的修正與 develop 合併並 push 到 Github
    ```shell
    $ git checkout develop
    $ git merge <NEW_BRANCH_NAME>
    $ git push
    ```
