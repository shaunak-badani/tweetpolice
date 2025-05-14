# TweetPolice

> Fine tuning an LLM model to identify if the text contains hate speech or not.

- This application is deployed and can be found [here](https://tweetpolice.azurewebsites.net/).

### Dataset

- The dataset was obtained from the git repository for [hate speech and offensive language](https://github.com/t-davidson/hate-speech-and-offensive-language).

### How to run

- Make virtual environment
- Install requirements:
    ```
    pip install -r requirements.txt
    ```
    - Please note that the above requirements are for torch with Rocm support (AMD GPU). Please install the corresponding cuda libraries for NVIDIA GPU.

- Run the fine tuning script:
    ```
    python main.py
    ```