# Dad Joke Generator

This project is a fun demonstration of fine-tuning a smaller language model, `distilgpt2`, to generate dad jokes. The trained model is served through a simple web interface created with Streamlit, allowing users to generate jokes and provide feedback.

## Code Structure

The repository is organized as follows:

* **`run.py`**: This is the main training script. It handles loading the dad jokes from a CSV, cleaning the data, setting up the dataset, and fine-tuning the `distilgpt2` model.
* **`gui.py`**: This script launches a Streamlit web application. It loads the fine-tuned model and provides an interface for users to generate jokes and give feedback.
* **`fine_tuned_distilgpt2_dad_jokes_only/`**: This directory contains the output of the training script—the fine-tuned model and tokenizer files ready for use.
* **`jokes_feedback_for_rlhf.csv`**: This CSV file stores the jokes and the user feedback (thumbs up/down) gathered from the GUI.

## Setup Instructions

To get this project running locally, follow these steps ([[video demo](url)](https://github.com/user-attachments/assets/ce91d9ba-4bda-4e78-a3d7-2cde4cae0c30)):



1.  **Clone the Repository**
2.  **Install Dependencies**: Make sure you have Python 3.8+ installed. You can install the necessary packages using pip:
    ```bash
    pip install torch transformers pandas streamlit
    ```
3.  **Run the GUI**: To start the web application and generate jokes from the pre-trained model, run:
    ```bash
    streamlit run gui.py
    ```
    If you want to train the model yourself, you will first need to run the training script:

    ```bash
    python run.py
    ```

## Feedback for Reinforcement Learning

This application includes a feature to collect user feedback on the generated jokes. When a user clicks the "👍 Good" or "👎 Meh" button, the joke and the corresponding feedback ("up" or "down") are saved to a CSV file named `jokes_feedback_for_rlhf.csv`.

This collected data is valuable for future enhancements. It is formatted to be suitable for Reinforcement Learning from Human Feedback (RLHF), a technique that can be used to further refine the model's performance by training it to generate jokes that are more likely to receive positive feedback.
