# LLM Summarization Tool README
This LLM Text Summarization tool allows you to summarize text from either a .txt or .docx file or from inputs.

Note: I run this app in a 16gb ram, and 8gb Vram computer and consumes about 6gb of Vram for the chosen model which is Llama-2 7B parameters(https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ)

## Installation
1. Clone the repository:

    ```bash
    git clone https://github.com/Vaaanc/llm-summarization-tool.git
    cd llm-summarization-tool
    ```

2. Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the Streamlit app, execute the following command in your terminal:

```bash
streamlit run app.py
