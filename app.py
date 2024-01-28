import torch
import streamlit as st

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from helpers import read_text

def main():
    llm_model = LLMModel("TheBloke/Llama-2-7B-chat-GPTQ")
    with st.spinner("Model loading..."):
        llm_model.initialize_pipeline()
        
    st.title("LLM Summarize App")
    
    # Upload file input
    uploaded_file = st.file_uploader("Choose a file", ['docx', 'txt'])
    
    st.write("Or")
    
    # Text area for user input
    user_input = st.text_area("Input text")
    
    # Button to summarize the text
    if st.button("Summarize"):
        # Summarize only if there is uploaded file w/ text on it or text input
        if uploaded_file:
            text = read_text(uploaded_file)
            if not text:
                st.warning("Uploaded file doesn't contain text")
                return
            
        elif user_input:
            text = user_input
        else:
            # Handle the case where neither file nor text input is available
            st.warning("Please upload a file or enter text to summarize.")
            return
        
        with st.spinner("Summarizing..."):
            # Summarize the text when the button is clicked
            summarized_text = llm_model.summarize_text(text)

            # Display the processed text
            st.divider()
            st.write("**Summarized text below:**")
            st.write(summarized_text)
    
class LLMModel:
    def __init__(self, model_name):
        """
        Initialize the LLMModel instance.
        
        Args:
        - model_name (str): model name from Huggingface(https://huggingface.co/models) or the file location of a model
        """
        self.pipe = None
        self.model_name = model_name

    def summarize_text(_self, text):
        """ 
        Summarizes the text

        Args:
        - text (str): Sentence/Article to summarize

        Returns:
        - str: Summarized text
        """

        prompt_template = """
        Summarize the text below and generate clear and concise summary

        {text}

        Summary:

        """
        
        # Generate the summarized text using the pipeline
        with torch.no_grad():
            processed_text = _self.pipe(prompt_template.format(text=text))[0]['generated_text']
        return processed_text
    
    @st.cache_resource
    def load_model_and_tokenizer(_self):
        """
        loads the model and tokenizer
        """
        model = AutoModelForCausalLM.from_pretrained(_self.model_name, 
                                                     device_map="cuda", 
                                                     trust_remote_code=False, 
                                                     revision="main",
                                                     torch_dtype=torch.float16)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(_self.model_name, use_fast=True)
        
        return model, tokenizer
    
    def initialize_pipeline(_self):
        """
        initializes the pipeline for inference
        """
        # Load the pre-trained model and tokenizer
        model, tokenizer = _self.load_model_and_tokenizer()
        
        # Create the pipeline for text summarization
        _self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.75,
            top_p=0.95,
            top_k=40,
            return_full_text = False
        )
    
if __name__ == "__main__":
    main()