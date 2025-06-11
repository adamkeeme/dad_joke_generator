import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import os
import logging

# configs
MODEL_DIR = "fine_tuned_distilgpt2_dad_jokes_only"
FEEDBACK_FILE = "jokes_feedback_for_rlhf.csv"
MAX_JOKE_LENGTH_GENERATION = 64
GENERATION_TEMP = 0.8
GENERATION_TOP_K = 50
GENERATION_TOP_P = 0.9
GENERATION_NO_REPEAT_NGRAM_SIZE = 2

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# model and tokenizer
@st.cache_resource
def load_fine_tuned_model_and_tokenizer(model_dir):
    logger.info(f"Loading model and tokenizer from '{model_dir}'...")
    try:
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            st.error(f"Model directory '{model_dir}' not found or is empty.")
            logger.error(f"Model directory '{model_dir}' not found or empty.")
            return None, None
        
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        inference_device = "cpu" #default
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            inference_device = "mps"
        elif torch.cuda.is_available():
            inference_device = "cuda"
        model.to(torch.device(inference_device))
        logger.info(f"Using device for inference: {inference_device}")
        
        model.eval()
        logger.info("Fine-tuned model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading fine-tuned model: {e}")
        logger.error(f"Error loading fine-tuned model from {model_dir}: {e}", exc_info=True)
        return None, None

# generate joke
def generate_joke_text(model, tokenizer, prompt=""):
    if model is None or tokenizer is None:
        return "Model not loaded. Cannot generate joke."
    
    try:
        device = model.device

        if prompt:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        else:
            # explicitly create a starting point for generation when no prompt is given
            # this is crucial for getting different jokes on each run
            input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)

        current_prompt_length = input_ids.shape[1]
        
        logger.info(f"Generating joke. Prompt: '{prompt}'")
        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=MAX_JOKE_LENGTH_GENERATION + current_prompt_length,
            temperature=GENERATION_TEMP,
            top_k=GENERATION_TOP_K,
            top_p=GENERATION_TOP_P,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=GENERATION_NO_REPEAT_NGRAM_SIZE,
            do_sample=True # Explicitly ensuring sampling is on
        )

        if not output_sequences.tolist() or not output_sequences[0].tolist():
             logger.error("model.generate() returned empty sequence.")
             return "Could not generate (empty sequence from model)."

        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        if prompt and generated_text.startswith(prompt):
            final_text = generated_text[len(prompt):].strip()
        else:
            final_text = generated_text.strip()
        
        final_text = final_text.split('\n')[0].strip()
        logger.info(f"Final processed text for UI: '{final_text}'")
        
        return final_text if final_text else "Hmm, I'm speechless. Try again!"

    except Exception as e:
        logger.error(f"Error during joke generation: {e}", exc_info=True)
        return f"Oops, error during generation: {e}"

# feedback saving
def save_feedback(joke, reaction):
    try:
        new_entry = pd.DataFrame([{"timestamp": pd.Timestamp.now(), "joke": joke, "reaction": reaction}])
        if not os.path.exists(FEEDBACK_FILE):
            new_entry.to_csv(FEEDBACK_FILE, index=False)
        else:
            new_entry.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
        logger.info(f"Feedback saved: {reaction}")
    except Exception as e:
        st.error(f"Could not save feedback: {e}")
        logger.error(f"Error saving feedback: {e}", exc_info=True)

# streamlit
def main_gui():
    st.set_page_config(page_title="Dad-Joke Generator", layout="centered") # Centered for minimalism
    st.title("Dad-Joke Generator")

    model, tokenizer = load_fine_tuned_model_and_tokenizer(MODEL_DIR)

    if model is None or tokenizer is None:
        st.warning("Model not loaded. Please ensure training completed successfully.")
        return

    if 'current_joke' not in st.session_state:
        st.session_state.current_joke = ""
    
    if st.button("Tell me a joke!", type="primary", use_container_width=True):
        with st.spinner("Thinking..."):
            st.session_state.current_joke = generate_joke_text(model, tokenizer)

    if st.session_state.current_joke:
        st.markdown("---")
        st.markdown(f"> {st.session_state.current_joke}") 
        st.markdown("---")

        cols = st.columns(2)
        if cols[0].button("👍 Good", use_container_width=True):
            save_feedback(st.session_state.current_joke, "up")
            st.toast("Thanks! 😄", icon="👍")
        if cols[1].button("👎 Meh", use_container_width=True):
            save_feedback(st.session_state.current_joke, "down")
            st.toast("Noted! 😉", icon="👎")
            
if __name__ == "__main__":
    main_gui()