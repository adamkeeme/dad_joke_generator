import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, get_linear_schedule_with_warmup
import pandas as pd
import os
import logging
import random
from tqdm import tqdm

# base model & tokenizer
BASE_MODEL_NAME = "distilgpt2"
FINE_TUNED_MODEL_SAVE_PATH = "fine_tuned_distilgpt2_dad_jokes_only"
TOKENIZER_NAME = "gpt2"

# hyperparameters
NUM_EPOCHS = 4 
BATCH_SIZE = 8
LEARNING_RATE = 2e-5 
KL_DIVERGENCE_WEIGHT = 0
MAX_SEQ_LENGTH = 64

DAD_JOKES_CSV_PATH = "/Users/pc/Documents/GitHub/dad_joke_generator/dad_jokes.csv" 

# logging
LOG_FILE_NAME = "training_run_output.log" 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_NAME),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# helper function to clean joke text
def clean_joke_text(text):
    if not isinstance(text, str):
        return ""
    return text.strip()

# dataset
class JokesDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_masks = []
        logger.info(f"Tokenizing {len(texts)} jokes...")
        for text in tqdm(texts, desc="Tokenizing Data"):
            encoding = tokenizer(
                text + tokenizer.eos_token,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            self.input_ids.append(encoding['input_ids'].squeeze(0))
            self.attention_masks.append(encoding['attention_mask'].squeeze(0))
        logger.info("Tokenization done.")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.input_ids[idx].clone() 
        }

def load_and_prepare_data(tokenizer, max_seq_len, num_jokes_to_use=None):
    logger.info(f"Loading data from CSV: {DAD_JOKES_CSV_PATH}")
    if not os.path.exists(DAD_JOKES_CSV_PATH):
        logger.error(f"Data file not found: {DAD_JOKES_CSV_PATH}. Please check the path.")
        raise FileNotFoundError(f"Data file not found: {DAD_JOKES_CSV_PATH}")
    try:
        df_dad_jokes = pd.read_csv(DAD_JOKES_CSV_PATH)
        if 'joke' not in df_dad_jokes.columns:
            logger.error(f"'joke' column not found in {DAD_JOKES_CSV_PATH}. Available columns: {df_dad_jokes.columns.tolist()}")
            raise ValueError("'joke' column missing from CSV.")
        
        df_dad_jokes.dropna(subset=['joke'], inplace=True)
        jokes = df_dad_jokes['joke'].astype(str).tolist()
    except Exception as e:
        logger.error(f"Error loading or parsing CSV {DAD_JOKES_CSV_PATH}: {e}")
        raise

    logger.info(f"Loaded {len(jokes)} raw jokes.")
    cleaned_jokes = [clean_joke_text(joke) for joke in jokes if clean_joke_text(joke)]
    logger.info(f"Retained {len(cleaned_jokes)} cleaned jokes for training.")
    
    if not cleaned_jokes:
        logger.error("No jokes available after cleaning. Cannot train.")
        raise ValueError("No valid joke data to train on after cleaning.")

    if num_jokes_to_use and 0 < num_jokes_to_use < len(cleaned_jokes):
        logger.info(f"Sampling {num_jokes_to_use} jokes for training.")
        cleaned_jokes = random.sample(cleaned_jokes, num_jokes_to_use)
    
    return JokesDataset(cleaned_jokes, tokenizer, max_seq_len)



def train_model():
    logger.info("Starting training")

    # CUDA for quest, mps for mac, cpu for backup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built(): 
        device = torch.device("mps")
        logger.info("Using MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_and_prepare_data(tokenizer, MAX_SEQ_LENGTH)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    student_model_config = GPT2Config.from_pretrained(BASE_MODEL_NAME) 
    student_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_NAME, config=student_model_config)
    student_model.resize_token_embeddings(len(tokenizer)) 
    student_model.to(device)

    # teacher model for KL loss calculation
    teacher_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_NAME)
    teacher_model.resize_token_embeddings(len(tokenizer))
    teacher_model.to(device)
    teacher_model.eval()

    optimizer = AdamW(student_model.parameters(), lr=LEARNING_RATE)
    num_training_steps = len(train_dataloader) * NUM_EPOCHS
    num_warmup_steps = int(0.05 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    student_model.train()
    logger.info(f"Starting training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        total_lm_loss_epoch = 0
        total_kl_loss_epoch = 0 

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs_student = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            lm_loss = outputs_student.loss

            kl_div_loss_value = 0.0
            if KL_DIVERGENCE_WEIGHT > 0 or True:
                with torch.no_grad():
                    outputs_teacher = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                prob_teacher = F.softmax(outputs_teacher.logits, dim=-1)
                log_prob_student = F.log_softmax(outputs_student.logits, dim=-1)
                kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean', log_target=False)
                kl_div_loss = kl_loss_fn(log_prob_student, prob_teacher)
                kl_div_loss_value = kl_div_loss.item()
            
            training_loss = lm_loss + KL_DIVERGENCE_WEIGHT * kl_div_loss_value 
            training_loss.backward()
            optimizer.step()
            scheduler.step()

            total_lm_loss_epoch += lm_loss.item()
            total_kl_loss_epoch += kl_div_loss_value
            progress_bar.set_postfix({"LM Loss": f"{lm_loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_lm_loss = total_lm_loss_epoch / len(train_dataloader)
        avg_kl_loss_calc = total_kl_loss_epoch / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} Summary: Avg LM Loss: {avg_lm_loss:.4f}, Avg KL Loss (calc): {avg_kl_loss_calc:.4f}, Final LR: {scheduler.get_last_lr()[0]:.2e}")

    logger.info(f"Training done. Saving model to {FINE_TUNED_MODEL_SAVE_PATH}")
    if not os.path.exists(FINE_TUNED_MODEL_SAVE_PATH):
        os.makedirs(FINE_TUNED_MODEL_SAVE_PATH)
    student_model.save_pretrained(FINE_TUNED_MODEL_SAVE_PATH)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_SAVE_PATH)
    logger.info("Model and tokenizer saved successfully.")

if __name__ == "__main__":
    try:
        if not os.path.exists(FINE_TUNED_MODEL_SAVE_PATH):
            os.makedirs(FINE_TUNED_MODEL_SAVE_PATH, exist_ok=True)
        elif not os.path.isdir(FINE_TUNED_MODEL_SAVE_PATH):
            logger.error(f"Save path {FINE_TUNED_MODEL_SAVE_PATH} exists but is not a directory.")
            exit(1)
        train_model()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"A critical error occurred: {e}")
        logger.error("Training aborted.")