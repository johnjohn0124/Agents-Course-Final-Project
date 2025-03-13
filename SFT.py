import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


TRAIN_PATH = "input_data/twenty-questions/train_transformed.json"
VAL_PATH = "input_data/twenty-questions/eval_transformed.json"
EVAL_PATH = "input_data/twenty-questions/eval_transformed.json"

class RewardDataset(Dataset):
    """
    Custom dataset for reward regression from the 20Q task.
    """

    def __init__(self, data_path, tokenizer):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj = self.data[idx]
        text = " ".join(traj["turns"])
        reward = float(traj["guessed"])  # Binary reward: 1 if guessed correctly, else 0

        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "reward": torch.tensor(reward, dtype=torch.float),
        }



class SFTRewardModel(nn.Module):
    """
    A GPT-2-based reward model for direct reward regression.
    """

    def __init__(self, model_name="gpt2-medium"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size

        # Replace the LM head with a linear layer that outputs a single scalar
        self.reward_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        reward = self.reward_head(last_hidden_state[:, -1, :])
        return reward.squeeze(-1)  # Output shape: (batch_size,)




def train_sft_reward_model(model, train_path, val_path, tokenizer, best_path, epochs=3, batch_size=4, lr=2e-5, device="cuda"):
    """
    Train the SFT reward model on the 20Q dataset.
    """
    train_dataset = RewardDataset(train_path, tokenizer)
    val_dataset = RewardDataset(val_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # Binary classification loss

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rewards = batch["reward"].to(device)

            optimizer.zero_grad()
            predicted_rewards = model(input_ids, attention_mask)
            loss = criterion(predicted_rewards, rewards)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                rewards = batch["reward"].to(device)

                predicted_rewards = model(input_ids, attention_mask)
                loss = criterion(predicted_rewards, rewards)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            print(f"PREV BEST LOSS: {best_val_loss}")
            print(f"NEW BEST LOSS: {avg_val_loss}")
            best_val_loss = avg_val_loss
            # TODO: save the model
            torch.save(model.state_dict(), best_path)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

    print("Training complete!")



def evaluate_sft_reward_model(model, eval_path, tokenizer, device="cuda"):
    """
    Evaluate the SFT reward model and compute reward estimate variance and agreement.
    """
    eval_dataset = RewardDataset(eval_path, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    model.to(device)
    model.eval()

    all_rewards = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            predicted_rewards = model(input_ids, attention_mask).cpu().numpy()
            all_rewards.append(predicted_rewards)

    all_rewards = torch.tensor(all_rewards).flatten()
    variance = all_rewards.var().item()
    agreement = (all_rewards > 0.5).float().mean().item()

    print(f"Reward Estimate Variance: {variance:.4f}")
    print(f"Reward Estimate Agreement: {agreement:.4f}")

    return {"variance": variance, "agreement": agreement}



import os
import json
import random

def load_data_subset(data_path, subset_size, seed):
    """Loads a subset of the dataset."""
    with open(data_path, "r") as f:
        data = json.load(f)
    random.seed(seed)
    random.shuffle(data)  # Shuffle to get a diverse subset
    return data[:subset_size]  # Return only the required subset

def run_sft_experiments(
    seeds=[42, 43, 44], 
    model_scales=["gpt2", "gpt2-medium", "gpt2-large"], 
    data_diets=[100, 500, 1000, 5000, 10000], 
    epochs=3, 
    save_dir="checkpoints/twenty-questions/sft/"
):
    """Runs training with different seeds, model scales, and dataset sizes."""
    os.makedirs(save_dir, exist_ok=True)  # Create checkpoint directory
    results = []

    for seed in seeds:
        torch.manual_seed(seed)
        
        for model_name in model_scales:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token  # Fix padding error

            for data_size in data_diets:
                print(f"\nTraining {model_name} with seed {seed} on {data_size} samples...")

                # Load a subset of the dataset
                # for now, keep the same seed
                train_data_subset = load_data_subset(TRAIN_PATH, data_size, seed=42)
                train_subset_path = f"input_data/twenty-questions/subsets/train_{data_size}.json"
                
                # Save subset for reference
                with open(train_subset_path, "w") as f:
                    json.dump(train_data_subset, f)

                # Define checkpoint path
                checkpoint_path = f"{save_dir}/{model_name}_seed{seed}_data{data_size}_final.pth"
                best_path = f"{save_dir}/{model_name}_seed{seed}_data{data_size}_best.pth"

                # Initialize model
                model = SFTRewardModel(model_name)

                # **Check if checkpoint exists to resume training**
                if os.path.exists(checkpoint_path):
                    print(f"Loading checkpoint for {model_name} with seed {seed} and data size {data_size}...")
                    model.load_state_dict(torch.load(checkpoint_path))
                    model.eval()

                # Train model on dataset subset
                train_sft_reward_model(model, train_subset_path, VAL_PATH, tokenizer, epochs=epochs, best_path=best_path)

                # **Save model checkpoint after training**
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at: {checkpoint_path}")

                # Evaluate model
                print(f"Evaluating {model_name} with seed {seed} on {data_size} samples...")
                eval_metrics = evaluate_sft_reward_model(model, EVAL_PATH, tokenizer)

                # Store results
                results.append({
                    "model": model_name,
                    "seed": seed,
                    "data_size": data_size,
                    "variance": eval_metrics["variance"],
                    "agreement": eval_metrics["agreement"],
                })

    print("All experiments completed!")
    return results



experiment_results = run_sft_experiments()