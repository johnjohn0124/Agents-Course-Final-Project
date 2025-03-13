import pdb
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
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
        # eot token id will be \n
        text = "\n".join(traj["turns"])
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
        # output: [B, T, 1]
        reward = self.reward_head(last_hidden_state[:, :, :])
        return reward.squeeze(-1)  # Output shape: (batch_size,)


def get_loss(input_ids, rewards, predicted_rewards, pad_token_id=50256):
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # Binary classification loss
    rewards_mask = (input_ids != pad_token_id)
    rewards_expanded = rewards.unsqueeze(1).expand(-1, input_ids.shape[1]).float()

    loss_unmasked = criterion(predicted_rewards, rewards_expanded)
    loss_masked = loss_unmasked * rewards_mask.float()
    loss = loss_masked.sum() / rewards_mask.float().sum()

    return loss


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

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rewards = batch["reward"].to(device)

            # the pad token id

            optimizer.zero_grad()
            predicted_rewards = model(input_ids, attention_mask)
            loss = get_loss(input_ids, rewards, predicted_rewards)
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
                loss = get_loss(input_ids, rewards, predicted_rewards)
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

    eot_results = []

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            input_ids = batch["input_ids"].to(device)
            input_hash = hash(input_ids)
            attention_mask = batch["attention_mask"].to(device)
            predicted_rewards = model(input_ids, attention_mask).squeeze().cpu().numpy()
            eot_idxs = torch.argwhere(input_ids.squeeze() == 198).squeeze()

            eot_rewards = [predicted_rewards[i] for i in eot_idxs]
            true_reward = batch['reward'].cpu().item()

            acc = np.mean((np.array(eot_rewards) > 0.5) == int(true_reward))
            var = np.var(true_reward)

            all_rewards.append(predicted_rewards)

            eot_results.append(
                {
                    'eot_rewards': [float(r) for r in eot_rewards],
                    'true reward': float(true_reward),
                    'accuracy': float(acc),
                    'variance': float(var),
                    'input_hash': input_hash
                }
            )

    all_var = np.mean([r['variance'] for r in eot_results])
    all_acc = np.mean([r['accuracy'] for r in eot_results])

    print(f"Reward Estimate Variance: {all_var:.4f}")
    print(f"Reward Estimate Accuracy: {all_acc:.4f}")

    return {"variance": all_var, "accuracy": all_acc, 'eot_results': eot_results}


import os
import json
import random
import torch

def load_data_subset(data_path, subset_size, seed):
    with open(data_path, "r") as f:
        data = json.load(f)
    random.seed(seed)
    random.shuffle(data)  
    return data[:subset_size]  

def compute_pairwise_agreement(all_rewards):
    n = all_rewards.shape[0] 
    pairwise_agreement = 0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):  # Compare every pair (i, j)
            pairwise_agreement += (all_rewards[i] > 0.5).eq(all_rewards[j] > 0.5).float().mean().item()
            count += 1

    return pairwise_agreement / count if count > 0 else 0  # Normalize by number of pairs

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
                final_path = f"{save_dir}/{model_name}_seed{seed}_data{data_size}_final.pth"
                best_path = f"{save_dir}/{model_name}_seed{seed}_data{data_size}_best.pth"

                # Initialize model
                model = SFTRewardModel(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

                # **Check if checkpoint exists to resume training**
                if not os.path.exists(best_path):
                    train_sft_reward_model(model, train_subset_path, VAL_PATH, tokenizer, epochs=epochs, best_path=best_path)
                    torch.save(model.state_dict(), final_path)
                    print(f"Checkpoint saved at: {final_path}")

                print(f"Loading checkpoint for {model_name} with seed {seed} and data size {data_size}...")
                model.load_state_dict(torch.load(best_path))
                model.eval()

                # **Save model checkpoint after training**

                # Evaluate model
                print(f"Evaluating {model_name} with seed {seed} on {data_size} samples...")
                all_rewards = evaluate_sft_reward_model(model, EVAL_PATH, tokenizer)

                # Compute variance and pairwise agreement
                variance = all_rewards['variance']
                accuracy = all_rewards['accuracy']
                eot_results = all_rewards['eot_results']

                # Store results
                results.append({
                    "model": model_name,
                    "seed": seed,
                    "data_size": data_size,
                    "variance": variance,
                    "accuracy": accuracy,
                    'results': eot_results 
                })

                with open('sft_results.json', 'w') as f:
                    json.dump(results, f)

    print("All experiments completed!")
    return results




if __name__=="__main__":
    experiment_results = run_sft_experiments()
    with open('sft_results.json', 'w') as f:
        json.dump(experiment_results, f)

