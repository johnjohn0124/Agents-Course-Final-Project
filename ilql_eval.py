import json
import os
import numpy as np
import random
import pdb
from copy import deepcopy
from tqdm import tqdm
from pprint import pprint
from omegaconf import OmegaConf
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from ilql_utils import ILQLModel, Trajectory

def evaluate_value_predictions(model_path, val_data_path, output_path, config_path):
    """
    Evaluate value function predictions from a trained ILQL model.
    
    Args:
        model_path: Path to the trained ILQL model checkpoint
        val_data_path: Path to validation data
        output_path: Where to save the evaluation results
        config_path: Path to configuration file
    """
    # Load configuration
    config = OmegaConf.load(config_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load model
    backbone_model = AutoModelForCausalLM.from_pretrained(config.model_name, 
                                                            device_map={'': torch.device(0)},
                                                            torch_dtype=torch.bfloat16)
    
    ilql_model = ILQLModel(backbone_model,
                            peft_config=None,
                            checkpoint_dir=model_path,
                            reward_dist_config=config.reward,
                            cql_loss_coeff=config.training.cql_loss_coeff,
                            mc_loss_coeff=config.training.mc_loss_coeff,
                            polyak_coeff=config.training.polyak_coeff,
                            gamma=config.training.gamma)
    
    # Load the trained checkpoint
    ilql_model.load_ilql_checkpoint(model_path)
    ilql_model.eval()
    
    # Load validation data
    with open(val_data_path, 'r') as f:
        val_data = json.load(f)
    
    trajectories = [Trajectory(d, tokenizer) for d in val_data]
    
    # Store results
    results = []
    
    for idx, trajectory in enumerate(tqdm(trajectories)):
        # Generate a unique ID for this trajectory
        trajectory_id = f"{trajectory.secret}_{idx}"
        
        # Get V values for the entire trajectory
        vs, tokens = ilql_model.observe_v_values(trajectory, tokenizer)
        
        # Extract turn-level predictions
        turn_predictions = []
        
        # Get positions at the end of assistant turns
        assistant_eot_idxs = trajectory.assistant_eot_idxs
        
        
        valid_predictions = 0
        # Extract V values at these positions
        for i, pos in enumerate(assistant_eot_idxs):
            if pos < vs.shape[1]:  # Make sure position is within bounds
                v_at_turn = vs[0, pos, 0].item()  # Get scalar value
                turn_predictions.append({
                    'turn_idx': i,
                    'token_pos': pos,
                    'v_value': v_at_turn,
                    'predicted_success': v_at_turn > 0  # Binary prediction
                })
                valid_predictions += 1
            else:
                print(f"Warning: Position {pos} out of bounds for V tensor with shape {vs.shape} in trajectory {idx}")
        
        # Store results for this trajectory
        result = {
            'trajectory_id': trajectory_id,
            'secret': trajectory.secret,
            'ground_truth_success': trajectory.guessed,
            'turn_predictions': turn_predictions
        }
        
        results.append(result)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Compute metrics
    compute_metrics(results, output_path)
    
    return results

def compute_metrics(results, output_path):
    """Compute evaluation metrics from the prediction results"""
    
    # Accuracy at each turn position
    turn_accuracies = {}
    
    for result in results:
        for pred in result['turn_predictions']:
            turn_idx = pred['turn_idx']
            correct = pred['predicted_success'] == result['ground_truth_success']
            
            if turn_idx not in turn_accuracies:
                turn_accuracies[turn_idx] = []
            
            turn_accuracies[turn_idx].append(correct)
    
    # Compute average accuracy at each turn
    avg_accuracies = {
        turn: sum(accs) / len(accs) 
        for turn, accs in turn_accuracies.items() 
        if accs  # Only include turns with data
    }
    
    # Compute overall accuracy
    all_preds = [pred for result in results for pred in result['turn_predictions']]
    overall_acc = sum(pred['predicted_success'] == result['ground_truth_success'] 
                     for result, pred_group in zip(results, [r['turn_predictions'] for r in results])
                     for pred in pred_group) / len(all_preds)
    
    # Output metrics
    metrics = {
        'overall_accuracy': overall_acc,
        'turn_accuracies': avg_accuracies
    }
    
    # Save metrics
    metrics_path = output_path.replace('.json', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def run_evaluations(config, val_data_path):
    # Configuration variables
    save_dir = os.path.join(config.saving.save_basedir,
                            config.run_group_name)
    seeds = [42]
    model_scales = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    data_diets = [100]
    
    results_dir = "./evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    for seed in seeds:
        for model in model_scales:
            for diet in data_diets:
                print(f"Start Evaluating: {model}_{seed}_{diet}")
                model_path = f"{save_dir}/{model}_{seed}_{diet}/final_checkpoint"
                
                if os.path.exists(model_path):
                    output_path = f"{results_dir}/{model}_{seed}_{diet}_results.json"
                    
                    evaluate_value_predictions(
                        model_path = model_path,
                        val_data_path = val_data_path,
                        output_path = output_path,
                        config_path = f"{save_dir}/{model}_{seed}_{diet}/config.yaml"
                    )
                    print(f"Evaluated: {model}_{seed}_{diet}")
                else:
                    print(f"Skipping evaluation for {model}_{seed}_{diet} - checkpoint not found")

def compute_cross_run_metrics(results_dir):
    """Compute metrics across different runs to analyze variance and agreement"""
    
    # Load all results files
    result_files = [f for f in os.listdir(results_dir) if f.endswith("_results.json")]
    all_results = {}
    
    for file in result_files:
        run_name = file.replace("_results.json", "")
        with open(os.path.join(results_dir, file), 'r') as f:
            all_results[run_name] = json.load(f)
    
    # Group results by trajectory ID across runs
    trajectory_predictions = {}
    
    for run_name, results in all_results.items():
        for trajectory in results:
            traj_id = trajectory['trajectory_id']
            
            if traj_id not in trajectory_predictions:
                trajectory_predictions[traj_id] = {
                    'ground_truth': trajectory['ground_truth_success'],
                    'runs': {}
                }
            
            trajectory_predictions[traj_id]['runs'][run_name] = {
                'turn_predictions': trajectory['turn_predictions']
            }
    
    # Compute variance and agreement metrics
    variance_by_turn = {}
    agreement_by_turn = {}
    
    for traj_id, traj_data in trajectory_predictions.items():
        # Skip trajectories that don't appear in all runs
        if len(traj_data['runs']) < len(all_results):
            continue
            
        # Get predictions at each turn across runs
        for turn_idx in range(20):  # Assuming up to 20 turns
            values_at_turn = []
            
            for run_name, run_data in traj_data['runs'].items():
                # Find prediction for this turn if it exists
                for pred in run_data['turn_predictions']:
                    if pred['turn_idx'] == turn_idx:
                        values_at_turn.append(pred['v_value'])
                        break
            
            if values_at_turn:
                # Compute variance
                if turn_idx not in variance_by_turn:
                    variance_by_turn[turn_idx] = []
                
                variance = np.var(values_at_turn)
                variance_by_turn[turn_idx].append(variance)
                
                # Compute agreement (% of pairs that agree on binary prediction)
                binary_preds = [v > 0 for v in values_at_turn]
                agreements = 0
                total_pairs = 0
                
                for i in range(len(binary_preds)):
                    for j in range(i+1, len(binary_preds)):
                        agreements += binary_preds[i] == binary_preds[j]
                        total_pairs += 1
                
                agreement_rate = agreements / total_pairs if total_pairs > 0 else float('nan')
                
                if turn_idx not in agreement_by_turn:
                    agreement_by_turn[turn_idx] = []
                
                agreement_by_turn[turn_idx].append(agreement_rate)
    
    # Average metrics across trajectories
    avg_variance_by_turn = {
        turn: sum(variances) / len(variances)
        for turn, variances in variance_by_turn.items()
    }
    
    avg_agreement_by_turn = {
        turn: sum(agreements) / len(agreements)
        for turn, agreements in agreement_by_turn.items()
    }
    
    # Save cross-run metrics
    cross_run_metrics = {
        'variance_by_turn': avg_variance_by_turn,
        'agreement_by_turn': avg_agreement_by_turn
    }
    
    with open(os.path.join(results_dir, "cross_run_metrics.json"), 'w') as f:
        json.dump(cross_run_metrics, f, indent=2)
    
    return cross_run_metrics