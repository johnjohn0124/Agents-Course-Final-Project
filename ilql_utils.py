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


def apply_chat_template(messages, tokenizer):
    if tokenizer.chat_template is None:
        message_str = ''
        for message in messages:
            message_str += f"{message['role']}: {message['content']}\n"
    else:
        message_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            return_tensors='pt', tokenize=False)
    
    return message_str

class Trajectory:

    def __init__(self, data_d, tokenizer):
        if not isinstance(data_d, dict):
            pdb.set_trace()
        self.data_d = data_d
        self.turns = self.data_d['turns']
        self.n_turns = (len(self.turns) - 1) // 2
        self.guessed = self.data_d['guessed']
        self.secret = self.data_d['secret']

        self.tokenizer = tokenizer
        self.eot_id = self.tokenizer.eos_token_id

        self._string = None
        self._tokens = None
        self._user_eot_idxs = None
        self._assistant_eot_idxs = None
        
        self._string = self.string
        self._tokens = self.tokens
        self.n_tokens = len(self.tokens)
        self._user_eot_idxs = self.user_eot_idxs
        self._assistant_eot_idxs = self.assistant_eot_idxs

    @property
    def string(self):
        if self._string is None:
            
            message_roles = ['User', 'Assistant']
        
            # subject to change
            messages = []
            for i, turn in enumerate(self.turns):
                role = message_roles[i % 2]
                messages.append({'role': role.lower(),
                                 'content': turn})
            
            message_str = apply_chat_template(messages, self.tokenizer)
            
            self._string = message_str.strip()
        return self._string

    @property
    def tokens(self):
        if self._tokens is None:
            self._tokens = self.tokenizer(self.string)['input_ids']
        return self._tokens

    @property
    def user_eot_idxs(self):
        if self._user_eot_idxs is None:
            eot_idxs = []
            counter = 0
            for i in range(len(self.tokens)):
                if self.tokens[i] == self.eot_id:
                    if counter >= 2 and counter % 2 == 1:
                        eot_idxs.append(i)
                    counter += 1
            self._user_eot_idxs = eot_idxs
        return self._user_eot_idxs

    @property
    def assistant_eot_idxs(self):
        if self._assistant_eot_idxs is None:
            eot_idxs = []
            counter = 0
            
            for i in range(len(self.tokens)):
                if self.tokens[i] == self.eot_id:
                    if counter >= 2 and counter % 2 == 0:
                        eot_idxs.append(i)
                    counter += 1
            self._assistant_eot_idxs = eot_idxs
        return self._assistant_eot_idxs

    def to_dict(self):
        return {
            'turns': self.turns,
            'guessed': self.guessed,
            'secret': self.secret,
        }

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def print_cuda_memory():
    if torch.cuda.is_available():
        # Get the current device (assuming you're using the default CUDA device)
        device = torch.cuda.current_device()
        
        # Get the total memory of the GPU
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        # Get the memory currently allocated by PyTorch
        allocated_memory = torch.cuda.memory_allocated(device)
        
        # Calculate the free memory
        free_memory = total_memory - allocated_memory
        
        print('Num GPUs available:', torch.cuda.device_count())
        print('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', None))
        print(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")
        print(f"Allocated GPU memory: {allocated_memory / 1024**3:.2f} GB")
        print(f"Free GPU memory: {free_memory / 1024**3:.2f} GB")
    else:
        print("CUDA is not available on this system.")

class ILQLModel(nn.Module):
    """
    This model is only in charge of Q and V values. The two will be branch heads off of the same model.
    """

    def __init__(self,
                 backbone_lm,
                 peft_config,
                 reward_dist_config=None,
                 checkpoint_dir=None,
                 polyak_coeff=0.005,
                 mc_loss_coeff=0.0,
                 gamma=0.99,
                tau=0.7,
                cql_loss_coeff=0.005,
                cql_temp=1.0,
                q_head_bias=None):
        super().__init__()

        if checkpoint_dir is not None:
            self.backbone_lm = PeftModel.from_pretrained(self.backbone_lm,
                                                         os.path.join(checkpoint_dir, 'backbone_lm'))

        if peft_config is None:
            self.backbone_lm = backbone_lm
        else:
            self.backbone_lm = get_peft_model(backbone_lm, peft_config)

        self.reward_dist_config = reward_dist_config

        self.q_head_bias = q_head_bias
        
        self.h_dim = self.backbone_lm.config.hidden_size
        self.polyak_coeff = polyak_coeff
        self.mc_loss_coeff = mc_loss_coeff
        self.peft_config = peft_config
        self.gamma = gamma
        self.tau = tau

        # alpha
        self.cql_loss_coeff = cql_loss_coeff
        self.cql_temp = cql_temp
        self.q_head_bias = q_head_bias

        self.device = self.backbone_lm.lm_head.weight.device

        self.q_head = self._clone_existing_lm_head()

        self.v_head = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim*2),
            nn.ReLU(),
            nn.Linear(self.h_dim*2, 1)
        ).to(self.device).to(torch.bfloat16)

        self.target_q_head = deepcopy(self.q_head)

    def _clone_existing_lm_head(self):
        cloned_w = nn.Parameter(self.backbone_lm.lm_head.weight.clone())
        cloned_b = nn.Parameter(self.backbone_lm.lm_head.bias.clone()) if self.backbone_lm.lm_head.bias else None

        linear = nn.Linear(cloned_w.shape[1], cloned_w.shape[0]).to(self.device)

        linear.weight = cloned_w
        linear.bias = cloned_b
        return linear

    def step_target_q_network(self):
        # takes a step polyak-wise
        with torch.no_grad():
            for target_param, q_net_param in zip(self.target_q_head.parameters(), self.q_head.parameters()):
                target_param.data.copy_(self.polyak_coeff*q_net_param.data + (1.0-self.polyak_coeff)*target_param.data)

    def get_all_values(self, *args, **kwargs):
        outputs = self.backbone_lm(*args, output_hidden_states=True, **kwargs)
        if self.q_head_bias in [None, "None"]:
            self.q_head_bias = -(outputs['logits'].mean().cpu().item())

        last_hidden_states = outputs.hidden_states[-1]

        q_vals = self.q_head(last_hidden_states) + self.q_head_bias
        v_vals = self.v_head(last_hidden_states)
        
        with torch.no_grad():
            target_q_vals = self.target_q_head(last_hidden_states) + self.q_head_bias

        return {
            'q': q_vals,
            'target_q': target_q_vals,
            'v': v_vals,
        }

    def get_q_values(self, *args, **kwargs):
        self.backbone_lm.set_adapter('default')
        last_hidden_states = self.backbone_lm(*args, output_hidden_states=True, **kwargs).hidden_states[-1]
        q_vals = self.q_head(last_hidden_states) + self.q_head_bias
        return q_vals

    def get_utterance_rewards(self, trajectory):
        """
        Given a trajectory, return a list corresponding to the rewards for each step.

        This is where the central experimental changes take place.

        Rely on reward_dist_config for the logic here.
        """

        cfg = self.reward_dist_config

        if cfg is None:
            return [0]*((len(trajectory.data_d['turns']) - 1 ) // 2)

        assert cfg.specified, 'Must specify a reward distribution type.'

        # if prefix-conditioned blocked reward, then forget everythng and just uniformly distribute the llm scores

        if cfg.use_blocked_prefix_reward:
            # TODO: change this to allow for FUDGE rewards
            llm_scores = trajectory.data_d['secret_single']
            return llm_scores

        if cfg.no_llm_reward:
            num_turns = (len(trajectory.data_d['turns']) // 2)
            llm_scores = [None]*num_turns
            llm_dist_values = [None]*num_turns

        else:
            reward_keyname = {
                'prefix': 'secret_single',
                'sequence': 'scores',
                'cot': 'scores_cot',
            }[cfg.llm_reward_type]

            llm_scores = trajectory.data_d[reward_keyname]
            llm_dist_values = F.softmax(torch.Tensor(llm_scores) / cfg.llm_guidance_temp, dim=-1).squeeze().tolist()
        

        if cfg.use_zero_one_reward:
            final_reward = 1 if trajectory.data_d['guessed'] else 0
        else:
            final_reward = -len(llm_scores)

        if cfg.no_base_reward:
            extrinsic_rewards = [0]*len(llm_scores)
        elif cfg.use_sparse_reward:
            extrinsic_rewards = [0]*(len(llm_scores) - 1) + [final_reward]
        elif cfg.use_uniform_reward:
            extrinsic_rewards = [final_reward/len(llm_scores)]*len(llm_scores)

        # now we combine them

        if cfg.no_llm_reward:
            intrinsic_rewards = [0]*len(llm_scores)
        elif cfg.use_unconditional_llm_reward:
            intrinsic_rewards = [cfg.llm_guidance_coeff*llm_dist_values[i] for i in range(len(llm_scores))]
        elif cfg.use_conditional_llm_reward:
            if trajectory.data_d['guessed']:
                intrinsic_rewards = [cfg.llm_guidance_coeff*llm_dist_values[i] for i in range(len(llm_scores))]
            else:
                intrinsic_rewards = [0]*len(llm_scores)

        # TODO: add non-onezero reward distribution

        out_rewards = [extrinsic_rewards[i] + intrinsic_rewards[i] for i in range(len(llm_scores))]

        return out_rewards


    def prepare_trajectory_data(self, trajectory):
        # for now, batch size of 1.
        # TODO: make this work for larger batch sizes
        
        message_roles = ['User', 'Assistant']
    
        # subject to change
        # rewards_u = trajectory['scores']
        rewards_u = self.get_utterance_rewards(trajectory)
    
        messages = []
        for i, turn in enumerate(trajectory.data_d['turns']):
            role = message_roles[i % 2]
            messages.append({'role': role.lower(),
                             'content': turn})
        
        message_str = apply_chat_template(messages, trajectory.tokenizer)
        tokens = trajectory.tokenizer(message_str, return_tensors='pt').to(self.device)['input_ids']
        # tokens = trajectory.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(self.device)

        tokens_f = tokens[0]
    
        # llama 3 specific
        if trajectory.tokenizer.chat_template is None:
            eot_id = trajectory.tokenizer.encode('\n')[0]
            eot_idxs = [i for i in range(len(tokens_f)) if tokens_f[i] == eot_id]
            assistant_eot_idxs = eot_idxs[::2]
            env_eot_idxs = eot_idxs[1::2]
        else:
            eot_idxs = [i for i in range(len(tokens_f)) if tokens_f[i] == trajectory.tokenizer.eos_token_id]
            env_eot_idxs = eot_idxs[1::2]
            assistant_eot_idxs = eot_idxs[2::2]
    
        # some nice fancy operations to create a mask in between steps only
        env_eot_mask = torch.zeros_like(tokens, dtype=torch.int, device=self.device)
        env_eot_mask[:, env_eot_idxs] = +1
    
        assistant_eot_mask = torch.zeros_like(tokens, dtype=torch.int, device=self.device)
        assistant_eot_mask[:, assistant_eot_idxs] = -1
    
        rewards = torch.zeros_like(tokens, dtype=torch.bfloat16, device=self.device)
    
        pos_tens = torch.Tensor(assistant_eot_idxs).int().to(self.device)
        rewards_tens = torch.Tensor(rewards_u).to(self.device).to(torch.bfloat16)

        # possibly change this? maybe make it uniform.
        rewards[0].index_put_((pos_tens,), rewards_tens)
    
        # we assume that none of these are the same
        # values of +1 mean the assistant is generating
        # values of 0 mean that the environment is generating
        # let's shift this over by 1 to keep the rewards

        # for prefix-conditioned rewards, we'll also mask out all of the assistant tokens
        # this keeps the reward from propagating between steps

        terminals_mask = torch.ones_like(tokens)[..., None]
        terminals_mask[:, max(assistant_eot_idxs):] = 0

        if self.reward_dist_config and self.reward_dist_config.use_blocked_prefix_reward:
            running_vals = torch.cumsum(env_eot_mask + assistant_eot_mask, dim=-1)
            block_mask = (running_vals > 0).int()[..., None]

            # block_mask = torch.roll(block_mask, shifts=1, dims=1)
            # block_mask[:, 0, :] = 0

            terminals_mask = terminals_mask * block_mask
        
        return {
            'model_inputs': {
                'input_ids': tokens,
            },
            'tokens': tokens,
            'rewards': rewards,
            'rewards_u': rewards_u,
            'terminals_mask': terminals_mask
        }

    def get_values_from_trajectory(self, trajectory, tokenizer):
        trajectory_data = self.prepare_trajectory_data(trajectory)
        model_inputs = trajectory_data['model_inputs']
        all_values = self.get_all_values(**model_inputs)

        return all_values

    def select_qs(self, qs, action_tokens, trajectory):
        return torch.gather(qs[:, :-1], -1, action_tokens[:, 1:].unsqueeze(-1))

    def select_vs(self, vs, trajectory):
        return vs

    def get_loss(self, trajectory, log_values=False):
        logs = {}
        trajectory_data = self.prepare_trajectory_data(trajectory)
        model_inputs = trajectory_data['model_inputs']

        all_values = self.get_all_values(**model_inputs)

        vs = all_values['v']
        qs = all_values['q']


        target_qs = all_values['target_q']

        rewards = trajectory_data['rewards']
        action_tokens = trajectory_data['tokens']
        terminals_mask = trajectory_data['terminals_mask']

        # q_{i} [t_{i+1}]

        qs_select = self.select_qs(qs, action_tokens, trajectory)
        target_qs_select = self.select_qs(qs, action_tokens, trajectory)
        vs_select = self.select_vs(vs, trajectory)


        # TODO: think about how this works between turns.
        target_qs_select = torch.gather(target_qs[:, :-1], -1, action_tokens[:, 1:].unsqueeze(-1))

        # TODO: modify this so that it's selecting out the next one

        q_loss = self.get_q_loss(vs_select[:, 1:], qs_select, rewards[:, 1:], terminals_mask)
        v_loss, v_loss_logs = self.get_v_loss(vs_select[:, :-1], target_qs_select, terminals_mask, trajectory.guessed)
        logs.update(v_loss_logs)
        cql_loss = self.get_cql_loss(qs[:, :-1], action_tokens[:, 1:], terminals_mask)

        logs['q_loss'] = q_loss.item()
        logs['v_loss'] = v_loss.item()
        logs['cql_loss'] = cql_loss.item()

        loss = q_loss + v_loss + self.cql_loss_coeff*cql_loss

        logs['loss'] = loss.item()

        logs['q_value_mean'] = qs_select.mean().item()
        logs['v_value_mean'] = vs_select.mean().item()

        if log_values:
            logs['q_values'] = wandb.Histogram(qs_select.flatten().float().cpu().detach().numpy())
            logs['v_values'] = wandb.Histogram(vs_select.flatten().float().cpu().detach().numpy())

        return loss, logs

    def observe_q_values(self, trajectory, tokenizer):
        trajectory_data = self.prepare_trajectory_data(trajectory)
        model_inputs = trajectory_data['model_inputs']

        all_values = self.get_all_values(**model_inputs)
        qs = all_values['q']

        action_tokens = trajectory_data['tokens']

        return qs, action_tokens

    def observe_v_values(self, trajectory, tokenizer):
        trajectory_data = self.prepare_trajectory_data(trajectory)
        model_inputs = trajectory_data['model_inputs']

        all_values = self.get_all_values(**model_inputs)
        vs = all_values['v']

        action_tokens = trajectory_data['tokens']

        return vs, action_tokens

    def get_q_loss(self, vs, qs, rs, terminals_mask):
        vs = vs.detach()

        # TODO: add inter-turn reward flow

        q_losses = ((terminals_mask[:, 1:]*vs * self.gamma + rs[..., None] - qs)**2) * terminals_mask[:, :-1]
        q_loss = q_losses.sum() / max(terminals_mask[:, :-1].sum().item(), 1.0)

        if getattr(self, 'DEBUG', False):
            for i in range(vs.shape[1]):
                print('INDEX', i)
                print('Q\t', qs[0, i])
                print('V\t', vs[0, i])
                print('R\t', rs[0, i])
                print('TERM\t', terminals_mask[:, :-1][0, i])
                print('TARGET:\t', terminals_mask[:, 1:][0, i]*vs[0, i] * self.gamma + rs[0, i])
                print('NIX NEXT:\t', terminals_mask[:, 1:][0, i])
                print('REWARD NEXT:\t', rs[0, i])
                print('LOSS:\t', q_losses[0, i])

                print('#'*50)

            pdb.set_trace()

        return q_loss

    def get_v_loss(self, vs, target_qs, terminals_mask, success):
        target_qs = target_qs.detach()
        expectile_losses = ((target_qs >= vs).int() * self.tau * (target_qs - vs)**2 + (target_qs < vs).int() * (1 - self.tau) * (target_qs - vs)**2) * terminals_mask[:, :-1]
        expectile_loss = expectile_losses.sum() / max(terminals_mask[:, :-1].sum().item(), 1.0)

        success_val = int(success)
        vs_flat = vs[..., 0]
        # let's mask out the terminals mask stuff here
        mc_targets = torch.ones_like(vs_flat, device=vs_flat.device)*success_val
        # POTENTIAL TODO: mask out the terminals here too (shouldn't affect much)

        mc_loss = F.binary_cross_entropy_with_logits(vs_flat, mc_targets, reduction='mean')

        if getattr(self, 'DEBUG', False):
            for i in range(vs.shape[1]):
                print('INDEX', i)
                print('V\t', vs[0, i])
                print('TARGET\t', target_qs[0, i])
                print('TERM\t', terminals_mask[:, :-1][0, i])
                print('LOSS\t', expectile_losses[0, i])
                print('#'*50)

            pdb.set_trace()

        loss_logs = {
            'mc_loss': mc_loss.cpu().item(),
            'expectile_loss': expectile_loss.cpu().item(),
            'v_acc': (vs_flat > 0).int().eq(mc_targets).float().mean().cpu().item()
        }

        return (1 - self.mc_loss_coeff)*expectile_loss + self.mc_loss_coeff*mc_loss, loss_logs

    def get_cql_loss(self, qs, action_tokens, terminals_mask):

        # TODO: potentially mask out user turns if we do inter-turn gradient flow

        d = qs.shape[-1]
        cql_losses = F.cross_entropy(qs.reshape(-1, d) / self.cql_temp, action_tokens.reshape(-1), reduction='none')
        cql_loss = (cql_losses[None, :] * terminals_mask[:, :-1, 0]).sum() / max(terminals_mask[:, :-1].sum().item(), 1.0)

        return cql_loss

    def save_checkpoint(self, path):
        os.makedirs(path, exist_ok=True)

        # save backbone adapter
        self.backbone_lm.save_pretrained(os.path.join(path, 'backbone_lm'))

        # save q network
        torch.save(self.q_head.state_dict(), os.path.join(path, 'q_head.pt'))

        # save v network
        torch.save(self.v_head.state_dict(), os.path.join(path, 'v_head.pt'))

        print(f'Checkpoint saved at {path}')

    def load_ilql_checkpoint(self, path):

        if not isinstance(self.backbone_lm, PeftModel):
            self.backbone_lm = PeftModel.from_pretrained(self.backbone_lm, os.path.join(path, 'backbone_lm'))
            self.backbone_lm.enable_adapter_layers()
        else:
            self.backbone_lm.load_adapter(os.path.join(path, 'default'))

        self.q_head.load_state_dict(torch.load(os.path.join(path, 'q_head.pt')))
        self.v_head.load_state_dict(torch.load(os.path.join(path, 'v_head.pt')))
        print('ILQL CHECKPOINT LOADED')

    def load_bc_checkpoint(self, path):
        # must load ilql before
        assert isinstance(self.backbone_lm, PeftModel)

        ckpt_dir = os.path.join(path, 'model')
        # self.backbone_lm = PeftModel.from_pretrained(self.backbone_lm, ckpt_dir)
        self.backbone_lm.load_adapter(ckpt_dir, 'bc')
        self.use_bc_adapter = True
        print('BC CHECKPOINT LOADED')

    def forward(self, beta, **model_inputs):
        self.backbone_lm.eval()
        with torch.no_grad():
            # if we have a bc adapter, let's selectively enable it

            if self.use_bc_adapter:
                # TODO: revert
                self.backbone_lm.set_adapter('bc')
                lm_logits = self.backbone_lm(**model_inputs).logits[:, -1:, :]

                self.backbone_lm.set_adapter('default')
            else:
                with self.backbone_lm.disable_adapter():
                    lm_logits = self.backbone_lm(**model_inputs).logits[:, -1:, :]
            torch.cuda.empty_cache()
            q_values = self.get_q_values(**model_inputs)[:, -1:, :]
            mixture = lm_logits + beta*q_values
        return mixture

    def generate_turn(self, trajectory, beta, max_new_tokens=64):
        # tokens = self.prepare_trajectory_for_generation(trajectory['turns'], tokenizer)
        orig_n_tokens = tokens.shape[-1]

        for _ in range(max_new_tokens):
            logits = self.forward(beta=beta, input_ids=tokens)
            torch.cuda.empty_cache()

            # for now, let's greedy-decode
            token = torch.argmax(logits[:, -1, :])

            if token == trajectory.eot_id:
                break
            tokens = torch.cat((tokens, token.view(1, 1)), dim=-1)

        out_tokens = tokens[:, orig_n_tokens:]

        return out_tokens



def train_ilql(config):
    ########################################
    # sync data

    assert torch.cuda.is_available()
    # reward_dist_config = replace(RewardDistributionConfig(**config['reward_config']))

    # load and display data
    def load_data(path, tokenizer, data_diet=None):
        with open(path, 'r') as f:
            data = json.load(f)

        if data_diet is not None:
            random.seed(config.training.data_seed)
            random.shuffle(data)
            data = data[:data_diet]

        out_data = []
        print('Loading data...')
        for traj_dict in tqdm(data):
            out_data.append(Trajectory(traj_dict, tokenizer))
        out_data = [d for d in out_data if d is not None]

        print('Data loaded!')


        return out_data

    torch.manual_seed(config.training.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train = load_data(os.path.join(config.data_path, 'train_transformed.json'),
                      tokenizer,
                      config.training.train_data_diet)
    val = load_data(os.path.join(config.data_path, 'eval_transformed.json'),
                    tokenizer,
                    config.training.val_data_diet)

    print('TRAIN DATA LENGTH:', len(train))
    print('VAL DATA LENGTH:', len(val))

    def collate_fn(data):
        return data[0]
    
    # TODO: data diet
    train_dl = DataLoader(train, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val, batch_size=1, shuffle=True, collate_fn=collate_fn)

    ex_traj = train[0]

    print('#'*50)
    print('EXAMPLE TRAJECTORY:')
    print('#'*50)
    pprint(ex_traj.string)
    print('#'*50)


    # load model and tokenizer

    print('BEFORE LOADING MODEL')
    print_cuda_memory()
    print('#'*50)

    print('LOADING MODEL...')
    model = AutoModelForCausalLM.from_pretrained(config.model_name,
                                                device_map={'': torch.device(0)},
                                                torch_dtype=torch.bfloat16)

    # develop ILQL model

    lora_config = LoraConfig(
        peft_type='LORA',
        task_type='CAUSAL_LM',
        r=config.training.lora_r,
        lora_alpha=config.training.lora_r
    )

    ilql = ILQLModel(model,
                    lora_config,
                    reward_dist_config = config.reward,
                    cql_loss_coeff=config.training.cql_loss_coeff,
                    mc_loss_coeff=config.training.mc_loss_coeff,
                    polyak_coeff=config.training.polyak_coeff,
                    gamma=config.training.gamma,
                    q_head_bias=config.training.head_bias)

    print('MODEL LOADED')

    config_dict = OmegaConf.to_container(config, resolve=True)


    if config.saving.use_wandb:
        wandb.init(project='agents',
                name=config.run_name,
                mode='online',
                config=config_dict)
        wandb.watch(ilql)

    optimizer = torch.optim.Adam(ilql.parameters(), lr=config.training.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.1*config.training.lr, T_max=config.training.n_epochs * len(train_dl))

    i = 0

    optimizer.zero_grad()



    def eval_loop(ilql, val_dl, tokenizer):
        ilql.eval()

        all_logs = {}

        with torch.no_grad():
            for traj in tqdm(val_dl, total=len(val_dl)):
                _, logs = ilql.get_loss(traj, tokenizer)
                for k, v in logs.items():
                    all_logs[k] = all_logs.get(k, []) + [v]


            for k, v in all_logs.items():
                if isinstance(v, list) or isinstance(v, np.ndarray) or isinstance(v, tuple):
                    try:
                        all_logs[k] = torch.tensor(v).mean().item()
                    except Exception as e:
                        all_logs[k] = v


        out_logs = {}

        for k, v in all_logs.items():
            out_logs[f"{k}_val"] = v

        ilql.train()
        return out_logs

    best_val_loss = float('inf')

    for epoch in range(config.training.n_epochs):
        ilql.train()

        pbar = tqdm(train_dl, total=len(train))

        for traj in pbar:
            i += 1
            logs = {}

            if i % config.training.gradient_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                ilql.step_target_q_network()

            scheduler.step()

            if i == 1:
                print('#'*50)
                print('EXAMPLE TRAJECTORY:')
                pprint(traj.string)
                print('#'*50)

            if i % config.saving.eval_freq == 0:
                print('EVALUATING')
                eval_logs = eval_loop(ilql, val_dl, tokenizer)
                if config.saving.use_wandb:
                    wandb.log(eval_logs)
                
                if eval_logs['loss_val'] < best_val_loss:
                    save_path = os.path.join(config.saving.save_dir, f'best_checkpoint')

                    with open(os.path.join(config.saving.save_dir, 'best_epoch.txt'), 'w') as f:
                        f.write(f"BEST EPOCH:\n{epoch + 1}\n")
                        f.write(f"BEST LOSS:\n{eval_logs['loss_val']}\n")
                    
                    ilql.save_checkpoint(save_path)
            
            loss, logs = ilql.get_loss(traj, log_values=(i % 100 == 0))

            logs['loss'] = loss.item()
            logs['global_step'] = i
            logs['lr'] = scheduler.get_last_lr()[0]

            loss_scaled = loss / config.training.gradient_accum_steps
            loss_scaled.backward()

            if config.saving.use_wandb:
                wandb.log(logs)
            pbar.set_description(f'Loss: {loss.item():.4f}')

            # saving logic

            if config.saving.save_model and i % config.saving.save_freq == 0:
                save_path = os.path.join(config.saving.save_dir, f'checkpoint_{i}')
                ilql.save_checkpoint(save_path)


        eval_logs = eval_loop(ilql, val, tokenizer)

        if eval_logs['loss_val'] < best_val_loss:
            save_path = os.path.join(config.saving.save_dir, f'best_checkpoint')

            with open(os.path.join(config.saving.save_dir, 'best_epoch.txt'), 'w') as f:
                f.write(f"BEST EPOCH:\n{epoch + 1}\n")
                f.write(f"BEST LOSS:\n{eval_logs['loss_val']}\n")
            
            ilql.save_checkpoint(save_path)

    save_path = os.path.join(config.saving.save_dir, f'final_checkpoint')
    ilql.save_checkpoint(save_path)

