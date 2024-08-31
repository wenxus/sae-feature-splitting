#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch as t
from torch import nn, Tensor
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np
import einops
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple
from tqdm.notebook import tqdm
from rich import print as rprint
from rich.table import Table
from pathlib import Path
import json

#%%
device = t.device("mps")
MAIN = __name__ == "__main__"

#%%
from transformer_lens import HookedTransformer, FactoredMatrix
from transformer_lens.hook_points import HookPoint

from transformer_lens.utils import (
    load_dataset,
    tokenize_and_concatenate,
    download_file_from_hf,
)

#%%
@dataclass
class AutoEncoderConfig:
    n_instances: int
    n_input_ae: int
    n_hidden_ae: int
    l1_coeff: float = 0.5
    tied_weights: bool = False


class AutoEncoder(nn.Module):
    W_enc: Float[Tensor, "n_instances n_input_ae n_hidden_ae"]
    W_dec: Float[Tensor, "n_instances n_hidden_ae n_input_ae"]
    b_enc: Float[Tensor, "n_instances n_hidden_ae"]
    b_dec: Float[Tensor, "n_instances n_input_ae"]

    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.W_enc = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_input_ae, cfg.n_hidden_ae))))
        if not(cfg.tied_weights):
            self.W_dec = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden_ae, cfg.n_input_ae))))
        self.b_enc = nn.Parameter(t.zeros(cfg.n_instances, cfg.n_hidden_ae))
        self.b_dec = nn.Parameter(t.zeros(cfg.n_instances, cfg.n_input_ae))
        self.to(device)

    def forward(self, h: Float[Tensor, "batch_size n_instances n_hidden"]):
        # Compute activations
        h_cent = h - self.b_dec
        acts = einops.einsum(
            h_cent, self.W_enc,
            "batch_size n_instances n_input_ae, n_instances n_input_ae n_hidden_ae -> batch_size n_instances n_hidden_ae"
        )
        acts = F.relu(acts + self.b_enc)

        # Compute reconstructed input
        h_reconstructed = einops.einsum(
            acts, (self.W_enc.transpose(-1, -2) if self.cfg.tied_weights else self.W_dec),
            "batch_size n_instances n_hidden_ae, n_instances n_hidden_ae n_input_ae -> batch_size n_instances n_input_ae"
        ) + self.b_dec

        # Compute loss, return values
        l2_loss = (h_reconstructed - h).pow(2).mean(-1) # shape [batch_size n_instances]
        l1_loss = acts.abs().sum(-1) # shape [batch_size n_instances]
        loss = (self.cfg.l1_coeff * l1_loss + l2_loss).mean(0).sum() # scalar

        return l1_loss, l2_loss, loss, acts, h_reconstructed

    @t.no_grad()
    def normalize_decoder(self) -> None:
        '''
        Normalizes the decoder weights to have unit norm. If using tied weights, we we assume W_enc is used for both.
        '''
        if self.cfg.tied_weights:
            self.W_enc.data = self.W_enc.data / self.W_enc.data.norm(dim=1, keepdim=True)
        else:
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=2, keepdim=True)

    @t.no_grad()
    def resample_neurons(
        self,
        h: Float[Tensor, "batch_size n_instances n_hidden"],
        frac_active_in_window: Float[Tensor, "window n_instances n_hidden_ae"],
        neuron_resample_scale: float,
    ) -> None:
        '''
        Resamples neurons that have been dead for `dead_neuron_window` steps, according to `frac_active`.
        '''
        pass # See below for a solution to this function

    def optimize(
        self,
        model,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = None,
        neuron_resample_window: Optional[int] = None,
        dead_neuron_window: Optional[int] = None,
        neuron_resample_scale: float = 0.2,
    ):
        pass
#%%
VERSION_DICT = {"run1": "gelu-2l_L1_16384_mlp_out_50"}

def load_autoencoder_from_huggingface(versions: List[str] = ["run1"]):
    state_dict = {}

    for version in versions:
        version_id = VERSION_DICT[version]
        # Load the data from huggingface (both metadata and state dict)
        sae_data: dict = download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version_id}_cfg.json")
        new_state_dict: dict = download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version_id}.pt", force_is_torch=True)
        
        print(f"{new_state_dict.keys()}")
        # Add new state dict to the existing one
        for k, v in new_state_dict.items():
            shape = new_state_dict[k].shape
            state_dict[k] = new_state_dict[k].reshape((1,) + shape)
    # Get data about the model dimensions, and use that to initialize our model (with 2 instances)
    d_model = 512
    dict_mult = 32
    n_hidden_ae = d_model * dict_mult

    cfg = AutoEncoderConfig(
        n_instances = 1,
        n_input_ae = d_model,
        n_hidden_ae = n_hidden_ae,
    )

    # Initialize our model, and load in state dict
    autoencoder = AutoEncoder(cfg)
    autoencoder.load_state_dict(state_dict)
    return autoencoder

#%%
def load_autoencoder(version, dir):
    sae_data: dict = (json.load(open(dir/(str(version)+"_cfg.json"), "r")))
    rprint(sae_data)
    d_mlp_out = sae_data["act_size"]
    dict_mult = sae_data["dict_mult"]
    n_hidden_ae = d_mlp_out * dict_mult
    cfg = AutoEncoderConfig(
        n_instances = 1,
        n_input_ae = d_mlp_out,
        n_hidden_ae = n_hidden_ae,
    )
    print(f"{d_mlp_out=}")
    print(f"{n_hidden_ae=}")

    autoencoder = AutoEncoder(cfg)
    new_state_dict: dict = t.load(dir/(str(version)+".pt"), map_location=t.device('mps'))
    state_dict = {}
    for k, v in new_state_dict.items():
            shape = new_state_dict[k].shape
            state_dict[k] = new_state_dict[k].reshape((1,) + shape)
    autoencoder.load_state_dict(state_dict)
    return autoencoder

#%%
model = HookedTransformer.from_pretrained("gelu-1l").to(device)

#%%
data = load_dataset("NeelNanda/c4-code-20k", split="train")
tokenized_data = tokenize_and_concatenate(data, model.tokenizer, max_length=128)
tokenized_data = tokenized_data.shuffle(2)
all_tokens = tokenized_data["tokens"]
print("Tokens shape: ", all_tokens.shape)
# %%
autoencoder = load_autoencoder(14, Path('fill in path'))

## %% # highest_activating_tokens for each feature
@t.inference_mode()
def highest_activating_tokens(
    tokens: Int[Tensor, "batch seq"],
    model: HookedTransformer,
    autoencoder: AutoEncoder,
    feature_idx: int,
    autoencoder_B: bool = False,
    k: int = 10,
) -> Tuple[Int[Tensor, "k 2"], Float[Tensor, "k"]]:
    '''
    Returns the indices & values for the highest-activating tokens in the given batch of data.
    '''
    batch, seq = tokens.shape
    _, cache = model.run_with_cache(tokens, names_filter=['blocks.0.hook_mlp_out'])
    mlp_out = cache['blocks.0.hook_mlp_out']

    if autoencoder_B:
        instance_idx = 1
    else:
        instance_idx = 0

    no_bias = mlp_out - autoencoder.b_dec[instance_idx]
    actsf_raw = einops.einsum(autoencoder.W_enc[instance_idx, :, feature_idx], no_bias, "d_model, batch seq d_model -> batch seq")
    actsf = einops.rearrange(actsf_raw, "batch seq -> (batch seq)")

    top_values, idxs_flat = actsf.topk(k=k, sorted=True)
    idxs = t.empty(k, 2)
    for i in range(k):
        idxs[i][0] = int(idxs_flat[i] // seq)
        idxs[i][1] = int(idxs_flat[i] % seq)
    return idxs.to(t.int), top_values


def display_top_sequences(top_acts_indices, top_acts_values, tokens, feature_idx):
    table = Table("Sequence", "Token", "Activation", title=f"Tokens which most activate this feature: {feature_idx}")
    for (batch_idx, seq_idx), value in zip(top_acts_indices, top_acts_values):
        seq = ""
        for i in range(max(seq_idx-5, 0), min(seq_idx+5, tokens.shape[1])):
            new_str_token = model.to_single_str_token(tokens[batch_idx, i].item()).replace("\n", "\\n")
            if i == seq_idx: new_str_token = f"[b u dark_orange]{new_str_token}[/]"
            seq += new_str_token
        table.add_row(seq, f"[b u dark_orange]{model.to_single_str_token(tokens[batch_idx, seq_idx].item())}[/]", f'{value:.2f}')
    rprint(table)

##%% 
def most_affected_logits(
    model: HookedTransformer,
    autoencoder: AutoEncoder,
    feature_idx: int,
    autoencoder_B: bool = False,
    k: int = 10,
):
    if autoencoder_B:
        instance_idx = 1
    else:
        instance_idx = 0
    w_logits = autoencoder.W_dec[instance_idx, feature_idx] @ model.W_U
    topk, top_inds = w_logits.topk(k=k)
    botk, bot_inds = w_logits.topk(k=k, largest=False)

    s = f"TOP boosted logits for feature {feature_idx}\n"
    for idx, value in zip(top_inds, topk):
        s += f"{value:.2f}, {model.to_single_str_token(idx.item())}\n"
        
    s += "\nBOTTOM\n"
    for idx, value in zip(bot_inds, botk):
        s += f"{value:.2f}, {model.to_single_str_token(idx.item())}\n"

    rprint(s)    

## %% # highest_activating_tokens_for_many_features
@t.inference_mode()
def highest_activating_tokens_for_many_features(
    tokens: Int[Tensor, "n_features_idxs seq"], # feature idx, seq
    model: HookedTransformer,
    autoencoder: AutoEncoder,
    feature_idxs: List[Int],
    autoencoder_B: bool = False,
    k: int = 10,
) -> Tuple[Int[Tensor, "k 2"], Float[Tensor, "k"]]:
    '''
    Returns the indices & values for the highest-activating tokens in the given batch of data.
    '''
    n_features, seq = tokens.shape
    _, cache = model.run_with_cache(tokens, names_filter=['blocks.0.hook_mlp_out'])
    mlp_out = cache['blocks.0.hook_mlp_out']

    if autoencoder_B:
        instance_idx = 1
    else:
        instance_idx = 0
    
    print(f"{autoencoder.W_enc.shape=}")
    print(f"{feature_idxs=}")

    no_bias = mlp_out - autoencoder.b_dec[instance_idx]
    actsf_raw = einops.einsum(autoencoder.W_enc[instance_idx, :, feature_idxs], no_bias, "dmodel n_features_idxs, n_features_idxs seq dmodel -> n_features_idxs seq")
    actsf = einops.rearrange(actsf_raw, "n_features_idxs seq -> (n_features_idxs seq)")

    top_values, idxs_flat = actsf.topk(k=k, sorted=True)
    idxs = t.empty(k, 2)
    for i in range(k):
        idxs[i][0] = int(idxs_flat[i] // seq) # feature idx
        idxs[i][1] = int(idxs_flat[i] % seq)
    return idxs.to(t.int), top_values

def display_top_sequences_and_features(top_acts_indices, top_acts_values, tokens, starting_f_idx):
    table = Table("Sequence", "Activation", "Feature idx" "token idx", title="Tokens which most activate this feature")
    for (fea_idx, seq_idx), value in zip(top_acts_indices, top_acts_values):
        # Get the sequence as a string (with some padding on either side of our sequence)
        seq = ""
        for i in range(max(seq_idx-5, 0), min(seq_idx+5, tokens.shape[1])):
            new_str_token = model.to_single_str_token(tokens[0, i].item()).replace("\n", "\\n")
            # Highlight the token with the high activation
            if i == seq_idx: new_str_token = f"[b u dark_orange]{new_str_token}[/]"
            seq += new_str_token
        # Print the sequence, and the activation value
        table.add_row(seq, f'{value:.2f}', f"feature {fea_idx + starting_f_idx}", f"{seq_idx}")
    rprint(table)

## %% # top 10 highest_activating_tokens_for_many_features
def get_highest_activating_tokens_for_many_features(m_tokens):
    feature_batch = 512
    autoencoder_B = False

    agg_tokens_dict = {}
    
    for i in range(8):
        range_s = i * feature_batch
        range_e = (i+1) * feature_batch
        repeat = range_e - range_s
        stacked_tokens = einops.repeat(m_tokens, "b s -> (repeat b) s", repeat = repeat)
        feature_idxs=range(range_s, range_e)
        tokens = stacked_tokens[feature_idxs]
        top_acts_indices, top_acts_values = highest_activating_tokens_for_many_features(
            tokens, model, autoencoder, feature_idxs=feature_idxs, autoencoder_B=autoencoder_B, k=15)
        for (f_idx, token_idx), value in zip(top_acts_indices, top_acts_values):
            if token_idx.item() not in agg_tokens_dict.keys():
                agg_tokens_dict[token_idx.item()] = []
            agg_tokens_dict[token_idx.item()].append(tuple((f_idx.item() + range_s, value.item())))
        display_top_sequences_and_features(top_acts_indices, top_acts_values, tokens, range_s)

    for key in agg_tokens_dict.keys():
        agg_tokens_dict[key] = sorted(agg_tokens_dict[key], key=lambda x: x[1], reverse=True)
    rprint(agg_tokens_dict)
    return agg_tokens_dict

#%% using specific prompts to inspect features
# m_tokens = model.to_tokens("I am a multi-millionair. I have multi-purpose assets.")
m_tokens = model.to_tokens("that encapsulated the long-standing liberal argument that ")
# m_tokens = model.to_tokens("A 3-point landing is an important maneuver to learn.")
# m_tokens = model.to_tokens("This year is 2022. Next year is 2023. Last year is 20")
# m_tokens = model.to_tokens('''from django.conf import settings
# settings.configure(
#     DEBUG=True,
#     INSTALLED_APPS=['django.contrib.admin','django.contrib.auth']
# )
# ''')

agg_tokens_dict = get_highest_activating_tokens_for_many_features(m_tokens)
# %% “-”
rprint(agg_tokens_dict[5])

# %%  # highest_activating_tokens for features
feature_idxs = [646] #[3185, 1038]
tokens = all_tokens[100:400]
autoencoder_B = False
for feature_idx in feature_idxs:
    top_acts_indices, top_acts_values = highest_activating_tokens(
        tokens, model, autoencoder, feature_idx=feature_idx, autoencoder_B=autoencoder_B)
    display_top_sequences(top_acts_indices, top_acts_values, tokens, feature_idx=feature_idx)

    most_affected_logits(model, autoencoder, feature_idx, autoencoder_B=autoencoder_B)
