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
from typing import Optional, Callable, List
from rich import print as rprint
from pathlib import Path
import json
from transformer_lens.utils import (
    download_file_from_hf,
)

#%%
device = t.device("mps")

MAIN = __name__ == "__main__"


#%% class AutoEncoder:
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
#%% load_autoencoder_from_huggingface
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
    # pprint.pprint(sae_data)
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

autoencoder = load_autoencoder(16, Path('path1'))
autoencoder2 = load_autoencoder(16, Path('path2'))
types = ['512-feature SAE', '4096-feature SAE']

features1_cnt = autoencoder.W_dec.shape[1]
features2_cnt = autoencoder2.W_dec.shape[1]

#%% plot features

import umap
import matplotlib.pyplot as plt
## UMAP hyperparameter n_neighbors=15, metric="cosine", min_dist=0.05
reducer = umap.UMAP(n_neighbors=15, metric="cosine", min_dist=0.05, random_state=1) # use min_dist=0.1 for clustering
# feature_data = autoencoder.W_dec.squeeze()
feature_data = t.concat((autoencoder.W_dec.squeeze(), autoencoder2.W_dec.squeeze()), dim=0)
embedding = reducer.fit_transform(feature_data.detach().cpu().numpy())
#%% cluster
import hdbscan

for i in range(len(types)):
    idx_s = i * features1_cnt
    idx_e = features1_cnt if i == 0 else features1_cnt + features2_cnt
    labels = hdbscan.HDBSCAN(
        min_cluster_size=3,
    ).fit_predict(embedding[idx_s:idx_e])
    clustered = (labels >= 0)
    plt.scatter(embedding[idx_s:idx_e][~clustered, 0],
                embedding[idx_s:idx_e][~clustered, 1],
                color=(0.5, 0.5, 0.5),
                s=0.1,
                alpha=0.5)
    plt.scatter(embedding[idx_s:idx_e][clustered, 0],
                embedding[idx_s:idx_e][clustered, 1],
                c='b' if i == 0 else 'r',
                s=0.1,)

plt.title(f"{types}")

#%% plot specific features
# print(embedding.shape) 
# z = t.ones(embedding.shape).detach().cpu().numpy()
# z[features1_cnt:] = embedding[features1_cnt:]
# z[240] = embedding[240]
# embedding = z

#%%
import pandas as pd
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral4

feature_df = pd.DataFrame(embedding, columns=('x', 'y'))
feature_df['feature_idx'] = list(range(features1_cnt)) + list(range(features2_cnt))
feature_df['type'] = types[0]
feature_df['type'][features1_cnt:features1_cnt+features2_cnt] = types[1]

datasource = ColumnDataSource(feature_df)
color_mapping = CategoricalColorMapper(factors=types,
                                       palette=('blue', 'red'))

plot_figure = figure(
    title='UMAP projection of 512-feature SAE (blue) VS 4096-feature SAE (red)',
    width=1200,
    height=1200,
    tools=('pan, wheel_zoom, reset')
)

plot_figure.add_tools(HoverTool(tooltips="""
<div>
    <div>
        <span style='font-size: 16px; color: #224499'>Type:</span>
        <span style='font-size: 18px'>@type</span>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>Feature index:</span>
        <span style='font-size: 18px'>@feature_idx</span>
    </div>
</div>
"""))

plot_figure.circle(
    'x',
    'y',
    source=datasource,
    color=dict(field='type', transform=color_mapping),
    line_alpha=0.6,
    fill_alpha=0.6,
    size=4
)
show(plot_figure)

# %%
