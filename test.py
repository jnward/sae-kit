# %%
%load_ext autoreload
%autoreload 2
import torch
from jsae.utils import cached_activation_generator, compute_metrics
from datasets import load_dataset, IterableDataset, Dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm
from jsae.sparse_autoencoder import SparseAutoencoder, Encoder, Decoder
from jsae.gradient_pursuit import gradient_pursuit, gradient_descent, gradient_descent_topk


# %%
device='mps'
hook_name = "blocks.8.hook_resid_pre"
release="gpt2-small-res-jb"

my_sae = SparseAutoencoder.from_pretrained(
    release,
    hook_name,
    device=device,
    k=100,
)

# %%
my_dataset = load_dataset(
    "openwebtext",
    split="train",
    streaming=True,
    trust_remote_code=True,
)
assert isinstance(my_dataset, IterableDataset)

my_model = HookedTransformer.from_pretrained(
    "gpt2-small",
    device=device,
    dtype=torch.bfloat16,
)

# %%
batch_size = 128

my_data_generator = cached_activation_generator(
    my_model,
    my_dataset,
    batch_size,
    16,
    16 * 8,
    128,
    1,
    hook_name,
)

# %%
num_tokens = 1_000_000
num_batches = num_tokens // batch_size

num_batches = 1

l0s = []
fvus = []
with torch.no_grad():
    for _ in tqdm(range(num_batches)):
        activations = next(my_data_generator)
        features = my_sae.encode(activations)
        reconstructions = my_sae.decode(features)

        l0, fvu = compute_metrics(activations, features, reconstructions)

        l0s.append(l0)
        fvus.append(fvu)

print(f"Mean L0: {sum(l0s) / len(l0s)}")
print(f"Mean FVU: {sum(fvus) / len(fvus)}")
        
# %%
gp_features = gradient_pursuit(
    activations,
    my_sae.decoder,
    k=67,
)
gp_reconstruction = my_sae.decode(gp_features)

l0, fvu = compute_metrics(activations, gp_features, gp_reconstruction)
print(f"L0: {l0}")
print(f"FVU: {fvu}")


# %%
gd_features = gradient_descent_topk(
    activations,
    my_sae.decoder,
    k=67,
    n_steps=4000,
    lr=0.01,
)

gd_reconstruction = my_sae.decode(gd_features)

l0, fvu = compute_metrics(activations, gd_features, gd_reconstruction)
print(f"L0: {l0}")
print(f"FVU: {fvu}")

# %%
gd_features = gradient_descent(
    activations,
    my_sae.decoder,
    n_steps=4000,
    lr=0.001,
    lambda_=0.05,
)

gd_reconstruction = my_sae.decode(gd_features)

l0, fvu = compute_metrics(activations, gd_features, gd_reconstruction)
print(f"L0: {l0}")
print(f"FVU: {fvu}")

# TODO: ^^^ this is bugged
# %%
