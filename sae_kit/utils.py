import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset

from typing import Union

AnyDataset = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]


def token_iter(
    model: HookedTransformer,
    dataset: Dataset | IterableDataset,
    batch_size: int,
    ctx_len: int,
):
    batch = []
    for example in dataset:
        text = example["text"]  # type: ignore
        assert model.tokenizer is not None
        tokens = model.tokenizer(
            text,
        )[
            "input_ids"
        ][  # type: ignore
            : ctx_len - 1
        ]
        tokens = torch.tensor([model.tokenizer.bos_token_id] + tokens)
        if len(tokens) != ctx_len:  # want one example per example
            continue
        if len(batch) < batch_size:
            batch.append(tokens)
        else:
            yield torch.stack(batch)
            batch = []


@torch.no_grad()
def get_activations(
    model: HookedTransformer, token_batch: torch.Tensor, hook_name: str
):
    with torch.no_grad():
        _, cache = model.run_with_cache(
            token_batch,
            return_type=None,
        )
        activations = cache[hook_name]
    return activations


def cache_activations_to_disk(
    model: HookedTransformer,
    dataset: Dataset | IterableDataset,
    n_activations: int,
    generator_batch_size=24,
    skip_first_n_tokens=1,
    ctx_len=128,
    max_acts_per_file=1_000_000,
    hook_name="blocks.8.hook_resid_pre",
):
    data_iter = token_iter(model, dataset, generator_batch_size, ctx_len)
    acts = []
    act_acc = 0
    file_acc = 0
    for _ in tqdm(
        range(n_activations // (generator_batch_size * (ctx_len - skip_first_n_tokens)))
    ):
        token_batch = next(data_iter)
        activations = get_activations(model, token_batch, hook_name)
        acts.append(activations)
        act_acc += generator_batch_size * ctx_len
        if act_acc >= max_acts_per_file:
            acts_cat = torch.cat(acts, dim=0)
            acts_cat = acts_cat[:, skip_first_n_tokens:, :]
            acts_cat = acts_cat.reshape(-1, acts_cat.size(-1))
            acts_cat = acts_cat[torch.randperm(acts_cat.size(0))]
            torch.save(acts_cat, f"acts/acts_{file_acc}.pt")
            file_acc += 1
            acts = []
            act_acc = 0
    if act_acc < max_acts_per_file:
        acts_cat = torch.cat(acts, dim=0)
        acts_cat = acts_cat[:, skip_first_n_tokens:, :]
        acts_cat = acts_cat.reshape(-1, acts_cat.size(-1))
        acts_cat = acts_cat[torch.randperm(acts_cat.size(0))]
        torch.save(acts_cat, f"acts/acts_{file_acc}.pt")


def cached_activation_generator(
    model: HookedTransformer,
    dataset: Dataset | IterableDataset,
    hook_name: str,
    activation_batch_size: int,
    generator_batch_size=24,
    examples_per_run=2048,
    ctx_len=128,
    skip_first_n_tokens=10,
):
    data_iter = token_iter(model, dataset, generator_batch_size, ctx_len)  # shuffle?
    batches_per_run = examples_per_run // generator_batch_size
    remainder_batch_size = examples_per_run % generator_batch_size
    while True:
        my_acts = []
        print("Generating new activations...")
        for _ in range(batches_per_run):
            token_batch = next(data_iter)
            activations = get_activations(model, token_batch, hook_name)
            my_acts.append(activations)
        if remainder_batch_size > 0:
            token_batch = next(data_iter)
            activations = get_activations(model, token_batch, hook_name)
            my_acts.append(activations[:remainder_batch_size])
        acts_cat = torch.cat(my_acts, dim=0)
        acts_cat = acts_cat[:, skip_first_n_tokens:, :]
        acts_cat = acts_cat.reshape(-1, acts_cat.size(-1))
        acts_cat = acts_cat[torch.randperm(acts_cat.size(0))]
        for i in range(0, len(acts_cat), activation_batch_size):
            yield acts_cat[i : i + activation_batch_size]


@torch.no_grad()
def compute_metrics(activations, features, reconstructions):
    l0 = (features > 0).float().sum(1).mean()

    e = reconstructions - activations
    total_variance = (activations - activations.mean(0)).pow(2).sum()
    squared_error = e.pow(2)
    fvu = squared_error.sum() / total_variance

    return l0.item(), fvu.item()
