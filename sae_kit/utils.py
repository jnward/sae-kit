import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset

from typing import Union

from pathlib import Path

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
    generator_batch_size: int = 24,
    skip_first_n_tokens: int = 1,
    ctx_len: int = 128,
    max_acts_per_file: int = 1_000_000,
    hook_name: str = "blocks.8.hook_resid_pre",
    dir: str = "acts",
):
    data_iter = token_iter(model, dataset, generator_batch_size, ctx_len)
    acts = []
    act_acc = 0
    file_acc = 0
    save_dir = Path(dir)
    save_dir.mkdir(parents=True, exist_ok=True)
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
            save_path = save_dir / f"acts_{file_acc}.pt"
            torch.save(acts_cat, save_path)
            file_acc += 1
            acts = []
            act_acc = 0
    if act_acc > 0:
        acts_cat = torch.cat(acts, dim=0)
        acts_cat = acts_cat[:, skip_first_n_tokens:, :]
        acts_cat = acts_cat.reshape(-1, acts_cat.size(-1))
        acts_cat = acts_cat[torch.randperm(acts_cat.size(0))]
        save_path = save_dir / f"acts_{file_acc}.pt"
        torch.save(acts_cat, save_path)


def disk_activation_generator(batch_size, num_files=None, dir="acts", skip_first_n=0):
    read_dir = Path(dir)
    current_batch = []
    if num_files is None:
        num_files = len(list(read_dir.glob("acts_*.pt")))
    print(f"Reading from {num_files} files", end="")
    if skip_first_n:
        print(f", skipping first {skip_first_n}")
    else:
        print()
    for file_id in range(skip_first_n, num_files):
        read_path = read_dir / f"acts_{file_id}.pt"
        # print("loading", read_path)
        acts = torch.load(read_path)
        for row in acts:
            current_batch.append(row.unsqueeze(0))
            if len(current_batch) == batch_size:
                yield torch.cat(current_batch, dim=0)
                current_batch = []


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
