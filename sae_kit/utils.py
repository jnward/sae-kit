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

def _save_activations_to_disk(activations_list, save_dir, file_idx, skip_first_n_tokens):
    """
    Helper function to save activations to disk.
    Used by both cached_activation_generator and cache_activations_to_disk.
    
    Parameters:
    - activations_list: List of activation tensors to save
    - save_dir: Directory to save to
    - file_idx: Index for the filename
    - skip_first_n_tokens: Number of tokens to skip from the beginning
    
    Returns:
    - None
    """
    acts_cat = torch.cat(activations_list, dim=0)
    acts_cat = acts_cat[:, skip_first_n_tokens:, :]
    acts_cat = acts_cat.reshape(-1, acts_cat.size(-1))
    # Apply random permutation (will be affected by torch.manual_seed)
    acts_cat = acts_cat[torch.randperm(acts_cat.size(0))]
    save_path = save_dir / f"acts_{file_idx}.pt"
    torch.save(acts_cat, save_path)
    return acts_cat.size(0)  # Return count for logging


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
    """
    Generate activations and save them to disk.
    
    Parameters:
    - model: The HookedTransformer model
    - dataset: Dataset to generate activations from
    - n_activations: Total number of activations to generate
    - generator_batch_size: Size of batches for generating activations
    - skip_first_n_tokens: Number of tokens to skip from the beginning
    - ctx_len: Context length for tokenization
    - max_acts_per_file: Maximum number of activations per file
    - hook_name: Name of the hook to extract activations from
    - dir: Directory to save activations to
    """
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
            saved_count = _save_activations_to_disk(acts, save_dir, file_acc, skip_first_n_tokens)
            print(f"Saved {saved_count} activations to {save_dir}/acts_{file_acc}.pt")
            file_acc += 1
            acts = []
            act_acc = 0
            
    if act_acc > 0:
        saved_count = _save_activations_to_disk(acts, save_dir, file_acc, skip_first_n_tokens)
        print(f"Saved {saved_count} activations to {save_dir}/acts_{file_acc}.pt")


def cached_activation_generator(
    model: HookedTransformer,
    dataset: Dataset | IterableDataset,
    hook_name: str,
    activation_batch_size: int,
    generator_batch_size=24,
    acts_per_run=1_000_000,  # Combined parameter (was examples_per_run and max_acts_per_file)
    ctx_len=128,
    skip_first_n_tokens=10,
    save_to_disk: str | None = None,
):
    """
    Generate activations and cache them in memory, with optional saving to disk.
    With the same random seed, this will produce identical files to cache_activations_to_disk.
    
    Parameters:
    - model: The HookedTransformer model
    - dataset: Dataset to generate activations from
    - hook_name: Name of the hook to extract activations from
    - activation_batch_size: Size of batches yielded to training
    - generator_batch_size: Size of batches for generating activations
    - acts_per_run: Maximum activations per run (and per file when saving)
    - ctx_len: Context length for tokenization
    - skip_first_n_tokens: Number of tokens to skip from the beginning
    - save_to_disk: Optional path to save activations (None = don't save)
    
    Yields:
    - Batches of activations for training
    """
    data_iter = token_iter(model, dataset, generator_batch_size, ctx_len)
    
    # Calculate how many batches to generate per run
    # Each token batch gives us generator_batch_size * (ctx_len - skip_first_n_tokens) tokens
    tokens_per_batch = generator_batch_size * (ctx_len - skip_first_n_tokens)
    batches_per_run = acts_per_run // tokens_per_batch
    
    # Set up disk saving if requested
    save_dir = None
    file_acc = 0
    if save_to_disk:
        save_dir = Path(save_to_disk)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Check if files already exist in directory and get the highest index
        existing_files = list(save_dir.glob("acts_*.pt"))
        if existing_files:
            existing_indices = [int(f.stem.split('_')[1]) for f in existing_files]
            file_acc = max(existing_indices) + 1
    
    while True:
        my_acts = []
        print(f"Generating new activations (batch size: {generator_batch_size}, batches: {batches_per_run})...")
        
        # Generate activations for this run
        for _ in range(batches_per_run):
            try:
                token_batch = next(data_iter)
                activations = get_activations(model, token_batch, hook_name)
                my_acts.append(activations)
            except StopIteration:
                print("Dataset exhausted.")
                if not my_acts:  # No activations generated
                    return
                break
        
        # Save to disk if requested
        if save_to_disk and my_acts:
            saved_count = _save_activations_to_disk(my_acts, save_dir, file_acc, skip_first_n_tokens)
            print(f"Saved {saved_count} activations to {save_dir}/acts_{file_acc}.pt")
            file_acc += 1
            
        # Process activations for training (we use the same permutation as when saving)
        acts_cat = torch.cat(my_acts, dim=0)
        acts_cat = acts_cat[:, skip_first_n_tokens:, :]
        acts_cat = acts_cat.reshape(-1, acts_cat.size(-1))
        acts_cat = acts_cat[torch.randperm(acts_cat.size(0))]
        
        # Yield batches for training
        for i in range(0, len(acts_cat), activation_batch_size):
            yield acts_cat[i : i + activation_batch_size]


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


@torch.no_grad()
def compute_metrics(activations, features, reconstructions):
    l0 = (features > 0).float().sum(1).mean()

    e = reconstructions - activations
    total_variance = (activations - activations.mean(0)).pow(2).sum()
    squared_error = e.pow(2)
    fvu = squared_error.sum() / total_variance

    return l0.item(), fvu.item()

################### Legacy generators ##########################
def _cached_activation_generator_old(
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

def _cache_activations_to_disk_old(
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
