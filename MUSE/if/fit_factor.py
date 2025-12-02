import argparse
import logging
import os
import sys
import torch
import time
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
from typing import Optional, List, Tuple
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Add grandparent directory for kronfluence (상대경로로 변경)
grandparent_dir = os.path.dirname(parent_dir)
tnpo_dir = os.path.dirname(grandparent_dir)  # tnpo directory
kronfluence_path = os.path.join(tnpo_dir, "kronfluence")
sys.path.insert(0, kronfluence_path)

# Import kronfluence
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.dataset import DataLoaderKwargs


# ============================================================================
# MUSE Utility Functions (copied to avoid relative import issues)
# ============================================================================

def read_text(file_path: str) -> str:
    """Read text from a .txt file."""
    if Path(file_path).suffix != '.txt':
        raise ValueError(f"Expected .txt file, got {file_path}")
    with open(file_path, 'r') as f:
        text: str = f.read()
    return text


def pad_or_trim_tensor(tensor, target_length, padding_value=0):
    """Pad or trim tensor to target length."""
    current_length = tensor.size(0)

    if current_length < target_length:
        # Padding
        padding_size = target_length - current_length
        padding_tensor = torch.full((padding_size,), padding_value, dtype=tensor.dtype)
        padded_tensor = torch.cat((tensor, padding_tensor))
        return padded_tensor

    elif current_length > target_length:
        # Trimming
        trimmed_tensor = tensor[:target_length]
        return trimmed_tensor

    else:
        # No change needed
        return tensor


# ============================================================================
# MUSE Dataset Classes (copied to avoid relative import issues)
# ============================================================================

class DefaultDataset(Dataset):
    """MUSE-style default dataset for language modeling."""

    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer | None = None,
        max_len: int | None = 4096,
        add_bos_token: bool = True
    ):
        if Path(file_path).suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data[0], str):
                self.strings = data
            elif isinstance(data[0], dict) and 'text' in data[0] \
                    and isinstance(data[0]['text'], str):
                self.strings = [d['text'] for d in data]
                if 'input_ids' in data[0]:
                    self.input_ids = [torch.tensor(d['input_ids']) for d in data]
                    return  # Done, since we have `input_ids` ready.
            else:
                raise ValueError("Format of this `.json` file is not recognized.")

            assert tokenizer is not None, "Tokenizer must be specified."

            self.input_ids = []
            for s in self.strings:
                encoding: torch.Tensor = tokenizer(
                    s,
                    add_special_tokens=add_bos_token,
                    return_tensors='pt'
                ).input_ids[0]
                encoding = pad_or_trim_tensor(
                    encoding,
                    target_length=max_len,
                    padding_value=tokenizer.pad_token_id
                )
                self.input_ids.append(encoding)

            return  # end if Path(file_path).suffix == '.json'

        assert Path(file_path).suffix == '.txt'

        tokens = tokenizer(read_text(file_path), add_special_tokens=False, return_tensors='pt').input_ids[0]
        assert len(tokens.shape) == 1, "Debug error: Tokens not 1-dimensional"

        if add_bos_token:
            self.input_ids = [
                F.pad(
                    tokens[i : i + max_len - 1], (1, 0),
                    value=tokenizer.bos_token_id
                )
                for i in range(0, len(tokens), max_len - 1)
            ]
        else:
            self.input_ids = [
                tokens[i : i + max_len]
                for i in range(0, len(tokens), max_len)
            ]

        # Rotate the tokens if the last `input_ids` isn't filled to max_len
        if len(self.input_ids[-1]) < max_len:
            self.input_ids[-1] = torch.concat(
                [self.input_ids[-1], self.input_ids[0]], dim=-1
            )[:max_len]

        # Original strings
        self.strings = tokenizer.batch_decode(self.input_ids, skip_special_tokens=True)

        pass  # def __init__()


    def __getitem__(self, index):
        return self.input_ids[index]


    def __len__(self):
        return len(self.input_ids)


    def get_collate_fn(self):

        def collate_fn(batch: List[torch.Tensor]):
            batch = torch.stack(batch)
            return {
                "input_ids": batch,
                "labels": batch.clone()
            }

        return collate_fn


class ForgetRetainDataset(DefaultDataset):
    """MUSE-style dataset for machine unlearning."""

    def __init__(
        self,
        forget_file_path: str,
        tokenizer: AutoTokenizer,
        retain_file_path: str | None = None,
        max_len: int = 4096,
        add_bos_token: bool = True
    ):
        self.forget_dataset = DefaultDataset(
            forget_file_path, tokenizer,
            max_len=max_len, add_bos_token=add_bos_token
        )

        self.retain_exists = retain_file_path is not None
        if self.retain_exists:
            self.retain_dataset = DefaultDataset(
                retain_file_path, tokenizer,
                max_len=max_len, add_bos_token=add_bos_token
            )

        self.tokenizer = tokenizer


    def __getitem__(self, index):
        if self.retain_exists:
            return (
                self.forget_dataset[index],
                self.retain_dataset[index % len(self.retain_dataset)]
            )
        else:
            return self.forget_dataset[index], None


    def __len__(self):
        return len(self.forget_dataset)


    def get_collate_fn(self):

        def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
            batch_forget = torch.stack([pair[0] for pair in batch])
            dict_forget = {
                "input_ids": batch_forget,
                "labels": batch_forget.clone(),
                "attention_mask": torch.ones_like(batch_forget)
            }

            if self.retain_exists:
                batch_retain = torch.stack([pair[1] for pair in batch])
                dict_retain = {
                    "input_ids": batch_retain,
                    "labels": batch_retain.clone(),
                    "attention_mask": torch.ones_like(batch_retain, dtype=torch.bool)
                }
            else:
                dict_retain = None

            return dict_forget, dict_retain

        return collate_fn

# Import local task definition
# Handle both directory names (utils.py or utils) for compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_py_dir = os.path.join(current_dir, "utils.py")
utils_dir = os.path.join(current_dir, "utils")

# Check which directory exists and import accordingly
if os.path.exists(utils_py_dir) and os.path.isdir(utils_py_dir):
    # utils.py directory exists (서버 환경)
    task_module_path = os.path.join(utils_py_dir, "task.py")
    if os.path.exists(task_module_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("task", task_module_path)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        LanguageModelingTask = task_module.LanguageModelingTask
    else:
        raise ImportError(f"Cannot find task.py in {utils_py_dir}")
elif os.path.exists(utils_dir) and os.path.isdir(utils_dir):
    # utils directory exists (로컬 환경)
    task_module_path = os.path.join(utils_dir, "task.py")
    if os.path.exists(task_module_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("task", task_module_path)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        LanguageModelingTask = task_module.LanguageModelingTask
    else:
        raise ImportError(f"Cannot find task.py in {utils_dir}")
else:
    raise ImportError("Cannot find utils or utils.py directory")

# Configure CUDA memory
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True",
)


def muse_data_collator(batch):
    """MUSE-style data collator for Kronfluence."""
    # batch is a list of tensors (input_ids)
    batch_tensor = torch.stack(batch)
    return {
        'input_ids': batch_tensor,
        'labels': batch_tensor.clone(),
        'attention_mask': torch.ones_like(batch_tensor)
    }


def load_model_and_tokenizer(model_name, tokenizer_name=None):
    """Load model and tokenizer (MUSE style - no model_family dependency)."""
    # Use model_name for tokenizer if not specified
    if tokenizer_name is None:
        tokenizer_name = model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Get num_layers from model config
    if hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model.config, 'n_layer'):
        num_layers = model.config.n_layer
    else:
        raise ValueError("Cannot determine number of layers from model config")

    # Auto-detect model family from model type
    model_type = model.config.model_type.lower()
    if 'llama' in model_type:
        family = 'llama2-7b'
    elif 'pythia' in model_type or 'gpt_neox' in model_type:
        family = 'pythia'
    elif 'phi' in model_type:
        family = 'phi'
    else:
        # Default to llama structure for most modern models
        logging.warning(f"Unknown model type '{model_type}', defaulting to llama2-7b structure")
        family = 'llama2-7b'

    return model, tokenizer, num_layers, family


def parse_factor_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fit Kronfluence factors (MUSE style)")

    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Tokenizer name or path (default: same as model_name)")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max sequence length (default: 4096)")

    # Data arguments (MUSE style)
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to data file (.json or .txt)")
    parser.add_argument("--add_bos_token", action="store_true", default=True,
                        help="Add BOS token to sequences")
    
    # Factor computation
    parser.add_argument("--factor_strategy", type=str, default="ekfac", choices=["ekfac", "kfac", "diagonal"])
    parser.add_argument("--use_half_precision", action="store_true", default=False)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./kronfluence_factors")
    
    # Partition arguments (for backward compatibility)
    parser.add_argument("--covariance_module_partitions", type=int, default=4)
    parser.add_argument("--lambda_module_partitions", type=int, default=4)
    parser.add_argument("--covariance_data_partitions", type=int, default=4)
    parser.add_argument("--lambda_data_partitions", type=int, default=4)
    
    return parser.parse_args()


def main():
    """Main function (MUSE style)."""
    args = parse_factor_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("🚀 Starting Kronfluence factor fitting (MUSE style)")

    # Load model and tokenizer
    model, tokenizer, num_layers, family = load_model_and_tokenizer(args.model_name, args.tokenizer_name)
    logging.info(f"Loaded model: {args.model_name} with {num_layers} layers")
    logging.info(f"Auto-detected model family: {family}")
    logging.info(f"Using max_length: {args.max_length}")

    # Load MUSE dataset
    train_dataset = DefaultDataset(
        file_path=args.data_file,
        tokenizer=tokenizer,
        max_len=args.max_length,
        add_bos_token=args.add_bos_token
    )
    logging.info(f"Loaded MUSE dataset with {len(train_dataset)} samples from {args.data_file}")

    # Setup task with model config (auto-detected family)
    task_config = {
        'model': {
            'num_layers': num_layers,
            'family': family
        }
    }
    task = LanguageModelingTask(config=task_config)
    model = prepare_model(model, task)

    # Setup accelerator
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[init_kwargs, ddp_kwargs])
    model = accelerator.prepare_model(model)

    os.makedirs(args.output_dir, exist_ok=True)

    # Create analyzer
    analyzer = Analyzer(
        analysis_name="if_results",
        model=model,
        task=task,
        output_dir=args.output_dir,
    )

    dataloader_kwargs = DataLoaderKwargs(
        num_workers=4,
        collate_fn=muse_data_collator,
        pin_memory=True
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Configure factors
    factor_args = FactorArguments(strategy=args.factor_strategy)
    factors_name = args.factor_strategy

    if args.use_half_precision:
        factor_args = all_low_precision_factor_arguments(strategy=args.factor_strategy, dtype=torch.bfloat16)
        factors_name += "_half"

    # Set partition parameters
    factor_args.covariance_module_partitions = args.covariance_module_partitions
    factor_args.lambda_module_partitions = args.lambda_module_partitions
    factor_args.covariance_data_partitions = args.covariance_data_partitions
    factor_args.lambda_data_partitions = args.lambda_data_partitions

    # No limit on examples
    factor_args.covariance_max_examples = None
    factor_args.lambda_max_examples = None

    # Fit all factors using standard Kronfluence
    logging.info("🚀 Fitting all factors (covariance, eigendecomposition, lambda)...")
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=args.train_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=False,
    )

    logging.info("✅ Factor fitting completed!")


if __name__ == "__main__":
    main()