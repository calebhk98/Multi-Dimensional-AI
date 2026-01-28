"""
Dataset wrapper for End-to-End Multi-Modal training.
Wraps the synthetic generator to produce full multi-modal samples.
"""

from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional
import torch

from src.data.synthetic_generator import SyntheticDataGenerator

class MultiModalDataset(Dataset):
    """
    Dataset that yields full multi-model samples (6 inputs, 4 outputs).
    """
    
    def __init__(self, generator: SyntheticDataGenerator, length: int = 1000):
        """
        ==============================================================================
        Function: __init__
        ==============================================================================
        Purpose:  Initializes the MultiModalDataset.
        
        Args:
            generator: SyntheticDataGenerator
                Generator instance to produce random samples.
            length: int
                Virtual length of the dataset (default: 1000).
        
        Returns:
            None
        
        Dependencies:
            - None
        
        Processing Workflow:
            1.  Store generator and length.
        
        ToDo:
            - None
        ==============================================================================
        """
        self.generator = generator
        self.length = length
        
    def __len__(self):
        """
        ==============================================================================
        Function: __len__
        ==============================================================================
        Purpose:  Returns the length of the dataset.
        
        Args:
            None
        
        Returns:
            int - Length of the dataset.
        
        Dependencies:
            - None
        
        Processing Workflow:
            1.  Return self.length.
        
        ToDo:
            - None
        ==============================================================================
        """
        return self.length
        
    def __getitem__(self, idx):
        """
        ==============================================================================
        Function: __getitem__
        ==============================================================================
        Purpose:  Generates a single full multi-modal sample.
        
        Args:
            idx: int
                Index of the sample (unused by synthetic generator).
        
        Returns:
            Dict[str, Dict[str, torch.Tensor]] - Sample dictionary with "inputs" and "targets".
        
        Dependencies:
            - self.generator.generate_sample()
        
        Processing Workflow:
            1.  Call generator.generate_sample().
        
        ToDo:
            - None
        ==============================================================================
        """
        return self.generator.generate_sample()

def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    ==============================================================================
    Function: multimodal_collate_fn
    ==============================================================================
    Purpose:  Collate function that handles nested dictionaries (like touch_data and animation).
    
    Args:
        batch: List[Dict[str, Any]]
            List of samples from the dataset.
    
    Returns:
        Dict[str, Any] - Batched dictionary with stacked tensors.
    
    Dependencies:
        - torch.stack
        - _collate_nested_dict
    
    Processing Workflow:
        1.  Identify all input and target keys present in the batch.
        2.  For each key:
            a. If it's a nested dict (touch/animation), call helper.
            b. If regular tensor, stack them (using zero padding if missing).
        3.  Return dictionary with batched inputs and targets.
    
    ToDo:
        - None
    ==============================================================================
    """
    # 1. Collect all keys present in the batch
    input_keys = set()
    target_keys = set()
    
    for sample in batch:
        input_keys.update(sample["inputs"].keys())
        target_keys.update(sample["targets"].keys())
        
    # 2. Build batched inputs
    batched_inputs = {}
    for key in input_keys:
        if key == "touch_data":
            batched_inputs[key] = _collate_nested_dict(batch, "inputs", key)
            continue
            
        batched_inputs[key] = _stack_tensors(batch, "inputs", key)

    # 3. Build batched targets
    batched_targets = {}
    for key in target_keys:
        if key == "animation":
             batched_targets[key] = _collate_nested_dict(batch, "targets", key)
             continue
        
        batched_targets[key] = _stack_tensors(batch, "targets", key)
        
    return {
        "inputs": batched_inputs,
        "targets": batched_targets
    }

def _stack_tensors(batch: List[Dict], root_key: str, key: str) -> torch.Tensor:
    """
    ==============================================================================
    Function: _stack_tensors
    ==============================================================================
    Purpose:  Helper to stack tensors from a batch, handling missing keys.
    
    Args:
        batch: List of samples.
        root_key: "inputs" or "targets".
        key: The key to stack.
    
    Returns:
        torch.Tensor - Stacked tensor.
    ==============================================================================
    """
    # Find reference tensor
    ref_tensor = None
    for sample in batch:
        if key in sample[root_key]:
            ref_tensor = sample[root_key][key]
            break
            
    tensors = []
    for sample in batch:
        if key in sample[root_key]:
            tensors.append(sample[root_key][key])
        else:
            tensors.append(torch.zeros_like(ref_tensor))
            
    return torch.stack(tensors)

def _collate_nested_dict(batch: List[Dict], root_key: str, nested_key: str) -> Dict[str, torch.Tensor]:
    """
    ==============================================================================
    Function: _collate_nested_dict
    ==============================================================================
    Purpose:  Helper to collate dict of tensors (like touch_data or animation).
    
    Args:
        batch: List
            List of samples.
        root_key: str
            "inputs" or "targets".
        nested_key: str
            The nested dictionary key (e.g., "touch_data").
    
    Returns:
        Dict - Collated nested dictionary.
    ==============================================================================
    """
    # Collect all subkeys
    subkeys = set()
    for sample in batch:
        if nested_key in sample[root_key]:
            subkeys.update(sample[root_key][nested_key].keys())
            
    collated = {}
    for sk in subkeys:
        collated[sk] = _stack_nested_tensors(batch, root_key, nested_key, sk)
    return collated

def _stack_nested_tensors(batch, root_key, nested_key, sk):
    """
    ==============================================================================
    Function: _stack_nested_tensors
    ==============================================================================
    Purpose:  Helper to stack nested tensors.
    
    Args:
        batch: Batch list.
        root_key: Root key.
        nested_key: Nested dict key.
        sk: Subkey.
        
    Returns:
        Stacked tensor.
    ==============================================================================
    """
    ref_tensor = None
    for sample in batch:
        if nested_key in sample[root_key] and sk in sample[root_key][nested_key]:
            ref_tensor = sample[root_key][nested_key][sk]
            break
    
    tensors = []
    for sample in batch:
        if nested_key in sample[root_key] and sk in sample[root_key][nested_key]:
            tensors.append(sample[root_key][nested_key][sk])
        else:
            tensors.append(torch.zeros_like(ref_tensor))
            
    return torch.stack(tensors)
