"""
Dataset wrapper for Pairwise Integration training.
Filters synthetic data to only include specific input->output pairs.
"""

from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Union
import torch
import torch.nn.functional as F
import random

from src.data.synthetic_generator import SyntheticDataGenerator

class PairwiseDataset(Dataset):
    """
    Dataset that yields samples for specific input->output modality pairs.
    """
    
    # Mapping from logical modality name to input keys expected by MultiModalCreature
    INPUT_MAPPING = {
        "vision": ["left_eye_image", "right_eye_image"],
        "audio": ["audio_waveform"],
        "internal_text": ["internal_voice_tokens"],
        "external_text": ["external_voice_tokens"],
        "touch": ["touch_data"],
        "proprioception": ["joint_positions", "joint_rotations"]
    }
    
    # Mapping from logical modality name to target keys expected by MultiModalCreature
    TARGET_MAPPING = {
        "internal_text": ["internal_text"],
        "external_text": ["external_text"],
        "audio": ["audio"],
        "animation": ["animation"]
    }

    # Mapping from logical modality name to target keys expected by MultiModalCreature
    TARGET_MAPPING = {
        "internal_text": ["internal_text"],
        "external_text": ["external_text"],
        "audio": ["audio"],
        "animation": ["animation"]
    }

    def __init__(self, generator: SyntheticDataGenerator, pairs: List[Tuple[str, str]], length: int = 1000):
        """
        Initialize PairwiseDataset.

        Args:
            generator (SyntheticDataGenerator): Generator instance.
            pairs (List[Tuple[str, str]]): List of (input_modality, output_modality) to train on.
                e.g. [("vision", "external_text"), ("external_text", "audio")]
            length (int): Virtual length of the dataset.
        """
        self.generator = generator
        self.pairs = pairs
        self.length = length
        
        # Validate pairs
        for input_mod, output_mod in pairs:
            if input_mod not in self.INPUT_MAPPING:
                raise ValueError(f"Unknown input modality: {input_mod}. Available: {list(self.INPUT_MAPPING.keys())}")
            if output_mod not in self.TARGET_MAPPING:
                raise ValueError(f"Unknown target modality: {output_mod}. Available: {list(self.TARGET_MAPPING.keys())}")
        
    def __len__(self):
        """
        Purpose:
            Returns the virtual length of the dataset.

        Workflow:
            1. Return configured length value

        ToDo:
            None

        Returns:
            int: Dataset length.
        """
        return self.length
        
    def __getitem__(self, idx):
        """
        Generate a single sample for a random pair from the list.

        Purpose:
            Generates a training sample with specific input->output modality pairing.

        Workflow:
            1. Randomly select an input-output pair
            2. Generate full sample from generator
            3. Filter to only include selected modalities
            4. Return formatted sample dict

        ToDo:
            None

        Args:
            idx: Sample index (unused, random pair selected instead).

        Returns:
            dict: Sample with 'inputs', 'targets', and 'task' keys.
        """
        # Select a random pair for this sample if multiple are defined
        # This effectively mixes tasks in the batch
        input_mod, output_mod = random.choice(self.pairs)
        
        # Generate full sample
        full_sample = self.generator.generate_sample()
        
        # Filter Inputs
        inputs = {}
        # 1. Add primary input modality
        input_keys = self.INPUT_MAPPING[input_mod]
        for key in input_keys:
            if key in full_sample["inputs"]:
                inputs[key] = full_sample["inputs"][key]
        
        # Filter Targets
        targets = {}
        target_keys = self.TARGET_MAPPING[output_mod]
        for key in target_keys:
            if key in full_sample["targets"]:
                targets[key] = full_sample["targets"][key]
                
        # Return dict matching Trainer expectations
        return {
            "inputs": inputs,
            "targets": targets,
            "task": f"{input_mod}_to_{output_mod}" # Optional metadata
        }

def pairwise_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that handles mixed keys by padding missing modalities with zeros.

    Purpose:
        Collates batch samples with potentially different modality keys.

    Workflow:
        1. Collect all unique input/target keys across batch
        2. Build batched inputs by stacking or zero-padding
        3. Build batched targets similarly
        4. Handle nested dicts (touch_data, animation) recursively

    ToDo:
        None

    Args:
        batch: List of sample dicts from dataset.

    Returns:
        dict: Batched data with 'inputs', 'targets', 'task' keys.
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
        # Find a reference tensor to look up shape/dtype
        ref_tensor = None
        for sample in batch:
            if key in sample["inputs"]:
                ref_item = sample["inputs"][key]
                # Handle nested dicts (e.g. touch_data) if necessary?
                # SyntheticGenerator returns touch_data as Tensors in 'touch_data' dict? 
                # PairwiseDataset might map 'touch' -> 'touch_data' which is a dict.
                # If so, we need recursive collation.
                # Let's check SyntheticGenerator output.
                # 'touch_data' is a dict of tensors.
                if isinstance(ref_item, dict):
                    # Recursive handling for nested dict
                    pass # See below
                else:
                    ref_tensor = ref_item
                break
        
        if ref_tensor is None and key == "touch_data":
             # Special handling for touch dict
             batched_inputs[key] = _collate_nested_dict(batch, "inputs", key)
             continue
             
        if ref_tensor is None: 
            continue # Should not happen given logic above

        # Stack
        tensors = []
        for sample in batch:
            if key in sample["inputs"]:
                tensors.append(sample["inputs"][key])
            else:
                # Create zero tensor
                tensors.append(torch.zeros_like(ref_tensor))
        
        batched_inputs[key] = torch.stack(tensors)

    # 3. Build batched targets (Similar logic)
    batched_targets = {}
    for key in target_keys:
        if key == "animation": # animation is nested dict
             batched_targets[key] = _collate_nested_dict(batch, "targets", key)
             continue

        ref_tensor = None
        for sample in batch:
            if key in sample["targets"]:
                ref_tensor = sample["targets"][key]
                break
        
        tensors = []
        for sample in batch:
            if key in sample["targets"]:
                tensors.append(sample["targets"][key])
            else:
                tensors.append(torch.zeros_like(ref_tensor))
        
        batched_targets[key] = torch.stack(tensors)
        
    return {
        "inputs": batched_inputs,
        "targets": batched_targets,
        "task": [s["task"] for s in batch]
    }

def _collate_nested_dict(batch, root_key, nested_key):
    """
    Helper to collate dict of tensors (like touch_data or animation).

    Purpose:
        Recursively collates nested dictionary of tensors.

    Workflow:
        1. Collect all subkeys across batch
        2. For each subkey, stack tensors or zero-pad

    ToDo:
        None

    Args:
        batch: List of sample dicts.
        root_key: Top-level key ('inputs' or 'targets').
        nested_key: Nested dict key ('touch_data' or 'animation').

    Returns:
        dict: Collated nested dictionary of stacked tensors.
    """
    # Collect all subkeys
    subkeys = set()
    for sample in batch:
        if nested_key in sample[root_key]:
            subkeys.update(sample[root_key][nested_key].keys())
            
    collated = {}
    for sk in subkeys:
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
        collated[sk] = torch.stack(tensors)
    return collated
