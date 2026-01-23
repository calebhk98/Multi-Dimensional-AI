"""
Unit tests for the VoiceTokenizer.
"""

import pytest
import torch
from src.encoders.voice_tokenizer import VoiceTokenizer


def test_tokenizer_initialization():
    """
    Purpose:
        Happy Path: Tokenizer Initialization.
        Verifies that the tokenizer initializes with the GPT-2 vocabulary 
        plus properly assigned special tokens (PAD, NULL).

    Workflow:
        1. Initialize VoiceTokenizer.
        2. Verify vocab size and special tokens.

    ToDo:
        - None
    """
    tokenizer = VoiceTokenizer()
    assert tokenizer.get_vocab_size() > 50000
    assert tokenizer.pad_token_id is not None
    assert tokenizer.null_token_id is not None

def test_tokenizer_encode_decode():
    """
    Purpose:
        Happy Path: Encode and Decode.
        Verifies that text can be encoded to tensors and decoded back to the original text.

    Workflow:
        1. Initialize VoiceTokenizer.
        2. Encode string "Hello, world!".
        3. Verify encoded keys and types.
        4. Decode back and verify content.

    ToDo:
        - None
    """
    tokenizer = VoiceTokenizer()
    text = "Hello, world!"
    
    # Encode
    encoded = tokenizer.encode(text)
    assert 'input_ids' in encoded
    assert 'attention_mask' in encoded
    assert isinstance(encoded['input_ids'], torch.Tensor)
    
    # Decode
    decoded = tokenizer.decode(encoded['input_ids'])
    assert text in decoded # Might have spaces/special chars, but core text should be there


def test_batch_processing():
    """
    Purpose:
        Happy Path: Batch Processing.
        Verifies batch encoding of multiple strings, ensuring they are padded
        to the max sequence length and return correct tensor shapes.

    Workflow:
        1. Initialize VoiceTokenizer.
        2. Encode batch of 3 texts.
        3. Verify output shape matches (batch_size, max_seq_len).
        4. Verify padding logic.

    ToDo:
        - None
    """
    tokenizer = VoiceTokenizer()
    texts = ["Hello", "World", "Testing batch"]
    
    batch = tokenizer.batch_encode(texts)
    input_ids = batch['input_ids']
    
    assert input_ids.shape[0] == 3
    assert input_ids.shape[1] == tokenizer.max_seq_length
    
    # Check padding
    # First token of padded sequences should be non-pad (or BOS)
    assert input_ids[0, 0] != tokenizer.pad_token_id
