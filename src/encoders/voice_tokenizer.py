"""
Voice tokenizer for converting text to tokens.
Uses GPT-2 tokenizer by default (BPE).
"""

import torch
from transformers import GPT2Tokenizer
from typing import List, Dict, Optional


class VoiceTokenizer:
    """
    Tokenizer for internal and external voice input.
    
    Converts text strings into token IDs that can be processed by the model.
    Uses GPT-2's BPE tokenizer which has a vocabulary of ~50k tokens.
    """
    
    # Special tokens
    NULL_TOKEN = "<NULL>"  # Represents no output
    PAD_TOKEN = "<PAD>"
    
    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_length: int = 512,
        add_special_tokens: bool = True,
    ):
        """
        Initialize voice tokenizer.
        
        Args:
            vocab_size: Size of vocabulary
            max_seq_length: Maximum sequence length
            add_special_tokens: Whether to add special tokens for null/padding
        """
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Load GPT-2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Add special tokens if needed
        if add_special_tokens:
            special_tokens = {
                'additional_special_tokens': [self.NULL_TOKEN, self.PAD_TOKEN]
            }
            self.tokenizer.add_special_tokens(special_tokens)
            self.vocab_size = len(self.tokenizer)
        
        # Token IDs for special tokens
        self.null_token_id = self.tokenizer.convert_tokens_to_ids(self.NULL_TOKEN)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.PAD_TOKEN)
        
        # Set padding token
        self.tokenizer.pad_token = self.PAD_TOKEN
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = True,
        padding: str = "max_length",
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Add BOS/EOS tokens
            truncation: Truncate to max_seq_length
            padding: Padding strategy
            return_tensors: Return format ('pt' for PyTorch)
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        encoding = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=self.max_seq_length,
            padding=padding if padding else False,
            return_tensors=return_tensors,
        )
        
        return encoding
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Tensor of token IDs [seq_len] or [batch, seq_len]
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text string
        """
        # Handle batch dimension
        if token_ids.dim() == 2:
            token_ids = token_ids[0]
        
        text = self.tokenizer.decode(
            token_ids.tolist(),
            skip_special_tokens=skip_special_tokens,
        )
        
        return text
    
    def batch_encode(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multiple texts.
        
        Args:
            texts: List of text strings
            **kwargs: Additional arguments for encode()
            
        Returns:
            Batched encoding dictionary
        """
        return self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            **kwargs
        )
    
    def batch_decode(
        self,
        token_ids: torch.Tensor,
        **kwargs
    ) -> List[str]:
        """
        Decode multiple token sequences.
        
        Args:
            token_ids: Tensor of token IDs [batch, seq_len]
            **kwargs: Additional arguments for decode()
            
        Returns:
            List of decoded text strings
        """
        return self.tokenizer.batch_decode(
            token_ids.tolist(),
            skip_special_tokens=True,
            **kwargs
        )
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size including special tokens."""
        return len(self.tokenizer)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.get_vocab_size()
