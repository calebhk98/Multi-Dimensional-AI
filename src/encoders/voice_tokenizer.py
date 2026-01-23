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
		==============================================================================
		Function: __init__
		==============================================================================
		Purpose:  Initializes the VoiceTokenizer using a pre-trained GPT-2 tokenizer.
		          Configures vocabulary size, sequence length, and special tokens
		          (pad, null) for processing voice inputs.

		Parameters:
		    - vocab_size: int
		        Target vocabulary size (default: 50257).
		    - max_seq_length: int
		        Maximum sequence length for encoding (default: 512).
		    - add_special_tokens: bool
		        Whether to add custom special tokens like <NULL> and <PAD> (default: True).

		Returns:
		    None

		Dependencies:
		    - transformers.GPT2Tokenizer

		Processing Workflow:
		    1.  Store configuration parameters.
		    2.  Load pre-trained 'gpt2' tokenizer.
		    3.  If `add_special_tokens` is True:
		        a. Add `<NULL>` and `<PAD>` tokens.
		        b. Update vocabulary size.
		    4.  Store IDs for special tokens.
		    5.  Set the padding token.

		ToDo:
		    - None

		Usage:
		    tokenizer = VoiceTokenizer()
		==============================================================================
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
		==============================================================================
		Function: encode
		==============================================================================
		Purpose:  Encodes a single text string into token IDs.

		Parameters:
		    - text: str
		        Input text string to encode.
		    - add_special_tokens: bool
		        Whether to add BOS/EOS tokens (default: True).
		    - truncation: bool
		        Whether to truncate to `max_seq_length` (default: True).
		    - padding: str
		        Padding strategy (default: "max_length").
		    - return_tensors: str
		        Return format, e.g., 'pt' for PyTorch tensors (default: "pt").

		Returns:
		    Dict[str, torch.Tensor] - Dictionary containing:
		        - "input_ids": Token IDs tensor.
		        - "attention_mask": Attention mask tensor.

		Dependencies:
		    - self.tokenizer

		Processing Workflow:
		    1.  Call `self.tokenizer` with provided arguments.
		    2.  Return the encoding result.

		ToDo:
		    - None

		Usage:
		    encoding = tokenizer.encode("Hello world")
		==============================================================================
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
		==============================================================================
		Function: decode
		==============================================================================
		Purpose:  Decodes a sequence of token IDs back into a text string.

		Parameters:
		    - token_ids: torch.Tensor
		        Tensor of token IDs, shape [seq_len] or [batch, seq_len].
		    - skip_special_tokens: bool
		        Whether to remove special tokens from the output string (default: True).

		Returns:
		    str - Decoded text string.

		Dependencies:
		    - self.tokenizer.decode

		Processing Workflow:
		    1.  Check dimensions of `token_ids`; if 2D, take the first sequence.
		    2.  Convert tensor to list.
		    3.  Decode list using `self.tokenizer.decode`.
		    4.  Return the resulting string.

		ToDo:
		    - None

		Usage:
		    text = tokenizer.decode(token_ids)
		==============================================================================
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
		==============================================================================
		Function: batch_encode
		==============================================================================
		Purpose:  Encodes a list of text strings into a batch of token IDs.

		Parameters:
		    - texts: List[str]
		        List of input text strings.
		    - **kwargs:
		        Additional arguments passed to `self.tokenizer` (e.g., padding, truncation).

		Returns:
		    Dict[str, torch.Tensor] - Batched encoding dictionary containing `input_ids`
		    and `attention_mask`.

		Dependencies:
		    - self.tokenizer

		Processing Workflow:
		    1.  Call `self.tokenizer` with list of texts.
		    2.  Set defaults for max_length, truncation, and padding if not overridden.
		    3.  Return the batched encoding.

		ToDo:
		    - None

		Usage:
		    batch_enc = tokenizer.batch_encode(["Hello", "World"])
		==============================================================================
		"""
		return self.tokenizer(
			texts,
			max_length=self.max_seq_length,
			truncation=True,
			padding="max_length",
			return_tensors="pt",
			**kwargs
		)
	
	def batch_decode(
		self,
		token_ids: torch.Tensor,
		**kwargs
	) -> List[str]:
		"""
		==============================================================================
		Function: batch_decode
		==============================================================================
		Purpose:  Decodes a batch of token ID sequences back into a list of strings.

		Parameters:
		    - token_ids: torch.Tensor
		        Tensor of token IDs [batch, seq_len].
		    - **kwargs:
		        Additional arguments passed to `self.tokenizer.batch_decode`.

		Returns:
		    List[str] - List of decoded text strings.

		Dependencies:
		    - self.tokenizer.batch_decode

		Processing Workflow:
		    1.  Convert `token_ids` to list.
		    2.  Call `self.tokenizer.batch_decode`.
		    3.  Return the list of strings.

		ToDo:
		    - None

		Usage:
		    texts = tokenizer.batch_decode(batch_ids)
		==============================================================================
		"""
		return self.tokenizer.batch_decode(
			token_ids.tolist(),
			skip_special_tokens=True,
			**kwargs
		)
	
	def get_vocab_size(self) -> int:
		"""
		==============================================================================
		Function: get_vocab_size
		==============================================================================
		Purpose:  Returns the current vocabulary size of the tokenizer.

		Parameters:
		    - None

		Returns:
		    int - Vocabulary size (including special tokens).

		Dependencies:
		    - len(self.tokenizer)

		Processing Workflow:
		    1.  Return the length of the tokenizer.

		ToDo:
		    - None

		Usage:
		    size = tokenizer.get_vocab_size()
		==============================================================================
		"""
		return len(self.tokenizer)
	
	def __len__(self) -> int:
		"""
		==============================================================================
		Function: __len__
		==============================================================================
		Purpose:  Returns the vocabulary size (allows `len(tokenizer)`).

		Parameters:
		    - None

		Returns:
		    int - Vocabulary size.

		Dependencies:
		    - self.get_vocab_size

		Processing Workflow:
		    1.  Call `self.get_vocab_size()`.

		ToDo:
		    - None

		Usage:
		    size = len(tokenizer)
		==============================================================================
		"""
		return self.get_vocab_size()
