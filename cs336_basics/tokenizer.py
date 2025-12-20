import regex as re
from typing import List, Iterable, Iterator, Dict, Tuple, Optional
import json

# Pretokenization pattern from the assignment
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """
    BPE Tokenizer for encoding text to token IDs and decoding token IDs to text.
    """
    
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and special tokens.
        
        Args:
            vocab: Dict mapping token ID to bytes
            merges: List of merge operations (pairs of bytes)
            special_tokens: Optional list of special tokens to add
        """
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        
        # Create inverse vocab for decoding
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Handle special tokens
        self.special_tokens = special_tokens if special_tokens else []
        
        # Add special tokens to vocab if not already present
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 256
        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in self.inverse_vocab:
                self.vocab[next_id] = token_bytes
                self.inverse_vocab[token_bytes] = next_id
                next_id += 1
        
        # Create merge priority dict for efficient encoding
        self.merge_priority = {pair: i for i, pair in enumerate(merges)}
        
        # Compile special token pattern for splitting
        # Sort special tokens by length (longest first) to handle overlapping tokens
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = '|'.join(re.escape(token) for token in sorted_tokens)
            self.special_pattern = re.compile(f'({special_pattern})')
        else:
            self.special_pattern = None
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Construct a Tokenizer from serialized vocabulary and merges files.
        
        Args:
            vocab_filepath: Path to vocabulary file
            merges_filepath: Path to merges file
            special_tokens: Optional list of special tokens
        
        Returns:
            Tokenizer instance
        """
        # Load vocabulary
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Convert string keys to int, string values to bytes
        vocab = {}
        for k, v in vocab_data.items():
            # Handle different serialization formats
            if isinstance(v, str):
                vocab[int(k)] = v.encode('latin-1')  # or 'utf-8' depending on how it was saved
            elif isinstance(v, list):
                vocab[int(k)] = bytes(v)
            else:
                vocab[int(k)] = v
        
        # Load merges
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges_data = json.load(f)
        
        # Convert merge data to list of tuples of bytes
        merges = []
        for merge in merges_data:
            if isinstance(merge, list) and len(merge) == 2:
                # Handle different serialization formats
                if isinstance(merge[0], str):
                    pair = (merge[0].encode('latin-1'), merge[1].encode('latin-1'))
                elif isinstance(merge[0], list):
                    pair = (bytes(merge[0]), bytes(merge[1]))
                else:
                    pair = (merge[0], merge[1])
                merges.append(pair)
        
        return cls(vocab, merges, special_tokens)
    
    def _apply_merges(self, word_bytes: List[bytes]) -> List[bytes]:
        """
        Apply BPE merges to a sequence of bytes.
        
        Args:
            word_bytes: List of individual byte objects
        
        Returns:
            List of merged byte sequences
        """
        if len(word_bytes) <= 1:
            return word_bytes
        
        # Keep applying merges until no more can be applied
        while len(word_bytes) > 1:
            # Find the earliest merge that applies
            best_pair = None
            best_pos = -1
            best_priority = float('inf')
            
            for i in range(len(word_bytes) - 1):
                pair = (word_bytes[i], word_bytes[i + 1])
                if pair in self.merge_priority:
                    priority = self.merge_priority[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_pair = pair
                        best_pos = i
            
            # If no merge found, we're done
            if best_pair is None:
                break
            
            # Apply the merge
            new_word = []
            i = 0
            while i < len(word_bytes):
                if i == best_pos:
                    # Merge this pair
                    merged = word_bytes[i] + word_bytes[i + 1]
                    new_word.append(merged)
                    i += 2
                elif i == best_pos + 1:
                    # Skip, already merged
                    i += 1
                else:
                    new_word.append(word_bytes[i])
                    i += 1
            
            word_bytes = new_word
        
        return word_bytes
    
    def encode(self, text: str) -> List[int]:
        """
        Encode an input text into a sequence of token IDs.
        
        Args:
            text: Input text string
        
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        token_ids = []
        
        # Split on special tokens first if they exist
        if self.special_pattern:
            # Use finditer to get both special tokens and text between them
            last_end = 0
            for match in self.special_pattern.finditer(text):
                # Process text before this special token
                if match.start() > last_end:
                    chunk = text[last_end:match.start()]
                    token_ids.extend(self._encode_chunk(chunk))
                
                # Process the special token itself
                special_token = match.group()
                if special_token in self.special_tokens:
                    token_bytes = special_token.encode('utf-8')
                    if token_bytes in self.inverse_vocab:
                        token_ids.append(self.inverse_vocab[token_bytes])
                
                last_end = match.end()
            
            # Process any remaining text after the last special token
            if last_end < len(text):
                chunk = text[last_end:]
                token_ids.extend(self._encode_chunk(chunk))
        else:
            # No special tokens, encode the whole text
            token_ids.extend(self._encode_chunk(text))
        
        return token_ids
    
    def _encode_chunk(self, text: str) -> List[int]:
        """
        Encode a chunk of text (no special tokens).
        
        Args:
            text: Text chunk to encode
        
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        token_ids = []
        
        # Pre-tokenize the chunk
        for match in re.finditer(PAT, text):
            word = match.group()
            
            # Convert to list of individual byte objects
            word_bytes = [bytes([b]) for b in word.encode('utf-8')]
            
            # Apply BPE merges
            merged_bytes = self._apply_merges(word_bytes)
            
            # Convert to token IDs
            for token_bytes in merged_bytes:
                if token_bytes in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[token_bytes])
                else:
                    # This shouldn't happen if vocab is complete
                    # Fall back to individual bytes
                    for b in token_bytes:
                        token_ids.append(b)
        
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.
        
        This is memory-efficient for large files.
        
        Args:
            iterable: An iterable of strings (e.g., file handle)
        
        Yields:
            Token IDs one at a time
        """
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        
        Args:
            ids: List of token IDs
        
        Returns:
            Decoded text string
        """
        if not ids:
            return ""
        
        # Concatenate all bytes
        byte_sequence = b''
        for token_id in ids:
            if token_id in self.vocab:
                byte_sequence += self.vocab[token_id]
            else:
                # Invalid token ID - skip or handle gracefully
                # For robustness, we can treat it as the byte value itself if < 256
                if token_id < 256:
                    byte_sequence += bytes([token_id])
        
        # Decode to string, replacing malformed bytes with replacement character
        try:
            return byte_sequence.decode('utf-8', errors='replace')
        except Exception:
            # Fallback
            return byte_sequence.decode('utf-8', errors='ignore')


# Helper functions for saving/loading vocab and merges

def save_vocab(vocab: Dict[int, bytes], filepath: str):
    """Save vocabulary to a JSON file."""
    # Convert bytes to list of ints for JSON serialization
    vocab_serializable = {k: list(v) for k, v in vocab.items()}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab_serializable, f)


def save_merges(merges: List[Tuple[bytes, bytes]], filepath: str):
    """Save merges to a JSON file."""
    # Convert bytes tuples to list of lists of ints
    merges_serializable = [[list(pair[0]), list(pair[1])] for pair in merges]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(merges_serializable, f)


def load_vocab(filepath: str) -> Dict[int, bytes]:
    """Load vocabulary from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    return {int(k): bytes(v) for k, v in vocab_data.items()}


def load_merges(filepath: str) -> List[Tuple[bytes, bytes]]:
    """Load merges from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        merges_data = json.load(f)
    return [(bytes(pair[0]), bytes(pair[1])) for pair in merges_data]