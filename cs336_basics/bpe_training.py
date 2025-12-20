import regex as re
from collections import defaultdict
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count

# Pretokenization pattern from the assignment
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer with optimized pair counting.
    
    Args:
        input_path: Path to the text file with training data
        vocab_size: Maximum vocabulary size
        special_tokens: List of special tokens to add to vocabulary
    
    Returns:
        vocab: Dict mapping token ID to bytes
        merges: List of merge operations (tuple of bytes pairs)
    """
    
    # Initialize vocabulary with all byte values (0-255)
    vocab = {i: bytes([i]) for i in range(256)}
    
    # Add special tokens to vocabulary
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode('utf-8')
        next_id += 1
    
    # Calculate number of merges needed
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []
    
    # Read and pretokenize the corpus
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split on special tokens to avoid merging across them
    if special_tokens:
        # Create regex pattern to split on special tokens
        special_pattern = '|'.join(re.escape(token) for token in special_tokens)
        chunks = re.split(f'({special_pattern})', text)
        # Filter out empty strings and special tokens themselves
        chunks = [chunk for chunk in chunks if chunk and chunk not in special_tokens]
    else:
        chunks = [text]
    
    # Pretokenize each chunk and count frequencies
    word_freqs = {}
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            word = match.group()
            # Convert to bytes as a tuple of individual byte objects
            word_bytes = tuple(bytes([b]) for b in word.encode('utf-8'))
            word_freqs[word_bytes] = word_freqs.get(word_bytes, 0) + 1
    
    # Perform BPE merges with incremental pair counting
    merges = []
    
    # Initial pair count
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freqs[pair] += freq
    
    for _ in range(num_merges):
        if not pair_freqs:
            break
        
        # Find most frequent pair (break ties lexicographically by taking max)
        best_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)
        
        # Add merged token to vocabulary
        merged = best_pair[0] + best_pair[1]
        vocab[next_id] = merged
        next_id += 1
        
        # Update word frequencies and pair counts incrementally
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            # Check if this word contains the pair to merge
            if len(word) < 2:
                new_word_freqs[word] = freq
                continue
            
            new_word = []
            i = 0
            changes = []  # Track where merges happened
            
            while i < len(word):
                # Check if current and next form the pair to merge
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(merged)
                    changes.append(i)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word_tuple = tuple(new_word)
            new_word_freqs[new_word_tuple] = freq
            
            # Update pair counts incrementally
            if changes:  # Only if word was modified
                # Remove old pairs
                for i in range(len(word) - 1):
                    old_pair = (word[i], word[i + 1])
                    pair_freqs[old_pair] -= freq
                    if pair_freqs[old_pair] <= 0:
                        del pair_freqs[old_pair]
                
                # Add new pairs
                for i in range(len(new_word_tuple) - 1):
                    new_pair = (new_word_tuple[i], new_word_tuple[i + 1])
                    pair_freqs[new_pair] += freq
        
        word_freqs = new_word_freqs
    
    return vocab, merges


def train_bpe_parallel(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = None
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train BPE with parallel pretokenization for faster processing.
    
    This version parallelizes the pretokenization step across multiple processes.
    """
    
    # Initialize vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    
    # Add special tokens
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode('utf-8')
        next_id += 1
    
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []
    
    # Read file and split on special tokens
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if special_tokens:
        special_pattern = '|'.join(re.escape(token) for token in special_tokens)
        chunks = re.split(f'({special_pattern})', text)
        chunks = [chunk for chunk in chunks if chunk and chunk not in special_tokens]
    else:
        chunks = [text]
    
    # Parallel pretokenization
    word_freqs = {}
    
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    
    if len(chunks) > 1 and num_processes > 1:
        # Split work across processes
        chunk_size = max(1, len(chunks) // num_processes)
        chunk_groups = [chunks[i:i + chunk_size] for i in range(0, len(chunks), chunk_size)]
        
        with Pool(num_processes) as pool:
            results = pool.map(pretokenize_chunks, chunk_groups)
        
        # Merge results
        for result in results:
            for word, freq in result.items():
                word_freqs[word] = word_freqs.get(word, 0) + freq
    else:
        # Single process
        for chunk in chunks:
            for match in re.finditer(PAT, chunk):
                word = match.group()
                word_bytes = tuple(bytes([b]) for b in word.encode('utf-8'))
                word_freqs[word_bytes] = word_freqs.get(word_bytes, 0) + 1
    
    # Perform merges with incremental pair counting
    merges = []
    
    # Initial pair count
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freqs[pair] += freq
    
    for _ in range(num_merges):
        if not pair_freqs:
            break
        
        best_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)
        
        merged = best_pair[0] + best_pair[1]
        vocab[next_id] = merged
        next_id += 1
        
        # Incremental update
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            if len(word) < 2:
                new_word_freqs[word] = freq
                continue
            
            new_word = []
            i = 0
            changes = []
            
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(merged)
                    changes.append(i)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word_tuple = tuple(new_word)
            new_word_freqs[new_word_tuple] = freq
            
            if changes:
                # Remove old pairs
                for i in range(len(word) - 1):
                    old_pair = (word[i], word[i + 1])
                    pair_freqs[old_pair] -= freq
                    if pair_freqs[old_pair] <= 0:
                        del pair_freqs[old_pair]
                
                # Add new pairs
                for i in range(len(new_word_tuple) - 1):
                    new_pair = (new_word_tuple[i], new_word_tuple[i + 1])
                    pair_freqs[new_pair] += freq
        
        word_freqs = new_word_freqs
    
    return vocab, merges


def pretokenize_chunks(chunks):
    """Helper function for parallel pretokenization."""
    word_freqs = {}
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            word = match.group()
            word_bytes = tuple(bytes([b]) for b in word.encode('utf-8'))
            word_freqs[word_bytes] = word_freqs.get(word_bytes, 0) + 1
    return word_freqs