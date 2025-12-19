from collections import defaultdict
from typing import Dict, List, Tuple

def pretokenize(text):
    words = text.split() # split on any whitespace
    freq = defaultdict(int)
    
    for word in words:
        byte_tuple = tuple(bytes([b]) for b in word.encode('utf-8'))
        freq[byte_tuple] += 1
    
    return dict(freq)

def count_pairs(word_freqs):
    """
    For each word in word_freqs, get the pair counts.
    
    :param word_freqs: Description
    """
    pair_counts = defaultdict(int)
    
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
    
    return dict(pair_counts)

def get_best_pair(pairs):
    best_pair = max(pairs.items(), key=lambda x: (x[1], x[0]))
    pair, count = best_pair
    return pair

def merge_pair(word_freqs, pair):
    new_word_freqs = {}
    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq
    return new_word_freqs

def train_bpe(text: str, num_merges: int) -> List[Tuple[bytes, bytes]]:
    """Train BPE and return the sequence of merges."""
    # Ensure text is a string
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}. Did you pass a file object? Use f.read() first.")
    
    word_freqs = pretokenize(text)
    merges = []
    
    # print("Initial word frequencies:")
    # for word, freq in word_freqs.items():
    #     word_str = ''.join(b.decode('utf-8') for b in word)
    #     print(f"  {word_str}: {freq}")
    # print()
    
    for merge_num in range(num_merges):
        pair_counts = count_pairs(word_freqs)
        if not pair_counts:
            break
        
        # Find the most frequent pair (with lexicographic tiebreaker)
        # best_pair = max(pair_counts.items(), 
        #                key=lambda x: (x[1], x[0]))  # Sort by count, then lexicographically
        # pair, count = best_pair
        pair = get_best_pair(pair_counts)
        
        # Format pair for display
        # pair_str = ' '.join(b.decode('utf-8') for b in pair)
        
        # print(f"Round {merge_num + 1}:")
        # print(f"  Most frequent pair: '{pair_str}' (count: {count})")
        
        merges.append(pair)
        word_freqs = merge_pair(word_freqs, pair)
        
        # print(f"  After merge:")
        # for word, freq in word_freqs.items():
        #     word_str = '|'.join(b.decode('utf-8') for b in word)
        #     print(f"    {word_str}: {freq}")
        # print()
    
    return merges

# def tokenize(word: str, merges: List[Tuple[bytes, bytes]]) -> List[str]:
#     """Tokenize a word using the trained BPE merges."""
#     # Start with individual bytes as bytes objects
#     tokens = [bytes([b]) for b in word.encode('utf-8')]
    
#     # Apply merges in order
#     for pair in merges:
#         new_tokens = []
#         i = 0
#         while i < len(tokens):
#             # Check if we can apply this merge
#             if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
#                 new_tokens.append(pair[0] + pair[1])
#                 i += 2
#             else:
#                 new_tokens.append(tokens[i])
#                 i += 1
#         tokens = new_tokens
    
#     # Convert to strings for display
#     return [token.decode('utf-8') for token in tokens]

# def format_merges(merges: List[Tuple[bytes, bytes]]) -> List[str]:
#     """Format merges as readable strings."""
#     return [f"{a.decode('utf-8')} {b.decode('utf-8')}" for a, b in merges]

# # Example from the problem
# corpus = """low low low low low
# lower lower widest widest widest
# newest newest newest newest newest newest"""

# print("=" * 60)
# print("BPE Training Example")
# print("=" * 60)
# print()

# # Train BPE with 6 merges
# merges = train_bpe(corpus, num_merges=6)

# print("=" * 60)
# print("Final Results")
# print("=" * 60)
# print()
# print("Merge sequence:")
# formatted_merges = format_merges(merges)
# for i, merge_str in enumerate(formatted_merges, 1):
#     print(f"  {i}. '{merge_str}'")

# print()
# print("Vocabulary additions:")
# vocab_additions = [m.replace(' ', '') for m in formatted_merges]
# print(f"  {vocab_additions}")

# print()
# print("Tokenizing 'newest' with the trained BPE:")
# tokens = tokenize("newest", merges)
# print(f"  Result: {tokens}")