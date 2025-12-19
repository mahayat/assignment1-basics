import regex as re
from collections import defaultdict
from multiprocessing import cpu_count

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = cpu_count()
SPLIT_TOKEN = "<|endoftext|>" 

def get_chunks(input_path):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, SPLIT_TOKEN.encode())
        # print(boundaries)
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            return chunk # temporary
        
def get_stories(chunk, special_tokens):
    pattern = "|".join(re.escape(token) for token in special_tokens)
    return re.split(pattern, chunk) # get list of stories
        
def get_pretokens(stories):
    return [token for story in stories for token in re.findall(PAT, story)]

def get_pretoken_bytes(pretokens):
    return [list(pretoken.encode('utf-8')) for pretoken in pretokens]

def get_pair_counts(pretoken_bytes):
    pair_counts = defaultdict(int)

    for pretoken_byte in pretoken_bytes:
        byte_len = len(pretoken_byte)
        for i in range(byte_len-1):
            pair_counts[(pretoken_byte[i], pretoken_byte[i+1])] += 1
    return pair_counts

def apply_merge(pretoken_bytes, pair, new_token_id):
    byte1, byte2 = pair

    new_pretoken_bytes = []

    for token in pretoken_bytes:
        new_token_list = []
        i = 0
        while i < len(token):
            # Check if we can merge at position i
            if i < len(token) - 1 and token[i] == byte1 and token[i + 1] == byte2:
                # Merge by adding the new token ID
                new_token_list.append(new_token_id)
                i += 2
            else:
                new_token_list.append(token[i])
                i += 1
        new_pretoken_bytes.append(new_token_list)
    
    return new_pretoken_bytes

def train_bpe(input_path, vocab_size, special_tokens):
    chunk = get_chunks(input_path)
    pretoken_bytes = get_pretoken_bytes(get_pretokens(get_stories(chunk, special_tokens)))

    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256

    for special_token in special_tokens:
        vocab[next_id] = special_token.encode('utf-8')
        next_id += 1 

    num_merges = vocab_size - len(vocab)
    merges = []   

    for merge_num in range(num_merges):
        
        pair_counts = get_pair_counts(pretoken_bytes)
        if not pair_counts:
            break 
        
        best_tuple = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        most_frequent_pair, count = best_tuple
        

        byte1, byte2 = most_frequent_pair
        new_token = vocab[byte1] + vocab[byte2]

        print(f'merge_num: {merge_num}, most_frequent_pair: {most_frequent_pair}, new_token: {new_token}')

        vocab[next_id] = new_token
        merges.append(most_frequent_pair)
        pretoken_bytes = apply_merge(pretoken_bytes, most_frequent_pair, next_id)
        next_id += 1
    return vocab, merges