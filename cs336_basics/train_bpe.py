import regex as re
from collections import defaultdict

from .pretokenization_example import find_chunk_boundaries

# split_special_token: splits the big text to chunks ---> each chunk goes to a process, each chunk can have multiple stories
# special_tokens: making sure we do not split beyond special_tokens ---> 

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_chunks(input_path: str, split_special_token: str, num_processes: int) -> str:
    """
    Creates chunks from input file. Each chunk is a string.
    """
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token.encode())
        print(boundaries)
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            return chunk # temporary
        
def split_chunk(chunk_text: str, special_tokens: str) -> list[str]:
    """
    Split the chunks to stories based on the special token. Each story is a string. 
    Tokenization should happen on each story.
    """
    pattern = "|".join(re.escape(token) for token in special_tokens)
    return re.split(pattern, chunk_text) # get list of stories
        

def get_all_pretokens(input_path: str,
              vocab_size: int,
              special_tokens: list[str] = ["<|endoftext|>"],
              num_processes: int = 20, # optional
              split_special_token: str = "<|endoftext|>", # optional
              ):
    
    chunk = get_chunks(input_path, split_special_token, num_processes)
    stories = split_chunk(chunk, special_tokens)
    all_pretokens = [token for story in stories for token in re.findall(PAT, story)]
    return all_pretokens


def get_pretoken_counter(all_pretokens):
    freq = defaultdict(int)
    for pretoken in all_pretokens:
        byte_tuple = tuple(bytes([b]) for b in pretoken.encode('utf-8'))
        freq[byte_tuple] += 1
    return dict(freq)

def train_bpe(input_path, vocab_size, special_tokens):
    pass