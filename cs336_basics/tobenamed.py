import regex as re
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from cs336_basics.pretokenization_example import find_chunk_boundaries

class TrainBPE:
    def __init__(self, input_path, vocab_size, special_tokens):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.NUM_PROCESSES = cpu_count()
        self.SPLIT_TOKEN = "<|endoftext|>" 
    
    @property
    def boundaries(self):
        with open(self.input_path, "rb") as f:
            return find_chunk_boundaries(f, self.NUM_PROCESSES, self.SPLIT_TOKEN.encode())

    def get_stories(self, chunk, special_tokens):
        pattern = "|".join(re.escape(token) for token in special_tokens)
        return re.split(pattern, chunk) # get list of stories
            
    def get_pretokens(self, stories):
        return [token for story in stories for token in re.findall(self.PAT, story)]

    def get_pretoken_bytes(self, pretokens):
        return [pretoken.encode('utf-8') for pretoken in pretokens]

    def pretoken_chunk_single(self, start_end_pair):
        start, end = start_end_pair 
        with open(self.input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return self.get_pretoken_bytes(self.get_pretokens(self.get_stories(chunk, self.special_tokens)))
        
    def pretoken_chunk_parallal(self):
        start_end_pairs = list(zip(self.boundaries[:-1], self.boundaries[1:]))
        with Pool(processes=self.NUM_PROCESSES) as pool:
            all_pretoken_bytes = pool.map(self.pretoken_chunk_single, start_end_pairs)
        flattened_pretoken_bytes = [pretoken for chunk in all_pretoken_bytes for pretoken in chunk]
        return flattened_pretoken_bytes

    def get_pair_counts(self, pretoken_bytes):
        pair_counts = defaultdict(int)
        for pretoken_byte in pretoken_bytes:
            byte_len = len(pretoken_byte)
            for i in range(byte_len-1):
                pair_counts[(pretoken_byte[i], pretoken_byte[i+1])] += 1
                # note: pretoken_byte[i] is integer, not bytes
        return pair_counts

    def apply_merge(self, pretoken_bytes, pair, new_token_id):
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

    def train_bpe(self):
        pretoken_bytes = self.pretoken_chunk_parallal()

        vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        for special_token in self.special_tokens:
            vocab[next_id] = special_token.encode('utf-8')
            next_id += 1 

        num_merges = self.vocab_size - len(vocab)
        merges = []   

        for merge_num in range(num_merges):
            pair_counts = self.get_pair_counts(pretoken_bytes)
            if not pair_counts:
                break 
            
            best_tuple = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
            most_frequent_pair, count = best_tuple
            

            byte1, byte2 = most_frequent_pair
            new_token = vocab[byte1] + vocab[byte2]

            print(f'merge_num: {merge_num}, most_frequent_pair: {most_frequent_pair}, new_token: {new_token}')

            vocab[next_id] = new_token
            merges.append((vocab[byte1], vocab[byte2]))
            pretoken_bytes = self.apply_merge(pretoken_bytes, most_frequent_pair, next_id)
            next_id += 1
        return vocab, merges