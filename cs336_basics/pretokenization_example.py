import os
import multiprocessing
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


# ## Usage
# file_path = "..."  # Replace with the actual file path
# with open(file_path, "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     def process_chunk(start_end):
#         start, end = start_end
#         with open(file_path, "rb") as f2:
#             f2.seek(start)
#             chunk = f2.read(end - start).decode("utf-8", errors="ignore")
#             # Run pre-tokenization on your chunk and return the counts
#             # For example, count pre-tokens
#             counts = {}  # Placeholder: implement your pre-tokenization logic here
#             return counts

#     pairs = list(zip(boundaries[:-1], boundaries[1:]))
#     with multiprocessing.Pool(processes=num_processes) as pool:
#         results = pool.map(process_chunk, pairs)

#     # Combine the results from all chunks
#     # For example, if counts are dictionaries, merge them
#     all_counts = {}
#     for res in results:
#         for key, val in res.items():
#             all_counts[key] = all_counts.get(key, 0) + val
