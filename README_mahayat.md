### Getting Started
- `git clone git@github.com:stanford-cs336/assignment1-basics.git`
- `cd ./assignment1-basics/`
- `uv sync` #failed
- `uv python pin 3.13`
- `cat .python-version`
- `uv sync`
- `uv run python -V` #Python 3.13.11
- `uv run pytest`
- `uv add --dev ipykernel`

### Tests
- `uv run pytest ./tests/test_train_bpe.py`