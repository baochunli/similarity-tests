# Evaluating the degree of similarity between two strings

## Setting up the Python environment using `uv`

First, we need to install `uv`:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

And make sure that `$HOME/.cargo/bin` is in the `$PATH` by revising `~/.zshrc` accordingly:

```sh
export PATH=$HOME/.cargo/bin:$PATH
```

Later on, whenever one needs to update the version of `uv`, the following command can be used:

```sh
uv self update
```

Optionally, one can pin to a particular Python version:

```sh
uv python pin 3.12
```

## Running the tests

```sh
uv run similarity.py
```
