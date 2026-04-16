# Contributing to CLUES

Thank you for your interest in contributing to CLUES!

## Reporting Issues

If you find a bug or have a feature request, please open an issue on
[GitHub Issues](https://github.com/Bayer-Group/clues-clinical-nlp/issues).

When reporting bugs, please include:

- A minimal reproducible example
- Your Python version and OS
- The full error traceback

## Submitting Changes

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Ensure your code follows the existing style
5. Submit a pull request against `main`

## Development Setup

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
# Clone your fork
git clone https://github.com/<your-username>/clues-clinical-nlp.git
cd clues-clinical-nlp

# Install dependencies
uv sync

# For Jupyter notebook support
uv sync --extra dev
```

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
