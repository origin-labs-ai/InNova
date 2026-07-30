# Contributing

## Getting Started

```bash
# Fork & clone
git clone https://github.com/xprimesamx/InNova
cd InNova

# Build for development
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure
```

## Code Style

- **Language**: C++20
- **Formatting**: Follow existing patterns (spaces, indentation)
- **Headers**: Document public API in header files
- **Names**: Use snake_case for functions, PascalCase for classes, UPPER for enums

## Development Workflow

1. Create a feature branch from `main`
2. Make your changes
3. Ensure all tests pass: `ctest --test-dir build --output-on-failure`
4. Submit a Pull Request

## Build Checks

```bash
# Debug build (for dev)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Release build (for benchmarks)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Tests must pass in both
ctest --test-dir build --output-on-failure
```

## Testing Guidelines

- Add tests for new functionality
- Tests go in `tests/` directory
- Use `test_ALL` naming convention
- Run `ctest -R test_your_feature -V` for specific tests

## Documentation

- Update relevant docs when changing functionality
- Docs live in `docs/` and `wiki/`
- Research notes go in `.research/`
- See [File Documentation Index](files/_index) for per-file docs

## License

This project is licensed under the terms in the repository license file.

See [docs/CONTRIBUTING.md](file:///c:/Users/thaku/Downloads/InNova/docs/CONTRIBUTING.md) for the full contributing guide.
