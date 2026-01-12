# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

katagocoreml is a standalone C++ library for converting KataGo neural network models to Apple Core ML format. It is a pure C++ implementation with no Python dependency, designed for embedding in native applications.

## Build Commands

```bash
# Prerequisites (install once)
brew install cmake protobuf abseil zlib

# Configure and build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)

# Build options
cmake -DKATAGOCOREML_BUILD_TESTS=OFF ..   # Skip tests
cmake -DKATAGOCOREML_BUILD_TOOLS=OFF ..   # Library only

# Run tests
cd build && ctest --output-on-failure

# Clean rebuild (if protobuf issues)
rm -rf CMakeCache.txt CMakeFiles _deps && cmake .. && make
```

## Architecture

The converter follows a three-stage pipeline:

1. **Parser** (`src/parser/KataGoParser.cpp`) - Reads KataGo .bin/.bin.gz files and produces `KataGoModelDesc` (defined in `src/types/KataGoTypes.hpp`). Handles gzip decompression, binary parsing, and weight extraction.

2. **MIL Builder** (`src/builder/MILBuilder.cpp`, `Operations.cpp`) - Converts `KataGoModelDesc` to Apple's MIL (Model Intermediate Language) protobuf format. Builds the neural network graph including trunk, policy head, and value head components.

3. **Serializer** (`src/serializer/CoreMLSerializer.cpp`, `WeightSerializer.cpp`) - Writes the MIL program and weights to .mlpackage format using vendored coremltools components (MILBlob, ModelPackage).

### Key Types

- `KataGoModelDesc` (`src/types/KataGoTypes.hpp`) - Complete model descriptor including trunk, policy head, value head, and layer descriptors
- `ConversionOptions` / `ModelInfo` (`include/katagocoreml/Options.hpp`) - Public API types
- `KataGoConverter` (`include/katagocoreml/KataGoConverter.hpp`) - Main public interface

### Vendored Dependencies

Located in `vendor/`:
- `mlmodel/format/` - Core ML protobuf definitions (compiled during build)
- `mlmodel/src/MILBlob/` - Binary weight serialization from coremltools
- `modelpackage/src/` - .mlpackage directory structure handling
- `deps/nlohmann/` - JSON library
- `deps/FP16/` - FP16 conversion utilities

## KataGo Model Versions

Supports versions 8-16. Key differences:
- v8: 1 policy channel, 4 score value channels
- v9-11: 1 policy channel, 6 score value channels
- v12-15: 2 policy channels, 6 score value channels
- v15+: Adds SGF metadata encoder
- v16: 4 policy channels

## Testing

Tests use Google Test (fetched via FetchContent):
- `tests/ParserTests.cpp` - KataGo binary parsing tests
- `tests/TypesTests.cpp` - Type system tests
