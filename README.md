# katagocoreml

**Standalone C++ library for converting KataGo neural network models to Apple Core ML format**

Pure C++ implementation with no Python dependency - designed for embedding in native applications.

## Installation

### Homebrew (Recommended)

```bash
brew tap chinchangyang/katagocoreml-cpp
brew install katagocoreml
```

### Build from Source

**Prerequisites:**
- macOS 10.15+ (for Core ML framework)
- C++17 compiler (Clang 10+ or GCC 9+)
- CMake 3.14+

**Install dependencies:**
```bash
brew install cmake protobuf abseil zlib
```

**Build:**
```bash
git clone https://github.com/chinchangyang/katagocoreml-cpp.git
cd katagocoreml-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

**Install:**
```bash
sudo make install
```

## Quick Start

### CLI Tool

```bash
# Show model info
katago2coreml --info model.bin.gz

# Convert to Core ML (19x19 board, FLOAT32)
katago2coreml model.bin.gz KataGo.mlpackage

# Convert 9x9 model with optimizations
katago2coreml -x 9 -y 9 --optimize-identity-mask model.bin.gz KataGo-9x9.mlpackage

# Use FLOAT16 for smaller model
katago2coreml --float16 model.bin.gz KataGo-fp16.mlpackage
```

### C++ API

```cpp
#include <katagocoreml/KataGoConverter.hpp>

int main() {
    using namespace katagocoreml;

    // Simple conversion
    KataGoConverter::convert("model.bin.gz", "KataGo.mlpackage");

    // With options
    ConversionOptions opts;
    opts.board_x_size = 19;
    opts.board_y_size = 19;
    opts.optimize_identity_mask = true;  // ~6.5% speedup
    opts.compute_precision = "FLOAT16";

    KataGoConverter::convert("model.bin.gz", "output.mlpackage", opts);

    return 0;
}
```

**Compile:**
```bash
clang++ -std=c++17 $(pkg-config --cflags katagocoreml) example.cpp \
    $(pkg-config --libs katagocoreml) -o example
```

## CLI Options

```
Usage: katago2coreml [options] <input.bin[.gz]> <output.mlpackage>

Options:
  -x, --board-x <size>     Board width (default: 19)
  -y, --board-y <size>     Board height (default: 19)
  --optimize-identity-mask Optimize for full board (~6.5% faster inference)
  --float16                Use FLOAT16 compute precision (smaller model)
  --info                   Show model info and exit (no conversion)
  -v, --verbose            Enable verbose output
  -h, --help               Show this help message

Supported KataGo model versions: 8-16
```

## API Reference

### ConversionOptions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `board_x_size` | int | 19 | Board width |
| `board_y_size` | int | 19 | Board height |
| `optimize_identity_mask` | bool | false | Skip mask ops (~6.5% speedup) |
| `compute_precision` | string | "FLOAT32" | "FLOAT32" or "FLOAT16" |
| `specification_version` | int | 6 | Core ML spec (6 = iOS 15+) |

### ModelInfo

```cpp
auto info = KataGoConverter::getModelInfo("model.bin.gz");
// info.name, info.version, info.num_blocks, info.trunk_channels, etc.
```

## KataGo Version Support

| Version | Policy Channels | Score Value Channels | Status |
|---------|-----------------|----------------------|--------|
| 8 | 1 | 4 | Supported |
| 9-11 | 1 | 6 | Supported |
| 12-15 | 2 | 6 | Supported |
| 16 | 4 | 6 | Supported |

## Build Options

```bash
# Release build (recommended)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Disable tests (faster build)
cmake -DKATAGOCOREML_BUILD_TESTS=OFF ..

# Library only (no CLI tool)
cmake -DKATAGOCOREML_BUILD_TOOLS=OFF ..
```

## Troubleshooting

**Missing protobuf:**
```bash
brew install protobuf
```

**Missing abseil:**
```bash
brew install abseil
```

**Clean build (if issues occur):**
```bash
rm -rf CMakeCache.txt CMakeFiles _deps
cmake .. && make
```

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.

This project includes third-party components (MILBlob, ModelPackage, nlohmann/json, FP16) under their respective licenses. See [NOTICE](NOTICE) for details.

## Credits

- [KataGo](https://github.com/lightvector/KataGo) - Go engine and model format
- [coremltools](https://github.com/apple/coremltools) - Python converter (source of this port)
- [Core ML](https://developer.apple.com/documentation/coreml) - Apple's ML framework
