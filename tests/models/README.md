# KataGo Test Models

This directory contains small neural network models for testing the katagocoreml converter. These models are for validation and testing purposes only.

## Included Models

### 1. g170e-b10c128-s1141046784-d204142634.bin.gz
- **Size**: ~11 MB
- **Architecture**: 10 blocks, 128 channels
- **Use**: Standard test model for comprehensive validation
- **Version**: Compatible with KataGo model version 8+

### 2. g170-b6c96-s175395328-d26788732.bin.gz
- **Size**: ~3.7 MB
- **Architecture**: 6 blocks, 96 channels
- **Use**: Smaller model for faster testing
- **Version**: Compatible with KataGo model version 8+

### 3. b5c192nbt-distilled.bin.gz
- **Size**: ~7.0 MB
- **Architecture**: 5 blocks, 192 channels with metadata encoder
- **Use**: Distilled human SL model (requires meta_input)
- **Version**: Tests metadata encoder functionality

## License and Attribution

### g170 Series Models (KataGo Project)

The g170 models (`g170-b6c96` and `g170e-b10c128`) are from the [KataGo project](https://github.com/lightvector/KataGo) by David J Wu ("lightvector").

**License**: CC0 (Public Domain)
- These are from the oldest KataGo training runs and are released into the public domain
- No restrictions on use

### b5c192nbt-distilled Model (Custom Training)

The b5c192nbt-distilled model is a custom-trained model included for testing purposes.

**License**: BSD-3-Clause (same as this project)
- Copyright © 2025 Chin-Chang Yang
- See project [LICENSE](../LICENSE) for full terms

### Other KataGo Models (Reference)

For reference, other KataGo models (not included in this directory) are available under an MIT-style license:
- Copyright © 2022 David J Wu ("lightvector")
- Full license text: https://katagotraining.org/network_license/

Permission is hereby granted, free of charge, to any person obtaining a copy of these neural network models and associated documentation files, to deal in the models without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the models, and to permit persons to whom the models are furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the models.

THE MODELS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE MODELS OR THE USE OR OTHER DEALINGS IN THE MODELS.

## Production Models

These are **small, weak test models** only. For actual Go gameplay, download the latest strong models from:

- **KataGo Releases**: https://github.com/lightvector/KataGo/releases
- **KataGo CDN**: https://d3dndmfyhecmj0.cloudfront.net/
- **Training Networks**: https://katagotraining.org/networks/

## Model Version Compatibility

The katagocoreml converter supports KataGo model versions 8-16:

| Version | Policy Channels | Score Value Channels | Features |
|---------|-----------------|----------------------|----------|
| 8       | 1               | 4                    | Basic    |
| 9-11    | 1               | 6                    | Extended score values |
| 12-15   | 2               | 6                    | Multi-channel policy |
| 15+     | 2               | 6                    | + SGF metadata encoder |
| 16      | 4               | 6                    | Enhanced policy |

## Usage in Tests

These models are referenced in `tests/conftest.py` as pytest fixtures:
- `standard_model_bin` → g170e-b10c128 model
- `smaller_model_bin` → g170-b6c96 model
- `distilled_model_bin` → b5c192nbt-distilled model

Run integration tests with:
```bash
pytest tests/test_cpp_vs_python.py -v
```

**Note**: C++ unit tests (`ctest`) do not require these model files.
