#  Copyright (c) 2025, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Integration tests to verify C++ katagocoreml library generates inference-equivalent
Core ML models compared to the Python coremltools/converters/katago converter.

The Python converter has been validated against KataGo's Eigen backend and serves
as the reference implementation. These tests ensure the C++ port produces models
with equivalent inference results within acceptable numerical tolerance.
"""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import pytest

if TYPE_CHECKING:
    import numpy as np

# Random seed base for deterministic test input generation
RANDOM_SEED_BASE = 42


# ==============================================================================
# Converter Wrappers
# ==============================================================================


def convert_with_cpp(
    exe_path: Path,
    model_path: Path,
    output_path: Path,
    board_size: int,
    optimize_mask: bool,
    float16: bool = False,
    float16_io: bool = False,
    min_batch_size: int = 1,
    max_batch_size: int = 1,
) -> None:
    """Convert KataGo model using C++ CLI tool.

    Args:
        exe_path: Path to katago2coreml executable
        model_path: Path to input .bin.gz model
        output_path: Path for output .mlpackage
        board_size: Board size (e.g., 9, 13, 19)
        optimize_mask: Enable optimize_identity_mask flag
        float16: Enable FLOAT16 precision
        float16_io: Enable FLOAT16 for inputs/outputs (requires float16=True)
        min_batch_size: Minimum batch size (default: 1)
        max_batch_size: Maximum batch size (default: 1, -1 for unlimited)

    Raises:
        subprocess.CalledProcessError: If conversion fails
    """
    cmd = [
        str(exe_path),
        "-x", str(board_size),
        "-y", str(board_size),
    ]

    if optimize_mask:
        cmd.append("--optimize-identity-mask")

    if float16:
        cmd.append("--float16")

    if float16_io:
        cmd.append("--float16-io")

    if min_batch_size != 1 or max_batch_size != 1:
        cmd.extend(["--dynamic-batch", f"{min_batch_size},{max_batch_size}"])

    cmd.extend([str(model_path), str(output_path)])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )


def convert_with_python(
    model_path: Path,
    output_path: Path,
    board_size: int,
    optimize_mask: bool,
    float16: bool = False,
) -> None:
    """Convert KataGo model using Python coremltools converter.

    Args:
        model_path: Path to input .bin.gz model
        output_path: Path for output .mlpackage
        board_size: Board size (e.g., 9, 13, 19)
        optimize_mask: Enable optimize_identity_mask flag
        float16: Enable FLOAT16 precision
    """
    import coremltools as ct
    from coremltools.converters.katago import convert

    precision = ct.precision.FLOAT16 if float16 else ct.precision.FLOAT32

    mlmodel = convert(
        str(model_path),
        board_x_size=board_size,
        board_y_size=board_size,
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=precision,
        optimize_identity_mask=optimize_mask,
    )

    mlmodel.save(str(output_path))


def create_batched_inputs(
    batch_size: int,
    board_size: int,
    has_meta_input: bool = False,
    start_idx: int = 0
) -> "Dict[str, np.ndarray]":
    """Create deterministic batched inputs for reproducibility.

    Each sample is generated with a consistent seed (42 + start_idx + sample_index) so that
    sample[i] is identical regardless of batch size. This enables per-sample
    consistency validation across different batch configurations.

    Args:
        batch_size: Number of samples in the batch (must be positive)
        board_size: Board dimension (e.g., 9, 13, 19)
        has_meta_input: Whether to include meta_input for metadata encoder
        start_idx: Starting index for seed calculation (default: 0)

    Returns:
        Dict with input arrays (spatial_input, global_input, input_mask, optionally meta_input)

    Raises:
        ValueError: If batch_size is not positive or start_idx is negative
    """
    import numpy as np

    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if start_idx < 0:
        raise ValueError(f"start_idx must be non-negative, got {start_idx}")

    spatial_list = []
    global_list = []
    meta_list = []

    for i in range(batch_size):
        # Use modulo to prevent integer overflow in seed calculation
        np.random.seed((RANDOM_SEED_BASE + start_idx + i) % (2**32))  # Consistent seed per sample index
        spatial_list.append(np.random.randn(1, 22, board_size, board_size))
        global_list.append(np.random.randn(1, 19))
        if has_meta_input:
            meta_list.append(np.zeros((1, 192)))

    inputs = {
        "spatial_input": np.concatenate(spatial_list, axis=0).astype(np.float32),
        "global_input": np.concatenate(global_list, axis=0).astype(np.float32),
        "input_mask": np.ones((batch_size, 1, board_size, board_size), dtype=np.float32),
    }

    if has_meta_input:
        inputs["meta_input"] = np.concatenate(meta_list, axis=0).astype(np.float32)

    return inputs


# ==============================================================================
# Test Classes
# ==============================================================================


class TestCppVsPythonConverterFP32:
    """Tests for C++ FP32 converter producing inference-equivalent models.

    Since binary equivalence between C++ and Python converters is difficult to achieve
    (due to metadata differences, protobuf field ordering, and operation serialization),
    we instead validate that both converters produce models with equivalent inference results.

    This approach validates what actually matters: that the models produce the same outputs
    for the same inputs within acceptable numerical tolerance.
    """

    @pytest.mark.parametrize(
        "model_name",
        [
            "g170e-b10c128-s1141046784-d204142634.bin.gz",  # Standard model
            "g170-b6c96-s175395328-d26788732.bin.gz",       # Smaller model
            "b5c192nbt-distilled.bin.gz",                   # Distilled model (human SL with metadata)
        ],
    )
    @pytest.mark.parametrize("board_size", [9, 13, 19])
    @pytest.mark.parametrize("optimize_mask", [False, True])
    def test_fp32_inference_equivalent(
        self,
        model_name: str,
        board_size: int,
        optimize_mask: bool,
        katago2coreml_exe: Path,
        all_test_models: dict,
        temp_output_dir: Path,
    ):
        """Test that C++ and Python FP32 models produce equivalent inference results.

        This test:
        1. Converts a KataGo model using both C++ and Python converters (FP32)
        2. Loads both models using coremltools
        3. Generates deterministic random inputs
        4. Runs inference on both models
        5. Compares outputs with FP32 tolerance

        Args:
            model_name: Name of the KataGo model file
            board_size: Board size (9, 13, or 19)
            optimize_mask: Whether to enable optimize_identity_mask
            katago2coreml_exe: Path to C++ CLI tool (fixture)
            all_test_models: Dict mapping model names to paths (fixture)
            temp_output_dir: Temporary directory for outputs (fixture)
        """
        import platform
        if platform.machine() != "arm64":
            pytest.skip("Core ML inference only available on Apple Silicon")

        import numpy as np

        # Skip if model not available
        if model_name not in all_test_models:
            pytest.skip(f"Model not available: {model_name}")

        model_path = all_test_models[model_name]

        # Generate unique output names based on configuration
        mask_suffix = "mask_true" if optimize_mask else "mask_false"
        cpp_output = temp_output_dir / f"cpp_fp32_{board_size}x{board_size}_{mask_suffix}.mlpackage"
        python_output = temp_output_dir / f"python_fp32_{board_size}x{board_size}_{mask_suffix}.mlpackage"

        # Convert with both converters
        convert_with_cpp(
            katago2coreml_exe,
            model_path,
            cpp_output,
            board_size,
            optimize_mask,
            float16=False,
        )

        convert_with_python(
            model_path,
            python_output,
            board_size,
            optimize_mask,
            float16=False,
        )

        # Load models
        import coremltools as ct
        cpp_model = ct.models.MLModel(str(cpp_output))
        python_model = ct.models.MLModel(str(python_output))

        # Generate deterministic random input
        np.random.seed(RANDOM_SEED_BASE)
        spatial_input = np.random.randn(1, 22, board_size, board_size).astype(np.float32)
        global_input = np.random.randn(1, 19).astype(np.float32)
        input_mask = np.ones((1, 1, board_size, board_size), dtype=np.float32)

        inputs = {
            "spatial_input": spatial_input,
            "global_input": global_input,
            "input_mask": input_mask,
        }

        # Add meta_input for human SL networks (models with metadata encoder)
        cpp_spec = cpp_model.get_spec()
        cpp_input_names = [inp.name for inp in cpp_spec.description.input]
        if "meta_input" in cpp_input_names:
            inputs["meta_input"] = np.zeros((1, 192), dtype=np.float32)

        # Run inference
        cpp_outputs = cpp_model.predict(inputs)
        python_outputs = python_model.predict(inputs)

        # Compare outputs with relative tolerance (matching cross-validation approach)
        # Use same tolerance strategy as validation_utils.py for consistency
        # Map output names to tolerance categories
        output_tolerances = {
            "policy_p2_conv": 0.015,       # policy: 1.5% relative
            "policy_pass_mul2": 0.015,     # pass_policy
            "policy_pass": 0.015,          # pass_policy (version-dependent name)
            "value_v3_bias": 0.01,         # value: 1% relative
            "value_ownership_conv": 0.03,  # ownership: 3% relative
            "value_sv3_bias": 0.015,       # score_value: 1.5% relative
        }
        default_tolerance = 0.03  # 3% for unrecognized outputs

        failed_outputs = []
        for key in python_outputs.keys():
            if key not in cpp_outputs:
                failed_outputs.append(f"Missing output key in C++ model: {key}")
                continue

            cpp_val = cpp_outputs[key]
            py_val = python_outputs[key]

            # Compute relative error: max_diff / max(abs(reference))
            max_diff = np.max(np.abs(cpp_val - py_val))
            max_ref = np.max(np.abs(py_val))
            if max_ref > 1e-8:
                rel_error = max_diff / max_ref
            else:
                rel_error = max_diff  # Absolute if reference is near zero

            tolerance = output_tolerances.get(key, default_tolerance)
            if rel_error > tolerance:
                failed_outputs.append(
                    f"Output '{key}': rel_error={rel_error:.4f} > tolerance={tolerance}"
                )

        if failed_outputs:
            pytest.fail("\n".join(failed_outputs))


class TestConverterSmoke:
    """Smoke tests for individual converter functionality."""

    def test_cpp_converter_help(self, katago2coreml_exe: Path):
        """Test that C++ CLI tool shows help without error."""
        result = subprocess.run(
            [str(katago2coreml_exe), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "katago2coreml" in result.stdout or "Usage" in result.stdout

    def test_cpp_converter_model_info(
        self,
        katago2coreml_exe: Path,
        standard_model_bin: Path,
    ):
        """Test that C++ CLI can read model info."""
        result = subprocess.run(
            [str(katago2coreml_exe), "--info", str(standard_model_bin)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Should output model information
        assert "version" in result.stdout.lower() or "blocks" in result.stdout.lower()

    def test_python_converter_import(self):
        """Test that Python converter can be imported."""
        try:
            from coremltools.converters.katago import convert
            assert callable(convert)
        except ImportError as e:
            pytest.skip(f"coremltools KataGo converter not available: {e}")


class TestCppDynamicBatch:
    """Tests for C++ converter dynamic batch functionality.

    Validates that dynamic batch models produce numerically equivalent outputs
    to fixed batch models. Uses intra-implementation validation (C++ vs C++).

    Test strategy:
    - Compare dynamic batch models against fixed (1,1) baseline
    - Test each dynamic model at batch=4
    - Run fixed (1,1) model 4 times independently for comparison
    - Validate per-sample numerical equivalence

    Config: smaller model (g170-b6c96), 9x9 board, FP32, optimize_mask=False
    """

    @pytest.mark.parametrize(
        "batch_config",
        [
            (1, 4),   # Small dynamic
            (1, 8),   # Medium dynamic
            (1, -1),  # Unlimited
        ],
        ids=["dynamic_1-4", "dynamic_1-8", "dynamic_unlimited"]
    )
    def test_dynamic_batch_equivalence(
        self,
        batch_config: tuple,
        katago2coreml_exe: Path,
        smaller_model_bin: Path,
        temp_output_dir: Path,
    ):
        """Test dynamic batch models produce equivalent outputs to fixed baseline.

        For each dynamic batch configuration:
        1. Convert model with dynamic batch settings
        2. Convert baseline model with fixed batch (1,1)
        3. Generate deterministic batched inputs (batch=4)
        4. Run inference on dynamic model (batch=4)
        5. Run inference on baseline model 4 times (batch=1 each)
        6. Compare outputs sample-by-sample

        Args:
            batch_config: Tuple of (min_batch_size, max_batch_size)
            katago2coreml_exe: Path to C++ CLI tool (fixture)
            smaller_model_bin: Path to smaller test model (fixture)
            temp_output_dir: Temporary directory for outputs (fixture)
        """
        import platform
        if platform.machine() != "arm64":
            pytest.skip("Core ML inference only available on Apple Silicon")

        import numpy as np
        import coremltools as ct

        min_batch, max_batch = batch_config
        board_size = 9
        test_batch_size = 4

        # Convert dynamic batch model
        batch_label = f"{max_batch}" if max_batch > 0 else "unlimited"
        dynamic_output = temp_output_dir / f"dynamic_1-{batch_label}.mlpackage"

        convert_with_cpp(
            katago2coreml_exe,
            smaller_model_bin,
            dynamic_output,
            board_size,
            optimize_mask=False,
            float16=False,
            min_batch_size=min_batch,
            max_batch_size=max_batch,
        )

        # Convert baseline fixed batch model
        baseline_output = temp_output_dir / "fixed_1-1.mlpackage"
        convert_with_cpp(
            katago2coreml_exe,
            smaller_model_bin,
            baseline_output,
            board_size,
            optimize_mask=False,
            float16=False,
            min_batch_size=1,
            max_batch_size=1,
        )

        # Load models
        dynamic_model = ct.models.MLModel(str(dynamic_output))
        baseline_model = ct.models.MLModel(str(baseline_output))

        # Check for meta_input
        spec = dynamic_model.get_spec()
        has_meta_input = "meta_input" in [inp.name for inp in spec.description.input]

        # Run dynamic model with batch=4
        dynamic_inputs = create_batched_inputs(test_batch_size, board_size, has_meta_input)
        dynamic_outputs = dynamic_model.predict(dynamic_inputs)

        # Validate output shapes have correct batch dimension
        for key in dynamic_outputs.keys():
            output_shape = dynamic_outputs[key].shape
            if len(output_shape) == 0 or output_shape[0] != test_batch_size:
                pytest.fail(
                    f"Output '{key}' has unexpected shape {output_shape}. "
                    f"Expected batch dimension (first dimension) to be {test_batch_size}"
                )

        # Run baseline model 4 times independently
        baseline_outputs_list = []
        for i in range(test_batch_size):
            single_input = create_batched_inputs(1, board_size, has_meta_input, start_idx=i)
            baseline_out = baseline_model.predict(single_input)
            baseline_outputs_list.append(baseline_out)

        # Compare per-sample outputs
        output_tolerances = {
            "policy_p2_conv": 0.015,
            "policy_pass_mul2": 0.015,
            "policy_pass": 0.015,
            "value_v3_bias": 0.01,
            "value_ownership_conv": 0.03,
            "value_sv3_bias": 0.015,
        }
        default_tolerance = 0.03

        failed_outputs = []
        for sample_idx in range(test_batch_size):
            for key in dynamic_outputs.keys():
                if key not in baseline_outputs_list[sample_idx]:
                    failed_outputs.append(f"Sample {sample_idx}: Missing key '{key}'")
                    continue

                dynamic_val = dynamic_outputs[key][sample_idx:sample_idx+1]
                baseline_val = baseline_outputs_list[sample_idx][key]

                max_diff = np.max(np.abs(dynamic_val - baseline_val))
                max_ref = np.max(np.abs(baseline_val))
                rel_error = max_diff / max_ref if max_ref > 1e-8 else max_diff

                tolerance = output_tolerances.get(key, default_tolerance)
                if rel_error > tolerance:
                    failed_outputs.append(
                        f"Sample {sample_idx}, '{key}': error={rel_error:.4f} > {tolerance} "
                        f"(max_diff={max_diff:.6f}, max_ref={max_ref:.6f})"
                    )

        if failed_outputs:
            pytest.fail(f"Dynamic batch {batch_config} failed:\n" + "\n".join(failed_outputs))


class TestCppVsPythonConverterFP16:
    """Tests for C++ FLOAT16 converter functionality.

    The C++ converter now implements the same approach as Python for FLOAT16:
    1. Model inputs remain FLOAT32
    2. Cast operations convert inputs to FLOAT16 for internal processing
    3. All intermediate operations use FLOAT16 weights and activations
    4. Cast operations convert outputs back to FLOAT32
    5. Weights are stored as FLOAT16 in the blob file

    This matches Python's `add_fp16_cast` pass behavior, producing models
    that Core ML can compile and execute with FP16 precision.
    """

    @pytest.mark.parametrize(
        "model_name",
        [
            "g170-b6c96-s175395328-d26788732.bin.gz",       # Smaller model
            "b5c192nbt-distilled.bin.gz",                   # Distilled model (human SL with metadata)
        ],
    )
    @pytest.mark.parametrize("board_size", [9, 19])
    def test_fp16_inference_equivalent(
        self,
        model_name: str,
        board_size: int,
        katago2coreml_exe: Path,
        all_test_models: dict,
        temp_output_dir: Path,
    ):
        """Test that C++ and Python FLOAT16 models produce equivalent inference results.

        This test verifies that the C++ converter with cast operations at input/output
        boundaries produces models that Core ML can compile and that produce inference
        results equivalent to the Python converter within FP16 tolerance.

        Args:
            model_name: Name of the KataGo model file
            board_size: Board size (9 or 19)
            katago2coreml_exe: Path to C++ CLI tool (fixture)
            all_test_models: Dict mapping model names to paths (fixture)
            temp_output_dir: Temporary directory for outputs (fixture)
        """
        import platform
        if platform.machine() != "arm64":
            pytest.skip("Core ML inference only available on Apple Silicon")

        import numpy as np

        # Skip if model not available
        if model_name not in all_test_models:
            pytest.skip(f"Model not available: {model_name}")

        model_path = all_test_models[model_name]

        # Convert with both converters
        cpp_output = temp_output_dir / f"cpp_fp16_{board_size}x{board_size}.mlpackage"
        python_output = temp_output_dir / f"python_fp16_{board_size}x{board_size}.mlpackage"

        convert_with_cpp(
            katago2coreml_exe,
            model_path,
            cpp_output,
            board_size,
            optimize_mask=False,
            float16=True,
        )

        convert_with_python(
            model_path,
            python_output,
            board_size,
            optimize_mask=False,
            float16=True,
        )

        # Load models
        import coremltools as ct
        cpp_model = ct.models.MLModel(str(cpp_output))
        python_model = ct.models.MLModel(str(python_output))

        # Generate random input
        np.random.seed(RANDOM_SEED_BASE)
        spatial_input = np.random.randn(1, 22, board_size, board_size).astype(np.float32)
        global_input = np.random.randn(1, 19).astype(np.float32)
        input_mask = np.ones((1, 1, board_size, board_size), dtype=np.float32)

        inputs = {
            "spatial_input": spatial_input,
            "global_input": global_input,
            "input_mask": input_mask,
        }

        # Add meta_input for human SL networks (models with metadata encoder)
        cpp_spec = cpp_model.get_spec()
        cpp_input_names = [inp.name for inp in cpp_spec.description.input]
        if "meta_input" in cpp_input_names:
            inputs["meta_input"] = np.zeros((1, 192), dtype=np.float32)

        # Run inference
        cpp_outputs = cpp_model.predict(inputs)
        python_outputs = python_model.predict(inputs)

        # Compare outputs with FP16 tolerance
        # FP16 has lower precision than FP32, so numerical differences up to ~0.5 are acceptable
        tolerance = 0.5
        for key in python_outputs.keys():
            if key not in cpp_outputs:
                pytest.fail(f"Missing output key in C++ model: {key}")

            cpp_val = cpp_outputs[key]
            py_val = python_outputs[key]

            max_diff = np.max(np.abs(cpp_val - py_val))
            if max_diff > tolerance:
                pytest.fail(
                    f"Output '{key}' differs by max {max_diff:.6f} "
                    f"(tolerance: {tolerance})"
                )

    def test_fp16_weight_size_reduced(
        self,
        katago2coreml_exe: Path,
        all_test_models: dict,
        temp_output_dir: Path,
    ):
        """Verify FLOAT16 weight.bin is approximately half the size of FLOAT32.

        This test ensures weights are actually being stored as FP16.
        """
        model_name = "g170-b6c96-s175395328-d26788732.bin.gz"  # Smaller model
        if model_name not in all_test_models:
            pytest.skip(f"Model not available: {model_name}")

        model_path = all_test_models[model_name]
        board_size = 9

        # Convert with FLOAT32
        fp32_output = temp_output_dir / "fp32_size_test.mlpackage"
        convert_with_cpp(
            katago2coreml_exe,
            model_path,
            fp32_output,
            board_size,
            optimize_mask=False,
            float16=False,
        )

        # Convert with FLOAT16
        fp16_output = temp_output_dir / "fp16_size_test.mlpackage"
        convert_with_cpp(
            katago2coreml_exe,
            model_path,
            fp16_output,
            board_size,
            optimize_mask=False,
            float16=True,
        )

        # Compare weight file sizes
        fp32_weights = fp32_output / "Data/com.apple.CoreML/weights/weight.bin"
        fp16_weights = fp16_output / "Data/com.apple.CoreML/weights/weight.bin"

        fp32_size = fp32_weights.stat().st_size
        fp16_size = fp16_weights.stat().st_size

        # FP16 weights should be roughly half the size of FP32
        # Allow some tolerance for blob header overhead
        expected_ratio = 0.5
        actual_ratio = fp16_size / fp32_size

        assert 0.45 < actual_ratio < 0.55, (
            f"FLOAT16 weight size ratio unexpected: {actual_ratio:.3f} "
            f"(expected ~{expected_ratio}). "
            f"FP32: {fp32_size} bytes, FP16: {fp16_size} bytes"
        )


class TestCppFP16IO:
    """Tests for C++ FLOAT16 I/O converter functionality.

    Tests the pure FP16 mode where model inputs and outputs are FLOAT16,
    not just internal computation. This requires iOS 16+ (spec version 7).

    Pure FP16 mode differences from mixed precision:
    1. Model inputs are FLOAT16 (not FLOAT32)
    2. No cast operations at input boundary
    3. Internal operations use FLOAT16
    4. No cast operations at output boundary
    5. Model outputs are FLOAT16 (not FLOAT32)
    """

    @pytest.mark.parametrize(
        "model_name",
        [
            "g170-b6c96-s175395328-d26788732.bin.gz",       # Smaller model
            "b5c192nbt-distilled.bin.gz",                   # Distilled model (human SL with metadata)
        ],
    )
    @pytest.mark.parametrize("board_size", [9, 19])
    def test_fp16_io_model_compiles(
        self,
        model_name: str,
        board_size: int,
        katago2coreml_exe: Path,
        all_test_models: dict,
        temp_output_dir: Path,
    ):
        """Test that FP16 I/O models compile and load successfully.

        Args:
            model_name: Name of the KataGo model file
            board_size: Board size (9 or 19)
            katago2coreml_exe: Path to C++ CLI tool (fixture)
            all_test_models: Dict mapping model names to paths (fixture)
            temp_output_dir: Temporary directory for outputs (fixture)
        """
        import platform
        if platform.machine() != "arm64":
            pytest.skip("Core ML inference only available on Apple Silicon")

        # Skip if model not available
        if model_name not in all_test_models:
            pytest.skip(f"Model not available: {model_name}")

        model_path = all_test_models[model_name]

        # Convert with FP16 I/O
        output = temp_output_dir / f"cpp_fp16_io_{board_size}x{board_size}.mlpackage"

        convert_with_cpp(
            katago2coreml_exe,
            model_path,
            output,
            board_size,
            optimize_mask=False,
            float16=True,
            float16_io=True,
        )

        # Load model
        import coremltools as ct
        model = ct.models.MLModel(str(output))
        spec = model.get_spec()

        # Verify specification version is 7 (iOS 16+)
        assert spec.specificationVersion == 7, (
            f"Expected spec version 7 for FP16 I/O, got {spec.specificationVersion}"
        )

    @pytest.mark.parametrize(
        "model_name",
        [
            "g170-b6c96-s175395328-d26788732.bin.gz",
        ],
    )
    def test_fp16_io_input_output_dtypes(
        self,
        model_name: str,
        katago2coreml_exe: Path,
        all_test_models: dict,
        temp_output_dir: Path,
    ):
        """Verify all inputs and outputs are FLOAT16 type.

        Args:
            model_name: Name of the KataGo model file
            katago2coreml_exe: Path to C++ CLI tool (fixture)
            all_test_models: Dict mapping model names to paths (fixture)
            temp_output_dir: Temporary directory for outputs (fixture)
        """
        import platform
        if platform.machine() != "arm64":
            pytest.skip("Core ML inference only available on Apple Silicon")

        # Skip if model not available
        if model_name not in all_test_models:
            pytest.skip(f"Model not available: {model_name}")

        model_path = all_test_models[model_name]
        board_size = 19

        # Convert with FP16 I/O
        output = temp_output_dir / "cpp_fp16_io_dtypes.mlpackage"

        convert_with_cpp(
            katago2coreml_exe,
            model_path,
            output,
            board_size,
            optimize_mask=False,
            float16=True,
            float16_io=True,
        )

        # Load model and check types
        import coremltools as ct
        model = ct.models.MLModel(str(output))
        spec = model.get_spec()

        # Check all inputs are FLOAT16 (65552)
        FLOAT16 = 65552
        for inp in spec.description.input:
            if inp.type.WhichOneof('Type') == 'multiArrayType':
                datatype = inp.type.multiArrayType.dataType
                assert datatype == FLOAT16, (
                    f"Input '{inp.name}' should be FLOAT16 (65552), got {datatype}"
                )

        # Check all outputs are FLOAT16
        for out in spec.description.output:
            if out.type.WhichOneof('Type') == 'multiArrayType':
                datatype = out.type.multiArrayType.dataType
                assert datatype == FLOAT16, (
                    f"Output '{out.name}' should be FLOAT16 (65552), got {datatype}"
                )

    @pytest.mark.parametrize(
        "model_name",
        [
            "g170-b6c96-s175395328-d26788732.bin.gz",
        ],
    )
    def test_fp16_io_inference(
        self,
        model_name: str,
        katago2coreml_exe: Path,
        all_test_models: dict,
        temp_output_dir: Path,
    ):
        """Test inference with FP16 inputs produces FP16 outputs.

        Args:
            model_name: Name of the KataGo model file
            katago2coreml_exe: Path to C++ CLI tool (fixture)
            all_test_models: Dict mapping model names to paths (fixture)
            temp_output_dir: Temporary directory for outputs (fixture)
        """
        import platform
        if platform.machine() != "arm64":
            pytest.skip("Core ML inference only available on Apple Silicon")

        import numpy as np

        # Skip if model not available
        if model_name not in all_test_models:
            pytest.skip(f"Model not available: {model_name}")

        model_path = all_test_models[model_name]
        board_size = 19

        # Convert with FP16 I/O
        output = temp_output_dir / "cpp_fp16_io_inference.mlpackage"

        convert_with_cpp(
            katago2coreml_exe,
            model_path,
            output,
            board_size,
            optimize_mask=False,
            float16=True,
            float16_io=True,
        )

        # Load model
        import coremltools as ct
        model = ct.models.MLModel(str(output))

        # Create FP16 inputs
        np.random.seed(RANDOM_SEED_BASE)
        spatial_input = np.random.randn(1, 22, board_size, board_size).astype(np.float16)
        global_input = np.random.randn(1, 19).astype(np.float16)
        input_mask = np.ones((1, 1, board_size, board_size), dtype=np.float16)

        inputs = {
            "spatial_input": spatial_input,
            "global_input": global_input,
            "input_mask": input_mask,
        }

        # Run inference
        outputs = model.predict(inputs)

        # Note: Core ML runtime may convert FP16 outputs to FP32 for Python API
        # The important part is that the model spec has FP16 outputs (verified in another test)
        # Runtime conversion doesn't affect on-device performance benefits

        # Verify output values are in reasonable range (no NaN/Inf)
        for name, value in outputs.items():
            assert not np.isnan(value).any(), f"Output '{name}' contains NaN values"
            assert not np.isinf(value).any(), f"Output '{name}' contains Inf values"

    @pytest.mark.parametrize(
        "model_name",
        [
            "g170-b6c96-s175395328-d26788732.bin.gz",
        ],
    )
    @pytest.mark.parametrize("board_size", [9, 19])
    @pytest.mark.parametrize("batch_config", [(1, 1), (1, 4)])
    def test_fp16_io_with_dynamic_batch(
        self,
        model_name: str,
        board_size: int,
        batch_config: tuple,
        katago2coreml_exe: Path,
        all_test_models: dict,
        temp_output_dir: Path,
    ):
        """Test FP16 I/O works with dynamic batch sizes.

        Args:
            model_name: Name of the KataGo model file
            board_size: Board size (9 or 19)
            batch_config: Tuple of (min_batch, max_batch)
            katago2coreml_exe: Path to C++ CLI tool (fixture)
            all_test_models: Dict mapping model names to paths (fixture)
            temp_output_dir: Temporary directory for outputs (fixture)
        """
        import platform
        if platform.machine() != "arm64":
            pytest.skip("Core ML inference only available on Apple Silicon")

        import numpy as np

        # Skip if model not available
        if model_name not in all_test_models:
            pytest.skip(f"Model not available: {model_name}")

        model_path = all_test_models[model_name]
        min_batch, max_batch = batch_config

        # Convert with FP16 I/O and dynamic batch
        output = temp_output_dir / f"cpp_fp16_io_batch_{min_batch}_{max_batch}_{board_size}x{board_size}.mlpackage"

        convert_with_cpp(
            katago2coreml_exe,
            model_path,
            output,
            board_size,
            optimize_mask=False,
            float16=True,
            float16_io=True,
            min_batch_size=min_batch,
            max_batch_size=max_batch,
        )

        # Load model
        import coremltools as ct
        model = ct.models.MLModel(str(output))

        # Test with different batch sizes
        for batch in [1, 2]:
            if batch > max_batch and max_batch > 0:
                continue

            # Create FP16 inputs with batch size
            np.random.seed(RANDOM_SEED_BASE + batch)
            spatial_input = np.random.randn(batch, 22, board_size, board_size).astype(np.float16)
            global_input = np.random.randn(batch, 19).astype(np.float16)
            input_mask = np.ones((batch, 1, board_size, board_size), dtype=np.float16)

            inputs = {
                "spatial_input": spatial_input,
                "global_input": global_input,
                "input_mask": input_mask,
            }

            # Run inference
            outputs = model.predict(inputs)

            # Note: Core ML runtime may convert FP16 outputs to FP32 for Python API
            # Verify batch dimension matches
            for name, value in outputs.items():
                assert value.shape[0] == batch, (
                    f"Output '{name}' batch dimension mismatch: expected {batch}, got {value.shape[0]}"
                )

            # Verify outputs are valid (no NaN/Inf)
            for name, value in outputs.items():
                assert not np.isnan(value).any(), f"Output '{name}' contains NaN values"
                assert not np.isinf(value).any(), f"Output '{name}' contains Inf values"


class TestCppMixedVsPureFP16:
    """Self-validation tests comparing mixed precision and pure FP16 I/O modes.

    Since the Python coremltools KataGo converter does not support float16_io mode,
    cross-validation against Python is not possible. Instead, these tests validate
    that the pure FP16 I/O mode produces inference results equivalent to the
    mixed precision mode (which has been validated against Python).

    Both modes produce outputs with the same names (e.g., policy_p2_conv, value_v3_bias),
    allowing direct comparison without name mapping.
    """

    @pytest.mark.parametrize(
        "model_name",
        [
            "g170-b6c96-s175395328-d26788732.bin.gz",       # Smaller model
            "b5c192nbt-distilled.bin.gz",                   # Distilled model (human SL with metadata)
        ],
    )
    @pytest.mark.parametrize("board_size", [9, 19])
    def test_mixed_vs_pure_fp16_inference(
        self,
        model_name: str,
        board_size: int,
        katago2coreml_exe: Path,
        all_test_models: dict,
        temp_output_dir: Path,
    ):
        """Test that pure FP16 I/O and mixed precision produce equivalent inference results.

        This test validates that the pure FP16 I/O mode produces results equivalent to
        the mixed precision mode (which has already been validated against Python).

        Args:
            model_name: Name of the KataGo model file
            board_size: Board size (9 or 19)
            katago2coreml_exe: Path to C++ CLI tool (fixture)
            all_test_models: Dict mapping model names to paths (fixture)
            temp_output_dir: Temporary directory for outputs (fixture)
        """
        import platform
        if platform.machine() != "arm64":
            pytest.skip("Core ML inference only available on Apple Silicon")

        import numpy as np

        # Skip if model not available
        if model_name not in all_test_models:
            pytest.skip(f"Model not available: {model_name}")

        model_path = all_test_models[model_name]

        # Convert with mixed precision (float16=True, float16_io=False)
        mixed_output = temp_output_dir / f"mixed_fp16_{board_size}x{board_size}.mlpackage"
        convert_with_cpp(
            katago2coreml_exe,
            model_path,
            mixed_output,
            board_size,
            optimize_mask=False,
            float16=True,
            float16_io=False,
        )

        # Convert with pure FP16 I/O (float16=True, float16_io=True)
        pure_output = temp_output_dir / f"pure_fp16_io_{board_size}x{board_size}.mlpackage"
        convert_with_cpp(
            katago2coreml_exe,
            model_path,
            pure_output,
            board_size,
            optimize_mask=False,
            float16=True,
            float16_io=True,
        )

        # Load models
        import coremltools as ct
        mixed_model = ct.models.MLModel(str(mixed_output))
        pure_model = ct.models.MLModel(str(pure_output))

        # Check for meta_input
        mixed_spec = mixed_model.get_spec()
        has_meta_input = "meta_input" in [inp.name for inp in mixed_spec.description.input]

        # Generate FP32 inputs (CoreML auto-converts for FP16 I/O model)
        inputs = create_batched_inputs(1, board_size, has_meta_input)

        # Run inference on both models
        mixed_outputs = mixed_model.predict(inputs)
        pure_outputs = pure_model.predict(inputs)

        # Compare outputs using relative tolerance (rtol) and absolute tolerance (atol)
        # FP16 has ~3 decimal digits of precision, so 1% relative tolerance is appropriate
        rtol, atol = 1e-2, 1e-2
        failed_outputs = []

        for name in pure_outputs.keys():
            if name not in mixed_outputs:
                failed_outputs.append(f"Missing output key in mixed model: {name}")
                continue

            pure_val = pure_outputs[name]
            mixed_val = mixed_outputs[name]

            if not np.allclose(pure_val, mixed_val, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(pure_val - mixed_val))
                failed_outputs.append(
                    f"Output '{name}' not close: max_diff={max_diff:.6f}, "
                    f"rtol={rtol}, atol={atol}"
                )

        if failed_outputs:
            pytest.fail("\n".join(failed_outputs))
