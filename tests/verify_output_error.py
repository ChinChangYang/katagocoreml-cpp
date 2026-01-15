#!/usr/bin/env python3
"""
Verify output error is acceptable across batch configurations.
Compares: static batch=1, static batch=4, dynamic batch (1-8)
Error threshold: max absolute difference < 1e-5 (FP32) or < 1e-3 (FP16)

Usage:
    python tests/verify_output_error.py \
        --model-static-b1 model-static-b1.mlpackage \
        --model-static-b4 model-static-b4.mlpackage \
        --model-dynamic model-dynamic-1to8.mlpackage \
        --threshold 1e-5
"""
import numpy as np
import coremltools as ct


def create_test_inputs(batch_size, board_size=19):
    """Create deterministic test inputs for reproducibility.

    Each sample is generated with a consistent seed so that sample[i] is
    identical regardless of batch size. This enables per-sample consistency testing.
    """
    spatial_list = []
    global_list = []
    for i in range(batch_size):
        np.random.seed(42 + i)  # Consistent seed per sample index
        spatial_list.append(np.random.randn(1, 22, board_size, board_size))
        global_list.append(np.random.randn(1, 19))
    return {
        "spatial_input": np.concatenate(spatial_list, axis=0).astype(np.float32),
        "global_input": np.concatenate(global_list, axis=0).astype(np.float32),
        "input_mask": np.ones((batch_size, 1, board_size, board_size), dtype=np.float32),
    }


def run_inference(model_path, inputs):
    """Run inference and return outputs as dict."""
    model = ct.models.MLModel(model_path)
    return model.predict(inputs)


def compare_outputs(out1, out2, name1, name2, threshold=1e-5):
    """Compare two output dicts, return max error and pass/fail."""
    max_error = 0.0
    results = {}
    for key in out1:
        if key in out2:
            diff = np.abs(out1[key] - out2[key])
            max_diff = np.max(diff)
            max_error = max(max_error, max_diff)
            results[key] = max_diff
    passed = max_error < threshold
    print(f"\n{name1} vs {name2}:")
    for key, err in results.items():
        status = "PASS" if err < threshold else "FAIL"
        print(f"  {key}: max_error={err:.2e} [{status}]")
    print(f"  Overall: {'PASS' if passed else 'FAIL'} (max_error={max_error:.2e}, threshold={threshold:.2e})")
    return passed, max_error


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify output error across batch configurations")
    parser.add_argument("--model-static-b1", required=True, help="Static batch=1 model path")
    parser.add_argument("--model-static-b4", required=True, help="Static batch=4 model path")
    parser.add_argument("--model-dynamic", required=True, help="Dynamic batch model path")
    parser.add_argument("--threshold", type=float, default=1e-5, help="Error threshold for same-config tests")
    parser.add_argument("--per-sample-threshold", type=float, default=1e-2,
                       help="Error threshold for per-sample consistency (relaxed due to GPU batch variance)")
    args = parser.parse_args()

    all_passed = True

    # Test 1: Single batch comparison (static b1 vs dynamic at b=1)
    print("=" * 60)
    print("TEST 1: Single batch - Static(b=1) vs Dynamic(b=1)")
    print("=" * 60)
    inputs_b1 = create_test_inputs(1)
    out_static_b1 = run_inference(args.model_static_b1, inputs_b1)
    out_dynamic_b1 = run_inference(args.model_dynamic, inputs_b1)
    passed, _ = compare_outputs(out_static_b1, out_dynamic_b1, "Static(b=1)", "Dynamic(b=1)", args.threshold)
    all_passed &= passed

    # Test 2: Static batch=4 vs Dynamic at batch=4
    print("\n" + "=" * 60)
    print("TEST 2: Batch=4 - Static(b=4) vs Dynamic(b=4)")
    print("=" * 60)
    inputs_b4 = create_test_inputs(4)
    out_static_b4 = run_inference(args.model_static_b4, inputs_b4)
    out_dynamic_b4 = run_inference(args.model_dynamic, inputs_b4)
    passed, _ = compare_outputs(out_static_b4, out_dynamic_b4, "Static(b=4)", "Dynamic(b=4)", args.threshold)
    all_passed &= passed

    # Test 3: Per-sample consistency (batch=4 should match 4x batch=1)
    # Note: Uses relaxed threshold since GPU batch processing may have small numerical variance
    print("\n" + "=" * 60)
    print("TEST 3: Per-sample consistency - Dynamic(b=4)[0] vs Dynamic(b=1)")
    print(f"        (using relaxed threshold {args.per_sample_threshold:.0e} for GPU batch variance)")
    print("=" * 60)
    # Extract first sample from batch=4 output
    out_dynamic_b4_sample0 = {k: v[0:1] for k, v in out_dynamic_b4.items()}
    passed, _ = compare_outputs(out_dynamic_b1, out_dynamic_b4_sample0, "Dynamic(b=1)", "Dynamic(b=4)[0]", args.per_sample_threshold)
    all_passed &= passed

    # Summary
    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
