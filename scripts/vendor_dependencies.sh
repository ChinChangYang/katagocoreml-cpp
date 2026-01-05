#!/bin/bash
# vendor_dependencies.sh
# Copy required files from parent coremltools repository into vendor/
#
# Usage: Run from the katagocoreml-cpp root directory:
#   ./scripts/vendor_dependencies.sh /path/to/coremltools

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-coremltools>"
    echo "Example: $0 ../KataGoCoremltools"
    exit 1
fi

COREMLTOOLS_ROOT="$1"
VENDOR_DIR="vendor"

if [ ! -d "$COREMLTOOLS_ROOT/mlmodel" ]; then
    echo "Error: $COREMLTOOLS_ROOT does not appear to be a coremltools directory"
    exit 1
fi

echo "Vendoring dependencies from $COREMLTOOLS_ROOT..."

# Create vendor directory structure
mkdir -p "$VENDOR_DIR/mlmodel/format"
mkdir -p "$VENDOR_DIR/mlmodel/src"
mkdir -p "$VENDOR_DIR/modelpackage"
mkdir -p "$VENDOR_DIR/deps"

# Copy proto files
echo "Copying proto files..."
cp "$COREMLTOOLS_ROOT"/mlmodel/format/*.proto "$VENDOR_DIR/mlmodel/format/"
cp "$COREMLTOOLS_ROOT/mlmodel/format/LICENSE.txt" "$VENDOR_DIR/mlmodel/format/"

# Copy MILBlob library
echo "Copying MILBlob library..."
cp -r "$COREMLTOOLS_ROOT/mlmodel/src/MILBlob" "$VENDOR_DIR/mlmodel/src/"

# Copy ModelPackage library (exclude Python bindings)
echo "Copying ModelPackage library..."
cp -r "$COREMLTOOLS_ROOT/modelpackage/src" "$VENDOR_DIR/modelpackage/"
rm -f "$VENDOR_DIR/modelpackage/src/ModelPackagePython.cpp"

# Copy header-only dependencies
echo "Copying nlohmann/json..."
cp -r "$COREMLTOOLS_ROOT/deps/nlohmann" "$VENDOR_DIR/deps/"

echo "Copying FP16..."
cp -r "$COREMLTOOLS_ROOT/deps/FP16" "$VENDOR_DIR/deps/"

echo ""
echo "Vendoring complete! Files copied to $VENDOR_DIR/"
echo ""
echo "Directory structure:"
find "$VENDOR_DIR" -type d | head -20
echo "..."
echo ""
echo "Total size:"
du -sh "$VENDOR_DIR"
