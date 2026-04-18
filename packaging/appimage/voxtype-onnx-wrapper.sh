#!/bin/sh
# Voxtype ONNX CPU-adaptive wrapper script
# Detects CPU capabilities and executes the appropriate ONNX binary variant

VOXTYPE_LIB="${VOXTYPE_LIB:-/usr/lib/voxtype}"

# Detect AVX-512 support (Linux-specific)
if [ -f /proc/cpuinfo ] && grep -q avx512f /proc/cpuinfo 2>/dev/null; then
    # Prefer AVX-512 binary if available
    if [ -x "$VOXTYPE_LIB/voxtype-onnx-avx512" ]; then
        exec "$VOXTYPE_LIB/voxtype-onnx-avx512" "$@"
    fi
fi

# Fall back to AVX2 (baseline for x86_64)
if [ -x "$VOXTYPE_LIB/voxtype-onnx-avx2" ]; then
    exec "$VOXTYPE_LIB/voxtype-onnx-avx2" "$@"
fi

# If we get here, no binary was found
echo "Error: No voxtype ONNX binary found in $VOXTYPE_LIB" >&2
echo "Please reinstall the package." >&2
exit 1
