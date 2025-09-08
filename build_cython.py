#!/usr/bin/env python3
"""Pre-compile Cython modules for multiprocessing compatibility."""

import numpy
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path

PYX_FILE = "src/models/bsum.pyx"


def build_cython_modules():
    """Build Cython extensions using cython command directly."""

    # Get paths
    current_dir = Path.cwd()
    pyx_file = current_dir / PYX_FILE

    if not pyx_file.exists():
        raise FileNotFoundError(f"Cython file not found: {pyx_file}")

    # Create temporary directory for compilation
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy pyx file to temp directory
        temp_pyx = temp_path / "bsum.pyx"
        shutil.copy2(pyx_file, temp_pyx)

        # Step 1: Convert .pyx to .cpp using cython
        print("Converting .pyx to .cpp...")
        cmd_cython = [
            sys.executable,
            "-m",
            "cython",
            "--cplus",
            "--3str",
            "-o",
            str(temp_path / "bsum.cpp"),
            str(temp_pyx),
        ]

        result = subprocess.run(cmd_cython, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Cython compilation failed: {result.stderr}")

        # Step 2: Compile .cpp to .so using gcc/g++
        print("Compiling .cpp to .so...")

        import sysconfig

        python_include = sysconfig.get_path("include")
        numpy_include = numpy.get_include()

        # Output file name
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        if not ext_suffix:
            ext_suffix = ".so"

        output_file = current_dir / "src" / "models" / f"bsum{ext_suffix}"

        cmd_compile = [
            "g++",
            "-shared",
            "-fPIC",
            "-O3",
            "-w",
            f"-I{python_include}",
            f"-I{numpy_include}",
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
            str(temp_path / "bsum.cpp"),
            "-o",
            str(output_file),
        ]

        result = subprocess.run(cmd_compile, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"C++ compilation failed: {result.stderr}")

        print(f"\u2713 Compiled to: {output_file}")


if __name__ == "__main__":
    print("Building Cython modules...")

    try:
        build_cython_modules()
        print("\u2713 Cython modules built successfully!")

        # Verify the module can be imported
        sys.path.insert(0, ".")
        from src.models.bsum import update_w

        print("Cython module can be imported successfully!")

    except ImportError as e:
        print(f"\u2717 Failed to import Cython module: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)
