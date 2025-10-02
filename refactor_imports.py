#!/usr/bin/env python3
"""
Refactor imports from old structure to pysrf.
Replaces:
- from models.admm import ADMM → from pysrf import SRF
- from cross_validation import ... → from pysrf import ...
- from _compute_prange import ... → from pysrf.bounds import ...
- ADMM( → SRF(
- observed_fraction → sampling_fraction
"""
import re
from pathlib import Path

def refactor_file(filepath):
    """Refactor imports in a single file."""
    try:
        content = filepath.read_text()
        original = content
        
        # Replace imports
        content = re.sub(
            r'from\s+models\.admm\s+import\s+ADMM',
            'from pysrf import SRF',
            content
        )
        content = re.sub(
            r'from\s+src\.models\.admm\s+import\s+ADMM',
            'from pysrf import SRF',
            content
        )
        content = re.sub(
            r'from\s+cross_validation\s+import',
            'from pysrf import',
            content
        )
        content = re.sub(
            r'from\s+src\.cross_validation\s+import',
            'from pysrf import',
            content
        )
        content = re.sub(
            r'from\s+_compute_prange\s+import',
            'from pysrf.bounds import',
            content
        )
        content = re.sub(
            r'from\s+src\._compute_prange\s+import',
            'from pysrf.bounds import',
            content
        )
        
        # Replace class usage (but not in strings/comments)
        content = re.sub(r'\bADMM\s*\(', 'SRF(', content)
        content = re.sub(r'=\s*ADMM\s*\(', '= SRF(', content)
        content = re.sub(r'model\s*=\s*ADMM', 'model = SRF', content)
        content = re.sub(r'estimator\s*=\s*ADMM', 'estimator = SRF', content)
        
        # Replace parameter names
        content = re.sub(r'observed_fraction\s*=', 'sampling_fraction=', content)
        content = re.sub(r'observed_fraction:', 'sampling_fraction:', content)
        content = re.sub(r'"observed_fraction"', '"sampling_fraction"', content)
        content = re.sub(r"'observed_fraction'", "'sampling_fraction'", content)
        
        if content != original:
            filepath.write_text(content)
            print(f"✓ Updated: {filepath}")
            return True
        else:
            print(f"  Skipped: {filepath} (no changes needed)")
            return False
    except Exception as e:
        print(f"✗ Error in {filepath}: {e}")
        return False

if __name__ == "__main__":
    base = Path(".")
    
    # Files to refactor
    files_to_update = [
        # Scripts
        "scripts/run_simulated_cross_validation.py",
        "scripts/run_embedding_generation.py",
        "scripts/run_things_behavior.py",
        "scripts/run_rsa_comparison.py",
        "scripts/run_simulated_denoising.py",
        "scripts/slurm/pipeline_core.py",
        "scripts/run_snmf_power.py",
        "scripts/sociopatterns.py",
        "scripts/ablation.py",
        "scripts/run_factorial.py",
        # Src
        "src/experiments/clustering/graph.py",
        "src/experiments/things/things.py",
        "src/pipeline/embedding_pipeline.py",
        "src/experiments/things/common.py",
        # Tests
        "tests/cv_test.py",
        "tests/test_mur.py",
        "tests/pbound_ultra.py",
        "tests/compare_pbound.py",
        "tests/timeadmm.py",
        "tests/test_admm_torch_vs_numpy.py",
        "tests/run_torch_benchmarks.py",
        "tests/run_vit.py",
        "tests/run_monkey.py",
        "tests/bench_bsum.py",
    ]
    
    print("=" * 60)
    print("Refactoring imports to use pysrf")
    print("=" * 60)
    
    updated_count = 0
    for file_path in files_to_update:
        full_path = base / file_path
        if full_path.exists():
            if refactor_file(full_path):
                updated_count += 1
        else:
            print(f"  Missing: {file_path}")
    
    print("=" * 60)
    print(f"Updated {updated_count} files")
    print("=" * 60)

