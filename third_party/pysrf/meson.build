project('pysrf', 'c', 'cython',
  version: '0.1.0',
  default_options: ['warning_level=1'],
)

py = import('python').find_installation(pure: false)

numpy_inc = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true,
).stdout().strip()

scipy_inc = run_command(py,
  ['-c', 'import os, scipy; print(os.path.dirname(os.path.dirname(scipy.__file__)))'],
  check: true,
).stdout().strip()

# Cython needs numpy/scipy include paths for cimport resolution
add_project_arguments('-I' + numpy_inc, language: 'cython')
add_project_arguments('-I' + scipy_inc, language: 'cython')

py.extension_module(
  '_bsum',
  'pysrf/_bsum.pyx',
  c_args: [
    '-I' + numpy_inc,
    '-O3',
    '-march=native',
    '-ffp-contract=off',
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
  ],
  install: true,
  subdir: 'pysrf',
)

py.install_sources(
  [
    'pysrf/__init__.py',
    'pysrf/_bsum.py',
    'pysrf/bounds.py',
    'pysrf/coherence.py',
    'pysrf/consensus.py',
    'pysrf/cross_validation.py',
    'pysrf/model.py',
  ],
  subdir: 'pysrf',
)
