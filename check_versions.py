import importlib, sys

packages = ['qiskit', 'qiskit_aer', 'qiskit_algorithms', 'numpy', 'scipy', 'joblib', 'matplotlib']
print(f'Python: {sys.version}')
print()
for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'{pkg:20s}: {version}')
    except ImportError as e:
        print(f'{pkg:20s}: NOT INSTALLED ({e})')
