# Installing pamo with Poetry

Since `pamo` has complex dependencies including CUDA extensions and local packages, here's how to install it using Poetry:

## Method: Use pip within Poetry environment

Poetry manages your environment, but we'll use `pip` for `pamo` dependencies because they require CUDA compilation and local package installation.

### Steps:

1. **Ensure Poetry environment is activated/synced:**
   ```bash
   poetry install  # Install your project dependencies first
   ```

2. **Install pamo dependencies using pip within Poetry:**
   ```bash
   # Install external dependencies
   poetry run pip install git+https://github.com/eliphatfs/cumesh2sdf.git
   poetry run pip install pdmc
   
   # Install Stage 2: Simplification (local package with CUDA extensions)
   cd pamo/simp_cuda
   poetry run pip install .
   cd ../..
   
   # Install Stage 3: Safe Projection
   cd pamo/simp_cuda/safe_project/warp_
   # Note: build_lib.py may need to be run first if it exists
   # poetry run python build_lib.py --cuda_path /usr/local/cuda
   poetry run pip install .
   cd ../../..
   
   cd pamo/simp_cuda/safe_project
   poetry run pip install .
   cd ../..
   ```

### Alternative: Create a script

You can also create a script `install_pamo.sh` that runs these commands:

```bash
#!/bin/bash
poetry run pip install git+https://github.com/eliphatfs/cumesh2sdf.git
poetry run pip install pdmc
cd pamo/simp_cuda && poetry run pip install . && cd ../..
cd pamo/simp_cuda/safe_project/warp_ && poetry run pip install . && cd ../../..
cd pamo/simp_cuda/safe_project && poetry run pip install . && cd ../..
```

### Why this approach?

- Poetry manages your main project dependencies and environment
- `pip` handles complex local packages with CUDA extensions better
- All packages are installed in the same Poetry-managed virtual environment
- Works around Poetry's limitations with local packages that need compilation

### Verify installation:

```bash
poetry run python -c "from pamo import PaMO; print('pamo installed successfully!')"
```

