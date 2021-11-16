# Installation
We recommend to start in a new conda environment or a new PyTorch docker container. We will only include the guide for conda environment at the monent.
1. Clone the repository.
    ```bash
    git clone <url>
   cd fast-image-retrieval
    ```
2. Create new conda environment. Deactivate your current if you are in any other environment(e.g. `base` env).
    ```bash
    conda deactivate
    conda create -n fast-image-retrieval python=3.8 -y
    conda activate fast-image-retrieval
    ```
3. Install the dependencies
   ```bash
   pip install -r requirements.txt
   # faiss gpu version, or `faiss-cpu` for cpu-only version
   conda install -c pytorch faiss-gpu -y
    ```

```{admonition} Operating Systems
Currently only Linux is tested.
`faiss-gpu` is not available for macOS.
```
