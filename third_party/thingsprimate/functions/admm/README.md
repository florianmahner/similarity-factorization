It is important to first compile cython correctly.

For this, please run

```bash
pyton build_cython.py
```

This project requires python 3.12 right now, but I guess it might also run with a version below that if you remove the 

```python
type NDArray = np.ndarray
```
in admm.py


Have a look at `example.py` to see how to directly cross validate the matrix and run the model.