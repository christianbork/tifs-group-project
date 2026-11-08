# Lessons

- Run scripts exactly as the user would (`.venv/Scripts/python.exe analysis/NN_x.py`); never add PYTHONPATH or other env workarounds that hide import errors.
- When generating Python through a Bash heredoc, avoid backslash escapes (the shell collapses \n into a real newline); use the Edit/Write tools for code containing escapes.
- Check every grid search (lags, hyper-parameters) for corner solutions; widen the grid or justify a fixed choice.
- In trade-level simulations, a single quote can be hit at most once per aggressive order; count fills per order, not per fill.
- Before changing results, reproduce the previous version exactly with the same package versions.
