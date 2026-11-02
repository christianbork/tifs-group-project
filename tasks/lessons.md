# Lessons

- Run scripts exactly as the user would (`.venv/Scripts/python.exe analysis/NN_x.py`); never add PYTHONPATH or other env workarounds that hide import errors.
- When generating Python through a Bash heredoc, avoid backslash escapes (the shell collapses \n into a real newline); use the Edit/Write tools for code containing escapes.
