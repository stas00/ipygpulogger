# Changes

## 0.1.1.dev0 (Work In Progress)

- pre-load pytorch CUDA structures to account for the first 0.5GB of GPU RAM eaten up
- switched to tracemalloc for general memory tracing

## 0.1.0 ()

- initial port from https://github.com/FrancescAlted/ipython_memwatcher
- add gpu support
- add cleaner report, compact version
- gc.collect support
