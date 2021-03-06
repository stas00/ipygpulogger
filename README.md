<!--
[![pypi ipygpulogger version](https://img.shields.io/pypi/v/ipygpulogger.svg)](https://pypi.python.org/pypi/ipygpulogger)
[![Conda ipygpulogger version](https://img.shields.io/conda/v/stason/ipygpulogger.svg)](https://anaconda.org/stason/ipygpulogger)
[![Anaconda-Server Badge](https://anaconda.org/stason/ipygpulogger/badges/platforms.svg)](https://anaconda.org/stason/ipygpulogger)
[![ipygpulogger python compatibility](https://img.shields.io/pypi/pyversions/ipygpulogger.svg)](https://pypi.python.org/pypi/ipygpulogger)
[![ipygpulogger license](https://img.shields.io/pypi/l/ipygpulogger.svg)](https://pypi.python.org/pypi/ipygpulogger)
-->

# ipygpulogger has been merged into ipyexperiments

Please see: https://github.com/stas00/ipyexperiments

# ipygpulogger

GPU Logger for jupyter/ipython (currently logging memory and time).

## About

This module is for those who need to track resource usage line by line in ipython or cell by cell in jupyter notebooks. You get the resource report automatically as soon as a command or a cell finished executing.

Currently this module logs GPU RAM, general RAM and execution time. But can be expanded to track other important thing. While there are various similar loggers out there, the main focus of this implementation is to help track GPU, whose main scarce resource is GPU RAM.

## Usage

It's trivial to use this module:

```
from ipygpulogger import IPyGPULogger
IPyGPULogger().start()
```

Then running a jupyter cell with some code, you get a report printed:
```
RAM: Consumed Peaked  Used Total  in 0.000s (In [4])
CPU:       45      0    170 MB
GPU:     2467      0   1465 MB
```

![Usage demo](https://raw.githubusercontent.com/stas00/ipygpulogger/master/docs/images/usage1.png)

## Installation

* dev:

   ```
   pip install git+https://github.com/stas00/ipygpulogger.git
   ```

<!--
will be available shortly
* pypi:

   ```
   pip install ipygpulogger
   ```
* conda:

   ```
   conda install -c fastai -c stason ipygpulogger
   ```
-->



## Requirements

Python 3.6+, and the rest of dependencies are automatically installed via pip (and soon conda).



## Demo

The easiest way to see how this framework works is to read the [demo notebook](https://github.com/stas00/ipygpulogger/blob/master/demo.ipynb).

## Backends

Currently supporting NVIDIA GPUs w/ `pytorch`, but it should be trivial to add support for other frameworks, like `tensorflow`, etc.

The only framework-specific code is framework preloading, device id access and cache clearing.

NVIDIA's data is read directly via `nvml` [nvidia-ml-py3](https://github.com/nicolargo/nvidia-ml-py3).

Please, note, that this module doesn't setup its `pip`/`conda` dependencies for the backend frameworks (e.g. `pytorch`), since you must have already installed those before attempting to use this module.


## API

```
from ipygpulogger import IPyGPULogger
```

1. Create a logger object:
   ```python
   il = IPyGPULogger(compact=False, gc_collect=True)
   ```
   Options:
   * `compact` - use compact one line printouts
   * `gc_collect` - correct memory usage reports. Don't use when tracking memory leaks (objects with circular reference).

2. Start logging if it wasn't started in the constructor
   ```python
   il.start()
   ```

3. Stop logging
   ```python
   il.stop()
   ```

4. Access the measured data directly (in not rounded MBs)
   ```python
   cpu_mem, gpu_mem, time_data = il.data
   print(cpu_mem)
   print(gpu_mem)
   print(time_data)

   ```
   gives:
   ```
   IPyGPULoggerMemory(used_delta=128.0062427520752, peaked_delta=128.0133180618286, used_total=2282)
   IPyGPULoggerMemory(used_delta=1024, peaked_delta=1024, used_total=3184)
   IPyGPULoggerTime(time_delta=0.806537389755249)
   ```
   This accessor returns 3 `namedtuple`s, so that you can access the data fields by name. For example, continuing from above.

   ```python
   print(cpu_mem.used_delta)
   print(gpu_mem.used_total)
   print(time_data.time_delta)
   ```
   or to unpack it:
   ```python
   cpu_mem_used_delta, cpu_mem_peaked_delta, cpu_mem_used_total = cpu_mem
   gpu_mem_used_delta, gpu_mem_peaked_delta, gpu_mem_used_total = gpu_mem
   ```

Please refer to the [demo notebook](https://github.com/stas00/ipygpulogger/blob/master/demo.ipynb) to see this API in action.


## Peak Memory Usage

Often, a function may use more RAM than if we were just to measure the memory usage before and after its execution, therefore this module uses a thread to take snapshots of its actual memory usage during its run. So when the report is printed you get to see the maximum memory that was required to run this function.

For example if the report was:

```
RAM: Consumed Peaked  Used Total in 0.000s (In [4])
CPU:        0      0    170 MB
GPU:     2567   1437   5465 MB
```

That means that when the function finished it consumed `2467 MB` of GPU RAM, as compared to the memory usage before its run. However, it actually needed a total of `4000 MB` of GPU RAM to be able to run (`2467`+`1437`). So if you didn't have `4000 MB` of free unfragmented RAM available it would have failed.

## Framework Preloading

You do need to be aware that some frameworks consume a big chunk of general and GPU RAM when they are used for the first time. For example `pytorch` `cuda` [eats up](
https://docs.fast.ai/dev/gpu.html#unusable-gpu-ram-per-process) about 0.5GB of GPU RAM and 2GB of general RAM on load (not necessarily on `import`, but usually later when it's used), so if your experiment started with doing a `cuda` action for the first time in a given process, expect to lose that much RAM - this one can't be reclaimed.

But `IPyGPULogger` does all this for you, for example, preloading `pytorch` `cuda` if the `pytorch` backend (default) is used. During the preloading it internally does:

   ```python
   import pytorch
   torch.ones((1, 1)).cuda() # preload pytorch with cuda libraries
   ```

## Cache Clearing

Before a snapshot of used GPU RAM is made, its cache is cleared, since otherwise there is no way to get any real GPU RAM usage. So this module gives very reliable data on GPU RAM (but also see [Temporary Memory Leaks](#temporary-memory-leaks).

For general RAM accounting, [tracemalloc](https://docs.python.org/3/library/tracemalloc.html) is used, which gives a correct report of how much RAM was allocated and peaked, which overcomes the interference of python internal caching. So if the code makes a large memory allocation, followed by its release and immediately a smaller allocation - the process's total memory usage won't change, yet it'll report how much RAM a specific cell has allocated.

## Temporary Memory Leaks

Modern (py-3.4+)` gc.collect()` handles circular references in objects, including those with custom `__del__` methods. So pretty much eventually, when `gc.collect()` arrives, all deleted objects get reclaimed. The problem is that in the environments like machine learning training, eventually is not good enough. Some objects that are no longer needed could easily hold huge chunks of GPU RAM and waiting for `gc.collect()` to arrive is very unreliable bad method of handling that. Moreover, allocating more GPU RAM before freeing RAM that is already not serving you leads to memory fragmentation, which is a very bad scenario - as you may have a ton of free GPU RAM, but none can be used. And at least at the moment, NVIDIA's CUDA doesn't have a *defrag* method.

In order to give you correct memory usage numbers, this module by default runs `gc.collect` before clearing GPU cache and taking a measurement of its used memory. But this could mask problems in your code, and if you turn this module off, suddenly the same code can't run on the same amount of GPU RAM.

So, make sure you compare your total GPU RAM consumption with and without `gc_collect=True` in the object `IPyGPULogger` constructor.


## Contributing

PRs with improvements and new features and Issues with suggestions are welcome.


## Testing

1. Install my fork of `pytest-ipynb` (the original one is no longer being maintained)
   ```
   pip install git+https://github.com/stas00/pytest-ipynb.git
   ```
2. Run the test suite
   ```
   make test
   ```


## Credits

This module was modelled after:

* [ipython_memwatcher](https://github.com/FrancescAlted/ipython_memwatcher) by Francesc Alted


## See Also

- [ipyexperiments](https://github.com/stas00/ipyexperiments) - jupyter/ipython experiment containers for GPU and general RAM re-use. This sister module's main purpose is to help calibrate hyper parameters in deep learning notebooks to fit the available GPU and general RAM, but, of course, it can be useful for any other use where memory limits is a constant issue. It is also useful for detecting memory leaks in your code.


## History

A detailed history of changes can be found [here](https://github.com/stas00/ipygpulogger/blob/master/CHANGES.md).
