import os, time, psutil, torch, gc
from collections import namedtuple
import threading
from IPython import get_ipython

have_cuda = 0
if torch.cuda.is_available():
    have_cuda = 1
    import pynvml
    pynvml.nvmlInit()

process = psutil.Process(os.getpid())

def gen_mem_used_get():
    "process used memory in MBs rounded down"
    return int(process.memory_info().rss/2**20)

def gpu_mem_used_get():
    "query nvidia for used memory for gpu in MBs (rounded down). If id is not passed, currently selected torch device is used. Clears pytorch cache before taking the measurements"
    torch.cuda.empty_cache() # clear cache to report the correct data
    id = torch.cuda.current_device()
    handle = pynvml.nvmlDeviceGetHandleByIndex(id)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return int(info.used/2**20)

# similar to gpu_mem_used_get, but doesn't do any checks, clearing caches,
# gc.collect, etc., to be lightening fast when run in a tight loop from a peak
# memory measurement thread.
def gpu_mem_used_get_fast(gpu_handle):
    info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return int(info.used/2**20)

class IPyGPULogger(object):

    def __init__(self, compact=False, gc_collect=True):

        self.compact = compact       # one line printouts
        self.gc_collect = gc_collect # don't use when tracking leaks

        self.keep_watching   = True
        self.watching_memory = True

        self.t1 = time.time() # will be set to current time later
        self.time_delta = 0

        self.gen_mem_used_peak   = -1
        self.gen_mem_used_peaked = -1
        self.gen_mem_used_delta  =  0

        self.gpu_mem_used_peak   = -1
        self.gpu_mem_used_peaked = -1
        self.gpu_mem_used_delta  =  0

        self.ipython = get_ipython()
        self.input_cells = self.ipython.user_ns['In']

        self._data = namedtuple(
            'Data',
            ['gen_mem_used_delta', 'gen_mem_peaked', 'gen_mem_used',
             'gpu_mem_used_delta', 'gpu_mem_peaked', 'gpu_mem_used',
            'time_delta'],
            )

        # initial measurement
        if gc_collect: gc.collect()
        self.gen_mem_used_prev = gen_mem_used_get()
        self.gpu_mem_used_prev = gpu_mem_used_get()

    @property
    def data(self):
        return self._data(
            self.gen_mem_used_delta, self.gen_mem_used_peaked, self.gen_mem_used_prev,
            self.gpu_mem_used_delta, self.gpu_mem_used_peaked, self.gpu_mem_used_prev,
            self.time_delta
        )

    def start(self):
        """Register memory profiling tools to IPython instance."""
        self.watching_memory = True
        self.ipython.events.register("pre_run_cell",  self.pre_run_cell)
        self.ipython.events.register("post_run_cell", self.watch_memory)
        return self

    def stop(self):
        """Unregister memory profiling tools from IPython instance."""
        self.watching_memory = False
        try: self.ipython.events.unregister("pre_run_cell",  self.pre_run_cell)
        except ValueError: pass
        try: self.ipython.events.unregister("post_run_cell", self.watch_memory)
        except ValueError: pass

    def watch_memory(self):
        if not self.watching_memory: return

        # calculate time delta using global t1 (from the pre-run
        # event) and current time
        self.time_delta = time.time() - self.t1

        self.keep_watching = False

        if self.gc_collect: gc.collect()

        gen_mem_used_new = gen_mem_used_get()
        self.gen_mem_used_delta = gen_mem_used_new - self.gen_mem_used_prev
        self.gen_mem_used_peaked = max(0, self.gen_mem_used_peak - gen_mem_used_new)

        gpu_mem_used_new = gpu_mem_used_get()
        self.gpu_mem_used_delta = gpu_mem_used_new - self.gpu_mem_used_prev
        self.gpu_mem_used_peaked = max(0, self.gpu_mem_used_peak - gpu_mem_used_new)

        num_commands = len(self.input_cells) - 1
        cmd = "In [{}]".format(num_commands)

        if (self.compact):
            print(f"Gen: {self.gen_mem_used_delta:0.0f}/{self.gen_mem_used_peaked:0.0f}/{gen_mem_used_new:0.0f} MB | GPU: {self.gpu_mem_used_delta:0.0f}/{self.gpu_mem_used_peaked:0.0f}/{gpu_mem_used_new:0.0f} MB | {self.time_delta:0.3f}s | {cmd} (Consumed/Peaked/Used Total)")
        else:
            print(f"RAM: Consumed Peaked  Used Total in {self.time_delta:0.3f}s ({cmd})")
            print(f"Gen:    {self.gen_mem_used_delta:5.0f}  {self.gen_mem_used_peaked:5.0f}    {gen_mem_used_new:5.0f} MB")
            print(f"GPU:    {self.gpu_mem_used_delta:5.0f}  {self.gpu_mem_used_peaked:5.0f}    {gpu_mem_used_new:5.0f} MB")

        self.gen_mem_used_prev = gen_mem_used_new
        self.gpu_mem_used_prev = gpu_mem_used_new


    def during_execution_memory_sampler(self):
        self.mem_used_peak = -1
        self.gpu_mem_used_peak = -1
        self.keep_watching = True

        # assuming the gpu is not switched to another one once the logger has
        # started, otherwise it would be measuring the wrong GPU
        # probably could provide an API to update the gpu id mid-way
        gpu_id = torch.cuda.current_device()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

        n = 0
        WAIT_BETWEEN_SAMPLES_SECS = 0.001
        MAX_ITERATIONS = 60.0 / WAIT_BETWEEN_SAMPLES_SECS
        while True:

            gen_mem_used = gen_mem_used_get()
            self.gen_mem_used_peak = max(gen_mem_used, self.gen_mem_used_peak)

            # no gc.collect, empty_cache here, since it has to be fast and we
            # want to measure only the peak memory usage
            gpu_mem_used = gpu_mem_used_get_fast(gpu_handle)
            self.gpu_mem_used_peak = max(gpu_mem_used, self.gpu_mem_used_peak)

            time.sleep(WAIT_BETWEEN_SAMPLES_SECS)
            if not self.keep_watching or n > MAX_ITERATIONS:
                # exit if we've been told our command has finished or if it has
                # run for more than a sane amount of time (e.g. maybe something
                # crashed and we don't want this to carry on running)
                if n > MAX_ITERATIONS:
                    print("✘ {} Something weird happened and this ran for too long, this thread is killing itself".format(__file__))
                break
            n += 1


    def pre_run_cell(self):
        """Capture current time before we execute the current command"""
        # start a thread that samples RAM usage until the current command finishes
        ipython_mem_used_thread = threading.Thread(target=self.during_execution_memory_sampler)
        ipython_mem_used_thread.daemon = True
        ipython_mem_used_thread.start()
        self.t1 = time.time()
