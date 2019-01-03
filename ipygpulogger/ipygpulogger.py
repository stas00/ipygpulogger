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

IPyGPULoggerData = namedtuple(
    'Data',
    ['gen_mem_used_delta', 'gen_mem_peaked', 'gen_mem_used',
     'gpu_mem_used_delta', 'gpu_mem_peaked', 'gpu_mem_used',
     'time_delta'],
)

class IPyGPULogger(object):

    def __init__(self, compact=False, gc_collect=True):

        self.compact = compact       # one line printouts
        self.gc_collect = gc_collect # don't use when tracking leaks

        self.peak_monitoring = False
        self.running         = False

        self.time_start = time.time() # will be set to current time later
        self.time_delta = 0

        self.gen_mem_used_peak   = -1
        self.gen_mem_used_peaked = -1
        self.gen_mem_used_delta  =  0

        self.gpu_mem_used_peak   = -1
        self.gpu_mem_used_peaked = -1
        self.gpu_mem_used_delta  =  0

        self.ipython = get_ipython()
        self.input_cells = self.ipython.user_ns['In']

        # initial measurements
        if gc_collect: gc.collect()
        self.gen_mem_used_prev = gen_mem_used_get()
        self.gpu_mem_used_prev = gpu_mem_used_get()


    @property
    def data(self):
        return IPyGPULoggerData(
            self.gen_mem_used_delta, self.gen_mem_used_peaked, self.gen_mem_used_prev,
            self.gpu_mem_used_delta, self.gpu_mem_used_peaked, self.gpu_mem_used_prev,
            self.time_delta
        )


    def start(self):
        """Register memory profiling tools to IPython instance."""
        self.running = True

        self.ipython.events.register("pre_run_cell",  self.pre_run_cell)
        self.ipython.events.register("post_run_cell", self.post_run_cell)

        return self


    def stop(self):
        """Unregister memory profiling tools from IPython instance."""
        if not self.running: return

        try: self.ipython.events.unregister("pre_run_cell",  self.pre_run_cell)
        except ValueError: pass
        try: self.ipython.events.unregister("post_run_cell", self.post_run_cell)
        except ValueError: pass

        self.running         = False
        self.peak_monitoring = False


    def pre_run_cell(self):
        if not self.running: return

        self.peak_monitoring = True

        # this thread samples RAM usage as long as the current cell is running
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

        # Capture current time before we execute the current command
        self.time_start = time.time()


    def post_run_cell(self):
        if not self.running: return

        # calculate time delta using global t1 (from the pre_run_cell event) and
        # current time
        self.time_delta = time.time() - self.time_start

        self.peak_monitoring = False

        if self.gc_collect: gc.collect()

        gen_mem_used_new = gen_mem_used_get()
        self.gen_mem_used_delta = gen_mem_used_new - self.gen_mem_used_prev
        self.gen_mem_used_peaked = max(0, self.gen_mem_used_peak - gen_mem_used_new)

        gpu_mem_used_new = gpu_mem_used_get()
        self.gpu_mem_used_delta = gpu_mem_used_new - self.gpu_mem_used_prev
        self.gpu_mem_used_peaked = max(0, self.gpu_mem_used_peak - gpu_mem_used_new)

        # not really useful, as the report is right next to the cell
        # cell_num = len(self.input_cells) - 1

        if (self.compact):
            print(f"Gen: {self.gen_mem_used_delta:0.0f}/{self.gen_mem_used_peaked:0.0f}/{gen_mem_used_new:0.0f} MB | GPU: {self.gpu_mem_used_delta:0.0f}/{self.gpu_mem_used_peaked:0.0f}/{gpu_mem_used_new:0.0f} MB | Time {self.time_delta:0.3f}s | (Consumed/Peaked/Used Total)")
        else:
            print(f"RAM: Consumed Peaked  Used Total | Exec time {self.time_delta:0.3f}s")
            print(f"Gen:    {self.gen_mem_used_delta:5.0f}  {self.gen_mem_used_peaked:5.0f}    {gen_mem_used_new:5.0f} MB |")
            print(f"GPU:    {self.gpu_mem_used_delta:5.0f}  {self.gpu_mem_used_peaked:5.0f}    {gpu_mem_used_new:5.0f} MB |")

        self.gen_mem_used_prev = gen_mem_used_new
        self.gpu_mem_used_prev = gpu_mem_used_new


    def peak_monitor_func(self):
        self.gen_mem_used_peak = -1
        self.gpu_mem_used_peak = -1

        gpu_id = torch.cuda.current_device()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

        while True:

            gen_mem_used = gen_mem_used_get()
            self.gen_mem_used_peak = max(gen_mem_used, self.gen_mem_used_peak)

            # no gc.collect, empty_cache here, since it has to be fast and we
            # want to measure only the peak memory usage
            gpu_mem_used = gpu_mem_used_get_fast(gpu_handle)
            self.gpu_mem_used_peak = max(gpu_mem_used, self.gpu_mem_used_peak)

            time.sleep(0.001) # 1msec

            if not self.peak_monitoring: break
