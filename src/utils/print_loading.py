import time
import threading
from functools import wraps
from typing import Callable

def print_loading_factory(
    text: str = "Now, Loading",
    dotnum: int = 10
):
    def print_loading(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            stop_event = threading.Event()
            thread = threading.Thread(
                target=_func_with_thread_stop,
                args=(func, stop_event, *args),
                kwargs=kwargs
            )
            thread.start()
            _print_loading(stop_event, text, dotnum)
            thread.join()

        return wrapper

    return print_loading

def _func_with_thread_stop(
    func: Callable,
    thread_stop_event: threading.Event,
    *args, **kwargs
):
    try:
        return func(*args, **kwargs)
    finally:
        thread_stop_event.set()

def _print_loading(
    thread_stop_event: threading.Event,
    text: str,
    dotnum: int
):
    """
    Note:
        この関数はthreadingでメインスレッドから制御しないと機能しない
        Pythonではstdoutはバッファリングされるらしく、
        print()においてflushを指定しないと、timeを指定しても上手く動作しない
    """
    i = 1
    print(f"{text}\033[s", end="")
    while not thread_stop_event.is_set():
        print(f"\033[u{i * '.'}\033[K", end="", flush=True)
        time.sleep(0.1)
        i = i+1 if i != min(i+1, dotnum) else 0
    print("\nDone!!")
