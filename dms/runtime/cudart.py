"""Minimal ctypes binding to libcudart, used instead of pycuda (forbidden on this board).

Every wrapper raises :class:`CudaError` on a non-zero CUDA status; none of them swallow
an error. Sizes are bytes, pointers are plain Python ints (device pointers and opaque
stream/event handles), and durations are milliseconds where they appear.

Failure behaviour: if ``libcudart.so`` cannot be loaded at all, :func:`lib` raises
:class:`CudaError` on first use - construction of a :class:`dms.runtime.TrtEngine` then
fails loudly rather than degrading to a silent no-op.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import threading
from typing import Optional, Tuple

CUDA_SUCCESS = 0
CUDA_ERROR_NOT_READY = 600

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3

# cudaHostAlloc flags (driver_types.h).
cudaHostAllocDefault = 0x00
cudaHostAllocPortable = 0x01
cudaHostAllocMapped = 0x02
cudaHostAllocWriteCombined = 0x04


class CudaError(RuntimeError):
    """A CUDA runtime call returned a non-success status."""


def _load() -> ctypes.CDLL:
    for name in ("libcudart.so", "libcudart.so.11.0", "libcudart.so.11"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    found = ctypes.util.find_library("cudart")
    if found:
        return ctypes.CDLL(found)
    raise CudaError("libcudart.so not found")


_lib: Optional[ctypes.CDLL] = None
_lib_lock = threading.Lock()

_VOIDPP = ctypes.POINTER(ctypes.c_void_p)

# name -> (argtypes, restype). restype is c_int for every status-returning call.
_SIGNATURES = (
    ("cudaMalloc", [_VOIDPP, ctypes.c_size_t]),
    ("cudaFree", [ctypes.c_void_p]),
    ("cudaHostAlloc", [_VOIDPP, ctypes.c_size_t, ctypes.c_uint]),
    ("cudaFreeHost", [ctypes.c_void_p]),
    ("cudaHostGetDevicePointer", [_VOIDPP, ctypes.c_void_p, ctypes.c_uint]),
    (
        "cudaMemcpyAsync",
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p],
    ),
    ("cudaStreamCreate", [_VOIDPP]),
    ("cudaStreamDestroy", [ctypes.c_void_p]),
    ("cudaStreamSynchronize", [ctypes.c_void_p]),
    ("cudaStreamQuery", [ctypes.c_void_p]),
    ("cudaEventCreate", [_VOIDPP]),
    ("cudaEventCreateWithFlags", [_VOIDPP, ctypes.c_uint]),
    ("cudaEventRecord", [ctypes.c_void_p, ctypes.c_void_p]),
    ("cudaEventQuery", [ctypes.c_void_p]),
    ("cudaEventSynchronize", [ctypes.c_void_p]),
    ("cudaEventDestroy", [ctypes.c_void_p]),
    ("cudaSetDevice", [ctypes.c_int]),
    ("cudaGetDevice", [ctypes.POINTER(ctypes.c_int)]),
    ("cudaRuntimeGetVersion", [ctypes.POINTER(ctypes.c_int)]),
    ("cudaDriverGetVersion", [ctypes.POINTER(ctypes.c_int)]),
    ("cudaMemGetInfo", [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]),
    ("cudaGetLastError", []),
)


def lib() -> ctypes.CDLL:
    """Return the loaded libcudart with argtypes bound. Raises CudaError if absent."""
    global _lib
    if _lib is not None:
        return _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        handle = _load()
        for name, argtypes in _SIGNATURES:
            fn = getattr(handle, name, None)
            if fn is None:
                # Older/newer runtimes may lack an optional entry point; the wrapper
                # below raises a clear CudaError when it is actually used.
                continue
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int
        handle.cudaGetErrorString.argtypes = [ctypes.c_int]
        handle.cudaGetErrorString.restype = ctypes.c_char_p
        _lib = handle
        return _lib


def _fn(name: str):
    handle = lib()
    fn = getattr(handle, name, None)
    if fn is None:
        raise CudaError("%s is not available in this libcudart" % name)
    return fn


def error_string(err: int) -> str:
    try:
        msg = lib().cudaGetErrorString(err)
    except CudaError:
        return str(err)
    return msg.decode("utf-8", "replace") if msg else str(err)


def check(err: int, what: str) -> None:
    if err != CUDA_SUCCESS:
        raise CudaError("%s: %s (%s)" % (what, error_string(err), err))


def set_device(index: int = 0) -> None:
    """Bind the calling thread to a device. Cheap and idempotent; call before allocating."""
    check(_fn("cudaSetDevice")(ctypes.c_int(int(index))), "cudaSetDevice")


def get_device() -> int:
    idx = ctypes.c_int(-1)
    check(_fn("cudaGetDevice")(ctypes.byref(idx)), "cudaGetDevice")
    return int(idx.value)


def runtime_version() -> int:
    """CUDA runtime version as 1000*major + 10*minor (11040 for CUDA 11.4)."""
    v = ctypes.c_int(0)
    check(_fn("cudaRuntimeGetVersion")(ctypes.byref(v)), "cudaRuntimeGetVersion")
    return int(v.value)


def driver_version() -> int:
    """CUDA driver version as 1000*major + 10*minor, or 0 when no driver is installed."""
    v = ctypes.c_int(0)
    check(_fn("cudaDriverGetVersion")(ctypes.byref(v)), "cudaDriverGetVersion")
    return int(v.value)


def mem_info() -> Tuple[int, int]:
    """(free_bytes, total_bytes) of device memory. Unified memory on Xavier."""
    free = ctypes.c_size_t(0)
    total = ctypes.c_size_t(0)
    check(_fn("cudaMemGetInfo")(ctypes.byref(free), ctypes.byref(total)), "cudaMemGetInfo")
    return int(free.value), int(total.value)


def malloc(nbytes: int) -> int:
    ptr = ctypes.c_void_p()
    check(_fn("cudaMalloc")(ctypes.byref(ptr), ctypes.c_size_t(int(nbytes))), "cudaMalloc")
    return int(ptr.value or 0)


def free(ptr: int) -> None:
    if ptr:
        check(_fn("cudaFree")(ctypes.c_void_p(ptr)), "cudaFree")


def host_alloc(nbytes: int, flags: int = cudaHostAllocDefault) -> int:
    """Page-locked host allocation. Required for a genuinely async cudaMemcpyAsync."""
    ptr = ctypes.c_void_p()
    check(
        _fn("cudaHostAlloc")(ctypes.byref(ptr), ctypes.c_size_t(int(nbytes)), ctypes.c_uint(int(flags))),
        "cudaHostAlloc",
    )
    return int(ptr.value or 0)


def free_host(ptr: int) -> None:
    if ptr:
        check(_fn("cudaFreeHost")(ctypes.c_void_p(ptr)), "cudaFreeHost")


def host_get_device_pointer(host_ptr: int) -> int:
    """Device-visible alias of a cudaHostAllocMapped allocation (zero-copy on Tegra)."""
    ptr = ctypes.c_void_p()
    check(
        _fn("cudaHostGetDevicePointer")(ctypes.byref(ptr), ctypes.c_void_p(host_ptr), ctypes.c_uint(0)),
        "cudaHostGetDevicePointer",
    )
    return int(ptr.value or 0)


def stream_create() -> int:
    s = ctypes.c_void_p()
    check(_fn("cudaStreamCreate")(ctypes.byref(s)), "cudaStreamCreate")
    return int(s.value or 0)


def stream_destroy(s: int) -> None:
    if s:
        check(_fn("cudaStreamDestroy")(ctypes.c_void_p(s)), "cudaStreamDestroy")


def stream_synchronize(s: int) -> None:
    """Block until every operation queued on the stream has completed."""
    check(_fn("cudaStreamSynchronize")(ctypes.c_void_p(s)), "cudaStreamSynchronize")


def stream_query(s: int) -> int:
    """CUDA_SUCCESS when the stream is idle, CUDA_ERROR_NOT_READY while work is pending."""
    return int(_fn("cudaStreamQuery")(ctypes.c_void_p(s)))


def event_create() -> int:
    e = ctypes.c_void_p()
    check(_fn("cudaEventCreate")(ctypes.byref(e)), "cudaEventCreate")
    return int(e.value or 0)


def event_record(event: int, stream: int) -> None:
    check(
        _fn("cudaEventRecord")(ctypes.c_void_p(event), ctypes.c_void_p(stream)),
        "cudaEventRecord",
    )


def event_query(event: int) -> int:
    return int(_fn("cudaEventQuery")(ctypes.c_void_p(event)))


def event_synchronize(event: int) -> None:
    check(_fn("cudaEventSynchronize")(ctypes.c_void_p(event)), "cudaEventSynchronize")


def event_destroy(event: int) -> None:
    if event:
        check(_fn("cudaEventDestroy")(ctypes.c_void_p(event)), "cudaEventDestroy")


def memcpy_async(dst: int, src: int, nbytes: int, kind: int, stream: int) -> None:
    check(
        _fn("cudaMemcpyAsync")(
            ctypes.c_void_p(dst),
            ctypes.c_void_p(src),
            ctypes.c_size_t(int(nbytes)),
            ctypes.c_int(int(kind)),
            ctypes.c_void_p(stream),
        ),
        "cudaMemcpyAsync",
    )


def clear_last_error() -> int:
    """Consume and return any sticky per-thread error so the next check is meaningful."""
    return int(_fn("cudaGetLastError")())
