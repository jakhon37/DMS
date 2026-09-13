"""TensorRT 8.5 execution wrapper: static engines, pinned IO, explicit lifecycle.

Contract
--------
* Engines must be fully static (no -1 in any binding shape). A dynamic engine is
  refused at load time rather than silently frozen to batch 1.
* Every host staging buffer is page-locked (``cudaHostAlloc``) so ``cudaMemcpyAsync``
  is a real DMA instead of a staged synchronous copy through a driver bounce buffer.
* At most one work item may be in flight per engine. ``submit`` raises
  :class:`EngineBusyError` on a second submit, because the host buffers are shared and
  a second submit would overwrite an input whose H2D is still queued.
* ``wait``/``infer`` synchronise on a CUDA event recorded after the D2H copies, so a
  returned array is genuinely complete. There is no hidden sync anywhere else.

Units: ``timeout_ms`` is milliseconds of wall time, ``nbytes`` are bytes.

Failure behaviour: load errors raise :class:`EngineLoadError`, enqueue/execute errors
raise :class:`TrtExecutionError`, a stalled stream raises :class:`DlaHangError`
(a ``TimeoutError``) after ``timeout_ms``. A partially constructed engine frees
everything it managed to allocate before re-raising, so a retry loop cannot leak.
"""
from __future__ import annotations

import ctypes
import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from dms.runtime import cudart

log = logging.getLogger("dms.runtime.trt")

_TRT_DTYPE = None


class EngineLoadError(RuntimeError):
    """The engine plan could not be read, deserialised, or bound."""


class TrtExecutionError(RuntimeError):
    """enqueue/execute returned failure, or the engine was used after close()."""


class EngineBusyError(RuntimeError):
    """submit() was called while a previous work item was still outstanding."""


class DlaHangError(TimeoutError):
    """The completion event did not fire within the requested timeout."""


def _dtype_map():
    global _TRT_DTYPE
    if _TRT_DTYPE is not None:
        return _TRT_DTYPE
    import tensorrt as trt

    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL: np.bool_,
    }
    if hasattr(trt.DataType, "INT64"):
        mapping[trt.DataType.INT64] = np.int64
    if hasattr(trt.DataType, "UINT8"):
        mapping[trt.DataType.UINT8] = np.uint8
    _TRT_DTYPE = mapping
    return mapping


class PinnedBuffer:
    """A page-locked host allocation exposed as a numpy array of the requested shape.

    The numpy view aliases driver-owned memory. ``free`` drops the view before the
    allocation goes away; using the array afterwards is a use-after-free, so callers
    must not retain it past ``TrtEngine.close()``.
    """

    __slots__ = ("ptr", "nbytes", "array")

    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype) -> None:
        nbytes = int(np.prod(shape)) * int(np.dtype(dtype).itemsize)
        if nbytes <= 0:
            raise EngineLoadError("refusing zero-sized binding shape %s" % (shape,))
        self.nbytes = nbytes
        self.ptr = cudart.host_alloc(nbytes)
        raw = (ctypes.c_ubyte * nbytes).from_address(self.ptr)
        self.array = np.frombuffer(raw, dtype=dtype).reshape(shape)

    def free(self) -> None:
        if self.ptr:
            self.array = None  # type: ignore[assignment]
            cudart.free_host(self.ptr)
            self.ptr = 0


@dataclass
class Binding:
    name: str
    is_input: bool
    dtype: np.dtype
    shape: Tuple[int, ...]
    nbytes: int
    host: np.ndarray
    device: int
    pinned: Optional[PinnedBuffer] = None


@dataclass
class WorkItem:
    """One outstanding inference. Owned by the engine that produced it.

    ``outputs`` are the owning engine pinned output buffers; they are only valid
    between completion and the next ``submit`` on that engine.
    """

    engine_id: str
    stream: int
    event: int
    outputs: Dict[str, np.ndarray]
    owner: Optional["TrtEngine"] = None


class TrtEngine:
    """Own one deserialised engine, its execution context, stream and IO buffers."""

    def __init__(self, engine_path: str, *, dla_core: Optional[int] = None, device: int = 0) -> None:
        import tensorrt as trt

        self.engine_path = engine_path
        self.dla_core = dla_core
        self.device_index = int(device)
        self._closed = False
        self._lock = threading.RLock()
        self._inflight: Optional[WorkItem] = None
        self._stream = 0
        self._bindings: Dict[str, Binding] = {}
        self._inputs: List[Binding] = []
        self._outputs: List[Binding] = []
        self._order: List[str] = []
        self._device_ptr_list: List[int] = []
        self._runtime = None
        self._engine = None
        self._ctx = None
        self._use_v3 = True

        try:
            cudart.set_device(self.device_index)
            self._logger = trt.Logger(trt.Logger.WARNING)
            self._runtime = trt.Runtime(self._logger)
            if dla_core is not None:
                ncores = int(getattr(self._runtime, "num_DLA_cores", 0) or 0)
                if int(dla_core) < 0 or int(dla_core) >= ncores:
                    raise EngineLoadError(
                        "ENGINE_FAIL dla_core=%s but this board reports %d DLA cores"
                        % (dla_core, ncores)
                    )
                self._runtime.DLA_core = int(dla_core)
            try:
                with open(engine_path, "rb") as f:
                    blob = f.read()
            except OSError as exc:
                raise EngineLoadError("ENGINE_FAIL cannot read %s: %s" % (engine_path, exc))
            engine = self._runtime.deserialize_cuda_engine(blob)
            del blob
            if engine is None:
                raise EngineLoadError(
                    "ENGINE_FAIL failed to deserialize %s (trt=%s) - the plan is not valid "
                    "for this TensorRT/cuDNN/driver combination" % (engine_path, trt.__version__)
                )
            self._engine = engine
            self._ctx = engine.create_execution_context()
            if self._ctx is None:
                raise EngineLoadError("ENGINE_FAIL create_execution_context failed for %s" % engine_path)
            self._use_v3 = hasattr(self._ctx, "execute_async_v3") and hasattr(engine, "num_io_tensors")
            self._stream = cudart.stream_create()
            self._bind(trt, engine)
        except BaseException:
            self.close()
            raise

        log.info(
            "loaded %s tensors=%s v3=%s dla=%s",
            engine_path,
            list(self._bindings),
            self._use_v3,
            dla_core,
        )

    # ---------------------------------------------------------------- binding

    def _bind(self, trt, engine) -> None:
        dmap = _dtype_map()
        if hasattr(engine, "num_io_tensors"):
            names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            modes = {n: engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in names}
            dtypes = {n: engine.get_tensor_dtype(n) for n in names}
            shapes = {n: engine.get_tensor_shape(n) for n in names}
        else:  # pragma: no cover - TRT < 8.5 shim, not this board
            names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
            modes = {n: engine.binding_is_input(n) for n in names}
            dtypes = {n: engine.get_binding_dtype(n) for n in names}
            shapes = {n: engine.get_binding_shape(n) for n in names}

        self._order = list(names)
        for name in names:
            trt_dt = dtypes[name]
            if trt_dt not in dmap:
                raise EngineLoadError("ENGINE_FAIL unsupported dtype %s on %s" % (trt_dt, name))
            dt = np.dtype(dmap[trt_dt])
            shape = tuple(int(x) for x in shapes[name])
            if any(s < 0 for s in shape):
                raise EngineLoadError(
                    "ENGINE_FAIL dynamic shape %s on %s - this runtime requires fully static engines"
                    % (shape, name)
                )
            pinned = PinnedBuffer(shape, dt)
            device = cudart.malloc(pinned.nbytes)
            self._bindings[name] = Binding(
                name=name,
                is_input=bool(modes[name]),
                dtype=dt,
                shape=shape,
                nbytes=pinned.nbytes,
                host=pinned.array,
                device=device,
                pinned=pinned,
            )
            if self._use_v3:
                self._ctx.set_tensor_address(name, device)
        self._inputs = [b for b in self._bindings.values() if b.is_input]
        self._outputs = [b for b in self._bindings.values() if not b.is_input]
        if not self._inputs or not self._outputs:
            raise EngineLoadError(
                "ENGINE_FAIL %s has %d inputs / %d outputs"
                % (self.engine_path, len(self._inputs), len(self._outputs))
            )
        self._device_ptr_list = [self._bindings[n].device for n in self._order]

    # ------------------------------------------------------------- properties

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._inputs[0].shape

    @property
    def input_name(self) -> str:
        return self._inputs[0].name

    @property
    def input_names(self) -> List[str]:
        return [b.name for b in self._inputs]

    @property
    def output_names(self) -> List[str]:
        return [b.name for b in self._outputs]

    def shape_of(self, name: str) -> Tuple[int, ...]:
        return self._bindings[name].shape

    @property
    def device_memory_bytes(self) -> int:
        """Execution-context scratch in bytes, on top of the deserialised weights."""
        if self._engine is None:
            return 0
        return int(getattr(self._engine, "device_memory_size", 0) or 0)

    @property
    def io_bytes(self) -> int:
        """Total bytes of pinned host (and mirrored device) IO staging for this engine."""
        return sum(b.nbytes for b in self._bindings.values())

    # ---------------------------------------------------------------- execute

    def _require_open(self) -> None:
        if self._closed:
            raise TrtExecutionError("engine %s is closed" % self.engine_path)

    def submit(self, inputs: Dict[str, np.ndarray]) -> WorkItem:
        """Queue H2D, enqueue, D2H and a completion event. One item in flight per engine."""
        with self._lock:
            self._require_open()
            if self._inflight is not None:
                raise EngineBusyError(
                    "engine %s already has an outstanding work item; wait() it first"
                    % self.engine_path
                )
            for b in self._inputs:
                arr = inputs.get(b.name)
                if arr is None:
                    if len(self._inputs) == 1 and len(inputs) == 1:
                        arr = next(iter(inputs.values()))
                    else:
                        raise KeyError("missing input %s for %s" % (b.name, self.engine_path))
                arr = np.asarray(arr)
                if arr.shape != b.shape:
                    raise ValueError("input %s expected %s got %s" % (b.name, b.shape, arr.shape))
                # copyto casts straight into the pinned staging buffer, no allocation.
                np.copyto(b.host, arr, casting="unsafe")
                cudart.memcpy_async(
                    b.device,
                    int(b.pinned.ptr),
                    b.nbytes,
                    cudart.cudaMemcpyHostToDevice,
                    self._stream,
                )
            try:
                if self._use_v3:
                    ok = self._ctx.execute_async_v3(self._stream)
                else:  # pragma: no cover - TRT < 8.5 shim
                    ok = self._ctx.execute_async_v2(self._device_ptr_list, self._stream)
            except Exception as exc:
                self._drain_stream()
                raise TrtExecutionError("enqueue failed for %s: %s" % (self.engine_path, exc))
            if not ok:
                self._drain_stream()
                raise TrtExecutionError(
                    "execute_async_v%d returned false for %s"
                    % (3 if self._use_v3 else 2, self.engine_path)
                )
            for b in self._outputs:
                cudart.memcpy_async(
                    int(b.pinned.ptr),
                    b.device,
                    b.nbytes,
                    cudart.cudaMemcpyDeviceToHost,
                    self._stream,
                )
            event = cudart.event_create()
            cudart.event_record(event, self._stream)
            item = WorkItem(
                engine_id=self.engine_path,
                stream=self._stream,
                event=event,
                outputs={b.name: b.host for b in self._outputs},
                owner=self,
            )
            self._inflight = item
            return item

    def wait(self, item: WorkItem, timeout_ms: float = 500.0, copy: bool = True) -> Dict[str, np.ndarray]:
        """Block until ``item`` completes and return its outputs.

        copy=True  (default) returns freshly allocated arrays the caller may keep.
        copy=False returns the engine pinned output buffers directly - zero
        allocation, but only valid until the next submit on this engine.
        """
        with self._lock:
            self._require_open()
            if item is not self._inflight:
                raise TrtExecutionError("work item does not belong to %s" % self.engine_path)
            try:
                _poll_event(item, timeout_ms)
            finally:
                self._inflight = None
            if copy:
                return {k: np.array(v, copy=True) for k, v in item.outputs.items()}
            return dict(item.outputs)

    @staticmethod
    def wait_all(items: List[WorkItem], timeout_ms: float = 500.0) -> List[Dict[str, np.ndarray]]:
        """Wait on one item per engine (the two-engine overlap case). Always copies.

        ``timeout_ms`` is the budget for the whole batch, not per item. Each owning
        engine in-flight slot is released as its item completes, so the caller may
        submit again immediately afterwards.
        """
        deadline = time.monotonic() + max(0.0, timeout_ms) / 1000.0
        out: List[Dict[str, np.ndarray]] = []
        for it in items:
            remaining_ms = max(0.0, (deadline - time.monotonic()) * 1000.0)
            try:
                _poll_event(it, remaining_ms)
            finally:
                _release(it)
            out.append({k: np.array(v, copy=True) for k, v in it.outputs.items()})
        return out

    def infer(
        self, inputs: Dict[str, np.ndarray], timeout_ms: float = 500.0, copy: bool = True
    ) -> Dict[str, np.ndarray]:
        """Synchronous single inference. See ``wait`` for the meaning of ``copy``."""
        return self.wait(self.submit(inputs), timeout_ms=timeout_ms, copy=copy)

    # --------------------------------------------------------------- teardown

    def _drain_stream(self) -> None:
        if getattr(self, "_stream", 0):
            try:
                cudart.stream_synchronize(self._stream)
            except cudart.CudaError as exc:
                log.error("stream sync failed for %s: %s", self.engine_path, exc)
        try:
            cudart.clear_last_error()
        except cudart.CudaError as exc:
            log.debug("cudaGetLastError unavailable: %s", exc)

    def close(self) -> None:
        """Idempotent, ordered teardown. Safe to call twice and from ``__del__``."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        item = getattr(self, "_inflight", None)
        self._drain_stream()
        if item is not None and item.event:
            try:
                cudart.event_destroy(item.event)
            except cudart.CudaError as exc:
                log.error("event destroy failed for %s: %s", self.engine_path, exc)
            item.event = 0
        self._inflight = None
        # Context and engine go before the memory the context was pointed at.
        self._ctx = None
        self._engine = None
        self._runtime = None
        for b in list(getattr(self, "_bindings", {}).values()):
            try:
                cudart.free(b.device)
            except cudart.CudaError as exc:
                log.error("cudaFree failed for %s.%s: %s", self.engine_path, b.name, exc)
            b.device = 0
            b.host = None  # type: ignore[assignment]
            if b.pinned is not None:
                try:
                    b.pinned.free()
                except cudart.CudaError as exc:
                    log.error("cudaFreeHost failed for %s.%s: %s", self.engine_path, b.name, exc)
                b.pinned = None
        self._bindings = {}
        self._inputs = []
        self._outputs = []
        self._device_ptr_list = []
        if getattr(self, "_stream", 0):
            try:
                cudart.stream_destroy(self._stream)
            except cudart.CudaError as exc:
                log.error("cudaStreamDestroy failed for %s: %s", self.engine_path, exc)
            self._stream = 0

    def __enter__(self) -> "TrtEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown ordering
        try:
            self.close()
        except Exception:
            pass


def _release(item: WorkItem) -> None:
    """Clear the owning engine in-flight slot so it accepts a new submit."""
    owner = item.owner
    if owner is None:
        return
    with owner._lock:
        if owner._inflight is item:
            owner._inflight = None


def _poll_event(item: WorkItem, timeout_ms: float) -> None:
    """Spin-with-yield on the completion event; raise DlaHangError past the deadline."""
    if not item.event:
        return
    deadline = time.monotonic() + max(0.0, timeout_ms) / 1000.0
    sleep_s = 0.0002
    while True:
        err = cudart.event_query(item.event)
        if err == cudart.CUDA_SUCCESS:
            cudart.event_destroy(item.event)
            item.event = 0
            return
        if err != cudart.CUDA_ERROR_NOT_READY:
            cudart.event_destroy(item.event)
            item.event = 0
            cudart.check(err, "cudaEventQuery")
        if time.monotonic() > deadline:
            raise DlaHangError(
                "engine %s did not complete within %.0f ms" % (item.engine_id, timeout_ms)
            )
        time.sleep(sleep_s)
        if sleep_s < 0.002:
            sleep_s *= 2.0
