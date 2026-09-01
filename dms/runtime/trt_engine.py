from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from dms.runtime import cudart

log = logging.getLogger("dms.runtime.trt")


def _wait_item(item: WorkItem, timeout_ms: float) -> Dict[str, np.ndarray]:
    deadline = time.monotonic() + timeout_ms / 1000.0
    while True:
        err = cudart.event_query(item.event)
        if err == cudart.CUDA_SUCCESS:
            cudart.event_destroy(item.event)
            item.event = 0
            return {k: np.copy(v) for k, v in item.outputs.items()}
        if err != cudart.CUDA_ERROR_NOT_READY:
            cudart.check(err, "cudaEventQuery")
        if time.monotonic() > deadline:
            raise DlaHangError("engine %s hung > %.0f ms" % (item.engine_id, timeout_ms))
        time.sleep(0.0005)

_TRT_DTYPE = None


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
    _TRT_DTYPE = mapping
    return mapping


class DlaHangError(TimeoutError):
    pass


@dataclass
class Binding:
    name: str
    is_input: bool
    dtype: np.dtype
    shape: Tuple[int, ...]
    nbytes: int
    host: np.ndarray
    device: int


@dataclass
class WorkItem:
    engine_id: str
    stream: int
    event: int
    outputs: Dict[str, np.ndarray]
    d2h_done: bool = False


class TrtEngine:
    def __init__(self, engine_path: str, *, dla_core: Optional[int] = None) -> None:
        import tensorrt as trt

        self.engine_path = engine_path
        self.dla_core = dla_core
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(engine_path, "rb") as f:
            blob = f.read()
        engine = self._runtime.deserialize_cuda_engine(blob)
        if engine is None:
            raise RuntimeError("failed to deserialize %s" % engine_path)
        self._engine = engine
        self._ctx = engine.create_execution_context()
        self._stream = cudart.stream_create()
        self._bindings: Dict[str, Binding] = {}
        dmap = _dtype_map()
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dt = dmap[engine.get_tensor_dtype(name)]
            shape = tuple(int(x) for x in engine.get_tensor_shape(name))
            if any(s < 0 for s in shape):
                raise RuntimeError("dynamic shape on %s — v1 requires static engines" % name)
            nbytes = int(np.prod(shape) * np.dtype(dt).itemsize)
            host = np.empty(shape, dtype=dt)
            device = cudart.malloc(nbytes)
            self._bindings[name] = Binding(
                name=name,
                is_input=is_input,
                dtype=np.dtype(dt),
                shape=shape,
                nbytes=nbytes,
                host=host,
                device=device,
            )
            self._ctx.set_tensor_address(name, device)
        self._inputs = [b for b in self._bindings.values() if b.is_input]
        self._outputs = [b for b in self._bindings.values() if not b.is_input]
        log.info("loaded %s tensors=%s", engine_path, list(self._bindings))

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._inputs[0].shape

    def submit(self, inputs: Dict[str, np.ndarray]) -> WorkItem:
        for b in self._inputs:
            arr = inputs.get(b.name)
            if arr is None:
                if len(self._inputs) == 1 and len(inputs) == 1:
                    arr = next(iter(inputs.values()))
                else:
                    raise KeyError("missing input %s" % b.name)
            arr = np.ascontiguousarray(arr, dtype=b.dtype)
            if arr.shape != b.shape:
                raise ValueError("input %s expected %s got %s" % (b.name, b.shape, arr.shape))
            np.copyto(b.host, arr)
            cudart.memcpy_async(
                b.device,
                int(b.host.ctypes.data),
                b.nbytes,
                cudart.cudaMemcpyHostToDevice,
                self._stream,
            )
        ok = self._ctx.execute_async_v3(self._stream)
        if not ok:
            raise RuntimeError("execute_async_v3 failed for %s" % self.engine_path)
        for b in self._outputs:
            cudart.memcpy_async(
                int(b.host.ctypes.data),
                b.device,
                b.nbytes,
                cudart.cudaMemcpyDeviceToHost,
                self._stream,
            )
        event = cudart.event_create()
        cudart.event_record(event, self._stream)
        outs = {b.name: b.host for b in self._outputs}
        return WorkItem(engine_id=self.engine_path, stream=self._stream, event=event, outputs=outs)

    def wait(self, item: WorkItem, timeout_ms: float = 500.0) -> Dict[str, np.ndarray]:
        return _wait_item(item, timeout_ms)

    @staticmethod
    def wait_all(items: List[WorkItem], timeout_ms: float = 500.0) -> List[Dict[str, np.ndarray]]:
        return [_wait_item(it, timeout_ms) for it in items]

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.wait(self.submit(inputs))

    def close(self) -> None:
        for b in self._bindings.values():
            cudart.free(b.device)
        cudart.stream_destroy(self._stream)
        self._bindings.clear()
