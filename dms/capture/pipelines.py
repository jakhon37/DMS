"""GStreamer pipeline descriptions for each source type.

``Gst.parse_launch`` takes a full pipeline *description*, so any config value
interpolated into it is parsed as pipeline syntax: a clip path containing a space
produces a malformed pipeline and a GLib parse error at startup, and a path
containing ``!`` would rewire the graph. Every value that comes from YAML or the
command line therefore goes through :func:`gst_quote`, which rejects the characters
that cannot be represented and quotes the rest.

Units: widths/heights are pixels, ``fps`` is frames per second.

Failure behaviour: :func:`build_pipeline` raises ``ValueError`` for an unknown source
type or an unrepresentable path/device. :func:`validate_source` is the stricter,
side-effecting check (does the clip exist?) that the app should run at startup so the
operator gets a config error instead of a GStreamer traceback.
"""
from __future__ import annotations

import os
import re
from typing import List

from dms.config.schema import AppConfig

#: Characters that cannot appear inside a quoted gst property value.
_FORBIDDEN = re.compile(r'[\x00-\x1f"\\\x7f]')

_V4L2_DEVICE = re.compile(r"^/dev/video\d+$")


def gst_quote(value: str, what: str = "value") -> str:
    """Return ``value`` as a quoted gst-launch property value.

    Raises ``ValueError`` when the value is empty or contains a character that would
    break out of the quoting (a quote, a backslash, or any control character).
    """
    if value is None or value == "":
        raise ValueError("%s must not be empty" % what)
    text = str(value)
    bad = _FORBIDDEN.search(text)
    if bad is not None:
        raise ValueError(
            "%s contains an unsupported character %r at index %d: %r"
            % (what, bad.group(0), bad.start(), text)
        )
    return '"%s"' % text


def _appsink_tail(cfg: AppConfig) -> str:
    """The common nvvidconv -> BGRx -> appsink tail shared by every source."""
    w, h = cfg.source.width, cfg.source.height
    # File replay must not drop: the decoder would EOS during TRT load / infer.
    # Live sources drop so the latest frame wins.
    drop = "false" if cfg.source.type == "file" else "true"
    max_buffers = 4 if cfg.source.type == "file" else 1
    return (
        "nvvidconv ! video/x-raw,width=%d,height=%d,format=BGRx ! "
        "appsink name=full emit-signals=false max-buffers=%d drop=%s sync=false"
        % (w, h, max_buffers, drop)
    )


def build_pipeline(cfg: AppConfig) -> str:
    """Single 720p BGRx appsink. nvvidconv cannot emit BGR on this Jetson."""
    w, h = cfg.source.width, cfg.source.height
    src = cfg.source.type
    if w <= 0 or h <= 0:
        raise ValueError("source.width/height must be positive, got %sx%s" % (w, h))
    if cfg.source.fps <= 0:
        raise ValueError("source.fps must be positive, got %s" % cfg.source.fps)
    caps = _appsink_tail(cfg)
    if src == "file":
        return "filesrc location=%s ! qtdemux ! h264parse ! nvv4l2decoder ! %s" % (
            gst_quote(cfg.source.path, "source.path"),
            caps,
        )
    if src == "test":
        return (
            "videotestsrc is-live=true ! "
            "video/x-raw,width=%d,height=%d,format=NV12,framerate=%d/1 ! %s"
            % (w, h, cfg.source.fps, caps)
        )
    if src == "usb":
        return (
            "v4l2src device=%s io-mode=2 ! "
            "image/jpeg,width=%d,height=%d,framerate=%d/1 ! "
            "nvv4l2decoder mjpeg=1 ! video/x-raw(memory:NVMM) ! %s"
            % (gst_quote(cfg.source.device, "source.device"), w, h, cfg.source.fps, caps)
        )
    if src == "csi":
        sensor_id = int(cfg.source.sensor_id)
        if sensor_id < 0:
            raise ValueError("source.sensor_id must be >= 0, got %s" % sensor_id)
        return (
            "nvarguscamerasrc sensor-id=%d wbmode=0 saturation=1.0 ! "
            "video/x-raw(memory:NVMM),width=%d,height=%d,format=NV12,framerate=%d/1 ! "
            "%s" % (sensor_id, w, h, cfg.source.fps, caps)
        )
    raise ValueError("unknown source.type %s" % src)


def validate_source(cfg: AppConfig) -> List[str]:
    """Check the configured source can plausibly be opened.

    Returns a list of non-fatal warnings (for example a USB device node that is not
    present right now - the supervisor will retry it). Raises ``ValueError`` for
    anything that can never work, so startup fails with a config message rather than
    a GLib parse error.
    """
    warnings: List[str] = []
    src = cfg.source.type
    if src == "file":
        gst_quote(cfg.source.path, "source.path")
        if not os.path.isfile(cfg.source.path):
            raise ValueError("source.path is not a file: %s" % cfg.source.path)
    elif src == "usb":
        device = cfg.source.device
        gst_quote(device, "source.device")
        if not _V4L2_DEVICE.match(device):
            raise ValueError("source.device must look like /dev/videoN, got %r" % device)
        if not os.path.exists(device):
            warnings.append("source.device %s does not exist yet" % device)
    elif src == "csi":
        if int(cfg.source.sensor_id) < 0:
            raise ValueError("source.sensor_id must be >= 0")
    elif src != "test":
        raise ValueError("unknown source.type %s" % src)
    if cfg.source.width <= 0 or cfg.source.height <= 0:
        raise ValueError("source.width/height must be positive")
    if cfg.source.fps <= 0:
        raise ValueError("source.fps must be positive")
    # build_pipeline is the real syntax check; run it once so a bad value is caught here.
    build_pipeline(cfg)
    return warnings
