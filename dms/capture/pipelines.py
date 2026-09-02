from __future__ import annotations

from dms.config.schema import AppConfig


def build_pipeline(cfg: AppConfig) -> str:
    """Single 720p BGRx appsink. nvvidconv cannot emit BGR on this Jetson."""
    w, h = cfg.source.width, cfg.source.height
    src = cfg.source.type
    # File replay must not drop: decoder would EOS during TRT load / infer.
    # Live sources drop so the latest frame wins.
    drop = "false" if src == "file" else "true"
    max_buffers = 4 if src == "file" else 1
    caps = (
        "nvvidconv ! video/x-raw,width=%d,height=%d,format=BGRx ! "
        "appsink name=full emit-signals=false max-buffers=%d drop=%s sync=false"
        % (w, h, max_buffers, drop)
    )
    if src == "file":
        return (
            "filesrc location=%s ! qtdemux ! h264parse ! nvv4l2decoder ! %s"
            % (cfg.source.path, caps)
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
            % (cfg.source.device, w, h, cfg.source.fps, caps)
        )
    if src == "csi":
        return (
            "nvarguscamerasrc sensor-id=%d wbmode=0 saturation=1.0 ! "
            "video/x-raw(memory:NVMM),width=%d,height=%d,format=NV12,framerate=%d/1 ! "
            "%s" % (cfg.source.sensor_id, w, h, cfg.source.fps, caps)
        )
    raise ValueError("unknown source.type %s" % src)
