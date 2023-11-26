from dataclasses import dataclass
from gpsextraction.dimensions import Dimension
from gpsextraction.timeunits import Timeunit, timeunits
import subprocess
import json
from typing import Optional
from array import array

@dataclass(frozen=True)
class MetaMeta:
    stream: int
    frame_count: int
    timebase: int
    frame_duration: int


@dataclass(frozen=True)
class VideoMeta:
    stream: int
    dimension: Dimension
    duration: Timeunit


@dataclass(frozen=True)
class AudioMeta:
    stream: int

@dataclass(frozen=True)
class StreamInfo:
    audio: Optional[AudioMeta]
    video: VideoMeta
    meta: MetaMeta

def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)

def invoke(cmd, **kwargs):
    try:
        return run(cmd, **kwargs, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise IOError(f"Error: {cmd}\n stdout: {e.stdout}\n stderr: {e.stderr}")

def find_frame_duration(filepath, data_stream_number, invoke=invoke):
    ffprobe_output = str(invoke(
        ["ffprobe",
         "-hide_banner",
         "-print_format", "json",
         "-show_packets",
         "-select_streams", str(data_stream_number),
         "-read_intervals", "%+#1",
         filepath]
    ).stdout)

    ffprobe_packet_data = json.loads(ffprobe_output)
    packet = ffprobe_packet_data["packets"][0]

    duration = int(packet["duration"])

    return duration

def find_number_frames(filepath, invoke=invoke):
    ffprobe_output = str(invoke(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
            "stream=nb_read_packets", "-of", "csv=p=0",
    filepath
        ]
    ).stdout)

    ffprobe_packet_data = json.loads(ffprobe_output)

    print(ffprobe_packet_data)

def find_streams(filepath, invoke=invoke, find_frame_duration=find_frame_duration) -> StreamInfo:
    ffprobe_output = str(invoke(["ffprobe", "-hide_banner", "-print_format", "json", "-show_streams", filepath]).stdout)

    ffprobe_json = json.loads(ffprobe_output)

    video_selector = lambda s: s["codec_type"] == "video"
    audio_selector = lambda s: s["codec_type"] == "audio"
    data_selector = lambda s: s["codec_type"] == "data" and s["codec_tag_string"] == "gpmd"

    def first_and_only(what, l, p):
        matches = list(filter(p, l))
        if not matches:
            raise IOError(f"Unable to find {what} in ffprobe output")
        if len(matches) > 1:
            raise IOError(f"Multiple matching streams for {what} in ffprobe output")
        return matches[0]

    def only_if_present(what, l, p):
        matches = list(filter(p, l))
        if matches:
            return first_and_only(what, l, p)

    streams = ffprobe_json["streams"]
    video = first_and_only("video stream", streams, video_selector)

    video_meta = VideoMeta(
        stream=int(video["index"]),
        dimension=Dimension(video["width"], video["height"]),
        duration=timeunits(seconds=float(video["duration"]))
    )

    audio = only_if_present("audio stream", streams, audio_selector)
    audio_meta = None
    if audio:
        audio_meta = AudioMeta(stream=int(audio["index"]))

    meta = first_and_only("metadata stream", streams, data_selector)

    data_stream_number = int(meta["index"])

    meta_meta = MetaMeta(
        stream=data_stream_number,
        frame_count=int(meta["nb_frames"]),
        timebase=int(meta["time_base"].split("/")[1]),
        frame_duration=find_frame_duration(filepath, data_stream_number, invoke)
    )

    return StreamInfo(audio=audio_meta, video=video_meta, meta=meta_meta)

def load_gpmd_from(filepath):
    track = find_streams(filepath).meta.stream
    if track:
        cmd = ["ffmpeg", "-hide_banner", '-y', '-i', filepath, '-codec', 'copy', '-map', '0:%d' % track, '-f',
               'rawvideo', "-"]
        result = run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0:
            raise IOError(f"ffmpeg failed code: {result.returncode} : {result.stderr.decode('utf-8')}")
        arr = array("b")
        arr.frombytes(result.stdout)
        return arr
