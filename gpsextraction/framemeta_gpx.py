import datetime
from datetime import timedelta

import gpxpy

from gpsextraction.entry import Entry
from gpsextraction.framemeta import FrameMeta
from gpsextraction.timeseries import Timeseries


def framemeta_to_gpx(fm: FrameMeta, step: timedelta = timedelta(seconds=0)):
    gpx = gpxpy.gpx.GPX()

    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    last_dt = datetime.datetime(year=1900, month=1, day=1,tzinfo=datetime.timezone.utc)

    for entry in fm.items():
        entry_dt = entry.dt

        if entry_dt > last_dt + step:
            last_dt = entry_dt

            gpx_segment.points.append(
                gpxpy.gpx.GPXTrackPoint(
                    time=entry_dt,
                    latitude=entry.point.lat,
                    longitude=entry.point.lon,
                    elevation=entry.alt.to("m").magnitude)
            )

    return gpx


def merge_gpx_with_gopro(gpx_timeseries: Timeseries, gopro_framemeta: FrameMeta):
    # pretty hacky merge
    # assume that the GPS timestamps in gopro are basically correct.
    # overwrite the location with the interpolated location from GPX
    # copy over any other attributes that are there
    # if no entry for that time exists in the GPX file, just ignore.

    if gpx_timeseries.min > gopro_framemeta.get(gopro_framemeta.max).dt:
        raise ValueError("GPX file seems to start after the end of the video")

    if gpx_timeseries.max < gopro_framemeta.get(gopro_framemeta.min).dt:
        raise ValueError("GPX file seems to finish before the start of the video")

    def processor(gopro_entry: Entry):
        try:
            gpx_entry = gpx_timeseries.get(gopro_entry.dt)
            return gpx_entry.items
        except ValueError:
            pass

    gopro_framemeta.process(processor)