"""Shared on-disk schema for alarm events produced by every detector.

All present and future algorithms should write their evidence through the
common alarm event store and use these fixed names.  The Web alarm gallery
uses ``alarm.jpg`` as the only display image for an event.
"""

EVENT_DIR_PREFIX = "event"
ALARM_IMAGE_FILENAME = "alarm.jpg"
ALARM_CLIP_FILENAME = "alarm_clip.mp4"
METADATA_FILENAME = "metadata.json"
ORIGINAL_IMAGE_FILENAME = "original.jpg"
RISK_CROP_FILENAME = "risk_crop.jpg"

EVENT_FILENAMES = (
    ALARM_IMAGE_FILENAME,
    ALARM_CLIP_FILENAME,
    METADATA_FILENAME,
    ORIGINAL_IMAGE_FILENAME,
    RISK_CROP_FILENAME,
)
