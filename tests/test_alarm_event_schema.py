import unittest

from alarm_event_schema import EVENT_FILENAMES


class AlarmEventSchemaTests(unittest.TestCase):
    def test_fixed_event_filenames(self):
        self.assertEqual(
            EVENT_FILENAMES,
            (
                "alarm.jpg",
                "alarm_clip.mp4",
                "metadata.json",
                "original.jpg",
                "risk_crop.jpg",
            ),
        )


if __name__ == "__main__":
    unittest.main()
