import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from server import Detection, MachineryHazardDetector


class AlarmLabelTests(unittest.TestCase):
    def test_chinese_class_label_uses_pillow_instead_of_opencv_text(self):
        detector = MachineryHazardDetector.__new__(MachineryHazardDetector)
        image = np.zeros((120, 320, 3), dtype=np.uint8)
        detection = Detection(
            class_id=6,
            class_name="挖掘机",
            confidence=0.93,
            box=(20, 45, 180, 110),
        )

        with patch.object(config, "DRAW_LABELS", True), patch("server.cv2.putText") as put_text:
            detector._draw_box(image, detection, (0, 0, 255), 2)

        put_text.assert_not_called()
        self.assertGreater(int(np.count_nonzero(image)), 0)

    def test_long_chinese_label_keeps_true_type_font(self):
        detector = MachineryHazardDetector.__new__(MachineryHazardDetector)
        draw = ImageDraw.Draw(Image.new("RGB", (40, 40)))

        font = detector._fit_alarm_font(draw, "机械设备" * 20, 12, 20)

        self.assertIsInstance(font, ImageFont.FreeTypeFont)


if __name__ == "__main__":
    unittest.main()
