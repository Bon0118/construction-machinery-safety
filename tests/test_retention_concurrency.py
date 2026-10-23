import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from server import AlarmImageStore


class RetentionConcurrencyTests(unittest.TestCase):
    def test_removed_entry_during_scan_does_not_stop_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = Mock()
            entry.name = 'event_removed_by_another_worker'
            entry.is_dir.return_value = True
            entry.stat.side_effect = FileNotFoundError()
            store = AlarmImageStore.__new__(AlarmImageStore)
            store.output_dir = root
            with patch('pathlib.Path.iterdir', return_value=iter([entry])), patch('config.ALARM_RETENTION_MODE', 'both'):
                store.enforce_retention()
