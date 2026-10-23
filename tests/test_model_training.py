import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from model_training import ModelTrainingManager, TrainingValidationError


class ModelTrainingManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manager = ModelTrainingManager(self.root / "platform.db", self.root / "training")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _archive(self, names="[person, head, helmet]"):
        source = self.root / "source"
        (source / "images" / "train").mkdir(parents=True)
        (source / "images" / "val").mkdir(parents=True)
        (source / "labels" / "train").mkdir(parents=True)
        (source / "labels" / "val").mkdir(parents=True)
        (source / "data.yaml").write_text(
            f"path: .\ntrain: images/train\nval: images/val\nnames: {names}\n", encoding="utf-8"
        )
        Image.new("RGB", (80, 80), (90, 100, 110)).save(source / "images" / "train" / "train.jpg")
        Image.new("RGB", (80, 80), (110, 100, 90)).save(source / "images" / "val" / "val.jpg")
        (source / "labels" / "train" / "train.txt").write_text(
            "0 0.500000 0.500000 0.800000 0.900000\n1 0.500000 0.250000 0.200000 0.200000\n",
            encoding="utf-8",
        )
        (source / "labels" / "val" / "val.txt").write_text(
            "2 0.500000 0.250000 0.220000 0.220000\n", encoding="utf-8"
        )
        archive = self.root / "helmet_dataset.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
            for path in source.rglob("*"):
                if path.is_file():
                    stream.write(path, path.relative_to(source).as_posix())
        return archive

    def test_import_local_yolo_zip_normalizes_and_validates_dataset(self):
        dataset = self.manager.import_dataset_archive(self._archive(), "现场安全帽样本.zip")

        self.assertTrue(dataset["training_ready"])
        self.assertEqual(dataset["source"], "local_zip_upload")
        self.assertEqual(dataset["original_name"], "现场安全帽样本.zip")
        self.assertEqual(dataset["split_counts"], {"train": 1, "val": 1})
        self.assertEqual(dataset["class_counts"], {"person": 1, "head": 1, "helmet": 1})
        dataset_dir = Path(dataset["data_yaml"]).parent
        self.assertTrue((dataset_dir / "data.yaml").is_file())
        self.assertEqual(len(list((dataset_dir / "images" / "train").glob("*.jpg"))), 1)
        self.assertEqual(len(list((dataset_dir / "labels" / "val").glob("*.txt"))), 1)
        self.assertEqual(len(self.manager.list_datasets()), 1)

    def test_import_rejects_wrong_class_order(self):
        archive = self._archive("[person, helmet, head]")
        with self.assertRaisesRegex(TrainingValidationError, "类别必须"):
            self.manager.import_dataset_archive(archive, archive.name)

if __name__ == "__main__":
    unittest.main()
