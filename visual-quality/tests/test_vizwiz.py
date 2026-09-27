import io
import unittest
import zipfile
from unittest.mock import MagicMock, patch

from visual_quality.vizwiz import _read_member, select_candidates


def annotation(name, **flaws):
    return {"image": name, "flaws": flaws, "unrecognizable": 5}


class VizWizSelectionTests(unittest.TestCase):
    def test_only_official_training_names_with_majority_defect_votes(self):
        rows = [
            annotation("VizWiz_train_00000001.jpg", BLR=3, NON=0),
            annotation("VizWiz_val_00000002.jpg", BLR=5),
            annotation("VizWiz_test_00000003.jpg", BLR=5),
            annotation("VizWiz_train_00000004.jpg", BLR=2),
            annotation("VizWiz_train_00000005.jpg", NON=5),
            annotation("VizWiz_train_00000006.jpg", OBS=3, NON=3),
            annotation("../VizWiz_train_00000007.jpg", BLR=5),
            annotation("VizWiz_train_00000008.jpg", OTH=5),
        ]
        selected = select_candidates(rows, seed=7)
        self.assertEqual([row["image"] for row in selected], ["VizWiz_train_00000001.jpg"])

    def test_unrecognizability_alone_does_not_select_a_photo(self):
        self.assertEqual(select_candidates([annotation("VizWiz_train_00000001.jpg")], 7), [])

    def test_selection_is_deterministic_unique_and_balances_defects(self):
        rows = [annotation(f"VizWiz_train_{number:08d}.jpg", BLR=4) for number in range(20)]
        rows.append(annotation("VizWiz_train_00000020.jpg", DRK=4))
        rows.append(rows[0])
        selected = select_candidates(rows, 19)
        self.assertEqual(selected, select_candidates(list(reversed(rows)), 19))
        self.assertEqual(len({row["image"] for row in selected}), 21)
        self.assertEqual(selected[1]["image"], "VizWiz_train_00000020.jpg")

    def test_invalid_votes_fail_instead_of_becoming_labels(self):
        for vote in [-1, 6, "3", True]:
            with self.subTest(vote=vote), self.assertRaises(ValueError):
                select_candidates([annotation("VizWiz_train_00000001.jpg", BLR=vote)], 7)


class VizWizRangeTests(unittest.TestCase):
    def setUp(self):
        stream = io.BytesIO()
        self.original = b"original image bytes" * 50
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("train/VizWiz_train_00000001.jpg", self.original)
            info = archive.getinfo("train/VizWiz_train_00000001.jpg")
        self.payload = stream.getvalue()
        self.member = {
            "offset": info.header_offset,
            "compressed_size": info.compress_size,
            "size": info.file_size,
            "compression": info.compress_type,
            "crc": info.CRC,
        }

    def response(self, status=206):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = status
        response.headers = {"Content-Range": f"bytes 0-{len(self.payload) - 1}/{len(self.payload)}"}
        response.raw = io.BytesIO(self.payload)
        return response

    def test_range_extraction_preserves_member_bytes_and_checks_crc(self):
        with (
            patch("visual_quality.vizwiz.ARCHIVE_SIZE", len(self.payload)),
            patch("visual_quality.vizwiz._session") as session,
        ):
            session.return_value.get.return_value = self.response()
            self.assertEqual(_read_member(self.member), self.original)
            session.return_value.get.return_value = self.response()
            with self.assertRaisesRegex(ValueError, "CRC"):
                _read_member({**self.member, "crc": 0})

    def test_server_that_ignores_range_is_rejected_before_reading(self):
        response = self.response(status=200)
        with patch("visual_quality.vizwiz._session") as session:
            session.return_value.get.return_value = response
            with self.assertRaisesRegex(ValueError, "bounded ZIP range"):
                _read_member(self.member)
        self.assertEqual(response.raw.tell(), 0)
