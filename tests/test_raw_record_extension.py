import os
import tempfile
import unittest
from unittest import mock

from phyling.decoder import decoder
from phyling.phyling_utils import is_raw_record_file
from phyling.phyling_utils import RAW_RECORD_MIME_TYPE


class RawRecordExtensionTest(unittest.TestCase):
    """A raw record is .phy or legacy .txt, compared case-insensitively."""

    def test_is_raw_record_file(self):
        """Maxi .phy, Mini .PHY and old firmware .txt / .TXT are raw records; other files are not."""
        for name in ("rec.phy", "M42_1.PHY", "data.txt", "M42_1.TXT", "/sd/rec/Rec.Phy"):
            self.assertTrue(is_raw_record_file(name), name)
        for name in ("data.csv", "data.json", "data.phyling", "phy", "rec.txt.gz", "noext"):
            self.assertFalse(is_raw_record_file(name), name)

    def test_mime_type(self):
        """The raw record MIME type is the vendor one."""
        self.assertEqual(RAW_RECORD_MIME_TYPE, "application/vnd.phyling.record")


class DecodeSaveExtensionTest(unittest.TestCase):
    """decodeSave / decodeSaveFolder take .phy and legacy .txt files in any case."""

    def test_decode_save_refuses_other_extension(self):
        """A non raw file raises before decoding."""
        with mock.patch.object(decoder, "decode") as decode, self.assertRaises(ValueError):
            decoder.decodeSave("data.csv")  # refused on its extension, the path is never opened
        decode.assert_not_called()

    def test_decode_save_writes_json_next_to_file(self):
        """The json output replaces the raw extension, whatever its case."""
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("M42_1.PHY", "rec.phy", "old.TXT"):
                with mock.patch.object(decoder, "decode", return_value={"ok": True}):
                    self.assertTrue(decoder.decodeSave(os.path.join(tmp, name), verbose=False))
            self.assertEqual(sorted(os.listdir(tmp)), ["M42_1.json", "old.json", "rec.json"])

    def test_decode_save_folder_filters_raw_files(self):
        """Only raw record files of the folder are decoded."""
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.phy", "B.PHY", "c.txt", "D.TXT", "e.csv", "f.json"):
                open(os.path.join(tmp, name), "wb").close()
            with mock.patch.object(decoder, "decodeSave", return_value=True) as decode_save:
                filenames = decoder.decodeSaveFolder(tmp + "/", verbose=False)
        self.assertEqual(sorted(filenames), ["B.PHY", "D.TXT", "a.phy", "c.txt"])
        self.assertEqual(decode_save.call_count, 4)
