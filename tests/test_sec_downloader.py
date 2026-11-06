import unittest

from ingestion.sec_downloader import (
    accession_to_path,
    build_filing_url,
    normalize_cik,
)


class TestSECDownloaderHelpers(unittest.TestCase):
    def test_normalize_cik(self):
        self.assertEqual(normalize_cik(320193), "0000320193")
        self.assertEqual(normalize_cik("789019"), "0000789019")

    def test_accession_to_path(self):
        self.assertEqual(accession_to_path("0000320193-24-000123"), "000032019324000123")

    def test_build_filing_url(self):
        url = build_filing_url(
            "0000320193",
            "0000320193-24-000123",
            "aapl-20240928.htm",
        )
        self.assertEqual(
            url,
            "https://www.sec.gov/Archives/edgar/data/320193/"
            "000032019324000123/aapl-20240928.htm",
        )


if __name__ == "__main__":
    unittest.main()
