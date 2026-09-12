# -*- coding: utf-8 -*-
"""
html_parser.py 的離線測試。

以 tests/fixtures/ 下兩份真實裁判書 HTML（取自司法院公開裁判書系統）為樣本：
  tpd_civil_judgment.html — 地方法院民事判決，標準 div 版型
  cc_ruling.html          — 憲法法庭裁定，text-pre 版型（特殊格式）
版型若被改版，這裡會先失敗。
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import html_parser  # noqa: E402
from html_parser import normalize_date, parse_html  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestNormalizeDate(unittest.TestCase):
    def test_roc_chinese(self):
        self.assertEqual(normalize_date("114年1月7日"), "2025-01-07")
        self.assertEqual(normalize_date("民國 114 年 12 月 31 日"), "2025-12-31")

    def test_roc_dotted(self):
        self.assertEqual(normalize_date("114.01.07"), "2025-01-07")
        self.assertEqual(normalize_date("114/1/7"), "2025-01-07")

    def test_ad_iso_passthrough(self):
        self.assertEqual(normalize_date("2025-01-07"), "2025-01-07")
        self.assertEqual(normalize_date("2025/01/07"), "2025-01-07")

    def test_garbage_returns_empty(self):
        # 舊版 DB 曾把「案由」誤存進日期欄，解析失敗必須回傳空字串而非拋錯
        self.assertEqual(normalize_date("清償借款"), "")
        self.assertEqual(normalize_date(""), "")


class TestParseCivilJudgment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = parse_html(
            _fixture("tpd_civil_judgment.html"), crawl_id=1,
            keyword="臺灣臺北地方法院-民事-判決",
            source_url="https://judgment.judicial.gov.tw/FJUD/data.aspx?ty=JD&id=x",
        )

    def test_metadata(self):
        self.assertEqual(self.data["case_number"],
                         "臺灣臺北地方法院 113 年度訴字第 6910 號民事判決")
        self.assertEqual(self.data["court"], "臺灣臺北地方法院")
        self.assertEqual(self.data["judgment_date"], "2025-01-07")   # 西元 ISO
        self.assertEqual(self.data["case_type"], "清償借款")
        self.assertEqual(self.data["judgment_type"], "判決")

    def test_parties(self):
        self.assertEqual(self.data["plaintiff"], "台新國際商業銀行股份有限公司")
        self.assertEqual(self.data["defendant"], "張沛晴")
        self.assertIn("原告：", self.data["party_roles"])
        self.assertIn("被告：", self.data["party_roles"])

    def test_judges_and_clerk(self):
        self.assertEqual(self.data["judges"], "匡偉")
        self.assertEqual(self.data["clerk"], "林鈞婷")

    def test_sections(self):
        self.assertTrue(self.data["verdict"].startswith("被告"))
        self.assertIn("給付原告", self.data["verdict"])
        self.assertGreater(len(self.data["full_text"]), 500)

    def test_keyword_label_carried(self):
        self.assertEqual(self.data["keyword"], "臺灣臺北地方法院-民事-判決")


class TestParseConstitutionalCourtRuling(unittest.TestCase):
    """憲法法庭是 text-pre 版型，欄位切法與一般法院不同。"""

    @classmethod
    def setUpClass(cls):
        cls.data = parse_html(_fixture("cc_ruling.html"), crawl_id=2)

    def test_metadata(self):
        self.assertEqual(self.data["court"], "憲法法庭")
        self.assertEqual(self.data["judgment_date"], "2025-04-30")
        self.assertEqual(self.data["judgment_type"], "裁定")

    def test_sections(self):
        self.assertEqual(self.data["verdict"], "本件不受理。")
        self.assertGreater(len(self.data["reasons"]), 200)

    def test_petitioners_go_to_plaintiff_field(self):
        self.assertIn("林薛彩雲", self.data["plaintiff"])
        self.assertIn("聲請人：", self.data["party_roles"])


class TestParseDegradesGracefully(unittest.TestCase):
    def test_stub_html_does_not_raise(self):
        data = parse_html("<html><head><title>x</title></head><body></body></html>",
                          crawl_id=3, case_number_hint="臺灣臺北地方法院 113 年度 訴 字第 1 號民事判決")
        # 取不到內容時以 hint 保底，且仍能判定法院與裁判種類
        self.assertEqual(data["court"], "臺灣臺北地方法院")
        self.assertEqual(data["judgment_type"], "判決")
        self.assertEqual(data["verdict"], "")

    def test_empty_html_does_not_raise(self):
        data = parse_html("", crawl_id=4)
        self.assertEqual(data["case_number"], "")
        self.assertEqual(data["judgment_type"], "")


class TestSaveJudgment(unittest.TestCase):
    """寫入暫存 DB，不碰真正的 judgments.db。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._db = html_parser.DB_PATH
        html_parser.DB_PATH = os.path.join(self.tmp.name, "t.db")
        html_parser.init_db()

    def tearDown(self):
        html_parser.DB_PATH = self._db
        self.tmp.cleanup()

    def test_insert_and_replace(self):
        data = parse_html(_fixture("tpd_civil_judgment.html"), crawl_id=1,
                          keyword="測試標籤")
        self.assertTrue(html_parser.save_judgment(data))
        self.assertTrue(html_parser.save_judgment(data))   # INSERT OR REPLACE
        conn = sqlite3.connect(html_parser.DB_PATH)
        rows = conn.execute(
            "SELECT case_number, judgment_type, keyword FROM judgments").fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "判決")
        self.assertEqual(rows[0][2], "測試標籤")


if __name__ == "__main__":
    unittest.main()
