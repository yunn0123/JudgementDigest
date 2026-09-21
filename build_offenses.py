"""從 html_cache 的附表建立 offenses 表（一列 = 一個被告的一個罪的一個宣告刑）。

不動 judgments 表，可重複執行（每次重建 offenses）。
用法: python build_offenses.py [--keyword ADV:TPD:M]
"""

import argparse
import sqlite3

from appendix_parser import extract_appendix_offenses

DB_PATH = "judgments.db"


def build(keyword: str = "ADV:TPD:M") -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        DROP TABLE IF EXISTS offenses;
        CREATE TABLE offenses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            crawl_id    INTEGER REFERENCES crawl_records(id),
            case_number TEXT,
            table_idx   INTEGER,
            row_no      TEXT,
            defendant   TEXT,
            law         TEXT,
            charge      TEXT,
            sentence    TEXT,
            months      INTEGER,
            days        INTEGER,
            fine        INTEGER,
            raw         TEXT
        );
        CREATE INDEX idx_offenses_case ON offenses(case_number);
    """)
    rows = conn.execute(
        "SELECT c.id, c.case_number, c.html_file, j.defendant FROM crawl_records c "
        "LEFT JOIN judgments j ON j.crawl_id = c.id "
        "WHERE c.keyword LIKE ? AND c.case_number LIKE '%判決'", (keyword + "%",)).fetchall()
    hit = n = 0
    for crawl_id, case_number, html_file, defendants in rows:
        try:
            html = open(html_file, encoding="utf-8").read()
        except OSError:
            continue
        offs = extract_appendix_offenses(html, names=[n.strip() for n in (defendants or "").split("；") if n.strip()])
        if offs:
            hit += 1
        for o in offs:
            conn.execute(
                "INSERT INTO offenses (crawl_id, case_number, table_idx, row_no, defendant, law, charge,"
                " sentence, months, days, fine, raw) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (crawl_id, case_number, o["table"], o["no"], o["defendant"], o["law"], o["charge"],
                 o["sentence"], o["months"], o["days"], o["fine"], o["raw"]))
            n += 1
    conn.commit()
    conn.close()
    print(f"判決 {len(rows)} 筆，其中 {hit} 筆有附表宣告刑，共寫入 {n} 列")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyword", default="ADV:TPD:M")
    build(ap.parse_args().keyword)
