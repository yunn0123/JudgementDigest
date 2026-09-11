# -*- coding: utf-8 -*-
"""
自動分段批次爬蟲 — 依裁判日期遞迴細分，突破單一查詢 500 筆上限。

為什麼需要切分：
  司法院裁判書系統對「單一查詢結果集」有 500 筆的硬性上限（翻到第 25 頁即無下一頁），
  且結果按裁判日期由新到舊排序，被截斷的永遠是最舊的部分。
  結果頁的分群參數 gy/gc（法院 / 年度 / 審級）是單一互斥軸且不能疊加，
  各分群桶同樣受 500 限制，因此光靠分群無法取得完整資料。

解法：
  進階搜尋（default_AD.aspx）可把裁判日期寫進查詢本身，
  每個日期區間都會產生獨立的 q= hash、各自享有自己的 500 額度。
  因此只要把日期區間切到每段結果 < 500，就能完整取得。

策略（深度優先，嚴格由新到舊）：
  1. 初始段落：每段 = 一個公曆年，最新年在前
  2. 每段先讀摘要頁「查詢結果 N」：N >= 500 代表會被截斷 → 對半切分，較新的半段先處理
  3. N < 500 → 完整翻頁收集並下載
  4. 切到單日仍 >= 500 → 日期無法再細分，改用法院分群（gy=jcourt）在該日內逐桶收集；
     法院分群與日期條件不衝突，每個法院桶各自獨立計算 500 額度

用法:
    python crawl_batched.py 借名登記 -n 2000
    python crawl_batched.py 借名登記 -n 1000 --start-year 2018 --end-year 2023
    python crawl_batched.py 借名登記 -n 1000 --start-date 2020/01/01 --end-date 2024/12/31
"""

import argparse
import logging
import sqlite3
from datetime import date, timedelta
from typing import List, Optional, Tuple

from crawler import (
    search_and_crawl, build_driver, DB_PATH, init_db, _RESULT_LIMIT,
)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("crawler_batched.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# 每個日期區間一次最多取回的筆數；設為結果集上限，代表「該區間能拿的全部拿走」
MAX_PER_CALL = _RESULT_LIMIT


# ─── 工具函式 ─────────────────────────────────────────────────────────────────
def _db_count(keyword: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    n = conn.execute(
        "SELECT COUNT(*) FROM crawl_records WHERE keyword LIKE ?",
        (f"%{keyword}%",),
    ).fetchone()[0]
    conn.close()
    return n


def _fmt(d: date) -> str:
    return d.strftime("%Y/%m/%d")


def _generate_year_chunks(start_date: date, end_date: date) -> List[Tuple[date, date]]:
    """
    產生公曆年度對齊的 (chunk_start, chunk_end) 列表，由新到舊排序。
    只是初始種子，用來降低遞迴深度；真正的細分由 batched_crawl 依實際筆數決定。
    """
    chunks = []
    for year in range(end_date.year, start_date.year - 1, -1):
        cs = max(start_date, date(year, 1, 1))
        ce = min(end_date, date(year, 12, 31))
        if cs <= ce:
            chunks.append((cs, ce))
    return chunks


def _split_chunk(cs: date, ce: date) -> List[Tuple[date, date]]:
    """將 [cs, ce] 對半切分，回傳 [(較新的半段), (較舊的半段)]。"""
    span = (ce - cs).days + 1
    mid  = cs + timedelta(days=span // 2 - 1)
    return [(mid + timedelta(days=1), ce), (cs, mid)]


# ─── 主函式（深度優先，新 → 舊）──────────────────────────────────────────────
def batched_crawl(
    keyword: str,
    total_target: int,
    start_date: date,
    end_date: date,
    headless: bool,
) -> int:
    """
    依裁判日期遞迴細分爬取，嚴格由新到舊。

    使用 LIFO 堆疊：切分後把「較舊的半段」先推入、「較新的半段」後推入，
    於是較新的半段會先被彈出處理，確保整體順序始終是新 → 舊。
    """
    init_db()

    # 堆疊頂端 = 下一個要處理的段落。初始種子為年度對齊段落，
    # 最舊的年先推入，最新的年最後推入（故最先處理）。
    stack: List[Tuple[date, date]] = list(reversed(_generate_year_chunks(start_date, end_date)))
    total_new = 0
    processed = 0
    truncated_days: List[date] = []

    logger.info(
        "開始批次爬取：keyword=%r  target=%d  初始段數=%d  range=[%s → %s]  單段上限=%d",
        keyword, total_target, len(stack), _fmt(start_date), _fmt(end_date), MAX_PER_CALL,
    )

    # 整個批次共用一個 WebDriver，避免每段重建（每次重建約 10 秒）
    driver = build_driver(headless)
    try:
        while stack and total_new < total_target:
            cs, ce = stack.pop()
            processed += 1
            span_days = (ce - cs).days + 1
            single_day = span_days <= 1

            logger.info(
                "[段 %d] %s → %s (%d 天)  剩餘堆疊=%d  累計=%d/%d",
                processed, _fmt(cs), _fmt(ce), span_days, len(stack), total_new, total_target,
            )

            before = _db_count(keyword)
            # 只取到還差的筆數為止，避免最後一段大幅超收
            remaining = min(MAX_PER_CALL, total_target - total_new)
            res = search_and_crawl(
                keyword=keyword,
                max_results=remaining,
                headless=headless,
                start_date=_fmt(cs),
                end_date=_fmt(ce),
                driver=driver,
                # 單日已無法再細分 → 不再探測，直接把能拿的拿走
                skip_if_truncated=not single_day,
            )
            new = _db_count(keyword) - before
            total_new += new

            if res["truncated"] and not single_day:
                # 該區間超過 500 筆 → 對半切，較新的半段先處理
                newer, older = _split_chunk(cs, ce)
                stack.append(older)
                stack.append(newer)
                logger.info(
                    "  ↳ total=%s >= %d，切分為 %s→%s（先）與 %s→%s（後）",
                    res["total"], _RESULT_LIMIT,
                    _fmt(newer[0]), _fmt(newer[1]), _fmt(older[0]), _fmt(older[1]),
                )
                continue

            if res["truncated"] and single_day:
                truncated_days.append(cs)
                logger.warning(
                    "  ⚠ %s 單日即有 %s 筆（>= %d），日期已無法再細分 → "
                    "已改用法院分群逐桶收集，取得 %d 筆",
                    _fmt(cs), res["total"], _RESULT_LIMIT, res["collected"],
                )

            logger.info(
                "  ✓ total=%s  下載=%d  新增=%d  累計=%d/%d",
                res["total"], res["collected"], new, total_new, total_target,
            )
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    logger.info("=" * 60)
    logger.info("批次爬取完成。處理段數=%d  總新增=%d 筆", processed, total_new)
    logger.info("=" * 60)
    if truncated_days:
        logger.warning(
            "以下 %d 個單日超過 %d 筆上限，資料可能不完整：%s",
            len(truncated_days), _RESULT_LIMIT,
            ", ".join(_fmt(d) for d in truncated_days[:10]),
        )
    return total_new


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="自動分段批次爬蟲（依裁判日期遞迴細分，由新到舊，突破 500 筆上限）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python crawl_batched.py 借名登記 -n 2000
  python crawl_batched.py 借名登記 -n 1000 --start-year 2018 --end-year 2023
  python crawl_batched.py 借名登記 -n 500  --start-date 2024/01/01 --end-date 2025/04/28
        """,
    )
    ap.add_argument("keyword", help="搜尋關鍵字")
    ap.add_argument(
        "-n", "--num", type=int, default=1000,
        help="目標筆數（預設: 1000）",
    )
    ap.add_argument(
        "--start-year", type=int, default=2015,
        help="裁判日期起始年（西元，預設: 2015，可被 --start-date 覆蓋）",
    )
    ap.add_argument(
        "--end-year", type=int, default=None,
        help="裁判日期結束年（西元，預設: 今年，可被 --end-date 覆蓋）",
    )
    ap.add_argument(
        "--start-date", default="",
        help="裁判日期起 YYYY/MM/DD（西元；覆蓋 --start-year）",
    )
    ap.add_argument(
        "--end-date", default="",
        help="裁判日期迄 YYYY/MM/DD（西元；覆蓋 --end-year）",
    )
    ap.add_argument(
        "--no-headless", action="store_true",
        help="顯示瀏覽器視窗（debug 用）",
    )
    args = ap.parse_args()

    today = date.today()

    if args.start_date:
        sd = date.fromisoformat(args.start_date.replace("/", "-"))
    else:
        sd = date(args.start_year, 1, 1)

    if args.end_date:
        ed = date.fromisoformat(args.end_date.replace("/", "-"))
    else:
        ed = date(args.end_year, 12, 31) if args.end_year else today

    if sd > ed:
        ap.error(f"起始日期 {_fmt(sd)} 晚於結束日期 {_fmt(ed)}")

    total = batched_crawl(
        keyword=args.keyword,
        total_target=args.num,
        start_date=sd,
        end_date=ed,
        headless=not args.no_headless,
    )
    print(f"\n完成！共新增 {total} 筆裁判書至資料庫。")
