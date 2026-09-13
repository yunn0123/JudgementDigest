# -*- coding: utf-8 -*-
"""
自動分段批次爬蟲 — 以公曆年度為單位切分搜尋，自動處理每次 500 筆上限。

採用廣度優先（BFS）策略：
  - 初始段落：每段 = 一個公曆年（與 search_and_crawl 的 ROC 年度篩選 URL 對齊）
  - 觸碰 500 上限且跨年的段落，下一輪切為兩半再抓
  - 同一年度內觸碰上限：已無法用年度 URL 細分，記錄警告後跳過

用法:
    python crawl_batched.py 借名登記 -n 2000
    python crawl_batched.py 借名登記 -n 1000 --start-year 2018 --end-year 2023
    python crawl_batched.py 借名登記 -n 1000 --start-date 2020/01/01 --end-date 2024/12/31
"""

import argparse
import logging
import sqlite3
from datetime import date, timedelta

from crawler import (
    search_and_crawl, advanced_search_and_crawl,
    resolve_court_codes, CATEGORY_CODE_MAP,
    DB_PATH, init_db,
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

# search_and_crawl 每次呼叫的最大筆數上限
# 設為 2000：允許每個「法院×年度」群組合計仍在此範圍內完整抓取
# 借名登記每年最多約 900 筆（~25 法院，每法院最多 ~100 筆），故 2000 足夠
MAX_PER_CALL = 2000
# 若某段抓到的筆數 >= HIT_LIMIT，視為可能有資料遺漏，記錄警告
# 以 MAX_PER_CALL 的 95% 為門檻
HIT_LIMIT  = int(MAX_PER_CALL * 0.95)
# 最小切分天數：低於此天數不再切分（避免無限細分）
MIN_SPLIT_DAYS = 7


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


def _generate_year_chunks(start_date: date, end_date: date) -> list:
    """
    產生公曆年度對齊的 (chunk_start, chunk_end) 列表，由新到舊排序。
    每段恰好對應一個公曆年（或起/迄年的不完整部分），
    與 search_and_crawl 內部的 ROC 年度篩選 URL 完全對齊，避免重複爬取。
    """
    chunks = []
    for year in range(end_date.year, start_date.year - 1, -1):
        cs = max(start_date, date(year, 1, 1))
        ce = min(end_date, date(year, 12, 31))
        if cs <= ce:
            chunks.append((cs, ce))
    return chunks


def _split_chunk(cs: date, ce: date) -> list:
    """將 [cs, ce] 對半切分，回傳 [(newer_half), (older_half)]（新的在前）。"""
    span = (ce - cs).days + 1
    mid  = cs + timedelta(days=span // 2 - 1)
    return [(mid + timedelta(days=1), ce), (cs, mid)]


# ─── 主函式（BFS）────────────────────────────────────────────────────────────
def batched_crawl(
    keyword: str,
    total_target: int,
    start_date: date,
    end_date: date,
    headless: bool,
    court: str = "",
    case_type: str = "",
    category: str = "",
) -> int:
    """
    廣度優先批次爬取。

    初始段落以公曆年為單位（對齊 search_and_crawl 的 ROC 年度篩選 URL）。
    每一輪依「今年 → 起始年」順序掃過所有待處理段落；
    觸碰 500 上限且跨年的段落才在下一輪被切成兩半重抓。
    累計新增筆數達 total_target 後停止。
    """
    init_db()

    # 初始段落：年度對齊，最新在前
    current_round: list = _generate_year_chunks(start_date, end_date)
    total_new   = 0
    round_no    = 0

    logger.info(
        "開始批次爬取：keyword=%r  target=%d  初始段數=%d（年度對齊）  range=[%s → %s]",
        keyword, total_target, len(current_round),
        _fmt(start_date), _fmt(end_date),
    )

    while current_round and total_new < total_target:
        round_no += 1
        next_round: list = []

        logger.info(
            "\n%s\n第 %d 輪：共 %d 段待處理  (累計 %d / %d)\n%s",
            "=" * 60, round_no, len(current_round), total_new, total_target, "=" * 60,
        )

        for idx, (cs, ce) in enumerate(current_round, 1):
            if total_new >= total_target:
                logger.info("已達目標 %d 筆，本輪提早結束。", total_target)
                break

            span_days = (ce - cs).days + 1
            logger.info(
                "  [%d/%d] %s → %s  (span=%dd)",
                idx, len(current_round), _fmt(cs), _fmt(ce), span_days,
            )

            before       = _db_count(keyword)
            chunk_target = min(MAX_PER_CALL, total_target - total_new)
            n_fetched = search_and_crawl(
                keyword=keyword,
                max_results=chunk_target,
                headless=headless,
                start_date=_fmt(cs),
                end_date=_fmt(ce),
                court=court,
                case_type=case_type,
                category=category,
            )
            new        = _db_count(keyword) - before
            total_new += new

            logger.info(
                "  fetched=%d  new=%d  累計=%d/%d",
                n_fetched, new, total_new, total_target,
            )

            # 觸碰上限 → 嘗試切分
            # 僅當本次呼叫用的是完整 MAX_PER_CALL（而非被剩餘目標數限縮）時才檢查，
            # 否則「剛好抓到 chunk_target 筆就停」會被誤判為觸頂。
            hit_limit_applicable = chunk_target == MAX_PER_CALL
            if hit_limit_applicable and n_fetched >= HIT_LIMIT and span_days > MIN_SPLIT_DAYS:
                if cs.year == ce.year:
                    # 同一年度：年度 URL 無法細分，記錄警告
                    logger.warning(
                        "  ⚠ 觸碰 500 上限（%d 年），但年度篩選已無法細分，"
                        "此年度可能有資料遺漏。",
                        cs.year,
                    )
                else:
                    # 跨年段落：切為兩半（各自對應不同年度）
                    halves = _split_chunk(cs, ce)
                    next_round.extend(halves)
                    logger.info(
                        "  ⚠ 觸碰 500 上限，下一輪切分為 %s→%s 和 %s→%s",
                        _fmt(halves[0][0]), _fmt(halves[0][1]),
                        _fmt(halves[1][0]), _fmt(halves[1][1]),
                    )
            elif hit_limit_applicable and n_fetched >= HIT_LIMIT:
                logger.warning(
                    "  ⚠ 觸碰 500 上限，但段落僅 %d 天，無法再切分，此區間可能有遺漏。",
                    span_days,
                )

        current_round = next_round

    logger.info(
        "\n%s\n批次爬取完成。總輪數=%d  總新增=%d 筆\n%s",
        "=" * 60, round_no, total_new, "=" * 60,
    )
    return total_new


# ─── 進階搜尋批次爬取（不需關鍵字）───────────────────────────────────────────
# 進階搜尋單次查詢上限（網站硬性限制，跟關鍵字搜尋一樣是 500 筆）
ADV_MAX_PER_CALL = 500
# 進階搜尋支援精確到「日」的日期區間，不像關鍵字搜尋只能靠 ROC 年度 URL 篩選，
# 因此最小切分天數可以一路切到 1 天（MIN_SPLIT_DAYS=7 是關鍵字搜尋模式的限制，
# 這裡不適用；沿用它會導致案件量大的法院/類別即使已知超過 500 筆也切不下去）。
ADV_MIN_SPLIT_DAYS = 1


def batched_advanced_crawl(
    court_codes:    list,
    category_codes: list,
    start_date:     date,
    end_date:       date,
    total_target:   int,
    headless:       bool,
) -> int:
    """
    用進階搜尋（Default_AD.aspx，不需關鍵字）做廣度優先批次爬取。

    進階搜尋支援任意精確日期區間（非僅 ROC 年度對齊），故直接對 [start_date, end_date]
    做二分切分以突破單次 500 筆上限，不像 batched_crawl 受限於「同一年度內無法再細分」。
    """
    init_db()
    keyword_tag = f"ADV:{','.join(court_codes)}:{','.join(category_codes)}"

    current_round: list = [(start_date, end_date)]
    total_new = 0
    round_no  = 0

    logger.info(
        "開始進階批次爬取：court=%s category=%s target=%d range=[%s → %s]",
        court_codes, category_codes, total_target, _fmt(start_date), _fmt(end_date),
    )

    while current_round and total_new < total_target:
        round_no += 1
        next_round: list = []

        logger.info(
            "\n%s\n第 %d 輪：共 %d 段待處理  (累計 %d / %d)\n%s",
            "=" * 60, round_no, len(current_round), total_new, total_target, "=" * 60,
        )

        for idx, (cs, ce) in enumerate(current_round, 1):
            if total_new >= total_target:
                logger.info("已達目標 %d 筆，本輪提早結束。", total_target)
                break

            span_days = (ce - cs).days + 1
            logger.info(
                "  [%d/%d] %s → %s  (span=%dd)",
                idx, len(current_round), _fmt(cs), _fmt(ce), span_days,
            )

            before       = _db_count(keyword_tag)
            chunk_target = min(ADV_MAX_PER_CALL, total_target - total_new)
            n_fetched, reported_total = advanced_search_and_crawl(
                court_codes=court_codes,
                category_codes=category_codes,
                start_date=_fmt(cs),
                end_date=_fmt(ce),
                max_results=chunk_target,
                headless=headless,
            )
            new        = _db_count(keyword_tag) - before
            total_new += new

            logger.info(
                "  fetched=%d  new=%d  reported_total=%s  累計=%d/%d",
                n_fetched, new, reported_total, total_new, total_target,
            )

            # 判斷是否需要切分：優先用網站回報的真實總數（reported_total），
            # 而非只看有沒有碰到我們自訂的 max_results 上限——翻頁可能中途靜默
            # 失敗，導致 n_fetched 遠低於上限卻仍不完整，光看上限會誤判為完整。
            exceeds_cap  = reported_total is not None and reported_total > ADV_MAX_PER_CALL
            undercounted = reported_total is not None and n_fetched < min(reported_total, chunk_target)
            needs_split  = exceeds_cap or undercounted

            if needs_split and span_days > ADV_MIN_SPLIT_DAYS:
                halves = _split_chunk(cs, ce)
                next_round.extend(halves)
                reason = "超過 500 筆上限" if exceeds_cap else "翻頁疑似未完整（重試後仍不足）"
                logger.info(
                    "  ⚠ %s，下一輪切分為 %s→%s 和 %s→%s",
                    reason,
                    _fmt(halves[0][0]), _fmt(halves[0][1]),
                    _fmt(halves[1][0]), _fmt(halves[1][1]),
                )
            elif needs_split:
                logger.warning(
                    "  ⚠ 此區間資料可能不完整（reported_total=%s, fetched=%d），"
                    "但段落僅 %d 天無法再切分。",
                    reported_total, n_fetched, span_days,
                )

        current_round = next_round

    logger.info(
        "\n%s\n進階批次爬取完成。總輪數=%d  總新增=%d 筆\n%s",
        "=" * 60, round_no, total_new, "=" * 60,
    )
    return total_new


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="自動分段批次爬蟲（廣度優先，年度對齊，自動處理 500 筆上限）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python crawl_batched.py 借名登記 -n 2000
  python crawl_batched.py 借名登記 -n 1000 --start-year 2018 --end-year 2023
  python crawl_batched.py 借名登記 -n 500  --start-date 2024/01/01 --end-date 2025/04/28
        """,
    )
    ap.add_argument("keyword", nargs="?", default="", help="搜尋關鍵字（--advanced 模式下不需要）")
    ap.add_argument(
        "-n", "--num", type=int, default=1000,
        help="目標筆數（預設: 1000）",
    )
    ap.add_argument(
        "--start-year", type=int, default=2015,
        help="搜尋起始年（預設: 2015，可被 --start-date 覆蓋）",
    )
    ap.add_argument(
        "--end-year", type=int, default=None,
        help="搜尋結束年（預設: 今年，可被 --end-date 覆蓋）",
    )
    ap.add_argument(
        "--start-date", default="",
        help="起始日期 YYYY/MM/DD（覆蓋 --start-year）",
    )
    ap.add_argument(
        "--end-date", default="",
        help="結束日期 YYYY/MM/DD（覆蓋 --end-year）",
    )
    ap.add_argument(
        "--no-headless", action="store_true",
        help="顯示瀏覽器視窗（debug 用）",
    )
    ap.add_argument("--court",     default="", help="法院名稱關鍵字（部分比對，例如「臺北」）")
    ap.add_argument("--case-type", default="", help="裁判字號案由代字（部分比對，例如「訴」「易」）")
    ap.add_argument("--category",  default="", help="裁判類別（部分比對，例如「刑事」「民事」「行政」）")
    ap.add_argument("--advanced", action="store_true",
                    help="使用進階搜尋（Default_AD.aspx），不需關鍵字，用 --court + --category 直接篩選")
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

    if args.advanced:
        court_codes    = resolve_court_codes(args.court) if args.court else []
        category_codes = [v for k, v in CATEGORY_CODE_MAP.items() if k in args.category] if args.category else []
        if args.court and not court_codes:
            ap.error(f"找不到符合「{args.court}」的法院代碼，請確認法院名稱。")
        total = batched_advanced_crawl(
            court_codes=court_codes,
            category_codes=category_codes,
            start_date=sd,
            end_date=ed,
            total_target=args.num,
            headless=not args.no_headless,
        )
    else:
        if not args.keyword:
            ap.error("非 --advanced 模式需要指定搜尋關鍵字。")
        total = batched_crawl(
            keyword=args.keyword,
            total_target=args.num,
            start_date=sd,
            end_date=ed,
            headless=not args.no_headless,
            court=args.court,
            case_type=args.case_type,
            category=args.category,
        )
    print(f"\n完成！共新增 {total} 筆裁判書至資料庫。")
