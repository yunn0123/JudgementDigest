"""
完整流程執行腳本（Pipeline）

步驟 1: crawl_batched  — 自動分段爬取（突破 500 筆上限）
步驟 2: html_parser    — 解析 HTML，結構化存入 SQLite
步驟 3: export_excel   — 匯出 Excel

使用方式:
    python pipeline.py <關鍵字> [-n 筆數] [選項]

範例:
    python pipeline.py 借名登記 -n 1000
    python pipeline.py 借名登記 -n 2000 --start-year 2018
    python pipeline.py 借名登記 -n 500  --start-date 2024/01/01 --end-date 2025/04/28
    python pipeline.py 借名登記 -n 100  --no-headless
    python pipeline.py 借名登記 -n 1000 --skip-crawl   # 只重新解析 + 匯出
"""

import argparse
import sys
import io
import time
from datetime import date, datetime

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def run(
    keyword: str,
    max_results: int,
    headless: bool,
    output: str,
    full_text: bool,
    start_date: date,
    end_date: date,
    skip_crawl: bool,
    court: str = "",
    case_type: str = "",
    category: str = "",
    advanced: bool = False,
) -> None:
    sep = "=" * 60

    print(sep)
    print("  司法院裁判書自動化流程")
    print(sep)
    if advanced:
        print("  模式      : 進階搜尋（不需關鍵字）")
    else:
        print(f"  關鍵字    : {keyword}")
    print(f"  目標筆數  : {max_results}")
    print(f"  日期範圍  : {start_date.strftime('%Y/%m/%d')} → {end_date.strftime('%Y/%m/%d')}")
    print(f"  無頭模式  : {headless}")
    if court:
        print(f"  法院篩選  : {court}")
    if case_type:
        print(f"  案由代字  : {case_type}")
    if category:
        print(f"  裁判類別  : {category}")
    if skip_crawl:
        print("  ⚠ --skip-crawl：跳過爬取，直接解析 + 匯出")
    print(sep)

    export_keyword = keyword
    court_codes: list = []
    category_codes: list = []
    if advanced:
        from crawler import resolve_court_codes, CATEGORY_CODE_MAP
        court_codes    = resolve_court_codes(court) if court else []
        category_codes = [v for k, v in CATEGORY_CODE_MAP.items() if k in category] if category else []
        if court and not court_codes:
            print(f"  [ERROR] 找不到符合「{court}」的法院代碼，請確認法院名稱。")
            sys.exit(1)
        export_keyword = f"ADV:{','.join(court_codes)}:{','.join(category_codes)}"

    # ── Step 1: 爬取（自動分段）────────────────────────────────────
    if not skip_crawl:
        print(f"\n【步驟 1/3】分段爬取裁判書…")
        t0 = time.time()
        try:
            if advanced:
                from crawl_batched import batched_advanced_crawl
                crawled = batched_advanced_crawl(
                    court_codes=court_codes,
                    category_codes=category_codes,
                    start_date=start_date,
                    end_date=end_date,
                    total_target=max_results,
                    headless=headless,
                )
            else:
                from crawl_batched import batched_crawl
                crawled = batched_crawl(
                    keyword=keyword,
                    total_target=max_results,
                    start_date=start_date,
                    end_date=end_date,
                    headless=headless,
                    court=court,
                    case_type=case_type,
                    category=category,
                )
        except Exception as exc:
            print(f"  [ERROR] 爬取失敗: {exc}")
            sys.exit(1)
        print(f"  完成：新增 {crawled} 筆  ({time.time()-t0:.1f}s)")
    else:
        print(f"\n【步驟 1/3】已跳過（--skip-crawl）")

    # ── Step 2: 解析 ──────────────────────────────────────────────
    print(f"\n【步驟 2/3】解析 HTML 並結構化…")
    t0 = time.time()
    try:
        from html_parser import parse_all_unparsed
        parsed = parse_all_unparsed()
    except Exception as exc:
        print(f"  [ERROR] 解析失敗: {exc}")
        sys.exit(1)
    print(f"  完成：解析 {parsed} 筆  ({time.time()-t0:.1f}s)")

    # ── Step 3: 匯出 ──────────────────────────────────────────────
    # 檔名不可含冒號等特殊字元，進階模式用 court/category 代碼組字串取代
    name_tag = f"ADV_{'_'.join(court_codes)}_{'_'.join(category_codes)}" if advanced else keyword
    out_path = output or f"judgments_{name_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    print(f"\n【步驟 3/3】匯出 Excel → {out_path}…")
    t0 = time.time()
    try:
        from export_excel import fetch_judgments, export_to_excel
        # limit=0 → 匯出所有符合條件的資料（不受 -n 限制）
        rows = fetch_judgments(limit=0, keyword=export_keyword)
        if not rows:
            print("  [WARN] 資料庫無資料，跳過匯出。")
        else:
            ok = export_to_excel(rows, out_path, include_full_text=full_text)
            if ok:
                print(f"  完成：匯出 {len(rows)} 筆  ({time.time()-t0:.1f}s)")
            else:
                print("  [WARN] 匯出失敗。")
    except Exception as exc:
        print(f"  [ERROR] 匯出失敗: {exc}")

    print(f"\n{sep}")
    print("  全部流程完成！")
    print(sep)


if __name__ == "__main__":
    today = date.today()

    ap = argparse.ArgumentParser(
        description="司法院裁判書完整自動化流程（含自動分段突破 500 筆上限）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python pipeline.py 借名登記 -n 1000
  python pipeline.py 借名登記 -n 2000 --start-year 2018 --end-year 2023
  python pipeline.py 借名登記 -n 500  --start-date 2024/01/01 --end-date 2025/04/28
  python pipeline.py 借名登記 -n 1000 --skip-crawl        # 只重解析 + 匯出
        """,
    )
    ap.add_argument("keyword", nargs="?", default="", help="搜尋關鍵字（--advanced 模式下不需要）")
    ap.add_argument(
        "-n", "--num", type=int, default=100,
        help="目標爬取筆數（預設: 100）",
    )

    # 日期選項
    date_grp = ap.add_argument_group("日期範圍（二擇一）")
    date_grp.add_argument(
        "--start-year", type=int, default=2015,
        help="搜尋起始年（預設: 2015）",
    )
    date_grp.add_argument(
        "--end-year", type=int, default=None,
        help="搜尋結束年（預設: 今年）",
    )
    date_grp.add_argument(
        "--start-date", default="",
        help="起始日期 YYYY/MM/DD（覆蓋 --start-year）",
    )
    date_grp.add_argument(
        "--end-date", default="",
        help="結束日期 YYYY/MM/DD（覆蓋 --end-year）",
    )
    # 其他選項
    ap.add_argument("--no-headless", action="store_true", help="顯示瀏覽器視窗（debug 用）")
    ap.add_argument("-o", "--output", default="",         help="輸出 Excel 檔名")
    ap.add_argument("--full-text",   action="store_true", help="Excel 含全文欄位")
    ap.add_argument("--court",      default="", help="法院名稱關鍵字（部分比對，例如「臺北」），只掃描符合的法院")
    ap.add_argument("--case-type",  default="", help="裁判字號案由代字（部分比對，例如「訴」「易」）")
    ap.add_argument("--category",   default="", help="裁判類別（部分比對，例如「刑事」「民事」「行政」）")
    ap.add_argument(
        "--skip-crawl", action="store_true",
        help="跳過爬取步驟，直接重新解析 + 匯出（用於只更新解析結果時）",
    )
    ap.add_argument(
        "--advanced", action="store_true",
        help="使用進階搜尋（Default_AD.aspx），不需關鍵字，用 --court + --category 直接篩選、完整抓取",
    )
    args = ap.parse_args()

    if not args.advanced and not args.keyword:
        ap.error("非 --advanced 模式需要指定搜尋關鍵字。")

    # 解析日期
    if args.start_date:
        sd = date.fromisoformat(args.start_date.replace("/", "-"))
    else:
        sd = date(args.start_year, 1, 1)

    if args.end_date:
        ed = date.fromisoformat(args.end_date.replace("/", "-"))
    else:
        ed = date(args.end_year, 12, 31) if args.end_year else today

    if sd > ed:
        ap.error(f"起始日期 {sd} 晚於結束日期 {ed}")

    run(
        keyword=args.keyword,
        max_results=args.num,
        headless=not args.no_headless,
        output=args.output,
        full_text=args.full_text,
        start_date=sd,
        end_date=ed,
        skip_crawl=args.skip_crawl,
        court=args.court,
        case_type=args.case_type,
        category=args.category,
        advanced=args.advanced,
    )
