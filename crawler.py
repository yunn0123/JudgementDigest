"""
Part 1 — 司法院裁判書自動爬蟲
目標網站: https://judgment.judicial.gov.tw/FJUD/default.aspx

使用方式:
    python crawler.py <關鍵字> [-n 筆數] [--no-headless]
                      [--start-date YYYY/MM/DD] [--end-date YYYY/MM/DD]
                      [--case-year-start 民國年] [--case-year-end 民國年]

範例:
    python crawler.py 詐欺 -n 20
    python crawler.py 勞動契約 -n 10 --start-date 2023/01/01 --end-date 2023/12/31 --no-headless
"""

import sqlite3
import time
import os
import re
import argparse
import logging
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from urllib.parse import unquote

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, WebDriverException
)
from webdriver_manager.chrome import ChromeDriverManager

# ─── 設定 ────────────────────────────────────────────────────────────────────
BASE_URL    = "https://judgment.judicial.gov.tw/FJUD/default.aspx"
BASE_URL_AD = "https://judgment.judicial.gov.tw/FJUD/default_AD.aspx"  # 進階搜尋（含日期篩選）
DB_PATH   = "judgments.db"
HTML_DIR  = "html_cache"
LOG_FILE  = "crawler.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─── 資料庫初始化 ─────────────────────────────────────────────────────────────
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crawl_records (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            case_number   TEXT    UNIQUE,
            court         TEXT,
            case_title    TEXT,
            judgment_date TEXT,
            source_url    TEXT,
            html_file     TEXT,
            keyword       TEXT,
            crawled_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            parsed        INTEGER   DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database ready: %s", DB_PATH)


# ─── WebDriver 設定 ───────────────────────────────────────────────────────────
def build_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--lang=zh-TW")
    # 停用圖片、字型等非必要資源，大幅減少等待時間
    opts.add_argument("--blink-settings=imagesEnabled=false")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    # eager：等待 DOMContentLoaded 即返回（HTML 已解析，圖片/追蹤器仍在背景載入）
    # 裁判書正文在初始 HTML 中，不需等完整頁面；比 "none" 穩定，不會卡住 find_element
    opts.page_load_strategy = "eager"

    # 避免偵測 WebDriver
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(90)

    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
    )

    # 封鎖字型、追蹤器等非必要請求
    driver.execute_cdp_cmd("Network.enable", {})
    driver.execute_cdp_cmd("Network.setBlockedURLs", {"urls": [
        "*.woff", "*.woff2", "*.ttf", "*.otf", "*.eot",
        "*.mp4", "*.webm", "*.mp3",
        "*google-analytics*", "*googletagmanager*",
        "*facebook.com/tr*", "*doubleclick*", "*adsystem*",
    ]})

    return driver


# ─── HTML 完整性檢查 ──────────────────────────────────────────────────────────
_MIN_COMPLETE_SIZE = 5_000  # 完整裁判書 HTML 最少應有 5 KB

def _is_html_content_complete(html: str) -> bool:
    """True 表示 HTML 字串包含完整裁判書正文（非 stub）。"""
    if len(html) < _MIN_COMPLETE_SIZE:
        return False
    return 'id="jud"' in html or '"htmlcontent"' in html


def _is_html_complete(html_file: str) -> bool:
    """
    True 表示 html_file 存在且包含完整裁判書內容（非 stub 頁面）。
    Stub 頁面：瀏覽器在 JS 注入內容前即停止，僅有 <head> 而無正文，通常 < 5 KB。
    以 id="jud" 或 htmlcontent 作為正文存在的標記。
    """
    if not html_file or not os.path.exists(html_file):
        return False
    if os.path.getsize(html_file) < _MIN_COMPLETE_SIZE:
        return False
    try:
        with open(html_file, encoding="utf-8", errors="ignore") as f:
            chunk = f.read(20_000)
        return 'id="jud"' in chunk or '"htmlcontent"' in chunk
    except OSError:
        return False


# ─── HTML 儲存 ────────────────────────────────────────────────────────────────
def save_html(html: str, case_number: str) -> str:
    safe = re.sub(r'[\\/*?:"<>|,\s]', "_", case_number)
    path = os.path.join(HTML_DIR, f"{safe}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ─── 資料庫寫入 ───────────────────────────────────────────────────────────────
def upsert_record(
    case_number: str,
    court: str,
    case_title: str,
    judgment_date: str,
    source_url: str,
    html_file: str,
    keyword: str,
) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    try:
        c.execute(
            """INSERT OR IGNORE INTO crawl_records
               (case_number, court, case_title, judgment_date, source_url, html_file, keyword)
               VALUES (?,?,?,?,?,?,?)""",
            (case_number, court, case_title, judgment_date, source_url, html_file, keyword),
        )
        conn.commit()
        if c.rowcount:
            logger.info("  Saved  → %s", case_number)
            return True
        logger.info("  Skip   → %s (already exists)", case_number)
        return False
    finally:
        conn.close()


def _url_case_key(url: str) -> Tuple[str, str, str]:
    """
    從 data.aspx?id=COURT,YEAR,TYPE,NUM,... 解析 (year, type, number)。
    用於驗證下載的 HTML 內容是否與預期案件對應。
    例：id=SCDV%2c115%2c消債更%2c37%2c... → ('115', '消債更', '37')
    """
    try:
        id_part = url.split("id=", 1)[1].split("&")[0]
        parts   = unquote(id_part).split(",")
        if len(parts) >= 4:
            return parts[1].strip(), parts[2].strip(), parts[3].strip()
    except Exception:
        pass
    return "", "", ""


def _html_matches_url(html: str, url: str) -> bool:
    """
    確認 HTML 頁面標題中包含 URL 所對應的年度、案件類型、案號。
    若三者（year, type, number）均出現在標題中，視為內容相符；
    否則視為 stale page（瀏覽器載入了上一筆案件的快取內容）。
    """
    year, ctype, cnum = _url_case_key(url)
    if not year:
        return True  # 無法解析 URL 時，略過驗證

    title_m = re.search(r"<title>\s*(.*?)\s*</title>", html[:2000], re.DOTALL)
    if not title_m:
        return True  # 無 title 時，略過驗證

    title_norm = re.sub(r"\s+", "", title_m.group(1))
    return all(k in title_norm for k in (year, ctype, cnum))


def _mark_recrawled(case_number: str, html_file: str) -> None:
    """Stub 重新爬取後：更新 html_file 路徑，並重設 parsed=0 觸發重新解析。"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE crawl_records SET html_file=?, parsed=0, crawled_at=CURRENT_TIMESTAMP "
        "WHERE case_number=?",
        (html_file, case_number),
    )
    conn.commit()
    conn.close()
    logger.info("  Updated → %s (stub replaced with complete HTML)", case_number)


# ─── 單筆裁判書頁面 ───────────────────────────────────────────────────────────
def crawl_detail_page(driver: webdriver.Chrome, url: str, case_number: str) -> Optional[str]:
    """
    導航至 url 並儲存 HTML。
    page_load_strategy="eager"：driver.get() 在 DOMContentLoaded 後返回；
    裁判書正文在初始 HTML 中，此時已可取得。

    Timeout 後偵測 stale page：若 driver.current_url 仍指向上一筆案件，
    代表本次導航尚未成功，直接跳過（避免用錯誤內容覆蓋目標檔案）。
    """
    try:
        driver.get(url)
    except TimeoutException:
        logger.warning("  Page load timeout for %s — checking URL", case_number)
        try:
            driver.execute_script("window.stop()")
        except Exception:
            pass
        time.sleep(0.8)

        # 比對 id= 參數前 20 字元：若不符表示瀏覽器未成功導航，跳過儲存
        def _id(u: str) -> str:
            return u.split("id=", 1)[1][:20] if "id=" in u else ""

        if _id(url) and _id(url) != _id(driver.current_url):
            logger.warning("  Stale page detected for %s (URL mismatch) — skipping", case_number)
            return None
    except Exception as exc:
        logger.error("  Navigation error for %s — %s", case_number, exc)
        return None

    try:
        html = driver.page_source

        # 若頁面尚未完整載入（stub：僅有 <head> 無正文），等待一次再試
        if not _is_html_content_complete(html):
            time.sleep(2.5)
            html = driver.page_source

        # 內容驗證：確認 HTML 頁面標題與預期案號相符（防止 stale page 覆蓋正確檔案）
        if not _html_matches_url(html, url):
            logger.warning(
                "  Content mismatch for %s — page title does not match URL key — skipping",
                case_number,
            )
            return None

        # 二次確認：若仍為 stub，跳過（不存入殘缺檔案）
        if not _is_html_content_complete(html):
            logger.warning(
                "  Stub HTML for %s (size=%d) — content not loaded — skipping",
                case_number, len(html),
            )
            return None

        html_file = save_html(html, case_number)
        return html_file
    except Exception as exc:
        logger.error("  Failed to save HTML for %s — %s", case_number, exc)
        return None


# ─── Debug 輔助 ───────────────────────────────────────────────────────────────
def _save_debug_html(driver: webdriver.Chrome, tag: str = "debug") -> None:
    path = f"{tag}_page.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    logger.info("Debug HTML → %s  (URL: %s)", path, driver.current_url)


# ─── Step A: 取得結果列表頁網址 ───────────────────────────────────────────────
# 搜尋後司法院頁面在 default.aspx 上以 AJAX 顯示摘要，
# 並提供 qryresultlst.aspx?ty=JUDBOOK&q=<hash> 的「查詢結果」連結。
_LIST_CSS = "a[href*='qryresultlst.aspx?ty=JUDBOOK']"

def _get_results_list_url(driver: webdriver.Chrome) -> Optional[str]:
    """等待並回傳完整結果列表頁的 URL（不含法院篩選參數的版本）。"""
    try:
        WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, _LIST_CSS))
        )
    except TimeoutException:
        logger.warning("qryresultlst link not found — saving debug HTML")
        _save_debug_html(driver, "search_result")
        return None

    links = driver.find_elements(By.CSS_SELECTOR, _LIST_CSS)
    # 優先選「不含 gy=」的那個（完整結果，非依法院篩選）
    for lnk in links:
        href = lnk.get_attribute("href") or ""
        if href and "gy=" not in href:
            logger.info("Results list URL: %s", href)
            return href
    # 備援：直接用第一個
    href = links[0].get_attribute("href") if links else None
    logger.info("Results list URL (fallback): %s", href)
    return href


def _collect_court_links(driver: webdriver.Chrome) -> List[Tuple[str, str]]:
    """
    從當前頁面收集所有「依法院」分群子連結。
    回傳 [(url, label), ...] 列表（依頁面出現順序）。
    法院代碼格式如 TPHV（高等法院民事）、TPDV（臺北地方法院民事）等。
    """
    court_links: List[Tuple[str, str]] = []
    seen: set = set()
    for lnk in driver.find_elements(By.CSS_SELECTOR, _LIST_CSS):
        href = lnk.get_attribute("href") or ""
        if "gy=jcourt&gc=" not in href:
            continue
        m = re.search(r"gy=jcourt&gc=([^&]+)", href)
        if not m:
            continue
        code = m.group(1)
        if code in seen:
            continue
        seen.add(code)
        # lnk.text may include a result-count on a second line — keep only the first line
        raw_label = lnk.text.strip()
        label = raw_label.split("\n")[0].strip() or code
        court_links.append((href, label))
    logger.info("Collected %d court sub-links", len(court_links))
    return court_links


def _case_number_year(url: str) -> Optional[int]:
    """
    從 data.aspx?ty=JD&id=COURT,YEAR,TYPE,NUM,DATE 解析「字號年度」（ROC）。
    例：id=TPHV,112,上,750,20260422 → 112
    ?gy=jyear&gc=<year> 依「字號年度」分群，用此函式保持一致。
    """
    try:
        id_part = url.split("id=", 1)[1].split("&")[0]
        parts   = unquote(id_part).split(",")
        if len(parts) >= 2:
            return int(parts[1].strip())
    except Exception:
        pass
    return None


def _judgment_date(url: str) -> str:
    """
    從 data.aspx?ty=JD&id=COURT,YEAR,TYPE,NUM,DATE 解析「裁判日期」，回傳 ISO(西元) YYYY-MM-DD。
    例：id=TPHV,112,上,750,20260422 → "2026-04-22"
    與 _case_number_year() 取的「案號年度」是兩回事：兩者可相差數年。
    解析失敗回傳空字串。
    """
    try:
        id_part = url.split("id=", 1)[1].split("&")[0]
        parts   = unquote(id_part).split(",")
        if len(parts) >= 5:
            raw = parts[4].strip()
            if len(raw) == 8 and raw.isdigit():
                return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    except Exception:
        pass
    return ""


# ─── 進階搜尋（default_AD.aspx）───────────────────────────────────────────────
# 簡易搜尋只能用 ?gy=/&gc= 分群，而 gy/gc 是「單一互斥軸」（法院、年度、審級三選一），
# 且每個結果集硬性上限 500 筆 —— 單靠分群無法取得超過 500 筆的完整資料。
# 進階搜尋可把裁判日期寫進查詢本身，每個日期區間產生獨立的 q= hash、
# 各自享有自己的 500 額度，因此切到每段 < 500 就能完整取得。
_RESULT_LIMIT = 500          # 單一結果集可翻取的硬性上限
_AD_DATE_FIELDS = ("dy1", "dm1", "dd1", "dy2", "dm2", "dd2")


def _roc_parts(date_str: str) -> Optional[Tuple[int, int, int]]:
    """西元 YYYY/MM/DD（或 YYYY-MM-DD）→ (民國年, 月, 日)。"""
    try:
        y, m, d = re.split(r"[/-]", date_str.strip())
        return int(y) - 1911, int(m), int(d)
    except Exception:
        return None


def _dismiss_alert(driver: webdriver.Chrome) -> str:
    """
    關閉可能出現的 JS alert 並回傳其文字。
    進階搜尋的表單驗證失敗會跳 alert（例如裁判字號填不完整），
    未處理會讓後續所有 driver 操作拋 UnexpectedAlertPresentException。
    """
    try:
        alert = driver.switch_to.alert
        text  = alert.text
        alert.accept()
        logger.warning("Dismissed alert: %s", text)
        return text
    except Exception:
        return ""


def _result_total(driver: webdriver.Chrome) -> Optional[int]:
    """
    從搜尋結果摘要頁取得總筆數（頁面顯示為「查詢結果 4082」）。
    取不到回傳 None（呼叫端應保守地當作可能超過上限）。
    """
    for css in ("div[id*='ount']", "span[id*='ount']", "#jud_count"):
        for el in driver.find_elements(By.CSS_SELECTOR, css):
            m = re.search(r"查詢結果\s*([\d,]+)", el.text or "")
            if m:
                return int(m.group(1).replace(",", ""))
    try:
        m = re.search(r"查詢結果\s*([\d,]+)",
                      driver.find_element(By.TAG_NAME, "body").text)
        if m:
            return int(m.group(1).replace(",", ""))
    except Exception:
        pass
    return None


def _ad_search(
    driver:     webdriver.Chrome,
    keyword:    str,
    start_date: str = "",
    end_date:   str = "",
) -> Tuple[Optional[str], Optional[int]]:
    """
    在進階搜尋頁送出「關鍵字 + 裁判日期區間」查詢。
    日期以民國年月日分別填入 dy1/dm1/dd1（起）與 dy2/dm2/dd2（迄）。
    回傳 (結果列表 URL, 總筆數)；查詢失敗回傳 (None, None)。
    """
    driver.get(BASE_URL_AD)
    time.sleep(2.5)
    _dismiss_alert(driver)

    if keyword:
        kw = driver.find_element(By.ID, "jud_kw")
        kw.clear()
        kw.send_keys(keyword)

    values = {}
    for label, ds in (("1", start_date), ("2", end_date)):
        if not ds:
            continue
        parts = _roc_parts(ds)
        if parts is None:
            logger.warning("Cannot parse date %r — ignored", ds)
            continue
        values[f"dy{label}"], values[f"dm{label}"], values[f"dd{label}"] = parts

    for fid in _AD_DATE_FIELDS:
        if fid in values:
            el = driver.find_element(By.ID, fid)
            el.clear()
            el.send_keys(str(values[fid]))

    driver.find_element(By.ID, "btnQry").click()
    time.sleep(3.5)

    alert_text = _dismiss_alert(driver)
    if alert_text:
        logger.error("Advanced search rejected: %s", alert_text)
        return None, None

    total = _result_total(driver)
    url   = _get_results_list_url(driver)
    logger.info("AD search [%s ~ %s] → total=%s",
                start_date or "*", end_date or "*", total)
    return url, total


# ─── Step B: 從結果列表頁解析個別案件 ────────────────────────────────────────
# qryresultlst.aspx 上每筆案件的連結指向
# data.aspx?ty=JD&id=<court>,<year>,<type>,<num>,<date>
_CASE_LINK_CSS = (
    "a[href*='data.aspx?ty=JD'],"
    "a[href*='data.aspx?ty=jd']"
)

# 從「裁判字號」全文（含法院名稱）解析法院名稱
# 例：「臺灣新竹地方法院 113 年度…」→「臺灣新竹地方法院」
_COURT_RE = re.compile(
    r'^(.*?(?:憲法法庭|少年及家事法院|地方法院|高等法院|最高行政法院|最高法院|高等行政法院'
    r'|行政法院|智慧財產及商業法院|智慧財產法院|海事法院)'
    r'(?:\s+\S+分院|\s+地方庭)?)'
)

def _parse_case_rows(driver: webdriver.Chrome) -> List[Dict]:
    """從 qryresultlst.aspx 擷取個別案件的連結與基本資訊。"""
    rows: List[Dict] = []
    try:
        WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, _CASE_LINK_CSS))
        )
    except TimeoutException:
        logger.warning("Case links not found on results list page — saving debug HTML")
        _save_debug_html(driver, "results_list")
        return rows

    seen: set = set()
    for lnk in driver.find_elements(By.CSS_SELECTOR, _CASE_LINK_CSS):
        href = lnk.get_attribute("href") or ""
        if not href or href in seen:
            continue
        seen.add(href)

        # 實際欄位順序：td[0]=序號  td[1]=裁判字號(含link)  td[2]=裁判日期  td[3]=案由
        # 無獨立法院欄 — 法院名稱內嵌在裁判字號文字的開頭
        case_number = lnk.text.strip()
        cm = _COURT_RE.match(case_number)
        court = cm.group(1).strip() if cm else ""
        try:
            tr  = lnk.find_element(By.XPATH, "./ancestor::tr[1]")
            tds = tr.find_elements(By.TAG_NAME, "td")
            rows.append({
                "case_number":   case_number,
                "court":         court,
                # 裁判日期以 URL id 第 5 欄為準（西元 ISO，可直接字串比較/排序）；
                # td 版型若變動仍可用，故保留 td 文字作為備援。
                "judgment_date": _judgment_date(href) or (
                    tds[2].text.strip() if len(tds) > 2 else ""),
                "case_title":    tds[3].text.strip() if len(tds) > 3 else "",
                "url":           href,
            })
        except Exception:
            rows.append({
                "case_number": case_number,
                "court": court, "judgment_date": _judgment_date(href),
                "case_title": "", "url": href,
            })

    logger.info("Found %d cases on current page", len(rows))
    return rows


# ─── Step C: 翻頁 ─────────────────────────────────────────────────────────────
def _go_next_page(driver: webdriver.Chrome) -> bool:
    """
    點擊「下一頁」並等待新頁案件連結出現。
    使用 JS click 避免元素被遮擋時的 ElementClickInterceptedException。
    """
    next_el = None
    for sel in ("a[title='下一頁']", "a.page-next", "[class*='nextpage'] a"):
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            if el.is_displayed() and el.is_enabled():
                next_el = el
                break
        except NoSuchElementException:
            pass
    if next_el is None:
        try:
            el = driver.find_element(By.LINK_TEXT, "下一頁")
            if el.is_displayed() and el.is_enabled():
                next_el = el
        except NoSuchElementException:
            pass
    if next_el is None:
        return False

    for _pg_attempt in range(2):
        try:
            driver.execute_script("arguments[0].click();", next_el)
        except Exception as exc:
            logger.warning("Next page click failed (attempt %d/2): %s", _pg_attempt + 1, exc)
            if _pg_attempt == 0:
                time.sleep(10)
                continue
            return False

        # 等待新頁的案件連結出現（舊連結會因 DOM 更新而失效）
        time.sleep(1)
        try:
            WebDriverWait(driver, 25).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, _CASE_LINK_CSS))
            )
            return True
        except TimeoutException:
            driver.execute_script("window.stop();")
            time.sleep(1)
            if _pg_attempt == 0:
                logger.warning("Next page load timed out — retrying after 10 s")
                time.sleep(10)
                # 重新尋找「下一頁」按鈕（DOM 可能已更新）
                next_el = None
                for sel in ("a[title='下一頁']", "a.page-next", "[class*='nextpage'] a"):
                    try:
                        el = driver.find_element(By.CSS_SELECTOR, sel)
                        if el.is_displayed() and el.is_enabled():
                            next_el = el
                            break
                    except NoSuchElementException:
                        pass
                if next_el is None:
                    return False
                continue
            return False  # 第二次逾時仍未載入 → 放棄翻頁
    return False


# ─── 主爬蟲函式 ───────────────────────────────────────────────────────────────
def search_and_crawl(
    keyword:     str,
    max_results: int  = 10,
    headless:    bool = True,
    start_date:  str  = "",
    end_date:    str  = "",
    case_year_start: Optional[int] = None,
    case_year_end:   Optional[int] = None,
    driver:      Optional[webdriver.Chrome] = None,
    skip_if_truncated: bool = False,
) -> Dict[str, object]:
    """
    以「關鍵字 + 裁判日期區間」搜尋並爬取裁判書。

    流程：
      1. default_AD.aspx 進階搜尋：關鍵字 + 裁判日期起迄 → 取得該區間專屬的 q= hash
      2. 讀摘要頁「查詢結果 N」判斷是否超過單一結果集上限（_RESULT_LIMIT=500）
      3. 未超過 → 翻頁收集所有案件連結；超過 → 回報 truncated 交由呼叫端切分日期區間
      4. 逐一下載案件 HTML

    參數：
      start_date / end_date       裁判日期起迄（西元 YYYY/MM/DD），伺服器端篩選
      case_year_start / _end      案號年度起迄（民國年），Python 層後過濾。
                                  進階搜尋的 jud_year 必須與字別、案號同時給定，
                                  無法單獨當範圍條件，故只能在本地過濾。
      driver                      傳入既有 driver 可跨多次呼叫重複使用（不會被關閉）
      skip_if_truncated           True 時，一旦偵測到結果被截斷就立即返回不翻頁

    回傳 {"collected": 實際下載數, "total": 該區間總筆數, "truncated": 是否被截斷}
    """
    os.makedirs(HTML_DIR, exist_ok=True)
    init_db()

    own_driver = driver is None
    if own_driver:
        driver = build_driver(headless)
    collected = 0
    total: Optional[int] = None
    truncated = False
    court_links: List[Tuple[str, str]] = []

    try:
        # ── 1. 進階搜尋取得該日期區間專屬的結果集 ─────────────────────
        base_results_url, total = _ad_search(driver, keyword, start_date, end_date)
        if not base_results_url:
            logger.error("Advanced search returned no results URL — aborting")
            return {"collected": 0, "total": total, "truncated": False}

        # ── 2. 判斷是否觸及單一結果集上限 ─────────────────────────────
        # 結果按裁判日期由新到舊排序，截斷時被砍掉的是最舊的部分。
        if total is not None and total >= _RESULT_LIMIT:
            truncated = True
            logger.warning(
                "Result set truncated: total=%d >= limit=%d  [%s ~ %s]",
                total, _RESULT_LIMIT, start_date or "*", end_date or "*",
            )
            if skip_if_truncated:
                logger.info("skip_if_truncated — returning for caller to split")
                return {"collected": 0, "total": total, "truncated": True}

            # 呼叫端表示已無法再切分日期（例如區間已縮到單日），
            # 最後手段：改用法院分群把同一個日期區間再切細。
            # gy=jcourt 與日期條件不衝突 —— 日期在查詢裡，法院是該結果集的分群，
            # 每個法院桶各自獨立計算 500 額度。
            # 必須在離開摘要頁前收集，導覽到 qryresultlst.aspx 後就看不到這些連結。
            court_links = _collect_court_links(driver)
            if court_links:
                logger.info("Fallback: subdividing by court (%d courts)", len(court_links))

        # ── 3. Phase A: 翻頁收集案件連結（新 → 舊）────────────────────
        logger.info("Phase A: collecting case URLs …")
        all_items: List[Dict] = []
        seen_urls: set = set()

        def _filter_by_case_year(items: List[Dict]) -> List[Dict]:
            """案號年度後過濾（與裁判日期是兩個不同維度，可各自獨立指定）。"""
            if case_year_start is None and case_year_end is None:
                return items
            lo = case_year_start if case_year_start is not None else -10 ** 6
            hi = case_year_end   if case_year_end   is not None else 10 ** 6
            return [
                it for it in items
                if _case_number_year(it.get("url", "")) is None
                or lo <= _case_number_year(it.get("url", "")) <= hi
            ]

        def _paginate(label: str) -> None:
            """從目前頁面一路翻頁收集，直到無下一頁或達 max_results。"""
            while len(all_items) < max_results:
                raw_items = _parse_case_rows(driver)
                if not raw_items:
                    break
                fresh = [it for it in _filter_by_case_year(raw_items)
                         if it.get("url") and it["url"] not in seen_urls]
                needed = max_results - len(all_items)
                for it in fresh[:needed]:
                    seen_urls.add(it["url"])
                    all_items.append(it)
                logger.info("  %d / %d collected%s", len(all_items), max_results, label)
                if len(all_items) >= max_results:
                    break
                if not _go_next_page(driver):
                    break
                time.sleep(0.6)

        if court_links:
            # 最後手段路徑：同一日期區間內再依法院逐桶收集
            for court_url, court_label in court_links:
                if len(all_items) >= max_results:
                    break
                logger.info("  [Court] %s", court_label)
                driver.get(court_url)
                time.sleep(2.0)
                _paginate(f"  [{court_label}]")
        else:
            driver.get(base_results_url)
            time.sleep(2.5)
            _paginate("")

        # 防護：摘要頁的總筆數若讀不到（頁面改版等），改以「恰好收滿上限」反推截斷。
        # 未指定案號年度過濾時，收到剛好 _RESULT_LIMIT 筆幾乎必然代表被截斷。
        if (total is None and not truncated and not court_links
                and case_year_start is None and case_year_end is None
                and len(all_items) >= _RESULT_LIMIT
                and max_results >= _RESULT_LIMIT):
            truncated = True
            logger.warning(
                "Total unknown but collected exactly %d — assuming truncated  [%s ~ %s]",
                _RESULT_LIMIT, start_date or "*", end_date or "*",
            )

        logger.info("Phase A complete — %d cases queued", len(all_items))

        # ── 6. Phase B: 下載每個案件 ─────────────────────────────────
        # 每 _SESSION_RENEW_EVERY 筆實際請求（非跳過）重建 WebDriver session，
        # 避免 Chrome renderer 因記憶體耗盡或伺服器限流而崩潰。
        _SESSION_RENEW_EVERY = 80   # 每 80 筆換一次 session
        _session_requests    = 0    # 本 session 已發出的請求數

        def _renew_driver() -> webdriver.Chrome:
            nonlocal driver
            logger.info("Renewing WebDriver session (will sleep 15 s) …")
            try:
                driver.quit()
            except Exception:
                pass
            time.sleep(15)
            driver = build_driver(headless)
            logger.info("New WebDriver session ready.")
            return driver

        logger.info("Phase B: downloading case pages …")
        for item in all_items:
            case_number = item["case_number"] or f"unknown_{collected + 1}"
            logger.info("[%d/%d] %s", collected + 1, len(all_items), case_number)

            # 查詢 DB 是否已有紀錄
            chk = sqlite3.connect(DB_PATH)
            existing = chk.execute(
                "SELECT html_file FROM crawl_records WHERE case_number=?", (case_number,)
            ).fetchone()
            chk.close()

            existing_html = existing[0] if existing else None

            if existing and _is_html_complete(existing_html):
                # 已爬取且 HTML 完整 → 跳過
                logger.info("  Skip   → %s (already downloaded, complete)", case_number)
                collected += 1
                time.sleep(0.2)
                continue

            if existing:
                # 紀錄存在但 HTML 不完整（stub 或遺失）→ 重新爬取
                logger.info("  Re-crawl → %s (stub/incomplete HTML detected)", case_number)

            # 定期重建 session
            if _session_requests > 0 and _session_requests % _SESSION_RENEW_EVERY == 0:
                driver = _renew_driver()

            html_file = None
            try:
                html_file = crawl_detail_page(driver, item["url"], case_number)
                _session_requests += 1
            except WebDriverException as exc:
                logger.warning("WebDriver error on %s: %s — restarting session and retrying",
                               case_number, exc)
                driver = _renew_driver()
                try:
                    html_file = crawl_detail_page(driver, item["url"], case_number)
                    _session_requests += 1
                except WebDriverException as exc2:
                    logger.error("Retry also failed for %s: %s — skipping", case_number, exc2)

            if html_file:
                if existing:
                    _mark_recrawled(case_number, html_file)
                else:
                    upsert_record(
                        case_number, item["court"], item["case_title"],
                        item["judgment_date"], item["url"], html_file, keyword
                    )
                collected += 1

            time.sleep(1.0)

    except WebDriverException as exc:
        logger.error("WebDriver error (outer): %s", exc, exc_info=True)
    finally:
        # 由呼叫端傳入的 driver 交還呼叫端管理（供跨區間重複使用），不在此關閉
        if own_driver:
            try:
                driver.quit()
            except Exception:
                pass

    logger.info("Crawl complete — collected %d / %d (total=%s, truncated=%s)",
                collected, max_results, total, truncated)
    return {"collected": collected, "total": total, "truncated": truncated}


# ─── 重新爬取 stub 紀錄 ───────────────────────────────────────────────────────
def recrawl_stubs(headless: bool = True) -> int:
    """找出所有 stub/不完整 HTML 紀錄並重新下載。"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, case_number, source_url, html_file FROM crawl_records "
        "WHERE source_url IS NOT NULL AND source_url != ''"
    ).fetchall()
    conn.close()

    to_recrawl = []
    for cid, case_number, source_url, html_file in rows:
        if not _is_html_complete(html_file):
            to_recrawl.append((cid, case_number, source_url))

    logger.info("Found %d stub/missing records to re-crawl", len(to_recrawl))
    if not to_recrawl:
        return 0

    os.makedirs(HTML_DIR, exist_ok=True)
    driver    = build_driver(headless)
    recrawled = 0
    _SESSION_RENEW_EVERY = 80
    try:
        for i, (cid, case_number, source_url) in enumerate(to_recrawl, 1):
            logger.info("[%d/%d] Re-crawling: %s", i, len(to_recrawl), case_number)
            if i > 1 and (i - 1) % _SESSION_RENEW_EVERY == 0:
                logger.info("Renewing WebDriver session (will sleep 15 s) …")
                try:
                    driver.quit()
                except Exception:
                    pass
                time.sleep(15)
                driver = build_driver(headless)
            try:
                new_html = crawl_detail_page(driver, source_url, case_number)
            except WebDriverException as exc:
                logger.warning("WebDriver error on %s: %s — restarting and retrying", case_number, exc)
                try:
                    driver.quit()
                except Exception:
                    pass
                time.sleep(15)
                driver = build_driver(headless)
                try:
                    new_html = crawl_detail_page(driver, source_url, case_number)
                except WebDriverException as exc2:
                    logger.error("Retry failed for %s: %s — skipping", case_number, exc2)
                    new_html = None
            if new_html:
                _mark_recrawled(case_number, new_html)
                recrawled += 1
            time.sleep(1.0)
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    logger.info("Re-crawled %d / %d stubs", recrawled, len(to_recrawl))
    return recrawled


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="司法院裁判書自動爬蟲")
    ap.add_argument("keyword", nargs="?", default="",      help="搜尋關鍵字")
    ap.add_argument("-n", "--num",   type=int, default=10, help="爬取筆數 (預設: 10)")
    ap.add_argument("--no-headless", action="store_true",  help="顯示瀏覽器視窗 (debug 用)")
    ap.add_argument("--start-date", default="",
                    help="裁判日期起 YYYY/MM/DD（西元，伺服器端精確篩選）")
    ap.add_argument("--end-date",   default="",
                    help="裁判日期迄 YYYY/MM/DD（西元，伺服器端精確篩選）")
    ap.add_argument("--case-year-start", type=int, default=None,
                    help="案號年度起（民國年，如 113）；與裁判日期是不同維度，於本地過濾")
    ap.add_argument("--case-year-end",   type=int, default=None,
                    help="案號年度迄（民國年，如 115）；與裁判日期是不同維度，於本地過濾")
    ap.add_argument("--recrawl-stubs", action="store_true",
                    help="重新爬取資料庫中所有 stub/不完整 HTML（不需關鍵字）")
    args = ap.parse_args()

    if args.recrawl_stubs:
        n = recrawl_stubs(headless=not args.no_headless)
        print(f"\n完成！共重新爬取 {n} 筆 stub 裁判書。")
    elif args.keyword:
        res = search_and_crawl(
            keyword=args.keyword,
            max_results=args.num,
            headless=not args.no_headless,
            start_date=args.start_date,
            end_date=args.end_date,
            case_year_start=args.case_year_start,
            case_year_end=args.case_year_end,
        )
        print("")
        print(f"完成！共爬取 {res['collected']} 筆裁判書"
              f"（該區間總筆數 {res['total']}）。")
        if res["truncated"]:
            print(f"  ⚠ 結果集達單一查詢上限 {_RESULT_LIMIT} 筆，較舊的資料未取得。")
            print("    請改用 crawl_batched.py，它會自動細分日期區間直到每段低於上限。")
    else:
        ap.print_help()
