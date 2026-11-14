"""Exact numbers from SEC's financial data (XBRL company facts)."""
import csv
import functools
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

DATA = Path(__file__).resolve().parent.parent / "data"

# The XBRL tags to try for each metric, in this order. A company may use any one of them.
METRIC_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "rd_expense": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "operating_income": ["OperatingIncomeLoss"],
}


@functools.lru_cache(maxsize=1)
def ciks() -> dict[str, str]:
    """Ticker -> CIK (SEC's company number), from data/filings.csv."""
    with open(DATA / "filings.csv", encoding="utf-8", newline="") as f:
        return {row["ticker"]: row["cik"] for row in csv.DictReader(f)}


@functools.lru_cache(maxsize=32)  # the file is large and does not change, so ask SEC once per company
def company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    # SEC requires a User-Agent with a name and an email. It comes from the environment.
    resp = httpx.get(url, headers={"User-Agent": os.environ["SEC_USER_AGENT"]}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def annual_value(ticker: str, metric: str, year: int) -> tuple[float | None, str | None]:
    """Return (value in dollars, tag) for a full-year amount, or (None, None) if SEC has no such number.

    The frame 'CY2025' marks a full calendar year. It picks the right value even though
    each 10-K also repeats the older years."""
    facts = company_facts(ciks()[ticker])["facts"].get("us-gaap", {})
    for tag in METRIC_TAGS[metric]:
        for values in facts.get(tag, {}).get("units", {}).values():
            for v in values:
                if v.get("frame") == f"CY{year}":
                    return v["val"], tag
    return None, None


def billions(value: float, year: int) -> str:
    """The project's number format: $9.4 billion (FY2025); a loss is -$3.7 billion (FY2025)."""
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value) / 1e9:.1f} billion (FY{year})"
