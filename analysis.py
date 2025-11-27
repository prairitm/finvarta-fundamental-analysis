#!/usr/bin/env python3
"""
Financial Analysis Script using OpenAI

This script extracts financial data from HTML content and analyzes it using
an OpenAI model configured to act as Aswath Damodaran.

Usage:
    python analysis-localllm.py --company IPL
    python analysis-localllm.py --html-file company_data.html
    python analysis-localllm.py --html-content "<html>...</html>"
"""

import os
import requests
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from bs4 import BeautifulSoup
from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:  # FastAPI API mode is optional
    FastAPI = None  # type: ignore
    HTTPException = None  # type: ignore
    BaseModel = None  # type: ignore


# System prompt for the LLM
SYSTEM_PROMPT = """Role: Act as Aswath Damodaran—a professor who ties a clear business narrative to the numbers. Be rigorous, calm, and transparent. Your job: read the FUNDAMENTAL HTML I paste, extract only what's in it, and decide if the company is worth investing in. If yes, explain *how* to invest (entry logic, tranche plan, and conditions to add/exit)—all based ONLY on the HTML.

Input: I will paste raw HTML of a company's fundamentals after this prompt. Use ONLY that HTML. Do not fetch or assume anything outside it.

Hard Rules
- Use only the pasted HTML. If a field is missing or ambiguous, write "Not available" and move on—do not infer, google, or guess.

- Keep Indian conventions when present (₹, lakhs/crores). Keep units consistent with the HTML.
- Show your math for any derived metric; cite the exact line/label from the HTML you used (e.g., source: "Profit & Loss → Sales").
- Do NOT run a DCF unless every required input (explicit in HTML) is available. Prefer simple, defensible, *HTML-anchored* valuations (e.g., EV/EBIT, P/E, P/B, FCF yield, 5Y median multiples if shown).
- No boilerplate filler. Keep it tight, transparent, and decision-oriented.

Output Format (Markdown)

# One-Glance Verdict (Damodaran-style)
- Verdict: **BUY / WATCH / AVOID** 
- Why in one line (link the **story** to the **numbers**). 
- Data Coverage & Confidence: **High / Medium / Low** (based on how complete the HTML is).

# Extracted Snapshot (from HTML only)
Provide a compact table with what's present in the HTML (else "Not available"):
- Business description (1–2 lines)
- Segment/vertical mix (if shown)
- Price, Market Cap
- TTM Sales, EBITDA, PAT
- ROCE, ROE
- Debt/Equity, Interest Coverage
- Operating Margin, Net Margin
- CFO, Capex, FCF (CFO − Capex)
- Working Capital/Receivable days (if shown)
- Promoter holding & pledging (if shown)
- Dividend Yield
- Valuation multiples provided (P/E, EV/EBIT, P/B, P/S, FCF yield, etc.)
- Any "Key Risks/Notes" section present in HTML
(After the table, list the exact HTML section/labels you pulled each item from.)

# Derived Metrics (show math)
Compute only if inputs are present in HTML; otherwise mark "Not available".
- Revenue CAGR (3Y/5Y)
- EBITDA margin, Net margin (TTM and last FY if both exist)
- FCF = CFO − Capex; **FCF margin** = FCF / Sales
- **Sales-to-Capital** ≈ ΔSales / (Capex + ΔWorking Capital) (if data allows)
- **Reinvestment Rate** ≈ Capex / CFO (or Capex / NOPAT if available)
- **Leverage**: Debt/Equity; **Coverage**: EBIT / Interest
- **Quality flags**: ROCE, ROE trend (multi-year if HTML shows it)

For each, show the formula and plug in the numbers you used; cite HTML labels.

# Narrative → Numbers (Damodaran bridge)
In 4–6 bullets, craft the *business story* grounded in the HTML tables:
- Where growth came from (segments/geography if shown)
- How margins behaved and why (from commentary if present)
- Reinvestment needs (Capex/CFO, Sales-to-Capital)
- Balance-sheet strength (debt, coverage, pledging)
- Governance/working-capital discipline (receivable days, RPTs if listed)
No speculation beyond the HTML.

# Valuation (HTML-anchored only)
Pick methods that the HTML supports (do NOT invent missing inputs):
- **Relative**: Compare current multiple(s) (P/E, EV/EBIT, P/B, P/S, FCF yield) to:
- (a) the company's own multi-year median/range if the HTML shows it,
- otherwise (b) note "peer/market comparison Not available".
- **Owner's earnings view**: If CFO and Capex exist, compute **FCF yield = FCF / Market Cap**.
- **Earnings power**: If EV and EBIT exist, compute **Earnings Yield = EBIT / EV**.
- **Optional DCF**: Only if the HTML provides *all* needed inputs (explicit growth, margins, reinvestment, discount rate). If even one is missing, write "DCF Not performed (inputs missing)."
Conclude with a short valuation summary: Cheap / Fair / Expensive relative to the **HTML-available** anchors.

# Quality & Risk Checklist
Mark each as Strong / Adequate / Weak / Not available (HTML only):
- Profitability (ROCE/ROE vs own history if shown)
- Growth durability (multi-year revenue & PAT trend)
- Reinvestment efficiency (Sales-to-Capital, FCF margin)
- Financial risk (Debt/Equity, Coverage)
- Working capital (Receivable days trend)
- Governance (promoter pledging, auditor notes, RPTs)
Briefly cite the HTML source for each call.

# Decision Rule (deterministic; HTML-based)
Apply the following gates using ONLY HTML numbers (state which passed/failed):
1) **Quality gate**: ROCE ≥ 15% *and* positive FCF (TTM or multi-year median) 
2) **Balance-sheet gate**: Debt/Equity ≤ 1 *and* Interest Coverage ≥ 3× 
3) **Valuation gate** (pick what's available in HTML):
- FCF Yield ≥ 5% **or** EV/EBIT ≤ its own 5Y median **or** P/E ≤ its own 5Y median
If ≥2 of the 3 gates pass → lean **BUY**; if 1 passes → **WATCH**; else **AVOID**. 
(If any metric is Not available, state "Gate cannot be evaluated".)

# If BUY → How to Invest (mechanical, HTML-tied)
Only populate this section if the Decision Rule says BUY.
- **Entry band (data-tied):** 
- If a 5Y median multiple is in HTML: Target **Entry Price** where Current Multiple ≤ (Median × 0.9) for a 10% margin of safety. Show the arithmetic. 
- If FCF Yield exists: Prefer entry when **FCF Yield ≥ 7%** (write current yield; compute price that would imply 7% using HTML FCF). Show math.
- **Tranche plan:** 
- 50% at first touch of entry band; 25% if it falls another 10% without thesis break; 25% when next quarter's HTML KPIs (ROCE, FCF) remain ≥ thresholds above. 
- **Position review / exit triggers (HTML-linked):** 
- Exit/Hold-reduce if ROCE drops below 12% **or** Debt/Equity > 1 **or** FCF turns negative for 2 consecutive periods, as per future HTML. 
- Re-rate to WATCH if valuation stretches to ≥ 1.3× own 5Y median multiple (if shown).
- **What could improve intrinsic value (from HTML only):** margin expansion, working-capital release, deleveraging, etc. (tie each to numbers in the HTML).

# What Could Break the Story
List 3–5 HTML-grounded risks (customer concentration, capex intensity, receivable days spike, pledging, cyclical end-markets, etc.), each with a single data citation from the HTML.

# Sources (from your input)
Bullet the exact HTML sections/labels used (no external links). Example:
- "Profit & Loss → Sales", "Balance Sheet → Borrowings", "Ratios → ROCE", "Shareholding → Pledging", etc.

# Final Call (1-liner)
State BUY / WATCH / AVOID with one crisp reason that ties the *story* to the *numbers*.

Now wait for my HTML. Remember: use ONLY the HTML I provide; if a data point is absent, write "Not available"."""


def estimate_tokens(text: str, conservative: bool = True) -> int:
    """
    Estimate token count. Uses more conservative estimate for HTML content.
    
    Args:
        text: Text to estimate tokens for
        conservative: If True, use ~2.5 chars/token (better for HTML).
                      If False, use ~4 chars/token (plain text).
        
    Returns:
        Estimated token count
    """
    chars_per_token = 2.5 if conservative else 4.0
    return int(len(text) / chars_per_token)


def _print_context_reduction_tips(params, include_sections: Optional[list]) -> None:
    """Suggest ways to cut down payload/context size."""
    max_years = max(1, getattr(params, "max_years", 5))
    max_quarters = max(1, getattr(params, "max_quarters", 8))
    reduced_years = max(1, max_years - 2)
    reduced_quarters = max(1, max_quarters - 4)
    suggestions = [
        f"  1. Reduce years: --max-years {max(3, reduced_years)}",
        f"  2. Reduce quarters: --max-quarters {max(4, reduced_quarters)}",
    ]
    if include_sections is None or len(include_sections) > 3:
        suggestions.append("  3. Limit sections: --sections profit-loss,balance-sheet,ratios")
    suggestions.extend(
        [
            "  4. Enable aggressive compression: --aggressive",
            "  5. Increase context limit: --max-context <new_limit>",
        ]
    )
    for line in suggestions:
        print(line, file=sys.stderr)


def _load_env_from_file(env_path: Path) -> None:
    """Basic .env loader when python-dotenv is unavailable."""
    if not env_path.is_file():
        return
    try:
        with env_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError as exc:
        print(f"Warning: Unable to read {env_path}: {exc}", file=sys.stderr)


def _env_flag(var_name: str, default: bool = False) -> bool:
    """Read boolean values from environment variables."""
    value = os.getenv(var_name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# Load environment variables from .env (python-dotenv preferred, fallback to manual parser)
PROJECT_ROOT = Path(__file__).resolve().parent
if load_dotenv:
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
else:
    _load_env_from_file(PROJECT_ROOT / ".env")


def extract_financial_data(
    html_content: str,
    max_years: int = 5,
    max_quarters: int = 8,
    include_sections: Optional[list] = None,
    aggressive: bool = False
) -> str:
    """
    Extract only essential financial data and create minimal HTML structure.
    
    Args:
        html_content: Raw HTML content from screener.in or similar source
        max_years: Maximum number of years of historical data to include (default: 5)
        max_quarters: Maximum number of quarters to include (default: 8)
        include_sections: List of section IDs to include. If None, includes all sections.
                         Valid sections: 'quarters', 'profit-loss', 'balance-sheet', 
                         'cash-flow', 'ratios', 'shareholding'
        aggressive: If True, summarize older data instead of full tables
        
    Returns:
        Cleaned HTML string containing only financial data
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Default sections to keep
    all_sections = ['quarters', 'profit-loss', 'balance-sheet', 'cash-flow', 'ratios', 'shareholding']
    sections_to_keep = include_sections if include_sections is not None else all_sections
    
    # Build minimal HTML structure
    html_parts = ['<html><body>']
    
    # Company name
    h1 = soup.find('h1')
    if h1:
        html_parts.append(f'<h1>{h1.get_text(strip=True)}</h1>')
    
    # Key ratios
    ratios_ul = soup.find('ul', id='top-ratios')
    if ratios_ul:
        html_parts.append('<h2>Key Ratios</h2><ul>')
        for li in ratios_ul.find_all('li'):
            name = li.find('span', class_='name')
            value = li.find('span', class_='value')
            if name and value:
                html_parts.append(f'<li>{name.get_text(strip=True)}: {value.get_text(strip=True)}</li>')
        html_parts.append('</ul>')
    
    # About section
    about = soup.find('div', class_='about')
    if about:
        html_parts.append('<h2>About</h2>')
        html_parts.append(f'<p>{about.get_text(strip=True)}</p>')
    
    # Pros and Cons
    pros = soup.find('div', class_='pros')
    cons = soup.find('div', class_='cons')
    if pros or cons:
        html_parts.append('<h2>Analysis</h2>')
        if pros:
            html_parts.append('<h3>Pros</h3><ul>')
            for li in pros.find_all('li'):
                html_parts.append(f'<li>{li.get_text(strip=True)}</li>')
            html_parts.append('</ul>')
        if cons:
            html_parts.append('<h3>Cons</h3><ul>')
            for li in cons.find_all('li'):
                html_parts.append(f'<li>{li.get_text(strip=True)}</li>')
            html_parts.append('</ul>')
    
    # Extract financial tables with filtering
    for section_id in sections_to_keep:
        section = soup.find('section', id=section_id)
        if section:
            h2 = section.find('h2')
            if h2:
                html_parts.append(f'<h2>{h2.get_text(strip=True)}</h2>')
            
            # Extract tables
            tables = section.find_all('table', class_='data-table')
            for table in tables:
                html_parts.append('<table>')
                # Headers
                thead = table.find('thead')
                if thead:
                    html_parts.append('<thead><tr>')
                    ths = thead.find_all('th')
                    
                    # Filter columns based on section type
                    if section_id == 'quarters':
                        # Keep first column (row labels) + last N quarters
                        columns_to_keep = [0] + list(range(max(1, len(ths) - max_quarters), len(ths)))
                    elif section_id in ['profit-loss', 'balance-sheet', 'cash-flow', 'ratios']:
                        # Keep first column (row labels) + TTM + last N years
                        # TTM is typically the last column before the years
                        columns_to_keep = [0]  # Always keep first column
                        # Find TTM column if exists
                        ttm_index = None
                        for i, th in enumerate(ths):
                            if th.get_text(strip=True).upper() == 'TTM':
                                ttm_index = i
                                break
                        if ttm_index is not None:
                            columns_to_keep.append(ttm_index)
                        # Add last N years (excluding TTM)
                        year_cols = [i for i in range(1, len(ths)) if i != ttm_index]
                        columns_to_keep.extend(year_cols[-max_years:])
                        columns_to_keep = sorted(set(columns_to_keep))
                    else:
                        # For shareholding and other sections, keep all columns
                        columns_to_keep = list(range(len(ths)))
                    
                    for i in columns_to_keep:
                        if i < len(ths):
                            html_parts.append(f'<th>{ths[i].get_text(strip=True)}</th>')
                    html_parts.append('</tr></thead>')
                    
                    # Body - filter rows to match filtered columns
                    tbody = table.find('tbody')
                    if tbody:
                        html_parts.append('<tbody>')
                        for tr in tbody.find_all('tr'):
                            html_parts.append('<tr>')
                            tds = tr.find_all(['td', 'th'])
                            for i in columns_to_keep:
                                if i < len(tds):
                                    cell_text = tds[i].get_text(strip=True)
                                    # Skip empty cells in aggressive mode
                                    if not (aggressive and not cell_text):
                                        html_parts.append(f'<td>{cell_text}</td>')
                            html_parts.append('</tr>')
                        html_parts.append('</tbody>')
                html_parts.append('</table>')
            
            # Growth tables (ranges-table) - keep all, they're small
            growth_tables = section.find_all('table', class_='ranges-table')
            if growth_tables:
                html_parts.append('<h3>Growth Metrics</h3>')
                for table in growth_tables:
                    html_parts.append('<table>')
                    for tr in table.find_all('tr'):
                        html_parts.append('<tr>')
                        for td in tr.find_all(['td', 'th']):
                            html_parts.append(f'<td>{td.get_text(strip=True)}</td>')
                        html_parts.append('</tr>')
                    html_parts.append('</table>')
    
    html_parts.append('</body></html>')
    
    return ''.join(html_parts)


def analyze_with_llm(
    financial_data: str,
    base_url: Optional[str],
    model: str,
    api_key: str
) -> str:
    """
    Send financial data to an OpenAI model for analysis.
    
    Args:
        financial_data: Cleaned HTML financial data
        base_url: Base URL for the OpenAI-compatible API (None => default)
        model: Model name to use (e.g., gpt-4o-mini)
        api_key: OpenAI API key
        
    Returns:
        Analysis response from the LLM
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    ) if base_url else OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": financial_data}
        ]
    )
    
    return response.choices[0].message.content


def load_html_from_file(file_path: Path) -> str:
    """Load HTML content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def parse_cookie_header(cookie_header: str) -> dict:
    """Convert a raw cookie header string into a dict for requests."""
    cookies = {}
    for part in cookie_header.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            cookies[k.strip()] = v.strip()
    return cookies


def build_screener_headers() -> dict:
    """Return browser-like headers for screener.in requests."""
    return {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "referer": "https://www.screener.in/",
        "sec-ch-ua": "\"Not/A)Brand\";v=\"8\", \"Chromium\";v=\"120\", \"Google Chrome\";v=\"120\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"macOS\"",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }


def fetch_company_html(company: str, cookie_header: Optional[str] = None, timeout: int = 20) -> str:
    """
    Download Screener HTML for the given company ticker.
    
    Args:
        company: Ticker/symbol as used on Screener (e.g., IPL)
        cookie_header: Raw cookie header string for authenticated access
        timeout: Request timeout in seconds
    """
    ticker = company.strip().upper()
    if not ticker:
        print("Error: --company value cannot be empty.", file=sys.stderr)
        sys.exit(1)
    
    url = f"https://www.screener.in/company/{ticker}/"
    headers = build_screener_headers()
    cookies = parse_cookie_header(cookie_header) if cookie_header else None
    
    print(f"Fetching Screener page for {ticker}...", file=sys.stderr)
    try:
        response = requests.get(url, headers=headers, cookies=cookies, timeout=timeout)
        response.raise_for_status()
    except requests.HTTPError as http_err:
        status = http_err.response.status_code if http_err.response else "unknown"
        print(f"Error: Failed to fetch Screener page (status {status}).", file=sys.stderr)
        if status == 403:
            print("Screener returned 403 (forbidden). You may need to provide authenticated cookies via --cookie-header or SCREENER_COOKIE_HEADER.", file=sys.stderr)
        elif status == 404:
            print(f"Screener cannot find ticker '{ticker}'. Double-check the symbol on screener.in.", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as req_err:
        print(f"Network error while fetching Screener page: {req_err}", file=sys.stderr)
        sys.exit(1)
    
    print(f"✅ Screener HTML fetched successfully for {ticker} ({len(response.text):,} characters).", file=sys.stderr)
    return response.text


def resolve_api_key(cli_key: Optional[str]) -> str:
    """
    Resolve the API key to use for OpenAI calls.
    
    Priority:
        1. CLI-provided key
        2. OPENAI_API_KEY environment variable
    """
    if cli_key:
        return cli_key
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    print("Error: No OpenAI API key provided. Use --api-key or set OPENAI_API_KEY.", file=sys.stderr)
    sys.exit(1)


def perform_analysis(params) -> Dict[str, Any]:
    """
    Core analysis workflow used by the FastAPI entrypoint (reusable elsewhere).
    
    Params should expose the same attributes defined in AnalysisRequest.
    Returns a dictionary containing analysis output and metadata.
    """
    # Determine HTML source (file, inline, or Screener fetch)
    html_content: Optional[str] = None
    html_source_desc: Optional[str] = None
    cookie_header = getattr(params, "cookie_header", None) or os.getenv("SCREENER_COOKIE_HEADER")
    
    html_file_param = getattr(params, "html_file", None)
    if html_file_param:
        html_path = html_file_param if isinstance(html_file_param, Path) else Path(html_file_param)
        html_content = load_html_from_file(html_path)
        html_source_desc = f"file: {html_path}"
    elif getattr(params, "html_content", None):
        html_content = params.html_content
        html_source_desc = "inline --html-content"
    elif getattr(params, "company", None):
        if not cookie_header:
            print("⚠️  No Screener cookies provided; attempting anonymous fetch (may fail for some users).", file=sys.stderr)
        html_content = fetch_company_html(params.company, cookie_header=cookie_header)
        html_source_desc = f"screener company {params.company.strip().upper()}"
    else:
        print("Error: Provide HTML input via html_file, html_content, or company parameter.", file=sys.stderr)
        raise SystemExit(1)
    
    print(f"HTML source: {html_source_desc}", file=sys.stderr)
    
    # Parse sections if provided
    include_sections = None
    sections_arg = getattr(params, "sections", None)
    if sections_arg:
        if isinstance(sections_arg, str):
            include_sections = [s.strip() for s in sections_arg.split(',')]
        elif isinstance(sections_arg, (list, tuple)):
            include_sections = [str(s).strip() for s in sections_arg]
        else:
            print("Error: --sections must be a comma-separated string or list.", file=sys.stderr)
            raise SystemExit(1)
        valid_sections = ['quarters', 'profit-loss', 'balance-sheet', 'cash-flow', 'ratios', 'shareholding']
        invalid = [s for s in include_sections if s not in valid_sections]
        if invalid:
            print(f"Error: Invalid sections: {', '.join(invalid)}", file=sys.stderr)
            print(f"Valid sections: {', '.join(valid_sections)}", file=sys.stderr)
            raise SystemExit(1)
    
    # Extract financial data
    print("Extracting financial data from HTML...", file=sys.stderr)
    financial_data = extract_financial_data(
        html_content,
        max_years=getattr(params, "max_years", 5),
        max_quarters=getattr(params, "max_quarters", 8),
        include_sections=include_sections,
        aggressive=getattr(params, "aggressive", False)
    )
    
    # Estimate token count (conservative for HTML content)
    system_tokens = estimate_tokens(SYSTEM_PROMPT, conservative=False)
    data_tokens = estimate_tokens(financial_data, conservative=True)
    total_tokens = system_tokens + data_tokens
    
    # Warn if user-set context may exceed server limits
    max_context = getattr(params, "max_context", 4096)
    if max_context > 4096:
        print(f"⚠️  Note: --max-context {max_context} specified, but many LLM servers cap at 4096 tokens.", file=sys.stderr)
        print("   If context errors persist, lower this value or increase the server context.", file=sys.stderr)
        print(file=sys.stderr)
    
    # Show HTML size statistics if requested
    if getattr(params, "show_stats", False):
        reduction_pct = ((len(html_content) - len(financial_data)) / len(html_content) * 100) if html_content else 0
        print(f"\nHTML Size Statistics:", file=sys.stderr)
        print(f"  Original: {len(html_content):,} characters", file=sys.stderr)
        print(f"  Cleaned:  {len(financial_data):,} characters", file=sys.stderr)
        print(f"  Reduction: {reduction_pct:.1f}%", file=sys.stderr)
        print(file=sys.stderr)
    
    # Always show token estimates (critical for context management)
    print(f"Token Estimates:", file=sys.stderr)
    print(f"  System prompt: ~{system_tokens:,} tokens", file=sys.stderr)
    print(f"  Financial data: ~{data_tokens:,} tokens", file=sys.stderr)
    print(f"  Total: ~{total_tokens:,} tokens", file=sys.stderr)
    print(f"  Context limit: {max_context:,} tokens", file=sys.stderr)
    
    # Pre-flight validation
    if total_tokens > max_context:
        print(f"\n⚠️  WARNING: Estimated tokens ({total_tokens:,}) exceed context limit ({max_context:,})", file=sys.stderr)
        print(f"\nSuggestions to reduce size:", file=sys.stderr)
        _print_context_reduction_tips(params, include_sections)
        print(f"\nProceeding anyway... (may fail)\n", file=sys.stderr)
    elif total_tokens > max_context * 0.9:
        print(f"\n⚠️  WARNING: Approaching context limit ({total_tokens:,} / {max_context:,} tokens)", file=sys.stderr)
        print(file=sys.stderr)
    else:
        print(f"  Status: ✅ Within context limit", file=sys.stderr)
        print(file=sys.stderr)
    
    # Preview mode
    if getattr(params, "preview", False):
        print("=" * 80)
        print("Preview of cleaned HTML (first 2000 characters):")
        print("=" * 80)
        print(financial_data[:2000])
        if len(financial_data) > 2000:
            print(f"\n... (truncated, total length: {len(financial_data):,} characters)")
        return {
            "preview": financial_data[:2000],
            "html_source": html_source_desc,
            "token_estimates": {
                "system": system_tokens,
                "financial": data_tokens,
                "total": total_tokens,
                "context_limit": max_context,
            }
        }
    
    api_key = resolve_api_key(getattr(params, "api_key", None))
    
    # Run analysis
    print("Sending to LLM for analysis...", file=sys.stderr)
    try:
        analysis = analyze_with_llm(
            financial_data,
            base_url=getattr(params, "base_url", None),
            model=getattr(params, "model", "gpt-4o-mini"),
            api_key=api_key
        )
        return {
            "analysis": analysis
        }
    except Exception as e:
        error_str = str(e)
        print(f"Error during LLM analysis: {e}", file=sys.stderr)
        
        # Check for context size errors
        if 'context' in error_str.lower() or 'exceed' in error_str.lower() or '400' in error_str:
            print(f"\n❌ Context size error detected!", file=sys.stderr)
            print(f"\nThe data is too large for the LLM's context window.", file=sys.stderr)
            print(f"\nTry these options to reduce size:", file=sys.stderr)
            _print_context_reduction_tips(params, include_sections)
        else:
            print("\nCheck your OpenAI credentials and network connectivity.", file=sys.stderr)
            print("  - Verify that the API key is valid and has access to the selected model.", file=sys.stderr)
            if not getattr(params, "base_url", None):
                print("  - If you are using the public OpenAI API, check https://status.openai.com/", file=sys.stderr)
            else:
                print(f"  - Custom endpoint: {getattr(params, 'base_url')}", file=sys.stderr)
        raise
    

app = None  # FastAPI app placeholder for uvicorn mode
if FastAPI and BaseModel:
    try:
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        CORSMiddleware = None
    class AnalysisRequest(BaseModel):
        """Schema for FastAPI requests."""
        html_file: Optional[str] = None
        html_content: Optional[str] = None
        company: Optional[str] = None
        cookie_header: Optional[str] = None
        base_url: Optional[str] = None
        model: Optional[str] = "gpt-4o-mini"
        api_key: Optional[str] = None
        show_stats: bool = False
        preview: bool = False
        max_years: int = 5
        max_quarters: int = 8
        sections: Optional[Union[str, List[str]]] = None
        aggressive: bool = False
        max_context: int = 4096

    app = FastAPI(title="Finvarta Fundamental Analysis API")

    if CORSMiddleware:
        allowed_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.post("/analyze")
    def analyze_via_api(payload: AnalysisRequest):
        """HTTP endpoint wrapper around perform_analysis."""
        try:
            return perform_analysis(payload)
        except SystemExit as exc:
            if HTTPException is None:
                raise
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health")
    def healthcheck():
        """Simple readiness probe."""
        return {"status": "ok"}


def _serve_with_uvicorn(host: str, port: int, reload_server: bool) -> None:
    """Start the FastAPI app using uvicorn."""
    if app is None:
        print(
            "Error: FastAPI is not available. Install fastapi and pydantic to use --serve.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        import uvicorn  # Local import keeps uvicorn optional for CLI use
    except ImportError:
        print("Error: uvicorn is not installed. Run `pip install uvicorn` to use --serve.", file=sys.stderr)
        sys.exit(1)
    uvicorn.run(app, host=host, port=port, reload=reload_server)


if __name__ == "__main__":
    default_host = os.getenv("ANALYSIS_SERVER_HOST", "0.0.0.0")
    default_port = int(os.getenv("ANALYSIS_SERVER_PORT", "8000"))
    reload_flag = _env_flag("ANALYSIS_SERVER_RELOAD", False)
    _serve_with_uvicorn(default_host, default_port, reload_flag)
