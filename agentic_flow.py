import html
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import openai
import pandas as pd
from urllib import error as urllib_error, parse as urllib_parse, request as urllib_request
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Stock(BaseModel):
        Security: str
        Symbol: str
        Mkt_Value: float
        Portfolio_Weight: float
        Price: Optional[float] = None
        Change_1m: Optional[float] = None
        Change_1y: Optional[float] = None


class NewsArticle(BaseModel):
        title: str
        url: str
        source: str
        summary: Optional[str] = None
        importance: float = 0.0
        related_symbol: Optional[str] = None


class Newsletter(BaseModel):
        top_stocks: List[Stock]
        articles: List[NewsArticle]


# ---------------------------------------------------------------------------
# Core extraction helpers
# ---------------------------------------------------------------------------

def agent_extract_pdf_text(pdf_path: str) -> str:
        """Extract raw text from a PDF using PyPDF2."""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
                text += page.extract_text() or ""
        return text


def agent_analyze_with_openai(text: str):
        """Use OpenAI to pull structured holdings data from PDF text."""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required to analyze the portfolio statement.")

        client = openai.OpenAI(api_key=api_key)
        prompt = (
                "Extract only the account balances from the following portfolio statement PDF text. "
                "Ignore trading activity and overall account balance. "
                "Return a JSON array where each object represents a security with these keys: "
                "'Security' (e.g. Apple), 'Symbol' (e.g. AAPL), 'Mkt Value', and '% of Total Portfolio'. "
                "Do not include any other information. "
                f"Here is the PDF text: {text[:40000]}"
        )

        response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        content = content.strip()

        if content.startswith("```"):
                content = _strip_code_fence(content)

        try:
                return json.loads(content)
        except Exception:
                logger.warning("OpenAI returned non-JSON content. Falling back to raw text.")
                return {"raw": content}


def agent_build_dataframe(analysis) -> pd.DataFrame:
        """Build a tidy dataframe from the LLM output."""
        if isinstance(analysis, dict) and "raw" in analysis:
                df = pd.DataFrame([analysis])
        else:
                df = pd.DataFrame(analysis)

        try:
                df_sorted = df.copy()
                df_sorted["Mkt Value"] = df_sorted["Mkt Value"].replace("[^0-9.]", "", regex=True).astype(float)
                df_sorted = df_sorted.sort_values(by="Mkt Value", ascending=False)
                df_sorted["% of Total Portfolio"] = (
                        df_sorted["% of Total Portfolio"].replace("[^0-9.]", "", regex=True).astype(float)
                )
                df_sorted["Cumulative Portfolio Weight"] = df_sorted["% of Total Portfolio"].cumsum()
        except Exception as exc:
                logger.warning("Falling back to unsorted dataframe: %s", exc)
                df_sorted = df
        return df_sorted
# ---------------------------------------------------------------------------
# Market data + news helpers
# ---------------------------------------------------------------------------

def agent_get_top_stocks(df: pd.DataFrame, threshold: float = 80.0, limit: int = 5) -> List[Stock]:
        stocks: List[Stock] = []
        cumulative = 0.0
        for _, row in df.iterrows():
                try:
                        market_value = float(row.get("Mkt Value", 0.0))
                except Exception:
                        market_value = 0.0
                try:
                        weight = float(row.get("% of Total Portfolio", 0.0))
                except Exception:
                        weight = 0.0
                stock = Stock(
                        Security=str(row.get("Security", "")),
                        Symbol=str(row.get("Symbol", "")),
                        Mkt_Value=market_value,
                        Portfolio_Weight=weight,
                )
                metrics = _fetch_stock_metrics(stock.Symbol)
                if metrics:
                        stock.Price = metrics.get("price")
                        stock.Change_1m = metrics.get("change_1m")
                        stock.Change_1y = metrics.get("change_1y")
                stocks.append(stock)
                cumulative += weight
                if cumulative >= threshold or len(stocks) >= limit:
                        break
        return stocks


def _fetch_stock_metrics(symbol: str) -> Optional[dict]:
        if not symbol:
                return None
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"range": "1y", "interval": "1d"}
        try:
                query = urllib_parse.urlencode(params)
                req = urllib_request.Request(f"{url}?{query}", headers={"User-Agent": "Mozilla/5.0"})
                with urllib_request.urlopen(req, timeout=10) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                result = (payload.get("chart", {}).get("result") or [None])[0]
                if not result:
                        return None
                timestamps = result.get("timestamp", [])
                closes = result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
                points = []
                for ts, close in zip(timestamps, closes):
                        if close is None:
                                continue
                        dt = datetime.fromtimestamp(ts)
                        points.append((dt, float(close)))
                if not points:
                        return None
                points.sort(key=lambda item: item[0])
                latest_date, latest_price = points[-1]
                price_1m = _price_on_or_before(points, latest_date - timedelta(days=30))
                price_1y = _price_on_or_before(points, latest_date - timedelta(days=365))
                return {
                        "price": round(latest_price, 2),
                        "change_1m": _percentage_change(price_1m, latest_price) if price_1m else None,
                        "change_1y": _percentage_change(price_1y, latest_price) if price_1y else None,
                }
        except Exception as exc:
                logger.warning("Failed to fetch Yahoo Finance data for %s: %s", symbol, exc)
                return None


def _price_on_or_before(points: List[Tuple[datetime, float]], target_date: datetime) -> Optional[float]:
        chosen: Optional[float] = None
        for dt, price in points:
                if dt <= target_date:
                        chosen = price
                else:
                        break
        if chosen is None and points:
                chosen = points[0][1]
        return chosen


def _percentage_change(previous: Optional[float], current: Optional[float]) -> Optional[float]:
        if previous is None or current is None or previous == 0:
                return None
        return round(((current - previous) / previous) * 100, 2)


def agent_search_news_tavily(symbol: str, api_key: str, max_results: int = 5) -> List[NewsArticle]:
        if not api_key:
                return []
        today = datetime.utcnow().date()
        three_days_ago = today - timedelta(days=3)

        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {api_key}"}
        query = f"{symbol} stock news after:{three_days_ago}"
        payload = {"query": query, "max_results": max_results}
        try:
                request_data = json.dumps(payload).encode('utf-8')
                req = urllib_request.Request(
                        url,
                        data=request_data,
                        headers={**headers, 'Content-Type': 'application/json'},
                        method='POST',
                )
                with urllib_request.urlopen(req, timeout=10) as response:
                        data = json.loads(response.read().decode('utf-8'))
        except urllib_error.HTTPError as http_exc:
                error_body = http_exc.read().decode('utf-8', errors='ignore')
                logger.error("Tavily API error for %s: %s - %s", symbol, http_exc.code, error_body)
                return []
        except Exception as exc:
                logger.error("Exception during Tavily search for %s: %s", symbol, exc)
                return []

        articles: List[NewsArticle] = []
        for item in data.get('results', []):
                pub_date = item.get('published_date', '')
                if pub_date:
                        try:
                                pub_dt = datetime.strptime(pub_date[:10], "%Y-%m-%d").date()
                                if pub_dt < three_days_ago:
                                        continue
                        except Exception:
                                pass
                articles.append(
                        NewsArticle(
                                title=item.get('title', ''),
                                url=item.get('url', ''),
                                source=item.get('source', ''),
                                summary=item.get('description', ''),
                        )
                )
        return articles


def agent_filter_articles(articles: List[NewsArticle], stock: Stock) -> List[NewsArticle]:
        credible_sources = {
                "Reuters",
                "Bloomberg",
                "WSJ",
                "CNBC",
                "Financial Times",
                "Yahoo Finance",
                "MarketWatch",
                "Seeking Alpha",
                "CNN",
        }
        exclude_sources = {"Benzinga", "TipRanks", "Forbes"}
        filtered = [
                article
                for article in articles
                if (any(src in article.source for src in credible_sources) or not article.source)
                and not any(ex in article.source for ex in exclude_sources)
        ]
        if not filtered:
                filtered = articles

        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key and filtered:
                try:
                        client = openai.OpenAI(api_key=openai_api_key)
                        for article in filtered:
                                system_prompt = (
                                        "You are a sharp financial editor who writes punchy newsletter blurbs."
                                )
                                user_prompt = (
                                        "Summarize the following news item for an investor update in 2-3 sentences. "
                                        "Keep it high-level, mention the expected impact for the stock, "
                                        "and stay neutral."
                                        f"\nStock: {stock.Security} ({stock.Symbol})"
                                        f"\nSource: {article.source}"
                                        f"\nTitle: {article.title}"
                                        f"\nExisting Summary: {article.summary or 'N/A'}"
                                )
                                response = client.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=[
                                                {"role": "system", "content": system_prompt},
                                                {"role": "user", "content": user_prompt},
                                        ],
                                        max_tokens=220,
                                        temperature=0.3,
                                )
                                summary = (response.choices[0].message.content or "").strip()
                                if summary.startswith("```"):
                                        summary = _strip_code_fence(summary)
                                article.summary = summary
                                article.importance = len(summary) + (stock.Portfolio_Weight or 0.0) * 2
                                article.related_symbol = stock.Symbol
                except Exception as exc:
                        logger.warning("Unable to summarize articles with OpenAI: %s", exc)

        for article in filtered:
                article.related_symbol = article.related_symbol or stock.Symbol
                if not article.importance:
                        article.importance = len(article.summary or "") + (stock.Portfolio_Weight or 0.0)
        filtered.sort(key=lambda a: a.importance, reverse=True)
        return filtered[:3]
# ---------------------------------------------------------------------------
# Newsletter creation
# ---------------------------------------------------------------------------

def agent_build_newsletter(df_sorted: pd.DataFrame, tavily_api_key: Optional[str]) -> Tuple[Newsletter, str]:
        top_stocks = agent_get_top_stocks(df_sorted)
        all_articles: List[NewsArticle] = []

        if tavily_api_key:
                for stock in top_stocks:
                        articles = agent_search_news_tavily(stock.Symbol, tavily_api_key, max_results=6)
                        filtered = agent_filter_articles(articles, stock)
                        for article in filtered:
                                article.importance += stock.Portfolio_Weight or 0.0
                        all_articles.extend(filtered)
        else:
                logger.warning("TAVILY_API_KEY not provided; skipping live news search.")

        all_articles.sort(key=lambda a: a.importance, reverse=True)
        newsletter = Newsletter(top_stocks=top_stocks, articles=all_articles[:5])
        markdown_content = agent_generate_markdown(newsletter)
        return newsletter, markdown_content


def agent_generate_markdown(newsletter: Newsletter) -> str:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        date_str = datetime.utcnow().strftime("%B %d, %Y")
        payload = {
                "generated_on": date_str,
                "stocks": [
                        {
                                "name": stock.Security,
                                "symbol": stock.Symbol,
                                "portfolio_weight": stock.Portfolio_Weight,
                                "market_value": stock.Mkt_Value,
                                "price": stock.Price,
                                "change_1m": stock.Change_1m,
                                "change_1y": stock.Change_1y,
                        }
                        for stock in newsletter.top_stocks
                ],
                "articles": [
                        {
                                "title": article.title,
                                "source": article.source,
                                "summary": article.summary,
                                "url": article.url,
                                "related_symbol": article.related_symbol,
                        }
                        for article in newsletter.articles
                ],
        }

        if not openai_api_key:
                logger.error("OPENAI_API_KEY not configured. Using fallback newsletter copy.")
                return _build_fallback_markdown(payload)

        client = openai.OpenAI(api_key=openai_api_key)
        system_prompt = (
                "You are an upbeat market analyst writing a breezy but insightful investor newsletter. "
                "Write in the style of Anand Sanwal's CB Insights and Emma Tucker's WSJ 10 Point: smart, "
                "punchy, and conversational. Always respond with valid Markdown only."
        )
        user_prompt = (
                "Craft today's portfolio newsletter. Include:"
                "\n- A sharp intro paragraph with a hook."
                "\n- A TL;DR section with 3-5 punchy bullets summarizing the biggest moves."
                "\n- A table titled 'Market Moves' with columns Stock, Price (USD), 1M %, 1Y %, Portfolio Weight %."
                "\n- A 'Headlines' section that walks through each article with descriptive subheads and inline links."
                "\n- A closing takeaway or action item."
                "\nKeep paragraphs short (max 3 sentences), avoid jargon, and make it fun to read."
                "\nUse the following structured data as your source of truth:\n"
                f"{json.dumps(payload, indent=2)}"
        )
        try:
                response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=1100,
                        temperature=0.4,
                )
                content = (response.choices[0].message.content or "").strip()
                if content.startswith("```"):
                        content = _strip_code_fence(content)
                return content
        except Exception as exc:
                logger.error("OpenAI newsletter generation failed: %s", exc)
        return _build_fallback_markdown(payload)


def _build_fallback_markdown(payload: dict) -> str:
        date_str = payload.get("generated_on", datetime.utcnow().strftime("%B %d, %Y"))
        stocks = payload.get("stocks", [])
        articles = payload.get("articles", [])
        lines = [f"# Portfolio Pulse — {date_str}"]
        if stocks:
                lines.append("\n## TL;DR")
                for stock in stocks[:4]:
                        lines.append(
                                f"- {stock.get('name')} ({stock.get('symbol')}): {float(stock.get('portfolio_weight', 0.0)):.2f}% of the portfolio."
                        )
                lines.append("\n## Market Moves")
                lines.append("| Stock | Price (USD) | 1M % | 1Y % | Portfolio Weight % |")
                lines.append("| --- | ---: | ---: | ---: | ---: |")
                for stock in stocks:
                        lines.append(
                                "| {name} ({symbol}) | {price} | {change_1m} | {change_1y} | {weight:.2f} |".format(
                                        name=stock.get("name", "N/A"),
                                        symbol=stock.get("symbol", "N/A"),
                                        price=_format_optional_number(stock.get("price")),
                                        change_1m=_format_optional_number(stock.get("change_1m"), suffix="%"),
                                        change_1y=_format_optional_number(stock.get("change_1y"), suffix="%"),
                                        weight=float(stock.get("portfolio_weight", 0.0)),
                                )
                        )
        if articles:
                lines.append("\n## Headlines")
                for article in articles:
                        title = article.get("title", "")
                        url = article.get("url")
                        source = article.get("source", "")
                        summary = article.get("summary", "")
                        if url:
                                heading = f"### [{title}]({url})"
                        else:
                                heading = f"### {title}"
                        lines.append(heading)
                        if source:
                                lines.append(f"*Source: {source}*")
                        if summary:
                                lines.append(summary)
        lines.append("\n## What to Watch")
        lines.append("Stay close to the tape and keep an eye on how earnings guidance shapes the next leg of performance.")
        return "\n".join(lines)
# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _markdown_to_html(markdown_content: str) -> str:
        lines = markdown_content.splitlines()
        html_lines: List[str] = []
        list_stack: List[str] = []
        table_rows: List[str] = []

        def close_lists():
                while list_stack:
                        html_lines.append(f"</{list_stack.pop()}>")

        def flush_table():
                nonlocal table_rows
                if table_rows:
                        html_lines.append(_render_table(table_rows))
                        table_rows = []

        for line in lines:
                stripped = line.strip()
                if not stripped:
                        close_lists()
                        flush_table()
                        continue
                if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 1:
                        table_rows.append(stripped)
                        continue
                else:
                        flush_table()
                if re.match(r"^[-*]\s+", stripped):
                        marker = "ul"
                        if not list_stack or list_stack[-1] != marker:
                                close_lists()
                                html_lines.append("<ul>")
                                list_stack.append(marker)
                        content = stripped[2:].strip()
                        html_lines.append(f"<li>{_format_inline(content)}</li>")
                        continue
                ordered_match = re.match(r"^\d+\.\s+(.*)", stripped)
                if ordered_match:
                        marker = "ol"
                        if not list_stack or list_stack[-1] != marker:
                                close_lists()
                                html_lines.append("<ol>")
                                list_stack.append(marker)
                        html_lines.append(f"<li>{_format_inline(ordered_match.group(1).strip())}</li>")
                        continue
                close_lists()
                header_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
                if header_match:
                        level = min(len(header_match.group(1)), 6)
                        html_lines.append(f"<h{level}>{_format_inline(header_match.group(2).strip())}</h{level}>")
                        continue
                html_lines.append(f"<p>{_format_inline(stripped)}</p>")

        close_lists()
        flush_table()
        return "\n".join(line for line in html_lines if line)


def _render_table(rows: List[str]) -> str:
        if not rows:
                return ""
        header = rows[0]
        body = rows[1:]
        if body and set(body[0].replace('|', '').strip()) <= {':', '-', ''}:
                body = body[1:]
        def split_row(row: str) -> List[str]:
                return [cell.strip() for cell in row.strip('|').split('|')]
        header_cells = [_format_inline(cell) for cell in split_row(header)]
        html_parts = ["<table>", "<thead>", "<tr>"]
        for cell in header_cells:
                html_parts.append(f"<th>{cell}</th>")
        html_parts.extend(["</tr>", "</thead>", "<tbody>"])
        for row in body:
                if not row.strip('|').strip():
                        continue
                html_parts.append("<tr>")
                for cell in split_row(row):
                        html_parts.append(f"<td>{_format_inline(cell)}</td>")
                html_parts.append("</tr>")
        html_parts.extend(["</tbody>", "</table>"])
        return "\n".join(html_parts)


def _format_inline(text: str) -> str:
        link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        result = []
        last_idx = 0
        for match in link_pattern.finditer(text):
                if match.start() > last_idx:
                        result.append(_format_simple(text[last_idx:match.start()]))
                label = match.group(1)
                url = match.group(2)
                result.append("<a href=\"{url}\" target=\"_blank\">{label}</a>".format(
                        url=html.escape(url, quote=True),
                        label=_format_simple(label)
                ))
                last_idx = match.end()
        if last_idx < len(text):
                result.append(_format_simple(text[last_idx:]))
        return ''.join(result)


def _format_simple(text: str) -> str:
        escaped = html.escape(text, quote=False)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"__(.+?)__", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"\*(.+?)\*", r"<em>\1</em>", escaped)
        escaped = re.sub(r"_(.+?)_", r"<em>\1</em>", escaped)
        escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)
        return escaped


def agent_render_newsletter_html(markdown_content: str) -> Tuple[str, str]:
        if not markdown_content:
                return "", ""
        html_body = _markdown_to_html(markdown_content)
        email_html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>Daily Portfolio Newsletter</title>
<style>
body {{
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        background: #f5f7fb;
        color: #101828;
        margin: 0;
        padding: 0;
}}
.email-wrapper {{
        max-width: 720px;
        margin: 0 auto;
        background: #ffffff;
        padding: 32px;
        border-radius: 16px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        line-height: 1.6;
}}
.email-wrapper h1, .email-wrapper h2, .email-wrapper h3 {{
        color: #0f172a;
        margin-top: 24px;
}}
.email-wrapper h1 {{
        font-size: 28px;
        margin-bottom: 12px;
}}
.email-wrapper h2 {{
        font-size: 20px;
        margin-bottom: 12px;
}}
.email-wrapper h3 {{
        font-size: 18px;
        margin-bottom: 8px;
}}
.email-wrapper table {{
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        font-size: 14px;
}}
.email-wrapper table th,
.email-wrapper table td {{
        border: 1px solid #e4e7ec;
        padding: 12px;
        text-align: left;
}}
.email-wrapper table th {{
        background: #f2f4f7;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 12px;
}}
.email-wrapper a {{
        color: #2563eb;
        text-decoration: none;
}}
.email-wrapper a:hover {{
        text-decoration: underline;
}}
.email-wrapper ul {{
        padding-left: 20px;
}}
.email-wrapper li {{
        margin-bottom: 8px;
}}
</style>
</head>
<body>
<div class='email-wrapper'>
{html_body}
</div>
</body>
</html>
"""
        preview_html = f"<div class='email-wrapper-preview'>{html_body}</div>"
        return email_html, preview_html


# ---------------------------------------------------------------------------
# Email delivery
# ---------------------------------------------------------------------------

def agent_send_newsletter_email(subject: str, markdown_content: str, html_content: str, recipient: str) -> bool:
        load_dotenv()
        api_key = os.getenv("SENDGRID_API_KEY")
        if not api_key:
                logger.error("SENDGRID_API_KEY not configured; skipping email send.")
                return False

        from_email = os.getenv("SEND_FROM_EMAIL", "mark@markkhoury.me")
        payload = {
                "personalizations": [{"to": [{"email": recipient}]}],
                "from": {"email": from_email},
                "subject": subject,
                "content": [
                        {"type": "text/plain", "value": markdown_content},
                        {"type": "text/html", "value": html_content},
                ],
        }
        request_data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
                "https://api.sendgrid.com/v3/mail/send",
                data=request_data,
                method='POST',
                headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                },
        )
        try:
                with urllib_request.urlopen(req, timeout=10) as response:
                        status = response.status
                        logger.info("SendGrid response status: %s", status)
                        return 200 <= status < 300
        except urllib_error.HTTPError as http_exc:
                error_body = http_exc.read().decode('utf-8', errors='ignore')
                logger.error("SendGrid API error: %s - %s", http_exc.code, error_body)
                return False
        except Exception as exc:
                logger.error("SendGrid request failed: %s", exc)
                return False


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _strip_code_fence(content: str) -> str:
        stripped = content.strip().strip("`")
        if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        if stripped.endswith("```"):
                stripped = stripped[:-3].strip()
        return stripped


def _format_optional_number(value: Optional[float], suffix: str = "") -> str:
        if value is None:
                return "—"
        return f"{value:.2f}{suffix}"
