import os
import openai
import pandas as pd
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pydantic import BaseModel, Field
import requests
from typing import List, Optional

# --- Agentic Pydantic Models ---
class Stock(BaseModel):
	Security: str
	Symbol: str
	Mkt_Value: float
	Portfolio_Weight: float

class NewsArticle(BaseModel):
	title: str
	url: str
	source: str
	summary: Optional[str] = None
	importance: float = 0.0

class Newsletter(BaseModel):
	top_stocks: List[Stock]
	articles: List[NewsArticle]

# --- Agent Functions ---
def agent_extract_pdf_text(pdf_path):
	reader = PdfReader(pdf_path)
	text = ""
	for page in reader.pages:
		text += page.extract_text() or ""
	return text

def agent_analyze_with_openai(text):
	load_dotenv()
	api_key = os.getenv("OPENAI_API_KEY")
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
		model="gpt-3.5-turbo",
		messages=[{"role": "user", "content": prompt}],
		max_tokens=1024,
		temperature=0.2
	)
	import json
	content = response.choices[0].message.content
	if content.strip().startswith('```'):
		content = content.strip()
		content = content.lstrip('`')
		if content.lower().startswith('json'):
			content = content[4:].strip()
		if content.endswith('```'):
			content = content[:-3].strip()
	try:
		result_json = json.loads(content)
	except Exception:
		result_json = {"raw": content}
	return result_json

def agent_build_dataframe(analysis):
	if isinstance(analysis, dict) and 'raw' in analysis:
		df = pd.DataFrame([analysis])
	else:
		df = pd.DataFrame(analysis)
	try:
		df_sorted = df.copy()
		df_sorted['Mkt Value'] = df_sorted['Mkt Value'].replace('[^0-9.]', '', regex=True).astype(float)
		df_sorted = df_sorted.sort_values(by='Mkt Value', ascending=False)
		df_sorted['% of Total Portfolio'] = df_sorted['% of Total Portfolio'].replace('[^0-9.]', '', regex=True).astype(float)
		df_sorted['Cumulative Portfolio Weight'] = df_sorted['% of Total Portfolio'].cumsum()
	except Exception:
		df_sorted = df
	return df_sorted

def agent_get_top_stocks(df, n=3):
	top = df.head(n)
	stocks = []
	for _, row in top.iterrows():
		stocks.append(Stock(
			Security=row.get('Security', ''),
			Symbol=row.get('Symbol', ''),
			Mkt_Value=row.get('Mkt Value', 0.0),
			Portfolio_Weight=row.get('% of Total Portfolio', 0.0)
		))
	return stocks

def agent_search_news_tavily(symbol, api_key, max_results=5):
	import logging
	logger = logging.getLogger(__name__)
	from datetime import datetime, timedelta
	url = "https://api.tavily.com/search"
	headers = {
		"Authorization": f"Bearer {api_key}"
	}
	# Only get news from last 3 days
	today = datetime.utcnow().date()
	three_days_ago = today - timedelta(days=3)
	# Tavily supports natural language queries, so add date filter
	query = f"{symbol} news after:{three_days_ago}"
	json_data = {
		"query": query,
		"max_results": max_results
	}
	try:
		logger.info(f"Searching news for symbol: {symbol} with Tavily API...")
		resp = requests.post(url, json=json_data, headers=headers)
		logger.info(f"Tavily API response status: {resp.status_code}")
		logger.info(f"Tavily API response text: {resp.text}")
		if resp.status_code == 200:
			data = resp.json()
			articles = []
			for item in data.get('results', []):
				# Only include articles published in last 3 days
				pub_date = item.get('published_date')
				if pub_date:
					try:
						pub_dt = datetime.strptime(pub_date[:10], "%Y-%m-%d").date()
						if pub_dt < three_days_ago:
							continue
					except Exception:
						pass
				articles.append(NewsArticle(
					title=item.get('title', ''),
					url=item.get('url', ''),
					source=item.get('source', ''),
					summary=item.get('description', '')
				))
			logger.info(f"Found {len(articles)} recent articles for {symbol}")
			return articles
		else:
			logger.error(f"Tavily API error: {resp.status_code} {resp.text}")
			return []
	except Exception as e:
		logger.error(f"Exception during Tavily news search: {e}")
		return []

def agent_filter_articles(articles: List[NewsArticle]):
	credible_sources = [
		"Reuters", "Bloomberg", "WSJ", "CNBC", "Financial Times", "Yahoo Finance",
		"MarketWatch", "Seeking Alpha", "CNN"
	]
	exclude_sources = ["Benzinga", "TipRanks", "Forbes"]
	filtered = [a for a in articles if any(src in a.source for src in credible_sources) and not any(ex in a.source for ex in exclude_sources)]
	if not filtered:
		filtered = [a for a in articles if not any(ex in a.source for ex in exclude_sources)]
	# Summarize and score impact using OpenAI
	load_dotenv()
	openai_api_key = os.getenv("OPENAI_API_KEY")
	if openai_api_key and filtered:
		try:
			client = openai.OpenAI(api_key=openai_api_key)
			for a in filtered:
				prompt = f"Summarize this news article for a daily investor newsletter. Estimate its impact on the stock price and portfolio.\nTitle: {a.title}\nSource: {a.source}\nSummary: {a.summary}"
				response = client.chat.completions.create(
					model="gpt-3.5-turbo",
					messages=[{"role": "user", "content": prompt}],
					max_tokens=256,
					temperature=0.3
				)
				summary = response.choices[0].message.content.strip()
				a.summary = summary
				# Score impact: crude proxy, count impactful words
				impact_keywords = ["upgrade", "downgrade", "earnings", "acquisition", "merger", "guidance", "forecast", "record", "lawsuit", "SEC", "investigation", "profit", "loss", "growth", "drop", "surge", "buyback", "dividend"]
				a.importance = sum([summary.lower().count(k) for k in impact_keywords]) + len(a.summary or "")
		except Exception as e:
			pass
	filtered.sort(key=lambda x: x.importance, reverse=True)
	return filtered[:3] if filtered else articles[:3]

def agent_build_newsletter(df_sorted, tavily_api_key):
	portfolio_weight = 0.0
	selected_stocks = []
	for _, row in df_sorted.iterrows():
		selected_stocks.append(Stock(
			Security=row.get('Security', ''),
			Symbol=row.get('Symbol', ''),
			Mkt_Value=row.get('Mkt Value', 0.0),
			Portfolio_Weight=row.get('% of Total Portfolio', 0.0)
		))
		portfolio_weight += row.get('% of Total Portfolio', 0.0)
		if portfolio_weight >= 80.0:
			break
	all_articles = []
	for stock in selected_stocks:
		articles = agent_search_news_tavily(stock.Symbol, tavily_api_key, max_results=10)
		filtered = agent_filter_articles(articles)
		# Weight by portfolio
		for a in filtered:
			a.importance += stock.Portfolio_Weight
		if filtered:
			all_articles.extend(filtered)
		else:
			all_articles.extend(articles[:3])
	all_articles.sort(key=lambda x: x.importance, reverse=True)
	# Always return at least 3 articles if available
	return Newsletter(top_stocks=selected_stocks, articles=all_articles[:3] if all_articles else [])

def agent_render_newsletter_html(newsletter: Newsletter):
	html = "<h2>Your Curated Portfolio Newsletter</h2>"
	html += "<h3>Top Stocks</h3><ul>"
	for stock in newsletter.top_stocks:
		html += f"<li>{stock.Security} ({stock.Symbol}): {stock.Portfolio_Weight:.2f}% of portfolio</li>"
	html += "</ul><h3>Top News (last 3 days)</h3><ol>"
	for article in newsletter.articles:
		html += f"<li><a href='{article.url}' target='_blank'>{article.title}</a> <br><em>{article.source}</em><br><strong>Summary:</strong> {article.summary}</li>"
	html += "</ol>"
	return html

def extract_pdf_text(pdf_path):
	reader = PdfReader(pdf_path)
	text = ""
	for page in reader.pages:
		text += page.extract_text() or ""
	return text

def analyze_with_openai(text):
	load_dotenv()
	api_key = os.getenv("OPENAI_API_KEY")
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
		model="gpt-3.5-turbo",
		messages=[{"role": "user", "content": prompt}],
		max_tokens=1024,
		temperature=0.2
	)
	import json
	content = response.choices[0].message.content
	# Robustly extract JSON from code block if present
	if content.strip().startswith('```'):
		content = content.strip()
		content = content.lstrip('`')
		if content.lower().startswith('json'):
			content = content[4:].strip()
		if content.endswith('```'):
			content = content[:-3].strip()
	try:
		result_json = json.loads(content)
	except Exception:
		result_json = {"raw": content}
	return result_json

def build_dataframe(analysis):

	if isinstance(analysis, dict) and 'raw' in analysis:
		df = pd.DataFrame([analysis])
	else:
		df = pd.DataFrame(analysis)
	try:
		df_sorted = df.copy()
		df_sorted['Mkt Value'] = df_sorted['Mkt Value'].replace('[^0-9.]', '', regex=True).astype(float)
		df_sorted = df_sorted.sort_values(by='Mkt Value', ascending=False)
		df_sorted['% of Total Portfolio'] = df_sorted['% of Total Portfolio'].replace('[^0-9.]', '', regex=True).astype(float)
		df_sorted['Cumulative Portfolio Weight'] = df_sorted['% of Total Portfolio'].cumsum()
	except Exception:
		df_sorted = df
	return df_sorted

# --- Pydantic Models and Agent Workflow ---
class Stock(BaseModel):
	Security: str
	Symbol: str
	Mkt_Value: float
	Portfolio_Weight: float

class NewsArticle(BaseModel):
	title: str
	url: str
	source: str
	summary: Optional[str] = None
	importance: float = 0.0

class Newsletter(BaseModel):
	top_stocks: List[Stock]
	articles: List[NewsArticle]

def get_top_stocks(df, n=3):
	# Assumes df_sorted from build_dataframe
	top = df.head(n)
	stocks = []
	for _, row in top.iterrows():
		stocks.append(Stock(
			Security=row.get('Security', ''),
			Symbol=row.get('Symbol', ''),
			Mkt_Value=row.get('Mkt Value', 0.0),
			Portfolio_Weight=row.get('% of Total Portfolio', 0.0)
		))
	return stocks

def search_news_tavily(symbol, api_key, max_results=5):
	# Tavily API docs: https://docs.tavily.com/
	url = "https://api.tavily.com/search"
	params = {
		"query": f"{symbol} stock news",
		"api_key": api_key,
		"max_results": max_results
	}
	resp = requests.get(url, params=params)
	if resp.status_code == 200:
		data = resp.json()
		articles = []
		for item in data.get('results', []):
			articles.append(NewsArticle(
				title=item.get('title', ''),
				url=item.get('url', ''),
				source=item.get('source', ''),
				summary=item.get('description', '')
			))
		return articles
	return []

def filter_articles(articles: List[NewsArticle]):
	# Reject speculative sources
	credible_sources = ["Reuters", "Bloomberg", "WSJ", "CNBC", "Financial Times", "Yahoo Finance"]
	filtered = [a for a in articles if any(src in a.source for src in credible_sources)]
	# Assign importance (simple: length of summary, can be improved)
	for a in filtered:
		a.importance = len(a.summary or "")
	filtered.sort(key=lambda x: x.importance, reverse=True)
	return filtered[:3]

def build_newsletter(df_sorted, tavily_api_key):
	top_stocks = get_top_stocks(df_sorted, n=3)
	all_articles = []
	for stock in top_stocks:
		articles = search_news_tavily(stock.Symbol, tavily_api_key)
		filtered = filter_articles(articles)
		all_articles.extend(filtered)
	# Sort by importance and portfolio weight
	all_articles.sort(key=lambda x: x.importance, reverse=True)
	return Newsletter(top_stocks=top_stocks, articles=all_articles[:3])

def render_newsletter_html(newsletter: Newsletter):
	html = "<h2>Your Curated Portfolio Newsletter</h2>"
	html += "<h3>Top Stocks</h3><ul>"
	for stock in newsletter.top_stocks:
		html += f"<li>{stock.Security} ({stock.Symbol}): {stock.Portfolio_Weight:.2f}% of portfolio</li>"
	html += "</ul><h3>Top News</h3><ol>"
	for article in newsletter.articles:
		html += f"<li><a href='{article.url}' target='_blank'>{article.title}</a> <br><em>{article.source}</em><br>{article.summary}</li>"
	html += "</ol>"
	return html

