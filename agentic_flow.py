import os
import openai
import pandas as pd
from dotenv import load_dotenv
from PyPDF2 import PdfReader

def prompt_pdf_path():
	path = input("Please enter the path to your PDF file: ")
	if not os.path.isfile(path) or not path.lower().endswith('.pdf'):
		print("Invalid file. Please provide a valid PDF file path.")
		return None
	return path

def extract_pdf_text(pdf_path):
	reader = PdfReader(pdf_path)
	text = ""
	for page in reader.pages:
		text += page.extract_text() or ""
	return text

def analyze_with_openai(text):
	load_dotenv()
	api_key = os.getenv("OPENAI_API_KEY")
	openai.api_key = api_key
	prompt = f"Analyze the following PDF content and return a JSON object summarizing its main points, topics, and any key data: {text[:4000]}"
	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=[{"role": "user", "content": prompt}],
		max_tokens=1024,
		temperature=0.2
	)
	# Expecting a JSON string in response
	import json
	content = response['choices'][0]['message']['content']
	try:
		result_json = json.loads(content)
	except Exception:
		print("Failed to parse JSON from OpenAI response. Returning raw content.")
		result_json = {"raw": content}
	return result_json

