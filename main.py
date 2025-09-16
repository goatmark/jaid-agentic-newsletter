def main():
	pdf_path = None
	while not pdf_path:
		pdf_path = prompt_pdf_path()
	print("Extracting text from PDF...")
	text = extract_pdf_text(pdf_path)
	print("Analyzing PDF content with OpenAI...")
	analysis = analyze_with_openai(text)
	print("Storing analysis as DataFrame...")
	df = pd.DataFrame([analysis])
	print(df)

import os
import openai
import pandas as pd
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from flask import Flask, request, render_template_string, redirect, url_for

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
		# Remove triple backticks and optional 'json' label
		content = content.strip()
		content = content.lstrip('`')
		if content.lower().startswith('json'):
			content = content[4:].strip()
		# Remove trailing backticks
		if content.endswith('```'):
			content = content[:-3].strip()
	try:
		result_json = json.loads(content)
	except Exception:
		result_json = {"raw": content}
	return result_json

HTML_FORM = '''
	<!doctype html>
	<title>Upload PDF for Analysis</title>
	<h1>Upload PDF File</h1>
	<form method=post enctype=multipart/form-data>
	  <input type=file name=file accept="application/pdf">
	  <input type=submit value=Upload>
	</form>
	{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
	{% if table %}<h2>Analysis DataFrame</h2>{{ table|safe }}{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	error = None
	table = None
	if request.method == 'POST':
		if 'file' not in request.files:
			error = 'No file part'
		else:
			file = request.files['file']
			if file.filename == '':
				error = 'No selected file'
			elif not file.filename.lower().endswith('.pdf'):
				error = 'File must be a PDF'
			else:
				filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
				file.save(filepath)
				text = extract_pdf_text(filepath)
				analysis = analyze_with_openai(text)
				# Expecting analysis to be a list of dicts
				if isinstance(analysis, dict) and 'raw' in analysis:
					df = pd.DataFrame([analysis])
				else:
					df = pd.DataFrame(analysis)
				# Try to sort by 'Mkt Value' descending
				try:
					df_sorted = df.copy()
					# Remove any non-numeric characters and convert to float for sorting
					df_sorted['Mkt Value'] = df_sorted['Mkt Value'].replace('[^0-9.]', '', regex=True).astype(float)
					df_sorted = df_sorted.sort_values(by='Mkt Value', ascending=False)
					# Convert '% of Total Portfolio' to float
					df_sorted['% of Total Portfolio'] = df_sorted['% of Total Portfolio'].replace('[^0-9.]', '', regex=True).astype(float)
					df_sorted['Cumulative Portfolio Weight'] = df_sorted['% of Total Portfolio'].cumsum()
				except Exception:
					df_sorted = df
				print(df_sorted)
				table = df_sorted.to_html()
	return render_template_string(HTML_FORM, error=error, table=table)

if __name__ == "__main__":
	app.run(debug=True)