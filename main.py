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
	prompt = f"Analyze the following PDF content and return a JSON object summarizing its main points, topics, and any key data: {text[:4000]}"
	response = client.chat.completions.create(
		model="gpt-3.5-turbo",
		messages=[{"role": "user", "content": prompt}],
		max_tokens=1024,
		temperature=0.2
	)
	import json
	content = response.choices[0].message.content
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
				df = pd.DataFrame([analysis])
				table = df.to_html()
	return render_template_string(HTML_FORM, error=error, table=table)

if __name__ == "__main__":
	app.run(debug=True)