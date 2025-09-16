from flask import Flask

old_data = """
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

"""

app = Flask(__name__)

@app.route('/')
def main():
    return "<p>Hello, world.</p>"