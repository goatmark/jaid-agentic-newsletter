from flask import Flask, request, render_template_string
from agentic_flow import agent_extract_pdf_text, agent_analyze_with_openai, agent_build_dataframe, agent_build_newsletter, agent_render_newsletter_html
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


UPLOAD_FOLDER = 'uploads'
import os
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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
    {% if newsletter %}<h2>Newsletter Preview</h2>{{ newsletter|safe }}{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	error = None
	table = None
	newsletter_html = None
	import time
	if request.method == 'POST':
		logger.info('Received POST request')
		start_total = time.time()
		if 'file' not in request.files:
			error = 'No file part'
			logger.error(error)
		else:
			file = request.files['file']
			logger.info(f'File received: {file.filename}')
			if file.filename == '':
				error = 'No selected file'
				logger.error(error)
			elif not file.filename.lower().endswith('.pdf'):
				error = 'File must be a PDF'
				logger.error(error)
			else:
				filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
				file.save(filepath)
				logger.info(f'File saved to {filepath}')
				print('Extracting PDF text...')
				start_pdf = time.time()
				text = agent_extract_pdf_text(filepath)
				end_pdf = time.time()
				print(f'PDF text extracted. Length: {len(text)}. Time: {end_pdf - start_pdf:.2f}s')
				logger.info(f'PDF extraction time: {end_pdf - start_pdf:.2f}s')
				logger.info('Analyzing PDF text with OpenAI...')
				start_openai = time.time()
				analysis = agent_analyze_with_openai(text)
				end_openai = time.time()
				print(f'OpenAI analysis result: {analysis}. Time: {end_openai - start_openai:.2f}s')
				logger.info(f'OpenAI analysis time: {end_openai - start_openai:.2f}s')
				start_df = time.time()
				df_sorted = agent_build_dataframe(analysis)
				end_df = time.time()
				print(f'DataFrame built. Columns: {df_sorted.columns.tolist()}. Time: {end_df - start_df:.2f}s')
				logger.info(f'DataFrame build time: {end_df - start_df:.2f}s')
				table = df_sorted.to_html()
				# Build newsletter preview
				from dotenv import load_dotenv
				load_dotenv()
				tavily_api_key = os.getenv("TAVILY_API_KEY")
				if tavily_api_key:
					print('Building newsletter...')
					start_news = time.time()
					newsletter = agent_build_newsletter(df_sorted, tavily_api_key)
					end_news = time.time()
					print(f'Newsletter object: {newsletter}. Time: {end_news - start_news:.2f}s')
					logger.info(f'Newsletter build time: {end_news - start_news:.2f}s')
					newsletter_html = agent_render_newsletter_html(newsletter)
				else:
					newsletter_html = "<p style='color:red;'>TAVILY_API_KEY not configured.</p>"
					logger.error('TAVILY_API_KEY not configured.')
		end_total = time.time()
		print(f'Total execution time: {end_total - start_total:.2f}s')
		logger.info(f'Total execution time: {end_total - start_total:.2f}s')
	return render_template_string(HTML_FORM, error=error, table=table, newsletter=newsletter_html)

if __name__ == "__main__":
	app.run(debug=True)