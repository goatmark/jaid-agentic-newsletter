

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
	if request.method == 'POST':
		logger.info('Received POST request')
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
				text = agent_extract_pdf_text(filepath)
				print('PDF text extracted. Length:', len(text))
				logger.info('Analyzing PDF text with OpenAI...')
				analysis = agent_analyze_with_openai(text)
				print('OpenAI analysis result:', analysis)
				df_sorted = agent_build_dataframe(analysis)
				print('DataFrame built. Columns:', df_sorted.columns.tolist())
				table = df_sorted.to_html()
				# Build newsletter preview
				from dotenv import load_dotenv
				load_dotenv()
				tavily_api_key = os.getenv("TAVILY_API_KEY")
				if tavily_api_key:
					print('Building newsletter...')
					newsletter = agent_build_newsletter(df_sorted, tavily_api_key)
					print('Newsletter object:', newsletter)
					newsletter_html = agent_render_newsletter_html(newsletter)
				else:
					newsletter_html = "<p style='color:red;'>TAVILY_API_KEY not configured.</p>"
					logger.error('TAVILY_API_KEY not configured.')
	return render_template_string(HTML_FORM, error=error, table=table, newsletter=newsletter_html)

if __name__ == "__main__":
	app.run(debug=True)