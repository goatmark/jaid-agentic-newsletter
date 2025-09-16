

from flask import Flask, request, render_template_string
from agentic_flow import extract_pdf_text, analyze_with_openai, build_dataframe, build_newsletter, render_newsletter_html

app = Flask(__name__)


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
				df_sorted = build_dataframe(analysis)
				table = df_sorted.to_html()
				# Build newsletter preview
				import os
				from dotenv import load_dotenv
				load_dotenv()
				tavily_api_key = os.getenv("TAVILY_API_KEY")
				if tavily_api_key:
					newsletter = build_newsletter(df_sorted, tavily_api_key)
					newsletter_html = render_newsletter_html(newsletter)
				else:
					newsletter_html = "<p style='color:red;'>TAVILY_API_KEY not configured.</p>"
	return render_template_string(HTML_FORM, error=error, table=table, newsletter=newsletter_html)

if __name__ == "__main__":
	app.run(debug=True)