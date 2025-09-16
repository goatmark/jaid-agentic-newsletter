import logging
import os
import time

from dotenv import load_dotenv
from flask import Flask, render_template_string, request

from agentic_flow import (
        agent_analyze_with_openai,
        agent_build_dataframe,
        agent_build_newsletter,
        agent_extract_pdf_text,
        agent_render_newsletter_html,
        agent_send_newsletter_email,
)

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

HTML_FORM = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Agentic Portfolio Newsletter</title>
<style>
body {
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        background: #eef2ff;
        color: #0f172a;
        margin: 0;
        padding: 40px 0;
}
.container {
        max-width: 960px;
        margin: 0 auto;
        background: #ffffff;
        padding: 32px 40px;
        border-radius: 24px;
        box-shadow: 0 20px 40px rgba(79, 70, 229, 0.15);
}
h1 {
        font-size: 32px;
        margin-bottom: 16px;
}
form {
        display: flex;
        gap: 12px;
        margin-bottom: 24px;
}
input[type="file"] {
        flex: 1;
        padding: 12px;
        border: 1px solid #c7d2fe;
        border-radius: 12px;
}
input[type="submit"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        cursor: pointer;
        font-weight: 600;
        transition: transform 0.2s ease;
}
input[type="submit"]:hover {
        transform: translateY(-2px);
}
.error {
        color: #ef4444;
        font-weight: 600;
}
.section {
        margin-top: 32px;
}
.table-wrapper {
        overflow-x: auto;
        border-radius: 12px;
        border: 1px solid #e0e7ff;
}
.table-wrapper table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
}
.table-wrapper th,
.table-wrapper td {
        padding: 12px;
        border-bottom: 1px solid #e0e7ff;
        text-align: left;
}
.table-wrapper th {
        background: #eef2ff;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 0.05em;
}
.markdown-block {
        background: #0f172a;
        color: #f8fafc;
        padding: 20px;
        border-radius: 16px;
        white-space: pre-wrap;
        overflow-x: auto;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 14px;
}
.newsletter-preview {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #e0e7ff;
        padding: 24px;
        box-shadow: inset 0 0 0 1px rgba(79, 70, 229, 0.05);
        max-height: 600px;
        overflow-y: auto;
}
.email-status {
        margin-top: 24px;
        font-weight: 600;
}
</style>
</head>
<body>
<div class="container">
        <h1>Agentic Portfolio Newsletter</h1>
        <p>Upload your latest portfolio statement PDF to spin up a punchy, email-ready market recap.</p>
        <form method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="application/pdf">
                <input type="submit" value="Generate Newsletter">
        </form>
        {% if error %}<p class="error">{{ error }}</p>{% endif %}
        {% if table %}
        <section class="section">
                <h2>Extracted Portfolio Snapshot</h2>
                <div class="table-wrapper">{{ table|safe }}</div>
        </section>
        {% endif %}
        {% if markdown %}
        <section class="section">
                <h2>Newsletter Markdown</h2>
                <pre class="markdown-block">{{ markdown }}</pre>
        </section>
        {% endif %}
        {% if newsletter_html %}
        <section class="section">
                <h2>Email Preview</h2>
                <div class="newsletter-preview">{{ newsletter_html|safe }}</div>
        </section>
        {% endif %}
        {% if email_status %}
        <p class="email-status">{{ email_status }}</p>
        {% endif %}
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def upload_file():
        error = None
        table_html = None
        newsletter_markdown = None
        newsletter_preview = None
        email_status = None

        if request.method == "POST":
                logger.info("Received POST request")
                start_total = time.time()
                try:
                        if "file" not in request.files:
                                error = "No file part"
                        else:
                                file = request.files["file"]
                                logger.info("File received: %s", file.filename)
                                if file.filename == "":
                                        error = "No selected file"
                                elif not file.filename.lower().endswith(".pdf"):
                                        error = "File must be a PDF"
                                else:
                                        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                                        file.save(filepath)
                                        logger.info("Saved uploaded file to %s", filepath)

                                        logger.info("Extracting PDF text")
                                        start_pdf = time.time()
                                        text = agent_extract_pdf_text(filepath)
                                        logger.info("PDF extraction completed in %.2fs", time.time() - start_pdf)

                                        logger.info("Analyzing holdings with OpenAI")
                                        start_openai = time.time()
                                        analysis = agent_analyze_with_openai(text)
                                        logger.info("OpenAI analysis completed in %.2fs", time.time() - start_openai)

                                        logger.info("Building dataframe")
                                        df_sorted = agent_build_dataframe(analysis)
                                        table_html = df_sorted.to_html(classes="data-table", index=False, border=0)

                                        tavily_api_key = os.getenv("TAVILY_API_KEY")
                                        if tavily_api_key:
                                                logger.info("Building newsletter content")
                                                newsletter, newsletter_markdown = agent_build_newsletter(df_sorted, tavily_api_key)
                                                if newsletter_markdown:
                                                        email_html, preview_html = agent_render_newsletter_html(newsletter_markdown)
                                                        newsletter_preview = preview_html
                                                        recipient = os.getenv("NEWSLETTER_RECIPIENT", "mark@markkhoury.me")
                                                        subject = "Your Daily Newsletter"
                                                        if email_html:
                                                                sent = agent_send_newsletter_email(
                                                                        subject,
                                                                        newsletter_markdown,
                                                                        email_html,
                                                                        recipient,
                                                                )
                                                                email_status = (
                                                                        f"Newsletter sent to {recipient}."
                                                                        if sent
                                                                        else f"Failed to send the newsletter to {recipient}."
                                                                )
                                                        else:
                                                                email_status = "Newsletter HTML could not be generated."
                                                else:
                                                        email_status = "Newsletter content could not be generated."
                                        else:
                                                error = "TAVILY_API_KEY not configured."
                except Exception as exc:
                        logger.exception("Failed to process newsletter generation: %s", exc)
                        error = "An unexpected error occurred while building the newsletter."
                finally:
                        logger.info("Total execution time: %.2fs", time.time() - start_total)

        return render_template_string(
                HTML_FORM,
                error=error,
                table=table_html,
                markdown=newsletter_markdown,
                newsletter_html=newsletter_preview,
                email_status=email_status,
        )


if __name__ == "__main__":
        app.run(debug=True)
