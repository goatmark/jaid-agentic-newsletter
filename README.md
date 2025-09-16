# jaid-agentic-newsletter
A newsletter to summarize only salient news for stocks in your portfolio

## To run on a Github codespace:

* Create `.env` in local directory, and add the following API keys:
    - `OPENAI_API_KEY=KEY`. This is needed for orchestrating and curating the newsletter 
    - `TAVILY_API_KEY=KEY`. This is needed for searching relevant stock news
    - `BREVO_API_KEY`. This is used for sending the newsletter to your inbox
    - Update `SEND_FROM_EMAIL` and `SEND_TO_EMAIL`. Note that `SEND_FROM_EMAIL` must be associated with a Brevo account
* Run `python main.py`
* Open the forwarded port
* Upload a PDF of a recent statement from Robinhood (available through https://robinhood.com/account/reports-statements/individual)



