# WhatsApp Cloud API setup for SahulatAI

## 1. Install dependencies

```powershell
pip install -r requirements.txt
```

## 2. Configure `.env`

Set:

```env
WHATSAPP_ACCESS_TOKEN=your_meta_access_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_VERIFY_TOKEN=make_your_own_random_string
WHATSAPP_API_VERSION=the_current_graph_api_version
WHATSAPP_APP_SECRET=your_meta_app_secret
```

Do not commit `.env` or your access token to GitHub.

## 3. Run the existing SahulatAI app

```powershell
streamlit run app.py
```

## 4. Run the WhatsApp webhook in another terminal

```powershell
uvicorn whatsapp_webhook:app --host 0.0.0.0 --port 8000
```

For Meta to reach your local PC, the webhook must have a public HTTPS URL.
For development you can use a tunnel such as ngrok or Cloudflare Tunnel.

Your webhook endpoint is:

`https://YOUR-PUBLIC-HTTPS-DOMAIN/webhook/whatsapp`

## 5. Configure Meta

In your Meta Developer app, add/configure the WhatsApp product and Webhooks.
Use:

- Callback URL: `https://YOUR-PUBLIC-HTTPS-DOMAIN/webhook/whatsapp`
- Verify token: exactly the same value as `WHATSAPP_VERIFY_TOKEN`

Subscribe the WhatsApp Business Account to the app/webhook so incoming
messages are delivered to the endpoint.

## 6. Message flow

WhatsApp user -> Meta Cloud API -> FastAPI webhook -> `run_turn()` ->
SahulatAI RAG/agent -> Meta Cloud API -> WhatsApp user.

The WhatsApp sender's phone number is used as the SahulatAI user identity,
so the existing SQLite conversation and memory system can keep the user's
context.

## 7. Important production notes

- Use HTTPS.
- Keep the access token and app secret in environment variables.
- Keep `WHATSAPP_APP_SECRET` configured so `X-Hub-Signature-256` is checked.
- The current code handles incoming TEXT messages first. Images/audio/documents
  can be added later.
- WhatsApp has messaging/template/policy rules; follow Meta's current
  WhatsApp Business Platform rules for production messaging.
