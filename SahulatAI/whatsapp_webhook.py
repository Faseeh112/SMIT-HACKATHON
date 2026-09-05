"""
SahulatAI WhatsApp Cloud API webhook.

Run:
    uvicorn whatsapp_webhook:app --host 0.0.0.0 --port 8000

The public webhook URL must be:
    https://YOUR-DOMAIN/webhook/whatsapp

Meta calls GET /webhook/whatsapp to verify the webhook and POST it
when a WhatsApp user sends a message.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from agent.agent import run_turn
from config.settings import settings
from database import db

logger = logging.getLogger("sahulatai.whatsapp")

app = FastAPI(title="SahulatAI WhatsApp Webhook")


def _check_config() -> None:
    missing = []
    if not settings.whatsapp_access_token:
        missing.append("WHATSAPP_ACCESS_TOKEN")
    if not settings.whatsapp_phone_number_id:
        missing.append("WHATSAPP_PHONE_NUMBER_ID")
    if not settings.whatsapp_verify_token:
        missing.append("WHATSAPP_VERIFY_TOKEN")
    if not settings.whatsapp_api_version:
        missing.append("WHATSAPP_API_VERSION")
    if missing:
        raise RuntimeError("Missing WhatsApp configuration: " + ", ".join(missing))


def _verify_signature(raw_body: bytes, signature: str | None) -> bool:
    """
    Verify Meta's X-Hub-Signature-256 when WHATSAPP_APP_SECRET is configured.
    If no app secret is configured, signature verification is skipped so the
    initial setup is easier; configure the secret before production use.
    """
    if not settings.whatsapp_app_secret:
        return True
    if not signature or not signature.startswith("sha256="):
        return False

    expected = hmac.new(
        settings.whatsapp_app_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    supplied = signature.split("=", 1)[1]
    return hmac.compare_digest(expected, supplied)


def _get_conversation_id(user_id: str) -> str:
    # Reuse the user's latest conversation so WhatsApp messages retain context
    # across webhook requests and server restarts.
    conversations = db.list_conversations(user_id, limit=1)
    if conversations:
        return conversations[0]["conversation_id"]

    db.ensure_user(user_id, preferred_language="en")
    return db.start_conversation(user_id)


def _extract_incoming_text(payload: dict[str, Any]) -> tuple[str, str] | None:
    """
    Return (WhatsApp sender number, text) for a normal incoming text message.
    Ignore status events, delivery events, and unsupported message types.
    """
    if payload.get("object") != "whatsapp_business_account":
        return None

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            messages = value.get("messages", [])
            if not messages:
                continue

            message = messages[0]
            sender = message.get("from")
            if not sender:
                continue

            if message.get("type") != "text":
                return sender, ""

            body = (message.get("text") or {}).get("body", "").strip()
            return sender, body

    return None


def send_whatsapp_text(recipient: str, text: str) -> dict:
    _check_config()

    url = (
        f"https://graph.facebook.com/"
        f"{settings.whatsapp_api_version}/"
        f"{settings.whatsapp_phone_number_id}/messages"
    )
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": text[:4096],
        },
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    if not response.ok:
        logger.error("WhatsApp send failed: %s %s", response.status_code, response.text[:1000])
        raise RuntimeError(
            f"WhatsApp API returned HTTP {response.status_code}: {response.text[:500]}"
        )
    return response.json()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "sahulatai-whatsapp"}


@app.get("/webhook/whatsapp", response_class=PlainTextResponse)
def verify_whatsapp_webhook(
    hub_mode: str | None = Query(default=None, alias="hub.mode"),
    hub_verify_token: str | None = Query(default=None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(default=None, alias="hub.challenge"),
):
    """
    Meta webhook verification endpoint.
    Meta sends hub.challenge when the configured verify token matches.
    """
    if (
        hub_mode == "subscribe"
        and hub_verify_token
        and hmac.compare_digest(hub_verify_token, settings.whatsapp_verify_token)
        and hub_challenge
    ):
        return hub_challenge

    raise HTTPException(status_code=403, detail="Webhook verification failed")


@app.post("/webhook/whatsapp")
async def receive_whatsapp_webhook(request: Request) -> dict[str, str]:
    raw_body = await request.body()

    if not _verify_signature(
        raw_body,
        request.headers.get("x-hub-signature-256"),
    ):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    incoming = _extract_incoming_text(payload)
    if incoming is None:
        # Acknowledge unrelated WhatsApp events.
        return {"status": "ignored"}

    sender, message = incoming

    # Acknowledge unsupported media messages without trying to send a blank
    # response. Text is the first integration target.
    if not message:
        return {"status": "ignored"}

    try:
        db.init_db()
        user_id = f"whatsapp:{sender}"
        db.ensure_user(user_id, preferred_language="en")
        conversation_id = _get_conversation_id(user_id)

        # Persist incoming message before the agent runs.
        db.add_message(conversation_id, "user", message)

        result = run_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=message,
        )

        db.add_message(conversation_id, "assistant", result.answer_text)
        send_whatsapp_text(sender, result.answer_text)

        return {"status": "processed"}

    except Exception:
        logger.exception("Failed to process WhatsApp message from %s", sender)
        # Return 200 so Meta does not repeatedly retry a permanently failed
        # application request. The error is logged for the developer.
        try:
            send_whatsapp_text(
                sender,
                "Sorry, I couldn't process your message right now. Please try again in a moment.",
            )
        except Exception:
            logger.exception("Also failed to send WhatsApp error message")
        return {"status": "error"}
