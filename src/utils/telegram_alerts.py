"""
Telegram Alert Utility

Sends daily trade scans and portfolio update logs directly to your phone.
"""

import logging
import sys
from pathlib import Path
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("telegram_alerts")


def send_telegram_message(message: str) -> bool:
    """
    Sends a text message using the Telegram Bot API.
    Returns True if successfully sent, False otherwise.
    """
    token = getattr(config, "TELEGRAM_BOT_TOKEN", "")
    chat_id = getattr(config, "TELEGRAM_CHAT_ID", "")
    
    if not token or not chat_id:
        logger.info("Telegram notification skipped: Bot Token or Chat ID not configured.")
        return False
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("Telegram notification sent successfully.")
            return True
        else:
            logger.error("Failed to send Telegram message: %s - %s", response.status_code, response.text)
            return False
    except Exception as exc:
        logger.error("Telegram API connection failed: %s", exc)
        return False
