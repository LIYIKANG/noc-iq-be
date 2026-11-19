from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from fastapi import HTTPException, status

from app.core.config import settings

logger = logging.getLogger(__name__)


class BridgeErrorCode(str, Enum):
    TIMEOUT = "TIMEOUT"
    REJECTED = "REJECTED"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    NETWORK_ERROR = "NETWORK_ERROR"
    CONTRACT_ERROR = "CONTRACT_ERROR"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    UNKNOWN = "UNKNOWN"


@dataclass
class BridgeError:
    code: BridgeErrorCode
    message: str
    retryable: bool
    upstream_error: Optional[Any] = None


@dataclass
class BridgeTimeoutConfig:
    sla_check_ms: int = 5000
    payment_ms: int = 30000
    balance_check_ms: int = 10000

    def for_operation(self, operation: str) -> int:
        mapping = {
            "sla_check": self.sla_check_ms,
            "payment": self.payment_ms,
            "balance_check": self.balance_check_ms,
        }
        return mapping.get(operation, self.payment_ms)


_TIMEOUT_CONFIG = BridgeTimeoutConfig(
    sla_check_ms=settings.BRIDGE_TIMEOUT_SLA_CHECK_MS,
    payment_ms=settings.BRIDGE_TIMEOUT_PAYMENT_MS,
    balance_check_ms=settings.BRIDGE_TIMEOUT_BALANCE_MS,
)


def get_timeout_config() -> BridgeTimeoutConfig:
    return _TIMEOUT_CONFIG


_TIMEOUT_KEYWORDS = ("timeout", "timed out", "deadline exceeded", "read timed out")
_NETWORK_KEYWORDS = ("connection", "dns", "reset", "refused", "unreachable", "network")
_REJECTED_KEYWORDS = ("rejected", "denied", "forbidden", "unauthorized", "not allowed")
_FUNDS_KEYWORDS = ("insufficient", "not enough", "balance", "insufficient funds")
_CONTRACT_KEYWORDS = ("contract", "soroban", "invoke", "wasm", "soroban_rpc")
# Phrases that say a response arrived but could not be read. Kept to whole
# phrases so that, for example, "invalid signature" is not swept in.
_MALFORMED_KEYWORDS = (
    "malformed",
    "unparseable",
    "invalid json",
    "not valid json",
    "invalid response",
    "invalid payload",
    "unexpected response",
    "could not parse",
    "cannot parse",
    "failed to parse",
    "failed to decode",
    "parse error",
    "decode error",
    "jsondecodeerror",
    "expecting value",
    "invalid xdr",
    "bad xdr",
    "schema mismatch",
    "missing required field",
)

# Only these codes may be advertised to clients as worth retrying.
RETRYABLE_BRIDGE_CODES = frozenset(
    {BridgeErrorCode.TIMEOUT, BridgeErrorCode.NETWORK_ERROR}
)


def map_bridge_error(raw_error: Any) -> BridgeError:
    """Classify a raw bridge failure.

    A timeout and a malformed response need opposite handling: after a timeout
    no answer arrived, so the call is worth retrying; a malformed response did
    arrive and will be just as unreadable next time, so it is terminal.

    Precedence, most reliable evidence first:

    1. The exception type. A ``TimeoutError`` is a timeout and a JSON or
       Unicode decode error is a malformed response, whatever the text says.
    2. Malformed-response wording, checked before timeout wording because a
       payload error often quotes the payload, and a quoted field such as
       ``timeout_ledger`` must not turn a terminal failure into a retryable one.
    3. The remaining keyword classes, in their original order.
    """
    error_str = str(raw_error).lower()

    if isinstance(raw_error, TimeoutError):
        return BridgeError(
            code=BridgeErrorCode.TIMEOUT,
            message=str(raw_error),
            retryable=True,
            upstream_error=raw_error,
        )

    if isinstance(raw_error, (json.JSONDecodeError, UnicodeDecodeError)) or any(
        kw in error_str for kw in _MALFORMED_KEYWORDS
    ):
        return BridgeError(
            code=BridgeErrorCode.MALFORMED_RESPONSE,
            message=str(raw_error),
            retryable=False,
            upstream_error=raw_error,
        )

    if any(kw in error_str for kw in _TIMEOUT_KEYWORDS):
        return BridgeError(
            code=BridgeErrorCode.TIMEOUT,
            message=str(raw_error),
            retryable=True,
            upstream_error=raw_error,
        )

    if any(kw in error_str for kw in _NETWORK_KEYWORDS):
        return BridgeError(
            code=BridgeErrorCode.NETWORK_ERROR,
            message=str(raw_error),
            retryable=True,
            upstream_error=raw_error,
        )

    if any(kw in error_str for kw in _REJECTED_KEYWORDS):
        return BridgeError(
            code=BridgeErrorCode.REJECTED,
            message=str(raw_error),
            retryable=False,
            upstream_error=raw_error,
        )

    if any(kw in error_str for kw in _FUNDS_KEYWORDS):
        return BridgeError(
            code=BridgeErrorCode.INSUFFICIENT_FUNDS,
            message=str(raw_error),
            retryable=False,
            upstream_error=raw_error,
        )

    if any(kw in error_str for kw in _CONTRACT_KEYWORDS):
        return BridgeError(
            code=BridgeErrorCode.CONTRACT_ERROR,
            message=str(raw_error),
            retryable=False,
            upstream_error=raw_error,
        )

    return BridgeError(
        code=BridgeErrorCode.UNKNOWN,
        message=str(raw_error),
        retryable=False,
        upstream_error=raw_error,
    )


# HTTP translation for bridge errors surfaced through the API. Retryable codes
# map to 503/504 and carry a Retry-After hint; terminal codes never do.
BRIDGE_ERROR_HTTP_STATUS: dict[BridgeErrorCode, int] = {
    BridgeErrorCode.TIMEOUT: status.HTTP_504_GATEWAY_TIMEOUT,
    BridgeErrorCode.NETWORK_ERROR: status.HTTP_503_SERVICE_UNAVAILABLE,
    BridgeErrorCode.REJECTED: status.HTTP_422_UNPROCESSABLE_ENTITY,
    BridgeErrorCode.INSUFFICIENT_FUNDS: status.HTTP_402_PAYMENT_REQUIRED,
    BridgeErrorCode.CONTRACT_ERROR: status.HTTP_502_BAD_GATEWAY,
    BridgeErrorCode.MALFORMED_RESPONSE: status.HTTP_502_BAD_GATEWAY,
    BridgeErrorCode.UNKNOWN: status.HTTP_500_INTERNAL_SERVER_ERROR,
}

BRIDGE_RETRY_AFTER_SECONDS = 5
UNKNOWN_BRIDGE_ERROR_MESSAGE = "Contract bridge returned an unrecognised error."
MALFORMED_BRIDGE_RESPONSE_MESSAGE = (
    "Contract bridge returned a response that could not be read."
)

# Codes whose raw text is never shown to clients. An unclassified error may
# carry upstream internals, and a malformed-response error usually quotes the
# payload it failed to read.
_CLIENT_SAFE_MESSAGES: dict[BridgeErrorCode, str] = {
    BridgeErrorCode.UNKNOWN: UNKNOWN_BRIDGE_ERROR_MESSAGE,
    BridgeErrorCode.MALFORMED_RESPONSE: MALFORMED_BRIDGE_RESPONSE_MESSAGE,
}


def http_status_for_bridge_code(code: Any) -> int:
    """Resolve the HTTP status for a bridge error code, falling back to 500."""
    try:
        return BRIDGE_ERROR_HTTP_STATUS[BridgeErrorCode(code)]
    except (ValueError, KeyError):
        return BRIDGE_ERROR_HTTP_STATUS[BridgeErrorCode.UNKNOWN]


def bridge_error_to_http(error: BridgeError) -> HTTPException:
    """Translate a BridgeError into an HTTPException without leaking upstream details."""
    try:
        code = BridgeErrorCode(error.code)
    except ValueError:
        code = BridgeErrorCode.UNKNOWN
    # Retryability is decided by the code's class, not only by the flag on the
    # error: a terminal code is never advertised as retryable, even if a caller
    # built the BridgeError with retryable=True.
    retryable = bool(error.retryable) and code in RETRYABLE_BRIDGE_CODES
    headers = {"Retry-After": str(BRIDGE_RETRY_AFTER_SECONDS)} if retryable else None
    return HTTPException(
        status_code=http_status_for_bridge_code(code),
        detail={
            "error_code": code.value,
            "message": _CLIENT_SAFE_MESSAGES.get(code, error.message),
            "retryable": retryable,
        },
        headers=headers,
    )
