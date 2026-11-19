"""
Issue #953 – Bridge error mapping to HTTP responses.

Covers app/services/contracts/bridge_error_mapper.py:
- Known contract error codes map to deterministic HTTP statuses.
- Unknown / unclassified codes fall back safely to 500 without leaking detail.
- Retryable and terminal errors are distinguished in status and headers.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.contracts.bridge_error_mapper import (
    BRIDGE_ERROR_HTTP_STATUS,
    BRIDGE_RETRY_AFTER_SECONDS,
    MALFORMED_BRIDGE_RESPONSE_MESSAGE,
    RETRYABLE_BRIDGE_CODES,
    UNKNOWN_BRIDGE_ERROR_MESSAGE,
    BridgeError,
    BridgeErrorCode,
    bridge_error_to_http,
    http_status_for_bridge_code,
    map_bridge_error,
)

RETRYABLE_CODES = {BridgeErrorCode.TIMEOUT, BridgeErrorCode.NETWORK_ERROR}
TERMINAL_CODES = set(BridgeErrorCode) - RETRYABLE_CODES


class TestKnownCodesMapped:
    @pytest.mark.parametrize(
        "code, expected_status",
        [
            (BridgeErrorCode.TIMEOUT, 504),
            (BridgeErrorCode.NETWORK_ERROR, 503),
            (BridgeErrorCode.REJECTED, 422),
            (BridgeErrorCode.INSUFFICIENT_FUNDS, 402),
            (BridgeErrorCode.CONTRACT_ERROR, 502),
            (BridgeErrorCode.MALFORMED_RESPONSE, 502),
            (BridgeErrorCode.UNKNOWN, 500),
        ],
    )
    def test_code_maps_to_status(self, code, expected_status):
        assert http_status_for_bridge_code(code) == expected_status

    def test_every_enum_member_has_explicit_mapping(self):
        assert set(BRIDGE_ERROR_HTTP_STATUS) == set(BridgeErrorCode)

    @pytest.mark.parametrize("code", list(BridgeErrorCode))
    def test_string_code_resolves_like_enum(self, code):
        assert http_status_for_bridge_code(code.value) == http_status_for_bridge_code(code)

    @pytest.mark.parametrize(
        "raw, expected_code, expected_status",
        [
            ("Request timed out after 30s", BridgeErrorCode.TIMEOUT, 504),
            ("connection refused by host", BridgeErrorCode.NETWORK_ERROR, 503),
            ("transaction rejected by validator", BridgeErrorCode.REJECTED, 422),
            ("insufficient funds in source account", BridgeErrorCode.INSUFFICIENT_FUNDS, 402),
            ("soroban invoke failed: wasm trap", BridgeErrorCode.CONTRACT_ERROR, 502),
        ],
    )
    def test_raw_error_round_trips_to_http(self, raw, expected_code, expected_status):
        bridge_error = map_bridge_error(raw)
        exc = bridge_error_to_http(bridge_error)

        assert bridge_error.code is expected_code
        assert exc.status_code == expected_status
        assert exc.detail["error_code"] == expected_code.value
        assert exc.detail["message"] == raw


class TestUnknownCodesFallBackSafely:
    @pytest.mark.parametrize("code", ["NOT_A_REAL_CODE", "", None, 42, "timeout"])
    def test_unrecognised_code_falls_back_to_500(self, code):
        assert http_status_for_bridge_code(code) == 500

    def test_unclassified_raw_error_maps_to_unknown_500(self):
        exc = bridge_error_to_http(map_bridge_error("something odd happened"))

        assert exc.status_code == 500
        assert exc.detail["error_code"] == BridgeErrorCode.UNKNOWN.value
        assert exc.detail["retryable"] is False

    def test_unknown_message_does_not_leak_upstream_text(self):
        secret = "stack: /srv/bridge/internal.py line 7 key=abc123"
        exc = bridge_error_to_http(map_bridge_error(secret))

        assert exc.detail["message"] == UNKNOWN_BRIDGE_ERROR_MESSAGE
        assert secret not in str(exc.detail)

    def test_bogus_code_on_bridge_error_is_coerced_to_unknown(self):
        error = BridgeError(code="BOGUS", message="raw upstream", retryable=True)  # type: ignore[arg-type]
        exc = bridge_error_to_http(error)

        assert exc.status_code == 500
        assert exc.detail["error_code"] == BridgeErrorCode.UNKNOWN.value
        assert exc.detail["retryable"] is False
        assert exc.headers is None

    def test_upstream_error_object_is_never_exposed(self):
        upstream = RuntimeError("connection reset")
        exc = bridge_error_to_http(map_bridge_error(upstream))

        assert set(exc.detail) == {"error_code", "message", "retryable"}


class TestRetryableVsTerminal:
    @pytest.mark.parametrize("code", sorted(RETRYABLE_CODES))
    def test_retryable_codes_get_retry_after_and_5xx(self, code):
        exc = bridge_error_to_http(BridgeError(code=code, message="m", retryable=True))

        assert exc.detail["retryable"] is True
        assert exc.headers == {"Retry-After": str(BRIDGE_RETRY_AFTER_SECONDS)}
        assert exc.status_code in (503, 504)

    @pytest.mark.parametrize("code", sorted(TERMINAL_CODES))
    def test_terminal_codes_have_no_retry_after(self, code):
        exc = bridge_error_to_http(BridgeError(code=code, message="m", retryable=False))

        assert exc.detail["retryable"] is False
        assert exc.headers is None
        assert exc.status_code not in (503, 504)

    def test_mapper_retryable_flag_matches_classification(self):
        samples = {
            "deadline exceeded": True,
            "dns lookup failed": True,
            "forbidden": False,
            "not enough balance": False,
            "contract panicked": False,
            "???": False,
        }
        for raw, expected in samples.items():
            error = map_bridge_error(raw)
            assert error.retryable is expected, raw
            assert bridge_error_to_http(error).detail["retryable"] is expected, raw


class TestTimeoutVersusMalformedResponse:
    """Issue #1053: a timeout is retryable, a malformed response is terminal."""

    @pytest.mark.parametrize(
        "raw",
        [
            "Request timed out after 30s",
            "soroban rpc timeout",
            "deadline exceeded",
            "read timed out",
        ],
    )
    def test_timeout_is_a_retryable_gateway_timeout(self, raw):
        error = map_bridge_error(raw)
        exc = bridge_error_to_http(error)

        assert error.code is BridgeErrorCode.TIMEOUT
        assert error.retryable is True
        assert exc.status_code == 504
        assert exc.headers == {"Retry-After": str(BRIDGE_RETRY_AFTER_SECONDS)}
        assert exc.detail["retryable"] is True

    @pytest.mark.parametrize(
        "raw",
        [
            "malformed contract response",
            "invalid JSON in bridge reply",
            "could not parse contract payload",
            "failed to decode result XDR",
            "invalid XDR returned by soroban rpc",
            "unexpected response shape from contract",
            "schema mismatch: result is not a map",
            "missing required field: status",
            "Expecting value: line 1 column 1 (char 0)",
        ],
    )
    def test_malformed_payload_is_a_terminal_upstream_error(self, raw):
        error = map_bridge_error(raw)
        exc = bridge_error_to_http(error)

        assert error.code is BridgeErrorCode.MALFORMED_RESPONSE
        assert error.retryable is False
        assert exc.status_code == 502
        assert exc.headers is None
        assert exc.detail == {
            "error_code": "MALFORMED_RESPONSE",
            "message": MALFORMED_BRIDGE_RESPONSE_MESSAGE,
            "retryable": False,
        }

    def test_the_two_classes_differ_in_code_status_and_retry(self):
        timeout = bridge_error_to_http(map_bridge_error("request timed out"))
        malformed = bridge_error_to_http(map_bridge_error("malformed response"))

        assert timeout.detail["error_code"] != malformed.detail["error_code"]
        assert timeout.status_code != malformed.status_code
        assert timeout.detail["retryable"] is True
        assert malformed.detail["retryable"] is False

    @pytest.mark.parametrize(
        "raw",
        [
            TimeoutError(),
            TimeoutError("could not parse anything before the deadline"),
        ],
    )
    def test_timeout_exception_type_wins_over_its_text(self, raw):
        error = map_bridge_error(raw)

        assert error.code is BridgeErrorCode.TIMEOUT
        assert error.retryable is True

    def test_json_decode_error_is_malformed_whatever_its_text_says(self):
        try:
            json.loads("request timed out")
        except json.JSONDecodeError as exc:
            error = map_bridge_error(exc)

        assert error.code is BridgeErrorCode.MALFORMED_RESPONSE
        assert error.retryable is False

    def test_unicode_decode_error_is_malformed(self):
        try:
            b"\xff\xfe".decode("utf-8")
        except UnicodeDecodeError as exc:
            error = map_bridge_error(exc)

        assert error.code is BridgeErrorCode.MALFORMED_RESPONSE

    def test_payload_text_mentioning_a_timeout_field_is_not_retried(self):
        raw = 'invalid response: missing required field "timeout_ledger"'

        error = map_bridge_error(raw)

        assert error.code is BridgeErrorCode.MALFORMED_RESPONSE
        assert bridge_error_to_http(error).detail["retryable"] is False

    def test_malformed_message_does_not_leak_the_payload(self):
        payload = '{"secret_key": "SB3X-EXAMPLE", "sta'
        exc = bridge_error_to_http(
            map_bridge_error(f"could not parse contract payload: {payload}")
        )

        assert exc.detail["message"] == MALFORMED_BRIDGE_RESPONSE_MESSAGE
        assert "SB3X-EXAMPLE" not in str(exc.detail)

    def test_the_raw_error_is_kept_for_server_side_logging(self):
        raw = "could not parse contract payload: {..."

        error = map_bridge_error(raw)

        assert error.message == raw
        assert error.upstream_error == raw

    def test_only_timeout_and_network_errors_are_retryable_classes(self):
        assert RETRYABLE_BRIDGE_CODES == RETRYABLE_CODES

    @pytest.mark.parametrize("code", sorted(TERMINAL_CODES))
    def test_a_terminal_code_flagged_retryable_is_still_not_advertised(self, code):
        exc = bridge_error_to_http(BridgeError(code=code, message="m", retryable=True))

        assert exc.detail["retryable"] is False
        assert exc.headers is None

    def test_contract_errors_are_still_classified_as_contract_errors(self):
        assert (
            map_bridge_error("soroban invoke failed: wasm trap").code
            is BridgeErrorCode.CONTRACT_ERROR
        )


class TestHttpResponseShape:
    """The HTTPException renders as a real API response through FastAPI."""

    @pytest.fixture
    def client(self):
        app = FastAPI()

        @app.get("/bridge/{raw}")
        def call_bridge(raw: str):
            raise bridge_error_to_http(map_bridge_error(raw))

        return TestClient(app)

    def test_retryable_response(self, client):
        resp = client.get("/bridge/request timed out")

        assert resp.status_code == 504
        assert resp.headers["Retry-After"] == str(BRIDGE_RETRY_AFTER_SECONDS)
        assert resp.json()["detail"] == {
            "error_code": "TIMEOUT",
            "message": "request timed out",
            "retryable": True,
        }

    def test_terminal_response(self, client):
        resp = client.get("/bridge/insufficient funds")

        assert resp.status_code == 402
        assert "Retry-After" not in resp.headers
        assert resp.json()["detail"]["retryable"] is False

    def test_unknown_response(self, client):
        resp = client.get("/bridge/mystery")

        assert resp.status_code == 500
        assert resp.json()["detail"]["error_code"] == "UNKNOWN"
        assert resp.json()["detail"]["message"] == UNKNOWN_BRIDGE_ERROR_MESSAGE

    def test_malformed_response(self, client):
        resp = client.get("/bridge/malformed contract response")

        assert resp.status_code == 502
        assert "Retry-After" not in resp.headers
        assert resp.json()["detail"] == {
            "error_code": "MALFORMED_RESPONSE",
            "message": MALFORMED_BRIDGE_RESPONSE_MESSAGE,
            "retryable": False,
        }
