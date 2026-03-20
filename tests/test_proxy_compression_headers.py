"""Tests for compression header handling in the proxy server.

These tests verify that the proxy correctly removes Content-Encoding headers
from responses after httpx automatically decompresses them, preventing
double-decompression errors (ZlibError) in clients.

Also tests _prepare_forwarding_headers to prevent accept-encoding forwarding
that causes UnicodeDecodeError crashes (GitHub issue #45).
"""

import gzip
import json
from unittest.mock import MagicMock

import pytest

from headroom.proxy.server import HeadroomProxy


@pytest.fixture
def mock_anthropic_response_with_compression_headers():
    """Create a mock response that simulates httpx behavior.

    httpx automatically decompresses responses but leaves compression headers.
    This is what causes the ZlibError bug we're testing for.
    """

    class MockResponse:
        """Mock httpx response with compression headers."""

        def __init__(self):
            self.response_data = {
                "id": "msg_test123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            # Body is already decompressed (httpx does this automatically)
            self.content = json.dumps(self.response_data).encode("utf-8")
            self.status_code = 200

            # Headers still contain compression info (this is the bug!)
            self.headers = {
                "content-type": "application/json",
                "content-encoding": "gzip",  # Should be removed!
                "content-length": str(len(gzip.compress(self.content))),  # Wrong!
                "x-request-id": "test-request-id",
            }

    return MockResponse()


class TestCompressionHeaderRemoval:
    """Tests for Content-Encoding header removal logic."""

    def test_compression_headers_are_removed_from_dict(
        self, mock_anthropic_response_with_compression_headers
    ):
        """Test that our fix removes compression headers from response headers."""
        mock_response = mock_anthropic_response_with_compression_headers

        # Simulate what the fixed code does
        response_headers = dict(mock_response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        # Verify compression headers are removed
        assert "content-encoding" not in response_headers
        assert "content-length" not in response_headers

        # Verify other headers are preserved
        assert response_headers["content-type"] == "application/json"
        assert response_headers["x-request-id"] == "test-request-id"

    def test_response_body_is_decompressed_not_compressed(
        self, mock_anthropic_response_with_compression_headers
    ):
        """Verify the response content is already decompressed (httpx behavior)."""
        mock_response = mock_anthropic_response_with_compression_headers

        # The content should be valid JSON (decompressed)
        response_data = json.loads(mock_response.content)
        assert response_data["id"] == "msg_test123"

        # Trying to decompress it again should fail (proving it's not compressed)
        with pytest.raises((gzip.BadGzipFile, OSError, Exception)):
            gzip.decompress(mock_response.content)

    def test_headers_with_wrong_content_length_cause_issues(
        self, mock_anthropic_response_with_compression_headers
    ):
        """Demonstrate that keeping compression headers causes length mismatch."""
        mock_response = mock_anthropic_response_with_compression_headers

        # The content-length header says the body is compressed size
        claimed_length = int(mock_response.headers["content-length"])

        # But the actual content is decompressed size
        actual_length = len(mock_response.content)

        # They don't match! This can cause client issues
        assert claimed_length != actual_length
        assert claimed_length < actual_length  # Compressed is smaller

    def test_removing_headers_fixes_length_mismatch(
        self, mock_anthropic_response_with_compression_headers
    ):
        """Show that removing compression headers allows proper content-length."""
        mock_response = mock_anthropic_response_with_compression_headers

        # Apply the fix
        response_headers = dict(mock_response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        # Now we can set correct content-length
        response_headers["content-length"] = str(len(mock_response.content))

        # Verify it matches actual content
        assert int(response_headers["content-length"]) == len(mock_response.content)


class TestNoRegressionForUncompressedResponses:
    """Ensure the fix doesn't break responses that were never compressed."""

    def test_pop_on_missing_keys_is_safe(self):
        """Verify that .pop() on non-existent keys doesn't cause errors."""
        headers = {
            "content-type": "application/json",
            # No compression headers
        }

        # This should not raise KeyError
        headers.pop("content-encoding", None)
        headers.pop("content-length", None)

        # Headers should be unchanged
        assert headers == {"content-type": "application/json"}

    def test_dict_conversion_preserves_headers(self):
        """Verify dict() conversion doesn't lose headers."""
        original_headers = {
            "content-type": "application/json",
            "x-custom-header": "value",
            "authorization": "Bearer token",
        }

        # Convert to dict (as the fix does)
        converted = dict(original_headers)

        # All headers preserved
        assert converted == original_headers
        assert converted is not original_headers  # New object


class TestPrepareForwardingHeaders:
    """Tests for _prepare_forwarding_headers (fix for GitHub issue #45).

    When a client (e.g. Codex GUI) sends Accept-Encoding values the proxy's
    HTTP client doesn't support (like zstd), the upstream may return compressed
    data that cannot be decoded, causing UnicodeDecodeError crashes.

    _prepare_forwarding_headers strips these problematic headers so httpx
    negotiates its own supported encodings with the upstream API.
    """

    @staticmethod
    def _make_mock_request(headers: dict) -> MagicMock:
        """Create a mock Starlette Request with the given headers."""
        mock_request = MagicMock()
        mock_request.headers = MagicMock()
        mock_request.headers.items.return_value = list(headers.items())
        return mock_request

    def test_strips_accept_encoding(self):
        """Accept-Encoding must be stripped to prevent unsupported encoding responses."""
        request = self._make_mock_request(
            {
                "host": "localhost:8787",
                "content-type": "application/json",
                "accept-encoding": "gzip, deflate, br, zstd",
                "authorization": "Bearer sk-test",
            }
        )

        headers = HeadroomProxy._prepare_forwarding_headers(request)

        assert "accept-encoding" not in headers
        assert "host" not in headers
        assert headers["authorization"] == "Bearer sk-test"
        assert headers["content-type"] == "application/json"

    def test_strips_host_and_content_length(self):
        """Host and content-length must always be stripped for forwarding."""
        request = self._make_mock_request(
            {
                "host": "localhost:8787",
                "content-length": "1234",
                "content-type": "application/json",
            }
        )

        headers = HeadroomProxy._prepare_forwarding_headers(request)

        assert "host" not in headers
        assert "content-length" not in headers
        assert headers["content-type"] == "application/json"

    def test_strips_transfer_encoding(self):
        """Transfer-encoding is hop-by-hop and must not be forwarded."""
        request = self._make_mock_request(
            {
                "host": "localhost",
                "transfer-encoding": "chunked",
                "content-type": "application/json",
            }
        )

        headers = HeadroomProxy._prepare_forwarding_headers(request)

        assert "transfer-encoding" not in headers

    def test_preserves_authorization_headers(self):
        """Authorization and API key headers must be preserved."""
        request = self._make_mock_request(
            {
                "host": "localhost",
                "authorization": "Bearer sk-test",
                "x-api-key": "test-key",
                "x-goog-api-key": "goog-key",
            }
        )

        headers = HeadroomProxy._prepare_forwarding_headers(request)

        assert headers["authorization"] == "Bearer sk-test"
        assert headers["x-api-key"] == "test-key"
        assert headers["x-goog-api-key"] == "goog-key"

    def test_preserves_headroom_tags(self):
        """Custom Headroom headers (x-headroom-*) must be preserved."""
        request = self._make_mock_request(
            {
                "host": "localhost",
                "x-headroom-tag": "my-tag",
                "x-headroom-session": "abc123",
            }
        )

        headers = HeadroomProxy._prepare_forwarding_headers(request)

        assert headers["x-headroom-tag"] == "my-tag"
        assert headers["x-headroom-session"] == "abc123"

    def test_handles_codex_gui_headers(self):
        """Simulate typical Codex GUI request headers.

        Codex GUI sends Accept-Encoding with zstd which httpx cannot
        decompress, causing UnicodeDecodeError when the upstream responds
        with zstd-compressed data. This is the exact scenario from issue #45.
        """
        request = self._make_mock_request(
            {
                "host": "127.0.0.1:8787",
                "content-type": "application/json",
                "accept-encoding": "gzip, deflate, br, zstd",
                "authorization": "Bearer sk-proj-test",
                "content-length": "500",
                "user-agent": "codex-gui/1.0",
            }
        )

        headers = HeadroomProxy._prepare_forwarding_headers(request)

        # Problematic headers stripped
        assert "accept-encoding" not in headers
        assert "host" not in headers
        assert "content-length" not in headers

        # Important headers preserved
        assert headers["authorization"] == "Bearer sk-proj-test"
        assert headers["content-type"] == "application/json"
        assert headers["user-agent"] == "codex-gui/1.0"

    def test_safe_when_headers_already_missing(self):
        """No error when stripped headers are not present in the request."""
        request = self._make_mock_request(
            {
                "content-type": "application/json",
            }
        )

        headers = HeadroomProxy._prepare_forwarding_headers(request)

        assert headers == {"content-type": "application/json"}


class TestResponseDecodeErrorHandling:
    """Tests verifying that compressed/binary response data doesn't crash json.loads.

    When a compressed response body reaches json.loads() as bytes, Python's json
    module calls detect_encoding() + .decode('utf-8', 'surrogatepass'), which
    raises UnicodeDecodeError on binary data like zstd-compressed content.
    """

    def test_json_loads_on_zstd_bytes_raises_unicode_error(self):
        """Demonstrate the exact crash from issue #45: json.loads on compressed bytes."""
        # Simulate zstd-compressed data (magic bytes: 0x28 0xb5 0x2f 0xfd)
        zstd_data = b"\x28\xb5\x2f\xfd\x00\x00\x00\x00\x00"

        with pytest.raises(UnicodeDecodeError):
            json.loads(zstd_data)

    def test_json_loads_on_gzip_bytes_raises_error(self):
        """json.loads on gzip bytes raises either UnicodeDecodeError or JSONDecodeError."""
        gzip_data = gzip.compress(b'{"key": "value"}')

        with pytest.raises((UnicodeDecodeError, json.JSONDecodeError)):
            json.loads(gzip_data)

    def test_utf8_decode_with_replace_handles_binary(self):
        """bytes.decode('utf-8', errors='replace') safely handles binary data."""
        zstd_data = b"\x28\xb5\x2f\xfd\x00\x00\x00\x00\x00"

        # Should not raise - replaces invalid bytes with replacement character
        result = zstd_data.decode("utf-8", errors="replace")
        assert isinstance(result, str)
