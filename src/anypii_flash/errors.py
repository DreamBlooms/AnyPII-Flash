from __future__ import annotations


class AnyPiiError(Exception):
    def __init__(
        self, message: str, status_code: int = 400, error_type: str = "invalid_request_error"
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type


class UnsupportedFeatureError(AnyPiiError):
    def __init__(self, message: str):
        super().__init__(message, status_code=400, error_type="unsupported_feature")


class ModelUnavailableError(AnyPiiError):
    def __init__(self, message: str):
        super().__init__(message, status_code=503, error_type="model_unavailable")
