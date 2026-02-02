from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional


def extract_asin(value: str) -> Optional[str]:
    candidate = value.strip()
    if re.fullmatch(r"[A-Z0-9]{10}", candidate):
        return candidate
    match = re.search(r"/dp/([A-Z0-9]{10})", candidate)
    if match:
        return match.group(1)
    match = re.search(r"/gp/product/([A-Z0-9]{10})", candidate)
    if match:
        return match.group(1)
    match = re.search(r"/product/([A-Z0-9]{10})", candidate)
    if match:
        return match.group(1)
    return None


class PriceProvider:
    def get_price(self, asin: str) -> Optional[float]:
        raise NotImplementedError


class NullPriceProvider(PriceProvider):
    def get_price(self, asin: str) -> Optional[float]:
        return None


@dataclass
class AmazonPAAPIConfig:
    access_key: str
    secret_key: str
    partner_tag: str
    host: str = "webservices.amazon.com"
    region: str = "us-east-1"


class AmazonPAAPIPriceProvider(PriceProvider):
    def __init__(self, config: AmazonPAAPIConfig):
        self.config = config

    @classmethod
    def from_env(cls) -> Optional["AmazonPAAPIPriceProvider"]:
        access_key = os.getenv("AMAZON_PAAPI_ACCESS_KEY")
        secret_key = os.getenv("AMAZON_PAAPI_SECRET_KEY")
        partner_tag = os.getenv("AMAZON_PAAPI_PARTNER_TAG")
        if not (access_key and secret_key and partner_tag):
            return None
        host = os.getenv("AMAZON_PAAPI_HOST", "webservices.amazon.com")
        region = os.getenv("AMAZON_PAAPI_REGION", "us-east-1")
        return cls(
            AmazonPAAPIConfig(
                access_key=access_key,
                secret_key=secret_key,
                partner_tag=partner_tag,
                host=host,
                region=region,
            )
        )

    def get_price(self, asin: str) -> Optional[float]:
        payload = {
            "ItemIds": [asin],
            "PartnerTag": self.config.partner_tag,
            "PartnerType": "Associates",
            "Marketplace": "www.amazon.com",
            "Resources": ["Offers.Listings.Price"],
        }
        body = json.dumps(payload)
        headers = self._signed_headers(body)
        url = f"https://{self.config.host}/paapi5/getitems"
        request = urllib.request.Request(url, data=body.encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None
        return self._extract_price(data)

    def _extract_price(self, data: dict) -> Optional[float]:
        items = data.get("ItemsResult", {}).get("Items", [])
        if not items:
            return None
        offers = items[0].get("Offers", {}).get("Listings", [])
        if not offers:
            return None
        price = offers[0].get("Price", {}).get("Amount")
        if price is None:
            return None
        try:
            return float(price)
        except (TypeError, ValueError):
            return None

    def _signed_headers(self, payload: str) -> dict:
        now = dt.datetime.utcnow()
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        canonical_uri = "/paapi5/getitems"
        canonical_querystring = ""
        content_type = "application/json; charset=utf-8"
        signed_headers = "content-type;host;x-amz-date;x-amz-target"
        canonical_headers = (
            f"content-type:{content_type}\n"
            f"host:{self.config.host}\n"
            f"x-amz-date:{amz_date}\n"
            "x-amz-target:com.amazon.paapi5.v1.ProductAdvertisingAPIv1.GetItems\n"
        )
        payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        canonical_request = (
            "POST\n"
            f"{canonical_uri}\n"
            f"{canonical_querystring}\n"
            f"{canonical_headers}\n"
            f"{signed_headers}\n"
            f"{payload_hash}"
        )
        algorithm = "AWS4-HMAC-SHA256"
        credential_scope = f"{date_stamp}/{self.config.region}/ProductAdvertisingAPI/aws4_request"
        string_to_sign = (
            f"{algorithm}\n{amz_date}\n{credential_scope}\n"
            f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        )
        signing_key = self._get_signature_key(
            self.config.secret_key, date_stamp, self.config.region, "ProductAdvertisingAPI"
        )
        signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        authorization_header = (
            f"{algorithm} Credential={self.config.access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        return {
            "Content-Type": content_type,
            "X-Amz-Date": amz_date,
            "X-Amz-Target": "com.amazon.paapi5.v1.ProductAdvertisingAPIv1.GetItems",
            "Authorization": authorization_header,
            "Host": self.config.host,
        }

    @staticmethod
    def _get_signature_key(key: str, date_stamp: str, region: str, service: str) -> bytes:
        k_date = hmac.new(f"AWS4{key}".encode("utf-8"), date_stamp.encode("utf-8"), hashlib.sha256).digest()
        k_region = hmac.new(k_date, region.encode("utf-8"), hashlib.sha256).digest()
        k_service = hmac.new(k_region, service.encode("utf-8"), hashlib.sha256).digest()
        return hmac.new(k_service, b"aws4_request", hashlib.sha256).digest()


def build_price_provider() -> PriceProvider:
    provider = AmazonPAAPIPriceProvider.from_env()
    if provider is None:
        return NullPriceProvider()
    return provider
