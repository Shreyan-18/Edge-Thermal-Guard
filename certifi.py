"""
Minimal certifi shim for requests and urllib3
"""
import ssl
import os

def where():
    return os.environ.get("SSL_CERT_FILE", "")

def contents():
    return ""
