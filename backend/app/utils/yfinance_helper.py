"""yfinance helper utilities"""
import yfinance as yf
import os

# Fix SSL certificate issue for Python 3.13
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()

# Configure yfinance to avoid Yahoo Finance blocking
try:
    import yfinance.const as yf_const
    if hasattr(yf_const, 'USER_AGENT'):
        yf_const.USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
except:
    pass


def create_yf_ticker(symbol: str, session=None):
    """Create a yfinance ticker - let yfinance handle anti-blocking with curl_cffi"""
    # yfinance >= 0.2.40 handles anti-blocking internally with curl_cffi
    # Don't pass session parameter - let yfinance create its own optimized session
    return yf.Ticker(symbol)



