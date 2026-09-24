from turtledemo.chaos import line

import httpx
YOUR_API_KEY="1Q... "
BASE = "https://lab.flashalpha.com"
H = {"X-Api-Key": "YOUR_API_KEY"}

# spot + reference values from the precomputed endpoint (Free tier)
ref = httpx.get(f"{BASE}/v1/exposure/gex/SPY", headers=H).json()
S = ref["underlying_price"]

# raw per-contract quotes run on the Growth plan
expiries = httpx.get(f"{BASE}/v1/options/SPY", headers=H).json()["expirations"]

net = 0.0
for exp in expiries:
    for c in httpx.get(f"{BASE}/optionquote/SPY", params={"expiry": exp}, headers=H).json():
        if c.get("gamma") and c.get("open_interest"):
            gex = c["gamma"] * c["open_interest"] * 100 * S * S * 0.01
            net += gex if c["type"] == "C" else -gex

print(f"hand-rolled net GEX: ${net/1e9:.2f}B per 1% move")
print(f"API net GEX:         ${ref['net_gex']/1e9:.2f}B, flip {ref['gamma_flip']}")

####  Call FlashAlpha from command line
# pip install FlashAlpha
#
# from flashalpha import FlashAlpha
# fa = FlashAlpha(api_key="YOUR_API_KEY")
# gex = fa.gex("AAPL", expiration="2026-09-25")
# print(f"Gamma flip: {gex['gamma_flip']}")

