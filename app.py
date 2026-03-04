import os
import json
import time
import base64
import streamlit as st
from openai import OpenAI
from playwright.sync_api import sync_playwright


DEFAULT_SYSTEM_PROMPT = """
You are MatchNarrator, a football explainer for general fans.

You will receive:
- match_url
- home team / away team / competition (may be provided)
- screenshots of our match page that contains our forecasts.

Your task:
1) Read the screenshots and extract these forecast numbers exactly as shown:
   - Value Tip: selection + confidence rating (1–5 stars)
   - Correct Score Probability: pick ONLY the most probable scoreline + its probability
   - Both Teams to Score: pick ONLY the most probable (Yes/No) + its probability
   - Match Goals Probability: include ALL listed options + probabilities
2) Write fan-friendly explanations (simple football language, short sentences).

Hard rules:
- No links, no references, no citations, no formulas, no technical model talk.
- Do not guess numbers. Only use what you can read in the screenshots.
- Return ONLY a single JSON object with exactly these keys (string values):
{
  "match_text": "...",
  "value_tip_text": "...",
  "correct_score_text": "...",
  "btts_text": "...",
  "match_goals_text": "..."
}
""".strip()


def get_client():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
        st.stop()
    return OpenAI(api_key=api_key)


def take_screenshots(url: str):
    """
    Returns a list of PNG bytes.
    Takes:
      - full page screenshot
      - a second screenshot after scrolling (helps if forecasts are lower)
    """
    images = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.goto(url, wait_until="networkidle", timeout=30000)

        # small wait for dynamic content
        page.wait_for_timeout(1500)

        # full page
        images.append(page.screenshot(full_page=True, type="png"))

        # scroll and take another
        page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
        page.wait_for_timeout(1500)
        images.append(page.screenshot(full_page=False, type="png"))

        browser.close()

    return images


def png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


st.set_page_config(page_title="KickForm POC (Screenshot)", layout="wide")
st.title("KickForm POC — Screenshot → Fan-friendly texts")

match_url = st.text_input("Match URL", value="")
home_team = st.text_input("Home team (optional)", value="")
away_team = st.text_input("Away team (optional)", value="")
competition = st.text_input("Competition (optional)", value="")

model = st.selectbox("Model", ["gpt-5.2", "gpt-5-mini"], index=0)
system_prompt = st.text_area("System prompt (editable)", value=DEFAULT_SYSTEM_PROMPT, height=340)

if st.button("Generate texts", type="primary"):
    if not match_url:
        st.warning("Paste a match URL first.")
        st.stop()

    with st.status("Taking screenshots…", expanded=False):
        imgs = take_screenshots(match_url)

    # show screenshots in the UI (nice for boss demo)
    st.subheader("Captured screenshots")
    for i, img in enumerate(imgs, start=1):
        st.image(img, caption=f"Screenshot {i}", use_container_width=True)

    client = get_client()

    content = [
        {
            "type": "text",
            "text": json.dumps({
                "match_url": match_url,
                "home_team": home_team,
                "away_team": away_team,
                "competition": competition
            })
        }
    ]

    for img in imgs:
        content.append({
            "type": "input_image",
            "image_url": png_bytes_to_data_url(img)
        })

    with st.status("Generating fan-friendly texts…", expanded=False):
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            reasoning={"effort": "low"},
            max_output_tokens=1200
        )

    raw = resp.output_text or ""
    try:
        result = json.loads(raw)
    except Exception:
        st.error("Model did not return valid JSON. Raw output below:")
        st.code(raw)
        st.stop()

    st.success("Done!")

    st.subheader("1) Whole match")
    st.write(result.get("match_text", ""))

    st.subheader("2) Value Tip")
    st.write(result.get("value_tip_text", ""))

    st.subheader("3) Correct Score")
    st.write(result.get("correct_score_text", ""))

    st.subheader("4) BTTS")
    st.write(result.get("btts_text", ""))

    st.subheader("5) Match Goals Probability")
    st.write(result.get("match_goals_text", ""))
