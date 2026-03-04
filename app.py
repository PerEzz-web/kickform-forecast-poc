import os
import json
import base64
import streamlit as st
from openai import OpenAI
from playwright.sync_api import sync_playwright

# ----------------------------
# Default prompt (editable in UI)
# ----------------------------
DEFAULT_SYSTEM_PROMPT = """
You are MatchNarrator, a football explainer for general fans.

You will receive:
- match_url
- optional home team / away team / competition
- screenshots of our match page that contains our forecasts.

Your task:
1) Read the screenshots and extract these forecast numbers exactly as shown:
   - Value Tip: selection + confidence rating (1–5 stars)
   - Correct Score Probability: choose ONLY the most probable scoreline + its probability
   - Both Teams to Score: choose ONLY the most probable (Yes/No) + its probability
   - Match Goals Probability: include ALL listed options + probabilities
2) Write fan-friendly explanations (simple football language, short sentences).

Hard rules:
- No links, no references, no citations, no source names, no formulas, no technical model talk.
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


# ----------------------------
# Helpers
# ----------------------------

def get_client():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets.")
        st.stop()
    return OpenAI(api_key=api_key)


def png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def take_screenshots(url: str):
    """
    Returns a list of PNG bytes.
    - full page screenshot
    - second screenshot after scrolling
    Uses safe Chromium flags for hosted environments (Streamlit Cloud).
    """
    images = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )

            page = browser.new_page(viewport={"width": 1280, "height": 720})

            # Load the page; 'networkidle' can hang on some sites, so we try networkidle first, then fallback.
            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
            except Exception:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)

            # Wait a moment for dynamic widgets
            page.wait_for_timeout(2000)

            # Screenshot 1: full page
            images.append(page.screenshot(full_page=True, type="png"))

            # Screenshot 2: scroll to middle and capture viewport
            page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
            page.wait_for_timeout(1500)
            images.append(page.screenshot(full_page=False, type="png"))

            browser.close()

    except Exception as e:
        # Show the real error (Streamlit otherwise redacts it)
        st.error("Playwright failed to capture screenshots.")
        st.code(repr(e))
        return []

    return images


def parse_json_or_show(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        # Try extracting a JSON object from surrounding text
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
        return None


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="KickForm POC — Screenshot → Texts", layout="wide")
st.title("KickForm POC — Screenshot → Fan-friendly forecast texts")

# Optional password gate
app_pw = st.secrets.get("APP_PASSWORD", "") or os.getenv("APP_PASSWORD", "")
if app_pw:
    entered = st.text_input("Password", type="password")
    if entered != app_pw:
        st.info("Enter the password to access the demo.")
        st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    match_url = st.text_input("Match URL", value="")

    home_team = st.text_input("Home team (optional)", value="")
    away_team = st.text_input("Away team (optional)", value="")
    competition = st.text_input("Competition (optional)", value="")

    model = st.selectbox("Model", ["gpt-5.2", "gpt-5-mini"], index=0)

with col2:
    system_prompt = st.text_area("System prompt (editable)", value=DEFAULT_SYSTEM_PROMPT, height=420)

st.divider()

if st.button("Generate texts", type="primary"):
    if not match_url.strip():
        st.warning("Please paste a match URL first.")
        st.stop()

    with st.status("Taking screenshots…", expanded=False):
        imgs = take_screenshots(match_url)

    if not imgs:
        st.stop()

    st.subheader("Captured screenshots (for demo)")
    for i, img in enumerate(imgs, start=1):
        st.image(img, caption=f"Screenshot {i}", use_container_width=True)

    client = get_client()

    # Build multimodal user content: JSON + images
    content = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "match_url": match_url,
                    "home_team": home_team,
                    "away_team": away_team,
                    "competition": competition,
                    "instructions": "Extract forecasts from screenshots and output the required JSON only.",
                }
            ),
        }
    ]

    for img in imgs:
        content.append({"type": "input_image", "image_url": png_bytes_to_data_url(img)})

    with st.status("Generating fan-friendly texts…", expanded=False):
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            reasoning={"effort": "low"},
            max_output_tokens=1400,
        )

    raw = getattr(resp, "output_text", "") or ""
    result = parse_json_or_show(raw)

    if not result:
        st.error("Model did not return valid JSON. Raw output below:")
        st.code(raw)
        st.stop()

    st.success("Done!")

    st.subheader("1) Whole match (general expectation)")
    st.write(result.get("match_text", ""))

    st.subheader("2) Value Tip (top pick)")
    st.write(result.get("value_tip_text", ""))

    st.subheader("3) Correct Score (most probable)")
    st.write(result.get("correct_score_text", ""))

    st.subheader("4) Both Teams To Score (most probable)")
    st.write(result.get("btts_text", ""))

    st.subheader("5) Match Goals Probability (all options)")
    st.write(result.get("match_goals_text", ""))

    with st.expander("Debug (parsed JSON)"):
        st.json(result)
