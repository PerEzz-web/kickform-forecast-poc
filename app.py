import os
import json
import streamlit as st
from urllib.parse import urlparse
import openai  # for openai.__version__
from openai import OpenAI, BadRequestError

# ----------------------------
# Defaults
# ----------------------------

DEFAULT_ALLOWED_DOMAINS = (
    "fotmob.com, sofascore.com, flashscore.com, whoscored.com, soccerway.com, "
    "fbref.com, theanalyst.com, transfermarkt.com, espn.com, onefootball.com"
)

DEFAULT_SYSTEM_PROMPT = """
You are MatchNarrator, a football research-and-writing agent for general fans.

INPUT:
- match_url: a link to our match page (contains our forecasts)
- allowed_domains: you must browse only these domains (domain allow-list)

TASK:
1) Use web search to open match_url and extract our forecast numbers exactly as shown on the page:
   - Value Tip (top pick) + confidence stars (1–5)
   - Correct Score Probability (ONLY the most probable scoreline)
   - Both Teams To Score (ONLY the most probable of Yes/No)
   - Match Goals Probability (describe ALL options within that market)
2) Use web search (still inside allowed domains) to gather light match context
   (recent form, head-to-head, team news if available).
3) Write simple, fan-friendly explanations for the numbers.

HARD RULES:
- Output MUST NOT include links, references/citations, source names, formulas, or technical/analytics jargon.
- Do NOT change probabilities you extracted. Copy numbers exactly.
- Do NOT invent injuries/lineups/transfers. If unclear, say it's not confirmed.
- Keep it easy to read and short.

OUTPUT:
Return ONLY valid JSON with exactly these keys (string values):
{
  "match_text": "...",
  "value_tip_text": "...",
  "correct_score_text": "...",
  "btts_text": "...",
  "match_goals_text": "..."
}

TEXT REQUIREMENTS:
- match_text: 5–9 sentences, overall match expectation.
- value_tip_text: 3–6 sentences. Start with: "<Pick> — <probability>% — <stars>"
- correct_score_text: 3–5 sentences, only most probable scoreline + probability.
- btts_text: 3–5 sentences, only most probable Yes/No + probability.
- match_goals_text: 5–9 sentences, MUST mention ALL match goals options + probabilities (bullets allowed).
""".strip()


# ----------------------------
# Helpers
# ----------------------------

def get_openai_client() -> OpenAI:
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets (recommended) or as an environment variable.")
        st.stop()
    return OpenAI(api_key=api_key)


def normalize_domains(domains_csv: str) -> list[str]:
    domains = [d.strip().lower() for d in (domains_csv or "").split(",") if d.strip()]
    domains = [d.replace("https://", "").replace("http://", "").strip().strip("/") for d in domains]

    # Remove leading www.
    domains = [d[4:] if d.startswith("www.") else d for d in domains]

    # Deduplicate, preserve order
    seen = set()
    out = []
    for d in domains:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def extract_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().strip()
        host = host.replace("www.", "")
        return host
    except Exception:
        return ""


def extract_response_text(resp) -> str:
    """
    Robustly extract assistant text even when resp.output_text is empty.
    """
    # First try the convenience property if present
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    parts = []
    output = getattr(resp, "output", None) or []
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None) or []
        for c in content:
            ctype = getattr(c, "type", None)
            if ctype in ("output_text", "text"):
                parts.append(getattr(c, "text", "") or "")
            elif ctype == "refusal":
                parts.append(getattr(c, "refusal", "") or "")

    return "\n".join([p for p in parts if p]).strip()


def stars(n: int) -> str:
    n = int(n or 0)
    n = max(0, min(5, n))
    return "★" * n + "☆" * (5 - n)


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="Match Forecast Explainer (POC)", layout="wide")
st.title("Match Forecast Explainer (POC)")

# Optional password gate
app_pw = ""
try:
    app_pw = st.secrets.get("APP_PASSWORD", "") or ""
except Exception:
    app_pw = os.getenv("APP_PASSWORD", "") or ""

if app_pw:
    entered = st.text_input("Password", type="password")
    if entered != app_pw:
        st.info("Enter the password to view the demo.")
        st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    match_url = st.text_input(
        "Match URL (page that contains your forecasts)",
        value="https://www.thepunterspage.com/kickform/premier-league/aston-villa-vs-fc-chelsea/nJBqa/"
    )

    allowed_domains_str = st.text_input(
        "Allowed domains for web search (comma-separated)",
        value=DEFAULT_ALLOWED_DOMAINS
    )

    include_match_domain = st.checkbox(
        "Also allow the match page’s domain (recommended so the agent can read your forecast page)",
        value=True
    )

    model = st.selectbox("Model", ["gpt-5-mini", "gpt-5.2", "gpt-5.2-pro"], index=0)
    max_tool_calls = st.slider("Max tool calls (limits browsing cost)", 1, 10, 5)

with col2:
    system_prompt = st.text_area("System prompt (editable)", value=DEFAULT_SYSTEM_PROMPT, height=430)

st.divider()

if st.button("Generate texts", type="primary"):
    if not match_url.strip():
        st.warning("Please paste a match URL.")
        st.stop()

    # Build allowlist
    allowed_domains = normalize_domains(allowed_domains_str)

    if include_match_domain:
        match_domain = extract_domain(match_url)
        if match_domain and match_domain not in allowed_domains:
            allowed_domains = [match_domain] + allowed_domains

    if not allowed_domains:
        st.warning("Allowed domains list is empty. Add at least one domain.")
        st.stop()

    # Create OpenAI client
    client = get_openai_client()

    payload = {
        "match_url": match_url,
        "allowed_domains": allowed_domains
    }

    with st.status("Generating…", expanded=False):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                tools=[
                    {
                        "type": "web_search",
                        "filters": {"allowed_domains": allowed_domains},
                        "search_context_size": "medium",
                    }
                ],
                tool_choice="required",
                max_tool_calls=max_tool_calls,
                reasoning={"effort": "low"},
                max_output_tokens=1000,
                text={"format": {"type": "json_object"}},
            )
        except BadRequestError as e:
            st.error("OpenAI request failed (BadRequest). Details below:")
            st.caption(f"openai sdk version: {openai.__version__}")
            st.caption(f"model: {model}")
            st.caption(f"allowed_domains count: {len(allowed_domains)}")
        
            # This usually contains the actual reason (e.g., model access, invalid param, etc.)
            if hasattr(e, "body") and e.body:
                st.json(e.body)
            else:
                st.code(str(e))
        
            st.stop()

    raw = extract_response_text(resp)

    # Debug/status info (safe)
    st.caption(f"Response status: {getattr(resp, 'status', 'unknown')}")

    if not raw:
        st.warning("No text returned. Debug info below:")
        st.write(
            {
                "status": getattr(resp, "status", None),
                "incomplete_details": getattr(resp, "incomplete_details", None),
                "output_items": len(getattr(resp, "output", []) or []),
            }
        )
        st.stop()

    # Parse JSON
    try:
        result = json.loads(raw)
    except Exception:
        st.error("Model did not return valid JSON. Raw output below:")
        st.code(raw)
        st.stop()

    # Render outputs
    st.subheader("1) Whole match (general expectation)")
    st.write(result.get("match_text", ""))

    st.subheader("2) Value Tip (top pick)")
    st.write(result.get("value_tip_text", ""))

    st.subheader("3) Correct Score Probability (most probable)")
    st.write(result.get("correct_score_text", ""))

    st.subheader("4) Both Teams to Score (most probable)")
    st.write(result.get("btts_text", ""))

    st.subheader("5) Match Goals Probability (all options)")
    st.write(result.get("match_goals_text", ""))

    # Optional: show parsed JSON for debugging
    with st.expander("Debug (parsed JSON)"):
        st.json(result)
