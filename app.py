import os
import json
import re
from urllib.parse import urlparse

import streamlit as st
import requests
from bs4 import BeautifulSoup

import openai  # just for version display in errors
from openai import OpenAI, BadRequestError


# ----------------------------
# Defaults
# ----------------------------

DEFAULT_ALLOWED_DOMAINS = (
    "fotmob.com, sofascore.com, flashscore.com, whoscored.com, soccerway.com, "
    "fbref.com, theanalyst.com, transfermarkt.com, espn.com, onefootball.com"
)

DEFAULT_SYSTEM_PROMPT = """
You are MatchNarrator, a football research-and-writing assistant for general fans.

You will receive:
- Match info (home team, away team, competition)
- Forecast numbers parsed from our match page (these are authoritative; DO NOT change them)
- Optional: web-search context from allowed football websites (recent form, team news etc.)

Your job:
Write 5 fan-friendly texts that explain/justify the numbers in plain English.

IMPORTANT RULES:
- Output MUST NOT include links, references/citations, source names, formulas, or technical model talk.
- Do NOT invent injuries/lineups/transfers. If uncertain, say it’s not confirmed/unclear.
- Keep it easy to read (short sentences, normal football language).
- Copy probabilities EXACTLY as provided (do not “fix” them).
- Return ONLY a single JSON object with exactly these keys (string values):
{
  "match_text": "...",
  "value_tip_text": "...",
  "correct_score_text": "...",
  "btts_text": "...",
  "match_goals_text": "..."
}

What each text should contain:
1) match_text:
   5–9 sentences. Overall match expectation / how it might play out.
   Use provided markets and (if available) light context from browsing.

2) value_tip_text:
   3–6 sentences.
   Start EXACTLY with: "<Selection> — <probability>% — <stars>"
   Example stars format: ★★☆☆☆

3) correct_score_text:
   3–5 sentences. ONLY the single most probable correct score + its probability.

4) btts_text:
   3–5 sentences. ONLY the most probable of Yes/No + its probability.

5) match_goals_text:
   5–9 sentences. Mention ALL match goals options and probabilities.
   Bullets are allowed here if it improves clarity.

Selection rules:
- “Most probable” = highest probability.
""".strip()

# Token budgets:
# If web_search is enabled, we do:
#   Step 1: web_search (no JSON mode allowed)
#   Step 2: finish (no tools, JSON mode on)
MAX_OUTPUT_TOKENS_SEARCH = 2800
MAX_OUTPUT_TOKENS_FINISH = 1400


# ----------------------------
# OpenAI helpers
# ----------------------------

def get_openai_client() -> OpenAI:
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets.")
        st.stop()
    return OpenAI(api_key=api_key)


def normalize_domains(domains_csv: str) -> list[str]:
    domains = [d.strip().lower() for d in (domains_csv or "").split(",") if d.strip()]
    domains = [d.replace("https://", "").replace("http://", "").strip().strip("/") for d in domains]
    domains = [d[4:] if d.startswith("www.") else d for d in domains]  # drop leading www.
    # dedupe preserve order
    seen, out = set(), []
    for d in domains:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def extract_response_text(resp) -> str:
    # convenience property
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    # robust walk
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


def extract_json_object(text: str):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


# ----------------------------
# Match-page parsing helpers
# ----------------------------

def parse_kickform_url(url: str):
    """
    Example:
    https://www.thepunterspage.com/kickform/premier-league/fc-brighton-hove-albion-vs-fc-arsenal/NbR6R/
    """
    m = re.search(r"/kickform/([^/]+)/([^/]+)/([^/]+)/?$", url)
    if not m:
        return None
    competition_slug, match_slug, match_id = m.group(1), m.group(2), m.group(3)

    parts = match_slug.split("-vs-")
    home = parts[0].replace("-", " ").title() if len(parts) > 0 else ""
    away = parts[1].replace("-", " ").title() if len(parts) > 1 else ""
    competition = competition_slug.replace("-", " ").title()

    # Optional cleanup (your example has "Fc " prefixes)
    home = home.replace("Fc ", "").strip()
    away = away.replace("Fc ", "").strip()

    return {"home_team": home, "away_team": away, "competition": competition, "match_id": match_id}


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _to_percent_number(s: str):
    # finds first X or X.Y before %
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    return float(m.group(1)) if m else None


def _stars_from_element(el) -> int | None:
    """
    Tries a few ways to detect star rating:
    - literal stars ★★★☆☆
    - "2/5" or "2 out of 5"
    - count of elements with 'active'/'filled' in classnames
    """
    if not el:
        return None

    text = _normalize_spaces(el.get_text(" ", strip=True))
    if "★" in text:
        return max(0, min(5, text.count("★")))

    m = re.search(r"(\d)\s*(?:/|out of)\s*5", text.lower())
    if m:
        return int(m.group(1))

    # Try to count active star-like children
    active = 0
    children = el.find_all(True)
    for c in children:
        cls = " ".join(c.get("class", [])).lower()
        if "star" in cls and ("active" in cls or "filled" in cls or "selected" in cls):
            active += 1
    if active > 0:
        return max(0, min(5, active))

    # Last fallback: aria-label like "2 out of 5"
    aria = (el.get("aria-label") or "").lower()
    m2 = re.search(r"(\d)\s*(?:/|out of)\s*5", aria)
    if m2:
        return int(m2.group(1))

    return None


def fetch_html(match_url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; KickFormPOC/1.0)"}
    r = requests.get(match_url, headers=headers, timeout=25)
    r.raise_for_status()
    return r.text


def extract_forecasts_from_match_page(match_url: str) -> dict:
    html = fetch_html(match_url)
    soup = BeautifulSoup(html, "html.parser")

    # --- Value Tip ---
    value_tip_selection = None
    value_tip_stars = None

    el_tip = soup.select_one(".value-tip__message")
    if el_tip:
        value_tip_selection = _normalize_spaces(el_tip.get_text(" ", strip=True))

    el_conf = soup.select_one(".value-tip__confidence-rating")
    if el_conf:
        value_tip_stars = _stars_from_element(el_conf)

    # --- Correct Score ---
    correct_scores = []
    for card in soup.select(".mp-correct-score-card"):
        text = _normalize_spaces(card.get_text(" ", strip=True))
        m_score = re.search(r"\b(\d{1,2})\s*-\s*(\d{1,2})\b", text)
        prob = _to_percent_number(text)
        if m_score and prob is not None:
            label = f"{m_score.group(1)}-{m_score.group(2)}"
            correct_scores.append({"label": label, "probability": prob})

    top_correct = max(correct_scores, key=lambda x: x["probability"]) if correct_scores else None

    # --- BTTS ---
    btts = {"Yes": None, "No": None}
    btts_card = soup.select_one(".mp-btts-card")
    if btts_card:
        txt = _normalize_spaces(btts_card.get_text(" ", strip=True)).lower()
        m_yes = re.search(r"\byes\b.*?(\d+(?:\.\d+)?)\s*%", txt)
        m_no = re.search(r"\bno\b.*?(\d+(?:\.\d+)?)\s*%", txt)
        if m_yes:
            btts["Yes"] = float(m_yes.group(1))
        if m_no:
            btts["No"] = float(m_no.group(1))

    btts_items = [(k, v) for k, v in btts.items() if v is not None]
    top_btts = max(btts_items, key=lambda kv: kv[1]) if btts_items else None

    # --- Match Goals Probability ---
    match_goals = {}
    outcome_cards = soup.select(".mp-outcome-card")
    best_card = None
    for card in outcome_cards:
        t = card.get_text(" ", strip=True)
        if ("Over 1.5" in t or "Over1.5" in t) and ("Under 1.5" in t or "Under1.5" in t):
            best_card = card
            break

    if best_card:
        t = _normalize_spaces(best_card.get_text(" ", strip=True))
        pairs = re.findall(r"\b(Over|Under)\s*(\d+(?:\.\d+)?)\b.*?(\d+(?:\.\d+)?)\s*%", t)
        for ou, line, prob in pairs:
            match_goals[f"{ou} {line}"] = float(prob)

    # --- Value tip probability (match it against match goals list, if possible) ---
    value_tip_prob = None
    if value_tip_selection and match_goals:
        m = re.search(r"\b(Over|Under)\s*(\d+(?:\.\d+)?)\b", value_tip_selection)
        if m:
            key = f"{m.group(1)} {m.group(2)}"
            value_tip_prob = match_goals.get(key)

    return {
        "value_tip": {
            "selection": value_tip_selection,
            "confidence_stars": value_tip_stars,
            "probability": value_tip_prob,
        },
        "correct_score_all": correct_scores,
        "correct_score_top": top_correct,
        "btts_all": btts,
        "btts_top": {"label": top_btts[0], "probability": top_btts[1]} if top_btts else None,
        "match_goals_all": match_goals,
        "raw_ok": True,
    }


def stars_str(n: int | None) -> str:
    if n is None:
        return "☆☆☆☆☆"
    n = max(0, min(5, int(n)))
    return "★" * n + "☆" * (5 - n)


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="KickForm Forecast Explainer (POC)", layout="wide")
st.title("KickForm Forecast Explainer (POC)")

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
        "KickForm match URL",
        value="https://www.thepunterspage.com/kickform/premier-league/fc-brighton-hove-albion-vs-fc-arsenal/NbR6R/"
    )

    # Auto-fill teams from URL, but allow override
    parsed = parse_kickform_url(match_url) if match_url else None
    default_home = parsed["home_team"] if parsed else ""
    default_away = parsed["away_team"] if parsed else ""
    default_comp = parsed["competition"] if parsed else ""

    home_team = st.text_input("Home team", value=default_home)
    away_team = st.text_input("Away team", value=default_away)
    competition = st.text_input("Competition", value=default_comp)

    st.write("---")

    use_web_search = st.checkbox(
        "Use web search for extra match context (form/news) from your allowed websites",
        value=True
    )

    allowed_domains_str = st.text_input(
        "Allowed domains for web search (comma-separated)",
        value=DEFAULT_ALLOWED_DOMAINS
    )
    allowed_domains = normalize_domains(allowed_domains_str)

    model = st.selectbox("Model", ["gpt-5-mini", "gpt-5.2", "gpt-5.2-pro"], index=0)
    max_tool_calls = st.slider("Max tool calls (limits browsing cost)", 1, 8, 3)

with col2:
    system_prompt = st.text_area("System prompt (editable)", value=DEFAULT_SYSTEM_PROMPT, height=520)

st.divider()

if st.button("Generate texts", type="primary"):
    if not match_url.strip():
        st.warning("Please paste a match URL.")
        st.stop()
    if not home_team.strip() or not away_team.strip():
        st.warning("Please ensure Home team and Away team are filled in.")
        st.stop()

    # 1) Parse forecasts from your page (cheap + reliable)
    with st.status("Step 1: Parsing forecasts from your match page…", expanded=False):
        try:
            forecast = extract_forecasts_from_match_page(match_url)
        except Exception as e:
            st.error("Failed to fetch/parse the match page.")
            st.code(str(e))
            st.stop()

    # Validate required fields
    vt = forecast.get("value_tip", {})
    top_cs = forecast.get("correct_score_top")
    top_btts = forecast.get("btts_top")
    mg = forecast.get("match_goals_all", {})

    missing = []
    if not vt.get("selection"):
        missing.append("Value Tip selection (.value-tip__message)")
    if vt.get("confidence_stars") is None:
        missing.append("Value Tip confidence stars (.value-tip__confidence-rating)")
    if not top_cs:
        missing.append("Correct Score top (.mp-correct-score-card)")
    if not top_btts:
        missing.append("BTTS top (.mp-btts-card)")
    if not mg:
        missing.append("Match Goals list (.mp-outcome-card with Over/Under 1.5/2.5/3.5)")

    if missing:
        st.error("Parsed page but some required forecast blocks were not found:")
        for m in missing:
            st.write(f"- {m}")
        with st.expander("Debug: what was parsed"):
            st.json(forecast)
        st.stop()

    # 2) Build payload to send to GPT
    # Fill Value Tip probability if missing but present in match goals
    value_tip_prob = vt.get("probability")
    if value_tip_prob is None:
        m = re.search(r"\b(Over|Under)\s*(\d+(?:\.\d+)?)\b", vt.get("selection", ""))
        if m:
            key = f"{m.group(1)} {m.group(2)}"
            value_tip_prob = mg.get(key)

    payload = {
        "match": {
            "home_team": home_team.strip(),
            "away_team": away_team.strip(),
            "competition": competition.strip(),
        },
        "value_tip": {
            "selection": vt["selection"],
            "probability": value_tip_prob,  # may still be None if not matchable
            "confidence_stars": vt["confidence_stars"],
            "confidence_stars_text": stars_str(vt["confidence_stars"]),
        },
        "markets": {
            "correct_score_top": top_cs,   # {"label":"1-2","probability":9}
            "correct_score_all": forecast.get("correct_score_all", []),
            "btts_all": forecast.get("btts_all", {}),
            "btts_top": top_btts,          # {"label":"Yes","probability":63}
            "match_goals_all": mg          # {"Over 1.5":89, ...}
        },
        "rules": {
            "no_links_no_references": True,
            "fan_friendly": True
        }
    }

    client = get_openai_client()

    # We can safely use JSON mode ONLY if we do NOT use web_search tools.
    # If web_search is enabled, we use a 2-step approach:
    #   Step 1: web_search (no JSON mode)
    #   Step 2: finish (JSON mode, no tools)
    system_prompt_runtime = (
        system_prompt.strip()
        + "\n\nIMPORTANT: Output ONLY a single JSON object and nothing else. No markdown."
    )

    result = None

    if not use_web_search:
        # One-step, JSON mode ON
        with st.status("Step 2: Writing texts (no web search)…", expanded=False):
            try:
                resp = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_prompt_runtime},
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                    text={"format": {"type": "json_object"}},
                    reasoning={"effort": "low"},
                    max_output_tokens=MAX_OUTPUT_TOKENS_FINISH,
                )
            except BadRequestError as e:
                st.error("OpenAI request failed (BadRequest).")
                st.caption(f"openai sdk version: {openai.__version__}")
                if hasattr(e, "body") and e.body:
                    st.json(e.body)
                else:
                    st.code(str(e))
                st.stop()

        raw = extract_response_text(resp)
        result = extract_json_object(raw)
        if not result:
            st.error("Could not parse JSON from the model output. Raw output below:")
            st.code(raw)
            st.stop()

    else:
        if not allowed_domains:
            st.error("Allowed domains list is empty. Add at least one domain.")
            st.stop()

        # Step 1: with web_search (NO JSON mode allowed)
        with st.status("Step 2/3: Browsing allowed football sites for context…", expanded=False):
            try:
                resp1 = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_prompt_runtime + "\n\nSEARCH BUDGET: Use minimal searching. Stop as soon as you have enough context."},
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                    tools=[
                        {
                            "type": "web_search",
                            "filters": {"allowed_domains": allowed_domains},
                            "search_context_size": "low",
                        }
                    ],
                    tool_choice="required",
                    max_tool_calls=max_tool_calls,
                    reasoning={"effort": "low"},
                    max_output_tokens=MAX_OUTPUT_TOKENS_SEARCH,
                )
            except BadRequestError as e:
                st.error("OpenAI request failed (BadRequest).")
                st.caption(f"openai sdk version: {openai.__version__}")
                if hasattr(e, "body") and e.body:
                    st.json(e.body)
                else:
                    st.code(str(e))
                st.stop()

        raw1 = extract_response_text(resp1)
        status1 = getattr(resp1, "status", "unknown")
        st.caption(f"Step 2 status: {status1}")

        # Try parse directly
        result = extract_json_object(raw1)

        # If incomplete or not parseable, do finish step
        if (not result) or (status1 == "incomplete") or (not raw1.strip()):
            with st.status("Step 3/3: Finalizing JSON output…", expanded=False):
                try:
                    resp2 = client.responses.create(
                        model=model,
                        previous_response_id=getattr(resp1, "id", None),
                        input=[
                            {"role": "system", "content": system_prompt.strip() + "\n\nIMPORTANT: Do NOT browse now. Output ONLY the final JSON object."},
                            {"role": "user", "content": "Return the final JSON now. No extra text."},
                        ],
                        text={"format": {"type": "json_object"}},
                        reasoning={"effort": "low"},
                        max_output_tokens=MAX_OUTPUT_TOKENS_FINISH,
                    )
                except BadRequestError as e:
                    st.error("OpenAI request failed (BadRequest) during finalization.")
                    st.caption(f"openai sdk version: {openai.__version__}")
                    if hasattr(e, "body") and e.body:
                        st.json(e.body)
                    else:
                        st.code(str(e))
                    st.stop()

            raw2 = extract_response_text(resp2)
            result = extract_json_object(raw2)
            if not result:
                st.error("Could not parse JSON from the model output. Raw output below:")
                st.code(raw2)
                st.stop()

    # 3) Render outputs
    st.success("Done!")

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

    with st.expander("Debug: parsed forecast payload (from your match page)"):
        st.json(payload)
