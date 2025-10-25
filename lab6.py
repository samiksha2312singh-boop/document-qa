import streamlit as st
from openai import OpenAI
import json

# ==============================
# Lab 6a: Streamlit App Setup
# ==============================
st.set_page_config(page_title="AI Fact-Checker + Citation Builder", page_icon="🧠")
st.title("🧠 AI Fact-Checker + Citation Builder")
st.write("Enter a factual claim and let the model verify it using live web sources.")

# Initialize OpenAI client (requires st.secrets["OPENAI_API_KEY"])
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Keep a session-state history
if "fact_history" not in st.session_state:
    st.session_state.fact_history = []


# ==============================
# Lab 6b: Core Function
# ==============================
def fact_check_claim(user_claim: str):
    """
    Accepts a user claim, queries the OpenAI Responses API using
    the web_search tool, and returns structured text or JSON-like output.
    (Compatible with SDK versions that do NOT support 'format' argument)
    """
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "system",
                "content": """You are a factual verification assistant.
For any given claim, search the web for credible sources and return
a structured JSON-like output with the following fields:

- claim
- verdict: True / False / Partly True
- explanation
- sources (list of URLs)
- confidence (0–1 based on source consistency)"""
            },
            {"role": "user", "content": user_claim}
        ],
        tools=[{"type": "web_search"}]
    )

    # --- Safe fallback for response parsing across SDK versions ---
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    elif hasattr(response, "output") and len(response.output) > 0:
        if hasattr(response.output[0], "content") and len(response.output[0].content) > 0:
            return response.output[0].content[0].get("text", "")
    return "Error: No valid response text found."


# ==============================
# Lab 6c: Streamlit Layout
# ==============================
user_claim = st.text_input("Enter a factual claim:", placeholder="e.g., Is dark chocolate healthy?")
if st.button("Check Fact"):
    if user_claim.strip() == "":
        st.warning("Please enter a claim to verify.")
    else:
        with st.spinner("Verifying claim..."):
            try:
                result_text = fact_check_claim(user_claim)
                st.subheader("🔍 Fact-Check Result (Raw Output)")
                st.text(result_text)

                # Attempt to parse JSON if possible
                try:
                    parsed = json.loads(result_text)
                    st.json(parsed)
                except Exception:
                    parsed = {}

                # 6d: Enhancements — clickable links + confidence + history
                if "confidence" in parsed:
                    conf = float(parsed.get("confidence", 0))
                    st.progress(conf)
                    st.caption(f"Confidence: {conf:.2f}")

                if "sources" in parsed:
                    st.markdown("**Sources:**")
                    for s in parsed["sources"]:
                        st.markdown(f"- [{s}]({s})")

                # Save to session history
                st.session_state.fact_history.append(
                    parsed if parsed else {"claim": user_claim, "raw": result_text}
                )

            except Exception as e:
                st.error(f"Error verifying claim: {e}")


# ==============================
# Lab 6d: History Display
# ==============================
if st.session_state.fact_history:
    st.divider()
    st.subheader("🕒 Session History")
    for i, item in enumerate(reversed(st.session_state.fact_history), 1):
        st.markdown(f"**{i}. {item.get('claim','(unknown claim)')}**")
        if "verdict" in item:
            st.write(f"Verdict: {item.get('verdict','?')}")
        if "explanation" in item:
            st.caption(item["explanation"])
        if "sources" in item:
            for s in item["sources"]:
                st.markdown(f" - [{s}]({s})")


# ==============================
# Lab 6e: Reflection & Discussion
# ==============================
with st.expander("💬 Reflection + Discussion"):
    st.markdown("""
**Q1 — How does the model’s reasoning differ from a standard chat model?**  
> The Responses API enables structured reasoning with web search and JSON-like outputs, focusing on factual verification instead of open-ended conversation.

**Q2 — Were the sources credible and diverse?**  
> Most responses reference reputable domains like BBC, WHO, or NASA.  
> However, source quality may vary depending on the query.

**Q3 — How does tool integration enhance trust and accuracy?**  
> Live web searches ground responses in verifiable data, improving transparency and allowing users to check cited evidence directly.
""")
