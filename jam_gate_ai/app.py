# =========================
# JAM Solver (FULLY FIXED + ENHANCED)
# =========================
import os
import re
import json
import glob
import unicodedata
import random
from typing import Dict, Any, List, Set, Tuple, Optional

import streamlit as st
import requests

# =========================
# Config
# =========================
DATA_DIR = r"data/questions_json"
SYLLABUS_MASTER_PATH = r"data/syllabus/syllabus.json"
MODEL_NAME = "deepseek-v3.1:671b-cloud"

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = (
    "You are a rigorous math tutor for JAM Mathematical Statistics.\n"
    "- Adhere to the official JAM MS syllabus whenever possible; if any off-syllabus method is used, explicitly note this in the 'Concepts:' section.\n"
    "- Never strip or alter LaTeX already present in the user prompt; integrate with it cleanly.\n"
    "- Prefer display math.\n"
    "- Output two sections in order:\n"
    "  1) Concepts: A short list of 2–6 key topics used (aligned to provided syllabus tags when given). If any step is off-syllabus, state it.\n"
    "  2) Solution: A concise yet rigorous derivation using LaTeX.\n"
    "- At the very end of the Solution, place the final option in a display box using: \\[\\boxed{\\text{final option here}}\\].\n"
)

# LaTeX Rendering
def convert_to_streamlit_latex(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'\\\[([\s\S]*?)\\\]', r'$$\1$$', text)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
    return text.strip()

def render_text_with_proper_latex(text: str):
    if not text:
        return
    parts = re.split(r'(\$\$.*?\$\$)', text, flags=re.DOTALL)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith('$$') and part.endswith('$$'):
            st.latex(part[2:-2])
        else:
            st.markdown(part, unsafe_allow_html=True)

# Data Loaders
@st.cache_data(show_spinner=False)
def load_json_files(folder: str, year: int) -> Dict[int, Dict[str, Any]]:
    questions = {}
    pattern = os.path.join(folder, f"*{year}*.json")
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("questions", [])
            for entry in items:
                qno = entry.get("question_number") or entry.get("qno") or entry.get("questionNo")
                if qno is not None:
                    questions[int(qno)] = entry
        except Exception as e:
            st.warning(f"Failed to load {filepath}: {e}")
    return questions

@st.cache_data(show_spinner="Loading syllabus...")
def load_master_syllabus_once(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Invalid syllabus.json: {e}")
        return None

# Syllabus Concept Detection
def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"[{}_^$#%~\\]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_concept_index(syllabus: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    alias_to_tag: Dict[str, str] = {}
    tag_to_labels: Dict[str, Set[str]] = {}

    for sec in syllabus.get("sections", []):
        sec_title = (sec.get("title") or "").strip()
        topics = sec.get("topics") or []
        tags = sec.get("canonical_tags") or []
        for t in tags:
            tag_to_labels.setdefault(t, set())
        for topic in topics:
            topic_clean = re.sub(r"[\(\)–—,:;/]+", " ", topic)
            tokens = [w.strip().lower() for w in topic_clean.split() if len(w) >= 3]
            for tag in tags:
                tag_to_labels[tag].add(f"{sec_title} • {topic}")
                for tok in tokens:
                    alias_to_tag[tok] = tag

    for canon, aliases in (syllabus.get("topic_aliases") or {}).items():
        canon_norm = _norm_text(canon)
        target_tag = next((t for t in syllabus.get("canonical_tags_index", []) if canon_norm.replace(" ", "-") in t.replace(" ", "-")), None)
        if target_tag:
            alias_to_tag[canon_norm] = target_tag
            for a in aliases:
                alias_to_tag[_norm_text(a)] = target_tag

    return alias_to_tag, tag_to_labels

def extract_concepts_from_question(
    question: Dict[str, Any],
    alias_to_tag: Dict[str, str],
    tag_to_labels: Dict[str, Set[str]],
    max_tags: int = 8
) -> Tuple[List[str], List[str]]:
    parts = [question.get("question_text", "")]
    opts = question.get("options") or {}
    for k in sorted(opts.keys()):
        parts.append(str(opts[k]))
    corpus = " \n ".join(parts)
    norm = _norm_text(corpus)

    found: List[str] = []
    seen: Set[str] = set()

    for word in set(norm.split()):
        tag = alias_to_tag.get(word)
        if tag and tag not in seen:
            seen.add(tag)
            found.append(tag)
            if len(found) >= max_tags:
                break

    if len(found) < max_tags:
        for alias, tag in alias_to_tag.items():
            if " " in alias and alias in norm and tag not in seen:
                seen.add(tag)
                found.append(tag)
                if len(found) >= max_tags:
                    break

    pretty: List[str] = []
    for tag in found[:max_tags]:
        labels = list(tag_to_labels.get(tag, []))
        pretty.append(f"{tag} — {labels[0]}" if labels else tag)

    return found[:max_tags], pretty

# UI Helpers
def concept_chips(tags: List[str]):
    if not tags:
        return
    colors = ["#e0f2fe", "#fef3c7", "#f0e6ff", "#d1fae5", "#fce7f3", "#fee2e2"]
    html = '<div style="display:flex; flex-wrap:wrap; gap:10px; margin:12px 0;">'
    for i, tag in enumerate(tags):
        color = colors[i % len(colors)]
        html += f'<span style="background:{color}; color:#0f172a; border:1px solid #cbd5e1; border-radius:16px; padding:6px 12px; font-size:0.92rem; font-weight:500;">{tag}</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# Model I/O
def call_ollama(messages: List[Dict[str, str]], model: str, temperature: float) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature
    }
    try:
        r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"Error: {str(e)}"

def parse_concepts_and_solution(resp: str) -> Tuple[str, str]:
    resp = re.sub(r'\r', '', resp)
    patterns = [
        r'(Concepts\s*[:\-]\s*)(.*?)(\n\s*Solution\s*[:\-])',
        r'(Concepts\s*Used\s*[:\-]\s*)(.*?)(\n\s*Solution\s*[:\-])',
    ]
    for pat in patterns:
        m = re.search(pat, resp, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(2).strip(), resp[m.end(3):].strip()
    return "", resp.strip()

def build_user_prompt_with_concepts(question: Dict[str, Any], section_tag: str) -> str:
    base = ["Problem:", question.get("question_text", "")]
    opts = question.get("options") or {}
    if opts:
        base.append("Options:")
        for k in sorted(opts.keys()):
            base.append(f"{k}) {opts[k]}")
    if section_tag:
        base.append(f"Syllabus section: {section_tag}")
    base.extend([
        "First, output 'Concepts:' with 2-6 key topics.",
        "Then 'Solution:' with full derivation.",
        "End with: \\[\\boxed{\\text{answer}}\\]"
    ])
    return "\n\n".join(base)

# SYLLABUS DISPLAY
def display_syllabus(syllabus_data: Dict[str, Any]):
    st.header(f"{syllabus_data.get('exam', 'IIT JAM Mathematical Statistics')} Syllabus")
    
    sections = syllabus_data.get("sections", [])
    if not sections:
        st.warning("No sections found.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sections", len(sections))
    with col2:
        total = sum(len(s.get("topics", [])) for s in sections)
        st.metric("Topics", total)
    
    st.markdown("---")
    search = st.text_input("Search topics", placeholder="e.g. Markov Chain")
    
    for sec in sections:
        title = sec.get("title", "Untitled")
        topics = sec.get("topics", [])
        
        if search:
            filtered = [t for t in topics if search.lower() in t.lower()]
            if not filtered:
                continue
        else:
            filtered = topics
        
        with st.expander(f"{title} ({len(filtered)} topics)", expanded=False):
            for i, topic in enumerate(filtered, 1):
                st.markdown(f"**{i}.** {topic}")

# APP SETUP
st.set_page_config(page_title="JAM Solver", page_icon="🧮", layout="wide")
st.title("🧮 JAM Mathematical Statistics Solver")

syllabus_master = load_master_syllabus_once(SYLLABUS_MASTER_PATH)
if not syllabus_master:
    st.error("syllabus.json NOT FOUND!")
    st.info(f"Expected at:\n`{SYLLABUS_MASTER_PATH}`")
    st.code("Create the file or fix the path.", language="bash")
    st.stop()

alias_to_tag, tag_to_labels = build_concept_index(syllabus_master)

# TABS (Added third tab for Section questions)
tab1, tab2, tab3 = st.tabs(["🧮 Question Solver", "📚 Syllabus", "📂 Section Questions"])

with tab2:
    display_syllabus(syllabus_master)
    with st.expander("View Raw JSON"):
        st.json(syllabus_master)


with tab3:
    st.header("Practice Questions by Syllabus Section")
    
    section_titles = ["— Select Section —"] + [section.get("title", "Untitled") for section in syllabus_master.get("sections", [])]
    selected_section = st.selectbox(
        "Select Section", 
        section_titles, 
        index=0,  # ← keeps it unselected by default
        key="sec_select_tab3"
    )
    
    # NEW: Question type filter
    qtype_options = ["All Types", "MCQ", "NAT", "MSQ"]
    selected_qtype = st.selectbox("Question Type", qtype_options, key="qtype_tab3")
    
    if selected_section and selected_section != "— Select Section —":
        all_questions = []
        years = sorted({int(re.search(r"(19|20)\d{2}", f).group()) for f in os.listdir(DATA_DIR) if re.search(r"(19|20)\d{2}", f)})
        for year in years:
            all_questions.extend(load_json_files(DATA_DIR, year).values())
        
        filtered_questions = [q for q in all_questions if q.get("syllabus_section") == selected_section]
        
        # Use "type" field (correct for your JSON)
        if selected_qtype != "All Types":
            filtered_questions = [q for q in filtered_questions if str(q.get("type", "")).strip().upper() == selected_qtype]
        
        if not filtered_questions:
            st.info(f"No {selected_qtype if selected_qtype != 'All Types' else ''} questions found for section: **{selected_section}**")
        else:
            sample_size = min(5, len(filtered_questions))
            sampled_questions = random.sample(filtered_questions, sample_size)
            
            for q in sampled_questions:
                st.markdown(f"### Question {q.get('question_number', 'N/A')}  (Year: {q.get('year', 'Unknown')})")
                st.markdown(convert_to_streamlit_latex(q.get("question_text", "")))
                
                opts = q.get("options")
                if opts:
                    for k in sorted(opts.keys()):
                        st.markdown(f"**{k})** {convert_to_streamlit_latex(opts[k])}")

                if q.get("solution"):
                    with st.expander("Click here for solution"):
                        sol = convert_to_streamlit_latex(q["solution"])
                        st.markdown(sol)

                # Solve with AI — NO CONCEPTS
                solve_container = st.container()
                with solve_container:
                    if st.button("Solve with AI", key=f"ai_{q.get('year')}_{q.get('question_number')}"):
                        with st.spinner("Solving..."):
                            user_prompt = build_user_prompt_with_concepts(q, selected_section)
                            messages = [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt}
                            ]
                            resp = call_ollama(messages, MODEL_NAME, temperature=0.1)

                        _, solution_text = parse_concepts_and_solution(resp)
                        st.markdown("### AI Solution")
                        st.markdown(convert_to_streamlit_latex(solution_text))

                st.markdown("---")
    else:
        st.info("Please select a syllabus section to begin practice.")


with tab1:
    st.caption("Powered by DeepSeek via Ollama")

    years = sorted({
        int(m.group()) for filename in os.listdir(DATA_DIR)
        for m in [re.search(r"(19|20)\d{2}", filename)] if m
    })

    if not years:
        st.error("No question files found!")
        st.stop()

    selected_year = st.selectbox("📅 Year", years, index=None, placeholder="Select year")
    if not selected_year:
        st.info("Select a year to begin.")
        st.stop()

    questions = load_json_files(DATA_DIR, selected_year)
    qnos = sorted(questions.keys())
    if not qnos:
        st.error(f"No questions for {selected_year}")
        st.stop()

    selected_qno = st.selectbox("❓ Question", qnos, index=None, placeholder="Select question")
    if not selected_qno:
        st.info("Select a question.")
        st.stop()

    question = questions[selected_qno]

    st.markdown("### Question")
    st.markdown(convert_to_streamlit_latex(question.get("question_text", "")))

    if question.get("options"):
        st.markdown("### Options")
        options = question["options"]
        for key in sorted(options.keys()):
            option_text = convert_to_streamlit_latex(options[key])
            st.markdown(f"**{key})** {option_text}")

    st.markdown("---")

    question_section = question.get("syllabus_section", "")

    if st.button("✨ Solve", type="primary", use_container_width=True):
        user_prompt = build_user_prompt_with_concepts(question, question_section)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        with st.spinner("Solving..."):
            resp = call_ollama(messages, MODEL_NAME, temperature=0.1)

        concepts_md, solution_text = parse_concepts_and_solution(resp)

        if question_section:
            st.markdown("### 📌 Syllabus Section")
            concept_chips([question_section])

        if concepts_md:
            st.markdown("### 🎯 Model Concepts")
            st.markdown(concepts_md)

        st.subheader("💡 Solution")
        clean_sol = convert_to_streamlit_latex(solution_text)
        st.markdown(clean_sol)

    correct = question.get("answer") or question.get("correct_option") or question.get("correct_answer")
    if correct and st.session_state.get("resp"):
        m = re.search(r'\\boxed\{(.*?)\}', st.session_state.resp)
        if m:
            ans = m.group(1).strip()
            if ans == str(correct).strip():
                st.success(f"✅ Correct Answer: {correct}")
            else:
                st.error(f"❌ Model: {ans} | Correct: {correct}")

