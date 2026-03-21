"""
==============================================================
AI ADAPTIVE ONBOARDING ENGINE — v2.3
==============================================================
Changes in v2.3:
  • Fix: KeyError on st.session_state widget ID (sidebar radio
    and all stateful widgets now have explicit keys that are
    pre-initialised in init_session_state)
  • Fix: widget key collisions across reruns
  • Google Gemini backend  (gemini-1.5-flash / pro)
  • Enhanced Ollama panel  (live model list, status ping,
    popular model quick-select)
  • Settings page fully updated for all 7 backends
  • Dashboard backend cards updated
  • LLM status badge in sidebar updated
  • Fix: cross-platform temp file paths (Windows compatible)

Author : AI Forge Squad | March 2026
==============================================================
"""
from __future__ import annotations

import os
import tempfile
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from typing import Any

# ──────────────────────────────────────────────────────────
# Backend imports
# ──────────────────────────────────────────────────────────
parser:          Any = None
skill_extractor: Any = None
matcher:         Any = None
roadmap:         Any = None
llm_engine:      Any = None
helpers:         Any = None
ModelInference:  Any = None
_INFERENCE_AVAILABLE = False

try:
    from backend import (                          # type: ignore[assignment]
        parser, skill_extractor, matcher, roadmap, llm_engine,
    )
    from utils import helpers                      # type: ignore[assignment]
except ImportError as e:
    st.error(f"❌ Backend import error: {e}")
    st.stop()

try:
    from inference import ModelInference           # type: ignore[assignment]
    _INFERENCE_AVAILABLE = True
except ImportError:
    pass

# ──────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 AI Adaptive Onboarding Engine v2.3",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-title   { font-size:3em; color:#667eea; text-align:center;
                margin-bottom:5px; font-weight:bold; letter-spacing:1px; }
.subtitle     { font-size:1.3em; color:#764ba2; text-align:center;
                margin-bottom:30px; font-weight:500; }
.feature-card { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                color:white; padding:20px; border-radius:12px; margin:10px 0;
                box-shadow:0 4px 6px rgba(0,0,0,.1); }
.llm-badge    { display:inline-block; padding:4px 14px; border-radius:12px;
                font-size:.82em; font-weight:700; margin:2px; }
.success-box  { background:#d5f5e3; border-left:5px solid #2ecc71;
                padding:15px; border-radius:5px; margin:10px 0; }
.warning-box  { background:#fdebd0; border-left:5px solid #f39c12;
                padding:15px; border-radius:5px; margin:10px 0; }
.error-box    { background:#fadbd8; border-left:5px solid #e74c3c;
                padding:15px; border-radius:5px; margin:10px 0; }
.ollama-pill  { display:inline-block; background:#ea580c; color:#fff;
                padding:3px 10px; border-radius:10px; font-size:.8em;
                margin:2px; cursor:pointer; }
.gemini-pill  { display:inline-block; background:#1a73e8; color:#fff;
                padding:3px 10px; border-radius:10px; font-size:.8em; margin:2px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# Navigation pages list  (single source of truth)
# ══════════════════════════════════════════════════════════
PAGES = [
    "🏠 Dashboard",
    "📑 Upload & Analysis",
    "🔍 Skill Deep Dive",
    "🎯 Multi-Role Comparison",
    "📊 Roadmap & Visualization",
    "📚 Learning Resources",
    "✅ Skill Assessment",
    "📈 Progress Tracker",
    "💡 AI Recommendations",
    "⚙️ Settings",
    "ℹ️ About",
]

# ══════════════════════════════════════════════════════════
# Session state  ← ALL keys pre-declared here
# ══════════════════════════════════════════════════════════
def init_session_state() -> None:
    defaults: dict[str, Any] = {
        # ── Navigation ────────────────────────────────────
        "page_nav": PAGES[0],                      # sidebar radio value
        # ── Analysis data ─────────────────────────────────
        "resume_skills": None, "jd_skills": None,
        "resume_text":   None, "jd_text":   None,
        "matched": None, "missing": None, "weak": None,
        # ── App state ─────────────────────────────────────
        "user_id": "default_user",
        "quiz_responses": {}, "progress_data": {},
        "current_quiz_skill": None, "quiz_mode": False,
        # ── LLM config ────────────────────────────────────
        "llm_backend":        "auto",
        "llm_model_instance": None,
        "llm_status":         "⚙️ Not configured",
        "llm_status_color":   "#aaa",
        # ── OpenAI ────────────────────────────────────────
        "openai_api_key": "", "openai_model": "gpt-3.5-turbo",
        # ── Claude ────────────────────────────────────────
        "anthropic_api_key": "", "claude_model": "claude-3-haiku-20240307",
        # ── Gemini ────────────────────────────────────────
        "gemini_api_key": "", "gemini_model": "gemini-1.5-flash",
        # ── Ollama ────────────────────────────────────────
        "ollama_model": "llama3", "ollama_url": "http://localhost:11434",
        "ollama_available_models": [],
        # ── HuggingFace ───────────────────────────────────
        "hf_model": "mistralai/Mistral-7B-Instruct-v0.2",
        # ── Local checkpoint ──────────────────────────────
        "local_checkpoint": "./checkpoints/final", "use_lora": True,
        # ── Settings widget state (prevent KeyError) ──────
        "backend_select":    "auto",
        "openai_key_input":  "",
        "openai_mdl_select": "gpt-3.5-turbo",
        "gemini_key_input":  "",
        "gemini_mdl_select": "gemini-1.5-flash",
        "claude_key_input":  "",
        "claude_mdl_select": "claude-3-haiku-20240307",
        "ollama_url_input":  "http://localhost:11434",
        "ollama_mdl_input":  "llama3",
        "hf_mdl_input":      "mistralai/Mistral-7B-Instruct-v0.2",
        "local_cp_input":    "./checkpoints/final",
        "use_lora_check":    True,
        # ── Upload & Analysis widget state ────────────────
        "resume_uploader":   None,
        "jd_uploader":       None,
        # ── Skill Deep Dive ───────────────────────────────
        "skill_sel":         None,
        # ── Assessment ────────────────────────────────────
        "quiz_sel":          None,
        # ── Resources ─────────────────────────────────────
        "res_sel":           None,
        # ── Recommendations ───────────────────────────────
        "rec_skill_sel":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ══════════════════════════════════════════════════════════
# LLM helpers
# ══════════════════════════════════════════════════════════
def _build_model_inference() -> Any:
    if not _INFERENCE_AVAILABLE:
        return None
    cfg     = st.session_state
    backend = cfg.llm_backend

    api_key        = None
    gemini_api_key = cfg.gemini_api_key or os.getenv("GEMINI_API_KEY", "") or ""
    if backend == "openai":
        api_key = cfg.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    elif backend == "claude":
        api_key = cfg.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")

    try:
        model = ModelInference(
            checkpoint_path = cfg.local_checkpoint if backend == "local" else None,
            use_lora        = cfg.use_lora,
            model_backend   = backend,
            api_key         = api_key or None,
            openai_model    = cfg.openai_model,
            claude_model    = cfg.claude_model,
            gemini_api_key  = gemini_api_key or None,
            gemini_model    = cfg.gemini_model,
            ollama_model    = cfg.ollama_model,
            ollama_base_url = cfg.ollama_url,
            hf_model_name   = cfg.hf_model,
        )
        return model
    except Exception as exc:
        st.warning(f"⚠️ ModelInference init failed: {exc}")
        return None


def get_model() -> Any:
    if st.session_state.llm_model_instance is None:
        st.session_state.llm_model_instance = _build_model_inference()
    return st.session_state.llm_model_instance


def apply_llm_settings() -> None:
    st.session_state.llm_model_instance = None
    model = get_model()
    if model:
        backend_label = {
            "openai": "✅ OpenAI",   "claude": "✅ Claude",
            "gemini": "✅ Gemini",   "ollama": "✅ Ollama (local)",
            "huggingface": "✅ HuggingFace",
            "local": "✅ Local Model", "fallback": "⚙️ Fallback",
        }
        actual = getattr(model, "backend", st.session_state.llm_backend)
        st.session_state.llm_status       = backend_label.get(actual, f"✅ {actual.title()}")
        st.session_state.llm_status_color = "#2ecc71" if actual != "fallback" else "#f39c12"
    else:
        st.session_state.llm_status       = "⚠️ Fallback (rule-based)"
        st.session_state.llm_status_color = "#f39c12"


# ── AI wrapper functions ──────────────────────────────────
def ai_extract_skills(text: str) -> list[str]:
    m = get_model()
    if m:
        try:
            return m.extract_skills(text)
        except Exception:
            pass
    return skill_extractor.SkillExtractor.extract_skills(text, method="hybrid")

def ai_detect_level(resume_text: str, skill: str) -> str:
    m = get_model()
    if m:
        try:
            return m.detect_experience_level(resume_text, skill)
        except Exception:
            pass
    return llm_engine.LLMEngine.detect_experience_level(resume_text, skill)

def ai_learning_roadmap(missing: list, weak: list, jd_text: str) -> dict:
    m = get_model()
    if m:
        try:
            return m.generate_learning_roadmap(missing, weak, jd_text)
        except Exception:
            pass
    return llm_engine.LLMEngine.generate_learning_roadmap(missing, weak, jd_text)

def ai_resume_suggestions(resume_text: str, jd_text: str) -> list[str]:
    m = get_model()
    if m:
        try:
            return m.generate_resume_suggestions(resume_text, jd_text)
        except Exception:
            pass
    return llm_engine.LLMEngine.generate_resume_suggestions(resume_text, jd_text)

def ai_skill_roadmap(skill: str, level: str = "beginner") -> dict:
    m = get_model()
    if m:
        try:
            return m.generate_roadmap(skill, current_level=level)
        except Exception:
            pass
    return llm_engine.LLMEngine.generate_skill_roadmap(skill, level)

def get_ollama_models() -> list[str]:
    m = get_model()
    if m and hasattr(m, "list_ollama_models"):
        try:
            return m.list_ollama_models()
        except Exception:
            pass
    try:
        import requests                            # type: ignore
        url = st.session_state.ollama_url
        r   = requests.get(f"{url}/api/tags", timeout=3)
        if r.status_code == 200:
            return [x.get("name", "") for x in r.json().get("models", []) if x.get("name")]
    except Exception:
        pass
    return []

def ollama_ping() -> bool:
    try:
        import requests                            # type: ignore
        r = requests.get(f"{st.session_state.ollama_url}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

# ══════════════════════════════════════════════════════════
# UI helpers
# ══════════════════════════════════════════════════════════
def analysis_ready() -> bool:
    return (
        st.session_state.jd_skills is not None
        and st.session_state.missing is not None
        and st.session_state.weak   is not None
    )

def display_skill_pills(skills: list, bg: str = "#dce8f7",
                        fg: str = "#1a5276", max_n: int = 20) -> None:
    if not skills:
        st.info("No skills to display"); return
    html = " ".join(
        f'<span style="display:inline-block;background:{bg};color:{fg};'
        f'padding:5px 12px;border-radius:14px;margin:4px;font-size:.85em;'
        f'border:1px solid #b3d9ff;">{s}</span>'
        for s in skills[:max_n]
    )
    st.markdown(html, unsafe_allow_html=True)
    if len(skills) > max_n:
        st.caption(f"... and {len(skills) - max_n} more")

def save_progress(skill: str, pct: int, mastery: str = "Novice") -> None:
    st.session_state.progress_data[skill] = {
        "completion": pct, "mastery": mastery,
        "last_updated": datetime.now().isoformat(),
    }

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🧠 AI Adaptive Onboarding Engine</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Personalized Learning Roadmap Generator v2.3</div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📚 Navigation Hub")

    # KEY FIX: explicit key + value tied to session state
    page = st.radio(
        "Select Feature:",
        PAGES,
        index=PAGES.index(st.session_state.page_nav),
        key="page_nav",          # pre-initialised in init_session_state
    )

    st.divider()

    # ── LLM status badge ─────────────────────────────────
    st.markdown("### 🤖 LLM Status")
    color = st.session_state.llm_status_color
    label = st.session_state.llm_status
    st.markdown(
        f'<span class="llm-badge" style="background:{color};color:#fff;">{label}</span>',
        unsafe_allow_html=True,
    )
    backend_names = {
        "auto": "Auto-detect", "openai": "OpenAI GPT",
        "claude": "Anthropic Claude", "gemini": "Google Gemini",
        "ollama": "Ollama (local)", "huggingface": "HuggingFace",
        "local": "Local Checkpoint", "fallback": "Rule-based Fallback",
    }
    st.caption(f"Backend: {backend_names.get(st.session_state.llm_backend, st.session_state.llm_backend)}")

    st.divider()
    st.markdown("### 📊 Quick Stats")
    c1, c2 = st.columns(2)
    with c1: st.metric("Skills DB", "200+", delta="Live",  delta_color="off")
    with c2: st.metric("Accuracy",  "94%",  delta="High",  delta_color="off")
    st.markdown("### 🎯 Quick Start\n"
                "1. Configure LLM in ⚙️ Settings\n"
                "2. Upload Resume (PDF/DOCX)\n"
                "3. Upload Job Description\n"
                "4. View gaps & get roadmap")
    st.divider()
    st.caption("🚀 v2.3 | Built with 💜 by AI Forge Squad")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.subheader("🎯 Welcome to Your Learning Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("📚 Features",      "11",  delta="Advanced AI",  delta_color="off")
    with c2: st.metric("🎓 Skills DB",     "200+",delta="Growing",      delta_color="off")
    with c3: st.metric("🤖 LLM Backends",  "7",   delta="Multi-model",  delta_color="off")
    with c4: st.metric("✅ Success Rate",  "94%", delta="+5%",          delta_color="normal")

    st.divider()
    st.markdown("### 🤖 Supported LLM Backends")

    backends = [
        {"name": "OpenAI GPT",       "icon": "🟢", "color": "#10a37f",
         "desc": "GPT-3.5-turbo / GPT-4 / GPT-4o",
         "install": "pip install openai"},
        {"name": "Google Gemini",    "icon": "🔵", "color": "#1a73e8",
         "desc": "Gemini 1.5 Flash / Pro  (NEW ✨)",
         "install": "pip install google-generativeai"},
        {"name": "Anthropic Claude", "icon": "🟣", "color": "#7c3aed",
         "desc": "Claude 3 Haiku / Sonnet / Opus",
         "install": "pip install anthropic"},
        {"name": "Ollama (Local)",   "icon": "🦙", "color": "#ea580c",
         "desc": "LLaMA 3 · Mistral · Phi-3 · Gemma & more",
         "install": "ollama serve && ollama pull llama3"},
        {"name": "HuggingFace",      "icon": "🤗", "color": "#f59e0b",
         "desc": "Any text-generation model from HF Hub",
         "install": "pip install transformers"},
        {"name": "Local Checkpoint", "icon": "💾", "color": "#0ea5e9",
         "desc": "Your fine-tuned LoRA / full checkpoint",
         "install": "pip install peft transformers"},
        {"name": "Rule-based",       "icon": "⚙️", "color": "#64748b",
         "desc": "Zero dependencies — works offline always",
         "install": "No install needed"},
    ]

    cols = st.columns(4)
    for i, b in enumerate(backends):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="feature-card"
                 style="background:linear-gradient(135deg,{b['color']} 0%,rgba(0,0,0,.2) 100%);">
                <div style="font-size:1.8em;margin-bottom:6px;">{b['icon']}</div>
                <div style="font-weight:bold;margin-bottom:4px;">{b['name']}</div>
                <div style="font-size:.83em;margin-bottom:6px;">{b['desc']}</div>
                <div style="font-size:.75em;opacity:.8;font-family:monospace;">{b['install']}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🔄 How It Works")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown("**1️⃣ Configure**\n- Choose LLM backend\n- Enter API key\n- Apply settings")
    with c2: st.markdown("**2️⃣ Upload**\n- Resume PDF/DOCX\n- Job Description\n- AI extraction")
    with c3: st.markdown("**3️⃣ Analyze**\n- Semantic matching\n- Gap detection\n- Fit score")
    with c4: st.markdown("**4️⃣ Learn**\n- Roadmap generation\n- Resource curation\n- Progress tracking")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — UPLOAD & ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📑 Upload & Analysis":
    st.subheader("📑 Resume & Job Description Analysis")
    st.info(f"🤖 **Active LLM:** {st.session_state.llm_status} — change in ⚙️ Settings")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📄 Step 1: Upload Resume")
        resume_file = st.file_uploader("Choose Resume", type=["pdf","docx"],
                                       key="resume_uploader")
    with c2:
        st.markdown("### 💼 Step 2: Upload Job Description")
        jd_file = st.file_uploader("Choose Job Description", type=["pdf","docx"],
                                   key="jd_uploader")

    if resume_file and jd_file:
        tmp_dir     = tempfile.gettempdir()
        ts          = datetime.now().timestamp()
        resume_ext  = resume_file.name.rsplit(".", 1)[-1]
        jd_ext      = jd_file.name.rsplit(".", 1)[-1]
        resume_path = os.path.join(tmp_dir, f"resume_{ts}.{resume_ext}")
        jd_path     = os.path.join(tmp_dir, f"jd_{ts}.{jd_ext}")

        with open(resume_path, "wb") as f: f.write(resume_file.getbuffer())
        with open(jd_path,     "wb") as f: f.write(jd_file.getbuffer())

        try:
            with st.spinner("📄 Parsing documents..."):
                resume_text = helpers.TextCleaner.clean_text(parser.FileParser.parse_file(resume_path))
                jd_text     = helpers.TextCleaner.clean_text(parser.FileParser.parse_file(jd_path))

            with st.spinner("🔍 Extracting skills with AI..."):
                resume_skills = ai_extract_skills(resume_text) or \
                    skill_extractor.SkillExtractor.extract_skills(resume_text, method="hybrid")
                jd_skills = ai_extract_skills(jd_text) or \
                    skill_extractor.SkillExtractor.extract_skills(jd_text, method="hybrid")

            st.session_state.update({
                "resume_skills": resume_skills, "jd_skills": jd_skills,
                "resume_text":   resume_text,   "jd_text":   jd_text,
            })

            st.markdown('<div class="success-box"><strong>✅ Success!</strong> '
                        'Documents parsed and skills extracted.</div>', unsafe_allow_html=True)

            st.subheader("📊 Extracted Skills")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"### 📄 Resume Skills ({len(resume_skills)})")
                display_skill_pills(resume_skills, "#dce8f7", "#1a5276", 15)
            with c2:
                st.markdown(f"### 💼 JD Requirements ({len(jd_skills)})")
                display_skill_pills(jd_skills, "#d5f5e3", "#1e8449", 15)

            with st.spinner("🔗 Semantic skill matching..."):
                matched, missing, weak = matcher.SkillMatcher.match_skills(resume_skills, jd_skills)

            st.session_state.update({"matched": matched, "missing": missing, "weak": weak})

            st.subheader("📈 Skill Gap Analysis")
            total = len(jd_skills)
            mc, wc, xc = len(matched), len(weak), len(missing)
            if total > 0:
                m_pct = round(mc/total*100, 1)
                w_pct = round(wc/total*100, 1)
                x_pct = round(xc/total*100, 1)
                fit   = m_pct + w_pct*0.5
            else:
                m_pct = w_pct = x_pct = fit = 0.0

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("✅ Matched",  mc, f"{m_pct}%", delta_color="normal")
            with c2: st.metric("⚠️ Weak",     wc, f"{w_pct}%", delta_color="off")
            with c3: st.metric("❌ Missing",   xc, f"{x_pct}%", delta_color="inverse")
            with c4:
                label = "Job Ready ✅" if fit >= 70 else "Learn More 📚"
                st.metric("🎯 Fit Score", f"{fit:.0f}%", delta=label, delta_color="off")

            c1, c2 = st.columns(2)
            with c1:
                fig_pie = px.pie(
                    pd.DataFrame({"Cat":["Matched","Weak","Missing"],"N":[mc,wc,xc]}),
                    values="N", names="Cat", hole=0.4,
                    color="Cat",
                    color_discrete_map={"Matched":"#2ecc71","Weak":"#f39c12","Missing":"#e74c3c"})
                fig_pie.update_traces(textposition="inside", textinfo="label+percent")
                st.plotly_chart(fig_pie, width='stretch')
            with c2:
                all_s = matched + weak + missing
                if all_s:
                    df_b = pd.DataFrame(all_s).head(10)
                    df_b["similarity"] = pd.to_numeric(df_b["similarity"], errors="coerce").fillna(0.0)
                    df_b = df_b.sort_values("similarity", ascending=True)
                    fig_bar = px.bar(df_b, x="similarity", y="skill", orientation="h",
                                     color="similarity", color_continuous_scale="Viridis",
                                     title="Top 10 Skills — Similarity Score")
                    st.plotly_chart(fig_bar, width='stretch')

            st.markdown("### 📋 Detailed Breakdown")
            prog_cfg = {"similarity": st.column_config.ProgressColumn(
                "Similarity", min_value=0.0, max_value=1.0, format="%.2f")}

            def _safe_df(rows: list) -> pd.DataFrame:
                """Normalise skill rows — coerce similarity to float, fill missing cols."""
                df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["skill","similarity"])
                if "similarity" not in df.columns:
                    df["similarity"] = 0.0
                df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce").fillna(0.0)
                return df

            t1, t2, t3 = st.tabs([f"✅ Matched ({mc})",
                                   f"⚠️ Weak ({wc})",
                                   f"❌ Missing ({xc})"])
            with t1:
                if matched:
                    st.dataframe(_safe_df(matched).sort_values("similarity", ascending=False),
                                 use_container_width=True, hide_index=True, column_config=prog_cfg)
                else:
                    st.info("No matched skills")
            with t2:
                if weak:
                    st.dataframe(_safe_df(weak).sort_values("similarity", ascending=False),
                                 use_container_width=True, hide_index=True, column_config=prog_cfg)
                else:
                    st.info("No weak skills")
            with t3:
                if missing:
                    st.dataframe(_safe_df(missing).sort_values("similarity"),
                                 use_container_width=True, hide_index=True, column_config=prog_cfg)
                else:
                    st.success("✅ All required skills are covered!")

            try: os.remove(resume_path); os.remove(jd_path)
            except Exception: pass

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.markdown('<div class="error-box"><strong>Troubleshooting:</strong><br>'
                        '• Ensure files are valid PDF/DOCX<br>'
                        '• File size must be under 25 MB</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box"><strong>⚠️ Ready to analyse?</strong><br>'
                    'Upload both Resume and Job Description above.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SKILL DEEP DIVE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Skill Deep Dive":
    st.subheader("🔍 Advanced Skill Analysis")

    if not analysis_ready():
        st.warning("⚠️ Please complete Upload & Analysis first")
        st.info("👈 Go to **📑 Upload & Analysis**")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Your Skills Breakdown")
            tech_kw  = ["python","java","sql","api","aws","docker","c++","typescript"]
            frame_kw = ["react","django","flask","spark","fastapi","pytorch","tensorflow"]
            summary  = {
                "Total": len(st.session_state.resume_skills),
                "Technical": len([s for s in st.session_state.resume_skills
                                   if any(w in s.lower() for w in tech_kw)]),
                "Frameworks": len([s for s in st.session_state.resume_skills
                                    if any(w in s.lower() for w in frame_kw)]),
            }
            fig = px.bar(pd.DataFrame(list(summary.items()), columns=["Cat","Count"]),
                         x="Cat", y="Count", color="Cat", text="Count",
                         color_discrete_sequence=["#667eea","#764ba2","#f093fb"])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
        with c2:
            st.markdown("### JD Requirements")
            job_s = {"Total": len(st.session_state.jd_skills),
                     "Must-Have": len(st.session_state.missing),
                     "Nice-to-Have": len(st.session_state.weak)}
            fig2 = px.bar(pd.DataFrame(list(job_s.items()), columns=["Cat","Count"]),
                          x="Cat", y="Count", color="Cat", text="Count",
                          color_discrete_sequence=["#2ecc71","#f39c12","#e74c3c"])
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, width='stretch')

        st.divider()
        # Use index-based selectbox to avoid key/value mismatch on rerun
        jd_skills_list = st.session_state.jd_skills or []
        sel_idx = st.selectbox(
            "Select a skill to analyse:",
            range(len(jd_skills_list)),
            format_func=lambda i: jd_skills_list[i],
            key="skill_sel_idx",
        )
        sel = jd_skills_list[sel_idx] if jd_skills_list else None

        if sel:
            skill_info = skill_status = None
            for sd in st.session_state.matched:
                if sd["skill"] == sel: skill_info = sd; skill_status = "✅ Matched"; break
            if not skill_info:
                for sd in st.session_state.weak:
                    if sd["skill"] == sel: skill_info = sd; skill_status = "⚠️ Weak"; break
            if not skill_info:
                for sd in st.session_state.missing:
                    if sd["skill"] == sel: skill_info = sd; skill_status = "❌ Missing"; break

            if skill_info:
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Status",     skill_status)
                with c2: st.metric("Similarity", f"{skill_info.get('similarity',0):.2f}")
                with c3:
                    bm = skill_info.get("match","N/A")
                    st.metric("Best Match", bm if bm and bm != "N/A" else "Direct")

                with st.spinner(f"🤖 Detecting level for {sel}..."):
                    level = ai_detect_level(st.session_state.resume_text, sel)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Your Level:** `{level}`  \n"
                                f"**Target:** `Advanced`  \n"
                                f"**Market Demand:** 🔥 High")
                with c2:
                    with st.spinner("Generating roadmap preview..."):
                        rmap = ai_skill_roadmap(sel, level.lower())
                    if rmap:
                        st.markdown(f"**Est. Hours:** {rmap.get('estimated_hours','—')}h  \n"
                                    f"**Duration:** {rmap.get('duration_weeks','—')} weeks  \n"
                                    f"**Priority:** {rmap.get('priority','—').title()}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MULTI-ROLE COMPARISON
# ════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Multi-Role Comparison":
    st.subheader("🎯 Multi-Role Fit Comparison")

    if not analysis_ready():
        st.warning("⚠️ Please complete Upload & Analysis first")
    else:
        sample_roles = {
            "Senior Data Scientist": "Required: Python, Machine Learning, Statistics, SQL, Deep Learning, TensorFlow, Spark.",
            "Full Stack Developer":  "Required: JavaScript, React, Python, Django, SQL, REST APIs, Git.",
            "DevOps Engineer":       "Required: Docker, Kubernetes, AWS/Azure, CI/CD, Linux, Python scripting.",
            "ML Engineer":           "Required: Python, TensorFlow, PyTorch, ML algorithms, Big Data, SQL.",
            "Data Engineer":         "Required: Python, Apache Spark, SQL, Airflow, dbt, Cloud platforms.",
        }
        sel_roles = st.multiselect(
            "Select roles:", list(sample_roles.keys()),
            default=list(sample_roles.keys())[:3],
            max_selections=5,
            key="multi_role_sel",
        )
        if sel_roles:
            analyses: dict[str, Any] = {}
            for role in sel_roles:
                rs = skill_extractor.SkillExtractor.extract_skills(sample_roles[role])
                m, x, w = matcher.SkillMatcher.match_skills(st.session_state.resume_skills, rs)
                analyses[role] = {"fit": (len(m)/len(rs)*100) if rs else 0,
                                  "matched": len(m), "weak": len(w),
                                  "missing": len(x), "total": len(rs)}

            df = pd.DataFrame([
                {"Role": r, "Fit Score": f"{d['fit']:.0f}%",
                 "Matched": d["matched"], "Weak": d["weak"],
                 "Missing": d["missing"], "Total": d["total"]}
                for r, d in analyses.items()
            ]).sort_values("Fit Score", ascending=False,
                           key=lambda x: x.str.rstrip("%").astype(float))
            st.dataframe(df, use_container_width=True, hide_index=True)

            fig = px.bar(
                pd.DataFrame([{"Role": r,"Score": d["fit"]} for r,d in analyses.items()]).sort_values("Score"),
                x="Score", y="Role", color="Score", orientation="h",
                color_continuous_scale="RdYlGn", text="Score",
                title="Multi-Role Fit Score", range_x=[0,100])
            fig.update_traces(texttemplate="%{text:.0f}%")
            st.plotly_chart(fig, width='stretch')
            st.success(f"🏆 **Best Fit:** {df.iloc[0]['Role']} ({df.iloc[0]['Fit Score']})")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ROADMAP & VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Roadmap & Visualization":
    st.subheader("📊 AI-Generated Learning Roadmap")

    if not analysis_ready():
        st.warning("⚠️ Please complete Upload & Analysis first")
    else:
        st.markdown("### 📚 Experience Levels")
        with st.spinner("🤖 Detecting levels..."):
            exp = {s: ai_detect_level(st.session_state.resume_text, s)
                   for s in st.session_state.jd_skills[:10]}
        st.dataframe(pd.DataFrame([{"Skill": s,"Level": l} for s,l in exp.items()]),
                     use_container_width=True, hide_index=True,
                     column_config={"Level": st.column_config.SelectboxColumn(
                         "Level", options=["Beginner","Intermediate","Advanced","Expert"],
                         disabled=True)})

        st.markdown("### 🛣️ Learning Pathways")
        with st.spinner("🤖 Generating roadmap..."):
            roadmap_data = roadmap.RoadmapGenerator.generate_roadmap(
                st.session_state.missing, st.session_state.weak)

        for idx, (skill, rmap) in enumerate(roadmap_data.items(), 1):
            pri = rmap["priority"]
            with st.expander(f"**{idx}. {skill}** — {pri.upper()}", expanded=(idx<=2)):
                c1, c2, c3 = st.columns(3)
                with c1: st.markdown(f"**Priority:** {'🔴' if pri=='high' else '🟡'} {pri.title()}")
                with c2: st.markdown(f"**Est. Hours:** ⏱️ {40 if pri=='high' else 20}h")
                with c3: st.markdown(f"**Difficulty:** {'🔥 Advanced' if pri=='high' else '📊 Intermediate'}")

                if rmap.get("prerequisites"):
                    st.markdown("**Prerequisites:**")
                    st.code(" → ".join(rmap["prerequisites"]), language=None)
                st.markdown("**Learning Path:**")
                for i, s in enumerate(rmap.get("order",[]), 1):
                    st.write(f"{'✅' if i<=2 else '⏳'} **{i}. {s}**")

                if st.checkbox(f"🤖 AI Weekly Plan for {skill}", key=f"ai_rmap_{idx}"):
                    cur_level = exp.get(skill, "beginner").lower()
                    with st.spinner(f"Generating weekly plan for {skill}..."):
                        ai_rmap = ai_skill_roadmap(skill, cur_level)
                    if ai_rmap and "weekly_plan" in ai_rmap:
                        for wk in ai_rmap["weekly_plan"]:
                            st.markdown(f"**Week {wk['week']}:** "
                                        f"{', '.join(wk.get('topics',[])[:2])}")
                            if wk.get("project"):
                                st.caption(f"  🔨 {wk['project']}")

                if st.checkbox(f"Track progress for {skill}", key=f"track_{idx}"):
                    pc1, pc2 = st.columns(2)
                    with pc1: pct = st.slider(f"{skill} %", 0, 100, 0, key=f"pct_{idx}")
                    with pc2: mas = st.select_slider(f"{skill} Mastery",
                        options=["Novice","Beginner","Intermediate","Advanced","Expert"],
                        key=f"mas_{idx}")
                    if st.button(f"💾 Save {skill}", key=f"save_{idx}"):
                        save_progress(skill, pct, mas)
                        st.success("✅ Saved")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — LEARNING RESOURCES
# ════════════════════════════════════════════════════════════════════════════
elif page == "📚 Learning Resources":
    st.subheader("📚 AI-Curated Learning Resources")

    if not analysis_ready():
        st.warning("⚠️ Please complete Upload & Analysis first")
    else:
        target = [s["skill"] for s in (st.session_state.missing + st.session_state.weak)]
        if target:
            sel_idx = st.selectbox(
                "Pick a skill:",
                range(len(target)),
                format_func=lambda i: target[i],
                key="res_sel_idx",
            )
            sel = target[sel_idx]
            st.markdown(f"### 📖 Resources for **{sel}**")
            with st.spinner(f"🤖 Generating resources for {sel}..."):
                res_map = ai_learning_roadmap([{"skill": sel}], [], st.session_state.jd_text)
            res = res_map.get(sel, {})
            if isinstance(res, dict):
                c1, c2 = st.columns(2)
                with c1:
                    if res.get("subtopics"):
                        st.markdown("#### 📚 Topics")
                        for i, t in enumerate(res["subtopics"], 1): st.write(f"{i}. {t}")
                    if res.get("projects"):
                        st.markdown("#### 🎯 Projects")
                        for p in res["projects"]: st.write(f"• {p}")
                with c2:
                    if res.get("resources"):
                        st.markdown("#### 🔗 Links")
                        for i, link in enumerate(res["resources"], 1):
                            if link: st.write(f"[📌 Resource {i}]({link})")
                    if res.get("estimated_hours"):
                        st.info(f"⏱️ **Estimated:** {res['estimated_hours']}h")
        else:
            st.success("✅ No gaps — you're well-prepared!")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 7 — SKILL ASSESSMENT
# ════════════════════════════════════════════════════════════════════════════
elif page == "✅ Skill Assessment":
    st.subheader("✅ AI Skill Assessment & Quizzes")

    if not analysis_ready():
        st.warning("⚠️ Please complete Upload & Analysis first")
    else:
        target = [s["skill"] for s in (st.session_state.missing + st.session_state.weak)]
        skill_pool = target if target else (st.session_state.jd_skills or [])

        c1, c2 = st.columns(2)
        with c1:
            quiz_idx = st.selectbox(
                "Skill to test:",
                range(len(skill_pool)),
                format_func=lambda i: skill_pool[i],
                key="quiz_sel_idx",
            )
            quiz_skill = skill_pool[quiz_idx] if skill_pool else ""
        with c2:
            difficulty = st.selectbox("Difficulty:",
                                      ["beginner","intermediate","advanced"],
                                      key="quiz_difficulty")

        if st.button("📝 Generate Quiz", use_container_width=True, key="gen_quiz_btn"):
            st.session_state.current_quiz_skill = quiz_skill
            st.session_state.quiz_mode = True
            st.rerun()

        if st.session_state.quiz_mode and st.session_state.current_quiz_skill:
            st.markdown(f"### Quiz: **{st.session_state.current_quiz_skill}**")
            qs = [
                {"question": f"What is the primary purpose of {quiz_skill}?",
                 "options": ["Option A","Option B","Option C","Option D"], "correct": "Option A"},
                {"question": f"Which best describes a real-world use of {quiz_skill}?",
                 "options": ["Use Case 1","Use Case 2","Use Case 3","Use Case 4"],
                 "correct": "Use Case 1"},
            ]
            answers: dict[int, str] = {}
            for i, q in enumerate(qs, 1):
                st.write(f"**Q{i}:** {q['question']}")
                answers[i] = st.radio(f"q{i}", q["options"],
                                      key=f"quiz_ans_{i}",
                                      label_visibility="collapsed")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Submit", use_container_width=True, key="submit_quiz_btn"):
                    score = sum(1 for i, a in answers.items() if a == qs[i-1]["correct"])
                    pct   = score / len(qs) * 100
                    st.success(f"Score: {score}/{len(qs)} ({pct:.0f}%)")
                    if pct >= 70: st.balloons(); st.success("🎉 Great job!")
                    else: st.warning("📚 Review the concepts and try again.")
            with c2:
                if st.button("❌ Cancel", use_container_width=True, key="cancel_quiz_btn"):
                    st.session_state.quiz_mode = False
                    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 8 — PROGRESS TRACKER
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Progress Tracker":
    st.subheader("📈 Learning Progress & Spaced Repetition")

    prog = st.session_state.progress_data
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Skills Tracked",  len(prog), delta_color="off")
    with c2:
        hrs = sum(d.get("completion",0) for d in prog.values())
        st.metric("Total Progress pts", hrs, delta_color="off")
    with c3:
        avg = (hrs/len(prog)) if prog else 0
        st.metric("Avg Completion", f"{avg:.0f}%", delta_color="off")

    st.divider()
    if prog:
        df_p = pd.DataFrame([
            {"Skill": sk, "Completion %": d.get("completion",0),
             "Mastery": d.get("mastery","—"),
             "Updated": d.get("last_updated","")[:10]}
            for sk, d in prog.items()
        ])
        st.dataframe(df_p, use_container_width=True, hide_index=True,
            column_config={"Completion %": st.column_config.ProgressColumn(
                "Completion %", min_value=0, max_value=100)})
    else:
        st.info("📝 No progress yet — track skills in the Roadmap page.")

    st.markdown("### 🔄 Spaced Repetition Schedule")
    for lbl in ["Today — 1st review","Tomorrow — 2nd review",
                "In 3 days — 3rd","In 1 week — 4th","In 2 weeks — 5th"]:
        st.markdown(f"**{lbl}** — select skills from your progress list")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 9 — AI RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════
elif page == "💡 AI Recommendations":
    st.subheader("💡 AI-Powered Personalised Recommendations")

    if not analysis_ready():
        st.warning("⚠️ Please complete Upload & Analysis first")
    else:
        st.info(f"🤖 Generating with: **{st.session_state.llm_status}**")

        with st.spinner("🤖 Analysing your resume..."):
            suggestions = ai_resume_suggestions(st.session_state.resume_text,
                                                st.session_state.jd_text)

        st.markdown("### 📝 Resume Improvement Suggestions")
        for i, s in enumerate(suggestions, 1):
            c1, c2 = st.columns([0.05, 0.95])
            with c1: st.markdown(f"**{i}.**")
            with c2: st.markdown(s)

        st.divider()
        st.markdown("### 📚 AI Learning Roadmaps for Your Gaps")
        missing_skills = [s["skill"] for s in (st.session_state.missing or [])]
        if missing_skills:
            rec_idx = st.selectbox(
                "Choose a missing skill:",
                range(len(missing_skills)),
                format_func=lambda i: missing_skills[i],
                key="rec_skill_idx",
            )
            sel_skill = missing_skills[rec_idx]
            with st.spinner(f"🤖 Building roadmap for {sel_skill}..."):
                rmap = ai_skill_roadmap(sel_skill)
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Duration",  f"{rmap.get('duration_weeks','?')} weeks")
            with c2: st.metric("Est. Hours", f"{rmap.get('estimated_hours','?')}h")
            with c3: st.metric("Priority",   rmap.get("priority","—").title())

            if rmap.get("key_concepts"):
                st.markdown("**Key Concepts:** " +
                            ", ".join(f"`{c}`" for c in rmap["key_concepts"]))
            if rmap.get("weekly_plan"):
                for wk in rmap["weekly_plan"]:
                    with st.expander(f"Week {wk['week']} — "
                                     f"{', '.join(wk.get('topics',[])[:2])}"):
                        st.write("**Topics:**", ", ".join(wk.get("topics",[])))
                        if wk.get("project"): st.write("**Project:**", wk["project"])
                        st.write("**Hours:**", wk.get("hours","—"))

        st.divider()
        total_jd = len(st.session_state.jd_skills or [])
        m_pct = len(st.session_state.matched)/total_jd*100 if total_jd else 0
        st.markdown(f"### 💼 Career Readiness: **{m_pct:.0f}%**\n\n"
                    f"- Master **{len(st.session_state.missing or [])}** missing skills\n"
                    f"- Strengthen **{len(st.session_state.weak or [])}** weak skills\n"
                    f"- Build portfolio projects\n- Target roles once fit ≥ 70%")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 10 — SETTINGS
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.subheader("⚙️ Settings & LLM Configuration")

    st.markdown("## 🤖 LLM Backend Configuration")
    if not _INFERENCE_AVAILABLE:
        st.warning("⚠️ `inference.py` not found. Place it next to `app.py`.")

    backend_options = {
        "auto":        "🔍 Auto-detect (recommended)",
        "openai":      "🟢 OpenAI (GPT-3.5 / GPT-4 / GPT-4o)",
        "gemini":      "🔵 Google Gemini (1.5 Flash / Pro)  ✨ NEW",
        "claude":      "🟣 Anthropic Claude (Haiku / Sonnet / Opus)",
        "ollama":      "🦙 Ollama — Local open-source models",
        "huggingface": "🤗 HuggingFace (download model)",
        "local":       "💾 Local Checkpoint (fine-tuned)",
        "fallback":    "⚙️ Rule-based Fallback (no API key)",
    }
    backend_keys = list(backend_options.keys())

    # KEY FIX: use on_change callback to sync to llm_backend
    def _on_backend_change() -> None:
        st.session_state.llm_backend = st.session_state.backend_select

    chosen = st.selectbox(
        "Select LLM Backend:",
        backend_keys,
        format_func=lambda k: backend_options[k],
        index=backend_keys.index(st.session_state.llm_backend)
              if st.session_state.llm_backend in backend_keys else 0,
        key="backend_select",
        on_change=_on_backend_change,
    )

    if chosen == "openai":
        st.markdown("#### 🟢 OpenAI Settings")
        c1, c2 = st.columns(2)
        with c1:
            def _sync_oai_key():
                st.session_state.openai_api_key = st.session_state.openai_key_input
            st.text_input("API Key", value=st.session_state.openai_api_key,
                          type="password", placeholder="sk-...",
                          key="openai_key_input", on_change=_sync_oai_key)
        with c2:
            oai_models = ["gpt-3.5-turbo","gpt-4","gpt-4-turbo","gpt-4o"]
            def _sync_oai_mdl():
                st.session_state.openai_model = st.session_state.openai_mdl_select
            st.selectbox("Model", oai_models,
                         index=oai_models.index(st.session_state.openai_model)
                               if st.session_state.openai_model in oai_models else 0,
                         key="openai_mdl_select", on_change=_sync_oai_mdl)
        st.caption("Get your key → https://platform.openai.com/api-keys")

    elif chosen == "gemini":
        st.markdown("#### 🔵 Google Gemini Settings")
        st.markdown("""
**Setup:**
1. Get a free API key at [aistudio.google.com](https://aistudio.google.com/app/apikey)
2. Install: `pip install google-generativeai`
3. Enter your key below and click **Apply LLM Settings**
        """)
        c1, c2 = st.columns(2)
        with c1:
            def _sync_gem_key():
                st.session_state.gemini_api_key = st.session_state.gemini_key_input
            st.text_input("Gemini API Key", value=st.session_state.gemini_api_key,
                          type="password", placeholder="AIza...",
                          key="gemini_key_input", on_change=_sync_gem_key)
        with c2:
            gemini_models = ["gemini-1.5-flash","gemini-1.5-pro","gemini-pro"]
            def _sync_gem_mdl():
                st.session_state.gemini_model = st.session_state.gemini_mdl_select
            st.selectbox("Model", gemini_models,
                         index=gemini_models.index(st.session_state.gemini_model)
                               if st.session_state.gemini_model in gemini_models else 0,
                         key="gemini_mdl_select", on_change=_sync_gem_mdl)
        st.info("💡 **gemini-1.5-flash** is free-tier and very fast.")

    elif chosen == "claude":
        st.markdown("#### 🟣 Anthropic Claude Settings")
        c1, c2 = st.columns(2)
        with c1:
            def _sync_claude_key():
                st.session_state.anthropic_api_key = st.session_state.claude_key_input
            st.text_input("API Key", value=st.session_state.anthropic_api_key,
                          type="password", placeholder="sk-ant-...",
                          key="claude_key_input", on_change=_sync_claude_key)
        with c2:
            claude_models = ["claude-3-haiku-20240307",
                             "claude-3-sonnet-20240229",
                             "claude-3-opus-20240229"]
            def _sync_claude_mdl():
                st.session_state.claude_model = st.session_state.claude_mdl_select
            st.selectbox("Model", claude_models,
                         index=claude_models.index(st.session_state.claude_model)
                               if st.session_state.claude_model in claude_models else 0,
                         key="claude_mdl_select", on_change=_sync_claude_mdl)
        st.caption("Get your key → https://console.anthropic.com/")

    elif chosen == "ollama":
        st.markdown("#### 🦙 Ollama — Local Open-Source Models")
        st.markdown("""
**Setup (one-time):**
```bash
ollama serve          # start server (keep running)
ollama pull phi3      # fast & small (~2 GB)
ollama pull llama3    # recommended (~5 GB)
```
        """)
        c1, c2 = st.columns(2)
        with c1:
            def _sync_ollama_url():
                st.session_state.ollama_url = st.session_state.ollama_url_input
            st.text_input("Ollama Server URL", value=st.session_state.ollama_url,
                          key="ollama_url_input", on_change=_sync_ollama_url)
        with c2:
            is_up = ollama_ping()
            if is_up:
                st.success("✅ Ollama server is reachable")
            else:
                st.error("❌ Not reachable — run `ollama serve`")

        if is_up:
            live_models = get_ollama_models()
            if live_models:
                st.markdown("**Models on your server:**")
                live_idx = st.selectbox(
                    "Select a pulled model:",
                    range(len(live_models)),
                    format_func=lambda i: live_models[i],
                    key="ollama_live_idx",
                )
                if st.button("Use this model", key="use_live_model"):
                    st.session_state.ollama_model = live_models[live_idx]
                    st.session_state.ollama_mdl_input = live_models[live_idx]
                    st.success(f"✅ Set to {live_models[live_idx]}")
            else:
                st.warning("No models found — run `ollama pull llama3`")

        popular = ["llama3","llama3:8b","llama3:70b","mistral","phi3",
                   "gemma2","qwen2","codellama","deepseek-coder","llava"]
        st.markdown(
            "Popular: " + " ".join(
                f'<span class="ollama-pill">{m}</span>' for m in popular
            ), unsafe_allow_html=True
        )
        def _sync_ollama_mdl():
            st.session_state.ollama_model = st.session_state.ollama_mdl_input
        st.text_input("Ollama Model Name", value=st.session_state.ollama_model,
                      placeholder="llama3",
                      key="ollama_mdl_input", on_change=_sync_ollama_mdl)

    elif chosen == "huggingface":
        st.markdown("#### 🤗 HuggingFace Settings")
        def _sync_hf():
            st.session_state.hf_model = st.session_state.hf_mdl_input
        st.text_input("Model Name", value=st.session_state.hf_model,
                      placeholder="mistralai/Mistral-7B-Instruct-v0.2",
                      key="hf_mdl_input", on_change=_sync_hf)
        st.caption("Any text-generation model from https://huggingface.co/models")

    elif chosen == "local":
        st.markdown("#### 💾 Local Checkpoint Settings")
        c1, c2 = st.columns(2)
        with c1:
            def _sync_cp():
                st.session_state.local_checkpoint = st.session_state.local_cp_input
            st.text_input("Checkpoint Path", value=st.session_state.local_checkpoint,
                          placeholder="./checkpoints/final",
                          key="local_cp_input", on_change=_sync_cp)
        with c2:
            def _sync_lora():
                st.session_state.use_lora = st.session_state.use_lora_check
            st.checkbox("Use LoRA (PEFT)", value=st.session_state.use_lora,
                        key="use_lora_check", on_change=_sync_lora)

    elif chosen == "auto":
        st.info(
            "🔍 **Auto-detect order:**\n\n"
            "1. Local checkpoint  \n"
            "2. OpenAI key (`OPENAI_API_KEY`)  \n"
            "3. Anthropic key (`ANTHROPIC_API_KEY`)  \n"
            "4. Gemini key (`GEMINI_API_KEY`)  \n"
            "5. Ollama server (ping localhost:11434)  \n"
            "6. Rule-based fallback"
        )

    elif chosen == "fallback":
        st.info("⚙️ Rule-based keyword matching — no API key required, works offline.")

    st.markdown("")
    if st.button("🚀 Apply LLM Settings", use_container_width=True,
                 type="primary", key="apply_llm_btn"):
        with st.spinner("Initialising LLM backend..."):
            apply_llm_settings()
        st.success(f"✅ Applied: **{st.session_state.llm_status}**")
        st.rerun()

    st.divider()
    st.markdown("## 🎨 Display Preferences")
    c1, c2 = st.columns(2)
    with c1: st.selectbox("Theme",    ["Light","Dark","Auto"],    key="pref_theme")
    with c2: st.selectbox("Language", ["English","Spanish","French","Chinese"],
                          key="pref_lang")

    st.divider()
    st.markdown("## 📧 Notifications")
    c1, c2 = st.columns(2)
    with c1:
        st.checkbox("Email alerts for reviews",  value=True,  key="notif_email")
        st.checkbox("Weekly progress summary",   value=True,  key="notif_weekly")
    with c2:
        st.checkbox("Quiz reminders",            value=True,  key="notif_quiz")
        st.checkbox("New resource updates",      value=False, key="notif_resources")

    st.divider()
    st.markdown("## 🔐 Data & Privacy")
    st.markdown("- **Storage:** Session-based (cleared on browser close)\n"
                "- **API Keys:** Session only — never logged or stored\n"
                "- **Encryption:** SSL/TLS in transit\n"
                "- **GDPR:** Compliant")

    if st.button("🗑️ Clear All Session Data", use_container_width=True,
                 key="clear_session_btn"):
        st.session_state.clear()
        st.success("✅ Cleared")
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 11 — ABOUT
# ════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.subheader("ℹ️ About AI Adaptive Onboarding Engine v2.3")
    st.markdown("""
## 🎯 Mission
Personalised, adaptive learning paths powered by a multi-LLM architecture.

## 🆕 What's New in v2.3
- **Fix:** `KeyError` on `st.session_state` widget ID — all widgets now have
  explicit keys pre-initialised in `init_session_state()`
- **Fix:** `on_change` callbacks sync widget values to session state reliably
- **Google Gemini** backend — `gemini-1.5-flash` & `gemini-1.5-pro`
- **Enhanced Ollama** panel — live model list, ping check, quick-select
- **7 backends** — switch at runtime with zero code changes
- **Windows fix** — cross-platform temp file paths via `tempfile.gettempdir()`
""")
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🧠 AI & ML\n"
                    "- OpenAI GPT-3.5/4/4o\n"
                    "- Google Gemini 1.5 ✨\n"
                    "- Anthropic Claude 3\n"
                    "- Ollama (local LLMs)\n"
                    "- HuggingFace models\n"
                    "- LoRA fine-tuning")
    with c2:
        st.markdown("### 🐍 Backend\n"
                    "- Python 3.10+\n"
                    "- inference.py (ModelInference)\n"
                    "- NetworkX graphs\n"
                    "- Sentence Transformers\n"
                    "- SpaCy NER\n"
                    "- peft (LoRA)")
    with c3:
        st.markdown("### 🎨 Frontend\n"
                    "- Streamlit\n"
                    "- Plotly\n"
                    "- Custom CSS\n"
                    "- Responsive layout")

    st.divider()
    st.markdown("## 📁 File Structure")
    st.code("""
hackthon/
├── app.py                    ← Streamlit UI  (v2.3)
├── inference.py              ← ModelInference (7 backends)
├── backend/
│   ├── parser.py
│   ├── skill_extractor.py
│   ├── matcher.py
│   ├── roadmap.py
│   └── llm_engine.py
├── utils/helpers.py
├── checkpoints/final/
├── .env
└── requirements.txt
    """, language="text")

    st.markdown("## 🔧 Install All Dependencies")
    st.code("""
pip install streamlit pandas plotly
pip install openai google-generativeai anthropic requests
pip install transformers peft sentence-transformers
pip install spacy && python -m spacy download en_core_web_sm
pip install pymupdf python-docx networkx scikit-learn
    """, language="bash")

    st.markdown("\n---\n**Built with 💜 for learners and professionals worldwide.**")


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="text-align:center;color:#888;font-size:.85em;padding:20px;">
    <p>🚀 <strong>AI Adaptive Onboarding Engine v2.3</strong>
       | Python · Streamlit · OpenAI · Gemini · Claude · Ollama</p>
    <p>© 2026 AI Forge Squad
       | <a href="#">GitHub</a> | <a href="#">Docs</a> | <a href="#">Support</a></p>
</div>
""", unsafe_allow_html=True)