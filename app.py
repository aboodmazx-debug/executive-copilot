import os
import sqlite3
from datetime import datetime
from typing import List
import streamlit as st
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

# إعداد عنوان الصفحة وتخطيطها العريض
st.set_page_config(page_title="Executive Brand-Voice Copilot", page_icon="🏢", layout="wide")

# تنسيق CSS لضبط اتجاه الواجهة بالكامل من اليمين لليسار (RTL)
st.markdown("""
    <style>
    /* اتجاه عام من اليمين لليسار */
    .stApp { direction: rtl; text-align: right; }
    .block-container { padding-top: 2rem; max-width: 1250px; }
    div[data-testid="stMarkdownContainer"], .stTextInput, .stTextArea, .stSelectbox,
    .stRadio, .stSlider label, div[data-testid="stCaptionContainer"], div[data-testid="stAlert"] {
        direction: rtl; text-align: right;
    }
    .stTextArea textarea, .stTextInput input { direction: rtl; text-align: right; line-height: 1.8; }
    /* القوائم النقطية: النقطة بجانب النص وبمسافات مضبوطة */
    div[data-testid="stMarkdownContainer"] ul, div[data-testid="stMarkdownContainer"] ol {
        direction: rtl; text-align: right; padding-right: 1.4rem; padding-left: 0; margin: 0.2rem 0 0.8rem 0;
    }
    div[data-testid="stMarkdownContainer"] li { margin-bottom: 0.2rem; padding-right: 0.3rem; }
    div[data-testid="stMarkdownContainer"] li p { margin: 0; }
    /* بطاقات التقييم */
    div[data-testid="stMetric"] {
        direction: rtl; text-align: right; background: rgba(128,128,128,0.10);
        border: 1px solid rgba(128,128,128,0.25); border-radius: 12px; padding: 0.8rem 1rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    div[data-testid="stMetricValue"] > div { white-space: normal; overflow: visible; text-overflow: clip; }
    div[data-testid="stMetricLabel"] { justify-content: flex-start; }
    /* الشريط المنزلق يبقى بشكله الطبيعي */
    div[data-testid="stSlider"] > div { direction: ltr; }
    div[data-testid="stSlider"] label { direction: rtl; }
    /* المصادر */
    div[data-testid="stExpander"] { direction: rtl; }
    </style>
""", unsafe_allow_html=True)

# إعداد قاعدة البيانات SQLite
DB_NAME = "executive_copilot.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meeting_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            raw_notes TEXT,
            summary TEXT,
            draft_email TEXT,
            clarity_score INTEGER,
            tone_alignment TEXT,
            verdict TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_log(raw_notes, summary, draft_email, clarity_score, tone_alignment, verdict):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO meeting_logs (timestamp, raw_notes, summary, draft_email, clarity_score, tone_alignment, verdict)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            raw_notes, summary, draft_email, clarity_score, tone_alignment, verdict
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        st.warning(f"تنبيه: تعذر حفظ السجل: {e}")

# تعريف قالب المخرجات المهيكل
class UnifiedExecutiveOutput(BaseModel):
    summary_and_actions: str = Field(description="القرارات المستخرجة وبنود العمل والمسؤولين والتواريخ والمخاطر بالتفصيل")
    draft_email: str = Field(description="مسودة الخطاب أو البريد الرسمي التنفيذي متضمناً العنوان والمحتوى والنبرة المحددة")
    clarity_score: int = Field(description="تقييم جودة ووضوح الخطاب من 1 إلى 10")
    tone_alignment: str = Field(description="مدى مطابقة النبرة المطلوبة (ممتاز، مقبول، يحتاج تحسين)")
    detected_strengths: List[str] = Field(description="أبرز نقاط القوة في الرسالة")
    potential_risks: List[str] = Field(description="أي غموض أو ملاحظات تدقيقية")
    verdict: str = Field(description="القرار النهائي: معتمد للإرسال / يحتاج مراجعة طفيفة / مرفوض")
    applied_guidelines: List[str] = Field(description="الإرشادات والسياسات المسترجعة من قاعدة المعرفة التي طُبقت فعلاً")

@st.cache_resource
def get_chain(api_key):
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        groq_api_key=api_key,
        temperature=0.2,
        max_tokens=2200
    )
    parser = PydanticOutputParser(pydantic_object=UnifiedExecutiveOutput)

    template = """أنت خبير حوكمة التواصل المؤسسي التنفيذي.
مهمتك: صياغة المخرجات بإيجاز واحترافية عالية مع الالتزام التام بإغلاق هيكل الـ JSON كاملاً دون انقطاع.

الإرشادات المؤسسية المسترجعة من قاعدة المعرفة (التزم بها في الصياغة والتدقيق ولا تخترع سياسات غير موجودة فيها):
{context}

المدخلات:
- الملاحظات: {raw_notes}
- النبرة: {tone}
- الجمهور: {audience}
- اللغة: {language}

المطلوب:
1. استخراج القرارات والمهام والمخاطر بنقاط موجزة ومركزة.
2. كتابة إيميل رسمي موجز ومباشر يعكس النبرة المطلوبة بدقة دون إطالة مفرطة، وفق دليل النبرة والقوالب الواردة في الإرشادات المسترجعة.
3. تدقيق الإيميل مقابل السياسات المسترجعة (مثل سياسات التأخير والميزانية) وكتابة تقرير التدقيق.
4. ذكر السياسات التي طُبقت فعلاً في applied_guidelines، وإغلاق كائن الـ JSON بشكل سليم 100%.

{format_instructions}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["raw_notes", "tone", "audience", "language", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return prompt | llm | parser

# ===================== طبقة RAG =====================
KB_DIR = "knowledge_base"

@st.cache_resource(show_spinner="جاري بناء قاعدة المعرفة (RAG)...")
def get_vectorstore():
    # 1) تحميل المستندات
    documents = DirectoryLoader(
        KB_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    ).load()
    # 2) التقسيم إلى مقاطع
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80, separators=["\n\n", "\n", ".", "،", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    # 3) التضمينات + 4) قاعدة المتجهات
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.from_documents(chunks, embeddings)

def retrieve_guidelines(query, k=4):
    # 5) المسترجع (Retriever)
    retriever = get_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever.invoke(query)

def format_docs(docs):
    return "\n\n".join(
        f"[المصدر: {os.path.basename(d.metadata.get('source', 'غير معروف'))}]\n{d.page_content}" for d in docs
    )

# العنوان الرئيسي والتحقق من المفتاح
st.title("🏢 نظام المساعد التنفيذي لصياغة وتدقيق المراسلات")
st.caption("تحويل الملاحظات غير المنظمة إلى مهام وإيميل رسمي مدقق ومعتمد (Executive Brand-Voice Copilot)")

api_key = os.environ.get("GROQ_API_KEY", "")
if not api_key:
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        api_key = ""

if not api_key:
    api_key = st.sidebar.text_input("🔑 أدخل مفتاح Groq API:", type="password")

col1, col2 = st.columns([1, 1], gap="large")

example_notes = "اجتماع طارئ لمشروع كلاود اكس: الميزانية فيها عجز 15%، فريق التطوير متأخر اسبوعين عن التسليم، احمد بيسلم تقرير الفحص يوم الثلاثاء، سارة بتراجع تكاليف السيرفرات غداً، قررنا تأجيل الاطلاق التجريبي لـ 15 نوفمبر ولازم نبلغ الإدارة اليوم."

with col1:
    st.subheader("📥 مدخلات الاجتماع")
    in_notes = st.text_area("ملاحظات الاجتماع (Raw Notes)", value=example_notes, height=150)
    in_tone = st.selectbox(
        "النبرة المستهدفة",
        [
            "رسمي تنفيذي (Executive)",
            "حازم ومستعجل (Urgent)",
            "دبلوماسي وتعاوني (Diplomatic)",
            "مهني وودي (Friendly)"
        ]
    )
    in_audience = st.text_input("الجمهور المستهدف", value="الإدارة التنفيذية ورؤساء الأقسام")
    in_lang = st.radio("اللغة", ["Arabic", "English"], horizontal=True)
    in_k = st.slider("عدد المقاطع المسترجعة (k)", 1, 8, 4)
    btn_run = st.button("🚀 بدء المعالجة والتوليد", type="primary", use_container_width=True)

with col2:
    st.subheader("📤 المخرجات والتدقيق")
    if btn_run:
        if not api_key:
            st.error("يرجى إدخال مفتاح Groq API أولاً.")
        elif len(in_notes.strip()) < 10:
            st.warning("يرجى إدخال ملاحظات واضحة للاجتماع تحتوي على تفاصيل كافية.")
        else:
            tone_dict = {
                "رسمي تنفيذي (Executive)": "Executive, highly formal, confident, concise, and strategic",
                "حازم ومستعجل (Urgent)": "Assertive, highly urgent, direct, emphasizing strict accountability",
                "دبلوماسي وتعاوني (Diplomatic)": "Diplomatic, highly professional, collaborative, and constructive",
                "مهني وودي (Friendly)": "Professional yet friendly, supportive, and motivating"
            }

            with st.spinner("جاري التحليل والصياغة والتدقيق..."):
                try:
                    # 1) الاسترجاع (Retrieval) من قاعدة المعرفة
                    docs = retrieve_guidelines(f"{in_tone} | {in_audience} | {in_notes}", k=in_k)
                    chain = get_chain(api_key)
                    # 2) التوليد المعزز (Augmented Generation)
                    res = chain.invoke({
                        "raw_notes": in_notes,
                        "tone": tone_dict[in_tone],
                        "audience": in_audience,
                        "language": in_lang,
                        "context": format_docs(docs)
                    })

                    save_log(in_notes, res.summary_and_actions, res.draft_email, res.clarity_score, res.tone_alignment, res.verdict)

                    st.text_area("1. الملخص وبنود العمل والقرارات", value=res.summary_and_actions, height=230)
                    st.text_area("2. مسودة الإيميل الرسمي", value=res.draft_email, height=320)

                    st.markdown("### 📋 تقييم جودة الخطاب")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("درجة الوضوح", f"{res.clarity_score}/10")
                    m2.metric("مطابقة النبرة", res.tone_alignment)
                    m3.metric("القرار النهائي", res.verdict)

                    st.markdown("**نقاط القوة:**")
                    for s in res.detected_strengths:
                        st.markdown(f"- {s}")

                    st.markdown("**الملاحظات ومواضع التدقيق:**")
                    if res.potential_risks:
                        for r in res.potential_risks:
                            st.markdown(f"- {r}")
                    else:
                        st.markdown("- لا توجد ملاحظات سلبية.")

                    st.markdown("**الإرشادات والسياسات المطبقة:**")
                    for g in res.applied_guidelines:
                        st.markdown(f"- {g}")

                    with st.expander("📚 مصادر RAG: المقاطع التي استرجعها النظام من قاعدة المعرفة", expanded=True):
                        for i, d in enumerate(docs, 1):
                            st.markdown(f"**{i}.** `{os.path.basename(d.metadata.get('source', ''))}`")
                            st.caption(d.page_content)

                    st.success("تم حفظ السجل تلقائياً في قاعدة بيانات SQLite.")
                except Exception as e:
                    st.error(f"حدث خطأ أثناء المعالجة: {e}")

st.divider()
with st.expander("🗄️ استعراض آخر السجلات المحفوظة في قاعدة البيانات (SQLite)"):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, clarity_score, tone_alignment, verdict FROM meeting_logs ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()
    conn.close()
    if rows:
        st.table([{"المعرف": r[0], "الوقت": r[1], "الوضوح": f"{r[2]}/10", "مطابقة النبرة": r[3], "القرار": r[4]} for r in rows])
    else:
        st.info("لا توجد سجلات محفوظة حتى الآن.")
