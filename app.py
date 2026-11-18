"""Step 3: Streamlit UI.  Run: streamlit run app.py"""
import streamlit as st
from rag import Retriever, answer, make_quiz, study_plan

st.set_page_config(page_title="GovExam AI Assistant", page_icon="📚")
st.title("📚 Government Exam Preparation Assistant")


@st.cache_resource
def load():
    return Retriever()


try:
    ret = load()
except Exception:
    st.error("Index not found. Run `python ingest.py` first.")
    st.stop()

exam = st.sidebar.selectbox("Exam", ["All"] + ret.exams())
exam = None if exam == "All" else exam
subject = st.sidebar.selectbox("Subject", ["All"] + ret.subjects(exam))
subject = None if subject == "All" else subject

tab1, tab2, tab3 = st.tabs(["Ask", "Quiz", "Study plan"])

with tab1:
    q = st.text_input("Ask a question from your study material")
    if q:
        with st.spinner("Searching..."):
            ans, hits = answer(ret, q, exam, subject)
        st.markdown(ans)
        with st.expander("Sources"):
            for i, h in enumerate(hits, 1):
                st.caption(f"[{i}] {h['file']} p.{h['page']} · score {h['score']:.2f}")
                st.write(h["text"][:300] + "...")

with tab2:
    topic = st.text_input("Quiz topic (e.g. Fundamental Rights)")
    n = st.slider("Questions", 3, 10, 5)
    if st.button("Generate quiz") and topic:
        st.session_state.quiz = make_quiz(ret, topic, n, exam, subject)
        if not st.session_state.quiz:
            st.warning("Not enough material on this topic.")
    quiz = st.session_state.get("quiz", [])
    for i, item in enumerate(quiz):
        st.radio(f"{i+1}. {item['q']}", item["options"], index=None, key=f"a{i}")
    if quiz and st.button("Submit"):
        score = 0
        for i, item in enumerate(quiz):
            pick = st.session_state.get(f"a{i}")
            ok = pick == item["options"][item["answer"]]
            score += ok
            st.write(("✅" if ok else "❌") + f" Q{i+1}: {item['options'][item['answer']]} — {item['explanation']}")
        st.success(f"Score: {score}/{len(quiz)}")

with tab3:
    plan_exam = st.text_input("Exam", exam or "UPSC Prelims")
    days = st.number_input("Days available", 7, 365, 90)
    hours = st.number_input("Hours per day", 1, 14, 6)
    weak = st.text_input("Weak subjects (optional)")
    if st.button("Generate plan"):
        st.markdown(study_plan(plan_exam, days, hours, weak))
