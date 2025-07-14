import streamlit as st
import typing
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig
from tools.llm import llm
from graph import build_graph
from prompts import system_message
from tools.composio_tools import get_composio_tools

load_dotenv()


st.set_page_config(page_title="Deep Research Agent", layout="wide")
st.title("🧠 Deep Research Agent")

topic = st.text_input("Enter your research topic:")
domain = st.text_input("Enter the domain (e.g., Health, Technology, etc.):")

if topic and domain and st.button("Start Research"):
    with st.spinner("Generating research questions..."):
        question_prompt = f"Generate exactly 3 specific yes/no research questions about the topic '{topic}' in the domain '{domain}'. Respond ONLY with a numbered list and NOTHING ELSE."
        q_messages = [{"role": "user", "content": question_prompt}]
        response = llm.invoke(q_messages)
        questions = [q.strip() for q in response.content.split('\n') if q.strip()]
    
    st.subheader("📌 Research Questions")
    for i, q in enumerate(questions, 1):
        st.markdown(f"**{i}. {q}**")

    # Step 2: Research each question
    answers = []
    for i, question in enumerate(questions):
        st.write(f"🔍 Researching: **{question}**")
        memory = MemorySaver()
        graph = build_graph().compile(checkpointer=memory)
        config = typing.cast(RunnableConfig, {"configurable": {"thread_id": str(i+1)}})

        messages = [{"role": "user", "content": f"{system_message}\n\nResearch question: {question}"}]
        answer = ""
        for chunk in graph.stream({"messages": messages}, config=config, stream_mode="values"):
            content = chunk["messages"][-1].content
            answer += content + "\n"
        answers.append((question, answer))
        st.success("Completed ✅")

    # Step 3: Compile answers into report
    final_memory = MemorySaver()
    final_graph = build_graph().compile(checkpointer=final_memory)
    final_config = typing.cast(RunnableConfig, {"configurable": {"thread_id": "final"}})

    qa_html = "\n".join(
        f"<h2>{i+1}. {q}</h2><p>{a}</p>" for i, (q, a) in enumerate(answers)
    )

    report_prompt = (
        f"You are a sophisticated research assistant. token limit 5000. Compile the following research findings into a professional, McKinsey-style report using HTML:\n\n"
        f"Topic: {topic}\nDomain: {domain}\n\n"
        f"Research Findings:\n{qa_html}\n\n"
        f"Create a Google Doc with this full report using GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN."
    )

    st.write("📄 Creating Final Report...")
    messages = [{"role": "user", "content": report_prompt}]
    report_output = ""
    for chunk in final_graph.stream({"messages": messages}, config=final_config, stream_mode="values"):
        content = chunk["messages"][-1].content
        report_output += content + "\n"

    st.markdown("---")
    st.subheader("✅ Report Created:")
    st.markdown(report_output)

    followup = st.text_input("Ask a follow-up question if needed:")
    if followup:
        followup_msgs = [{"role": "user", "content": followup}]
        for event in final_graph.stream({"messages": followup_msgs}, config=final_config, stream_mode="values"):
            st.markdown(event["messages"][-1].content)
