# Deep Research Agent 🧠📚

## Project Description 🌟

This is a powerful AI-driven Multi-Agent built with **Streamlit** and **LangGraph**, designed to conduct comprehensive research on user-specified topics within a chosen domain. It generates targeted research questions, performs in-depth analysis using AI-powered tools, and compiles findings into a professional, McKinsey-style HTML report, seamlessly saved to **Google Docs**. Leveraging the `composio_langgraph` library for tool integration and `langchain_groq` for language model interactions, this tool is perfect for researchers, analysts, or anyone seeking structured, high-quality insights. 🚀

With support for follow-up questions, it enables iterative refinement of research, making it a versatile solution for professional and academic use. 📊

## Features ✨

- **Input Flexibility** 📝: Specify a research topic and domain (e.g., Health, Technology) via an intuitive Streamlit web interface.
- **Automated Question Generation** ❓: Generates three specific yes/no research questions tailored to the topic and domain.
- **AI-Powered Research** 🤖: Uses the Meta LLaMA model and `COMPOSIO_SEARCH_TAVILY_SEARCH` for real-time web searches to gather accurate data.
- **Professional Reporting** 📄: Compiles findings into a polished, HTML-formatted, McKinsey-style report.
- **Google Docs Integration** 📑: Automatically saves reports to Google Docs using `GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN`.
- **Interactive Follow-Ups** 🔄: Ask follow-up questions to refine or expand research results.
- **State Management** 🧮: Employs LangGraph for seamless workflow orchestration and memory management.

## Tech Stack 🛠️

- **Python** 🐍: Core programming language.
- **Streamlit** 🌐: Powers the interactive web interface.
- **LangGraph** 🔗: Manages research workflow and state.
- **LangChain (langchain_groq)** 🤝: Interacts with the Meta LLaMA model.
- **Composio** 🔧: Enables web search (`COMPOSIO_SEARCH_TAVILY_SEARCH`) and Google Docs integration (`GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN`).
- **dotenv** 🔒: Securely manages API keys.
- **MemorySaver** 💾: Checkpoints and maintains research context across sessions.

## Project Structure 📂

```plaintext
├── app.py                  # Main Streamlit application 🚀
├── graph.py                # LangGraph workflow configuration 🔄
├── state.py                # Graph state definition 🧮
├── nodes/nodes.py          # Agent and tool nodes for the workflow 🤖
├── tools/composio_tools.py # Composio toolset configuration 🛠️
├── tools/llm.py            # Language model setup 🗣️
├── prompts.py              # System prompt for the research agent 📜
├── .env                    # Environment variables 🔒
└── README.md               # Project documentation 📖
```

## Installation 🛠️

1. **Clone the Repository** 📥:
   ```bash
   git clone https://github.com/zeeshier/deep-research-agent.git
   cd deep-research-agent
   ```

2. **Set Up a Virtual Environment** 🌍:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies** 📦:
   ```bash
   pip install streamlit langgraph langchain-groq composio-langgraph python-dotenv
   ```

4. **Configure Environment Variables** 🔑:
   Create a `.env` file in the project root and add your Groq API key:
   ```plaintext
   GROQ_API_KEY=your_groq_api_key
   ```

5. **Run the Application** 🚀:
   ```bash
   streamlit run app.py
   ```

## Usage 🎯

1. Open the app in your browser (default: `http://localhost:8501`) 🌐.
2. Enter a research topic (e.g., "AI in healthcare") and domain (e.g., "Health") 📝.
3. Click **Start Research** to generate questions and answers 🔍.
4. View the professional report, automatically saved to Google Docs 📑.
5. Ask follow-up questions to refine or expand the research 🔄.

## Example 📈

**Input**:
- Topic: AI-powered diagnostic tools
- Domain: Health

**Output**:
- **Research Questions** ❓:
  1. Are AI-powered diagnostic tools widely adopted in hospitals by 2025?
  2. Do AI diagnostic tools improve patient outcomes compared to traditional methods?
  3. Are there significant regulatory barriers to adopting AI diagnostic tools?
- **Report** 📄: A detailed HTML report with findings, saved to Google Docs.

## Contributing 🤝

We welcome contributions! Follow these steps to contribute:
1. Fork the repository 🍴.
2. Create a new branch (`git checkout -b feature/your-feature`) 🌿.
3. Commit your changes (`git commit -m "Add your feature"`) ✅.
4. Push to the branch (`git push origin feature/your-feature`) 🚀.
5. Open a pull request 📬.

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact 📧

For questions or feedback, open an issue on GitHub or contact the maintainer at [zeeshanwarraich51@gmail.com]. We'd love to hear from you! 😊
