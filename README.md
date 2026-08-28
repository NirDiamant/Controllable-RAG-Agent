# Sophisticated Controllable Agent for Complex RAG Tasks 🧠📚

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/nir-diamant-759323134/)
[![Twitter](https://img.shields.io/twitter/follow/NirDiamantAI?label=Follow%20@NirDiamantAI&style=social)](https://twitter.com/NirDiamantAI)
[![Discord](https://img.shields.io/badge/Discord-Join%20our%20community-7289da?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/cA6Aa4uyDX)


An advanced Retrieval-Augmented Generation (RAG) solution designed to tackle complex questions that simple semantic similarity-based retrieval cannot solve. This project showcases a sophisticated deterministic graph acting as the "brain" of a highly controllable autonomous agent capable of answering non-trivial questions from your own data.

![Demo](graphs/demo.gif)

<div align="center">

# 📖 [RAG Made Simple: the book that extends this repo](https://diamant-ai.com/rag-made-simple?code=RAGKING)

<a href="https://diamant-ai.com/rag-made-simple?code=RAGKING"><img src="https://raw.githubusercontent.com/NirDiamant/agents-towards-production/main/images/rag_book_best_seller.png" alt="RAG Made Simple - Amazon bestseller in Generative AI" width="500"></a>

The full reference: a 400-page visual guide that goes deeper than any notebook can. The **intuition** behind every technique, **side-by-side comparisons** of when each one wins (and when it quietly fails), and **diagrams** that make the tricky parts finally click.

**1,500+ copies sold · Hit #1 in Generative AI on Amazon at launch · ⭐ 4.6 stars**

📖 **PDF + EPUB · GitHub community price: 33% off with code RAGKING**

### 👉 [**Get RAG Made Simple (33% off with code RAGKING)**](https://diamant-ai.com/rag-made-simple?code=RAGKING)

</div>

---

🚀 Level up with my **[Agents Towards Production](https://github.com/NirDiamant/agents-towards-production)** repository. It delivers horizontal, code-first tutorials that cover every tool and step in the lifecycle of building production-grade GenAI agents, guiding you from spark to scale with proven patterns and reusable blueprints for real-world launches, making it the smartest place to start if you're serious about shipping agents to production.

📚 Explore my **[comprehensive guide on RAG techniques](https://github.com/NirDiamant/RAG_Techniques)** to complement this advanced agent implementation with many other RAG techniques.

🤖 Explore my **[GenAI Agents Repository](https://github.com/NirDiamant/GenAI_Agents)** to complement this advanced agent implementation with many other AI Agents implementations and tutorials.

## 📫 Stay Updated!

<div align="center">
<table>
<tr>
<td align="center">🚀<br><b>Cutting-edge<br>Updates</b></td>
<td align="center">💡<br><b>Expert<br>Insights</b></td>
<td align="center">🎯<br><b>Top 0.1%<br>Content</b></td>
</tr>
</table>

[![Subscribe to DiamantAI Newsletter](assets/subscribe-button.svg)](https://diamantai.substack.com/?r=336pe4&utm_campaign=pub-share-checklist)

*Join over 20,000 of AI enthusiasts getting unique cutting-edge insights and free tutorials!* ***Plus, subscribers get exclusive early access and special 33% discounts to my book and the upcoming RAG Techniques course!***
</div>

[![DiamantAI's newsletter](assets/substack_image.png)](https://diamantai.substack.com/?r=336pe4&utm_campaign=pub-share-checklist)


<!-- https://github.com/NirDiamant/Controllable-RAG-Agent/blob/main/assets/video_demo.mp4 -->
<!-- [![YouTube](http://i.ytimg.com/vi/_73OV1z3sTg/hqdefault.jpg)](https://www.youtube.com/watch?v=_73OV1z3sTg) -->

## 🌟 Key Features

- **Sophisticated Deterministic Graph**: Acts as the "brain" of the agent, enabling complex reasoning.
- **Controllable Autonomous Agent**: Capable of answering non-trivial questions from custom datasets.
- **Hallucination Prevention**: Ensures answers are solely based on provided data, avoiding AI hallucinations.
- **Multi-step Reasoning**: Breaks down complex queries into manageable sub-tasks.
- **Adaptive Planning**: Continuously updates its plan based on new information.
- **Performance Evaluation**: Utilizes `Ragas` metrics for comprehensive quality assessment.


## 🧠 How It Works
![Solution Schema](graphs/final_graph_schema.jpeg)

1. **PDF Loading and Processing**: Load PDF documents and split them into chapters.
2. **Text Preprocessing**: Clean and preprocess the text for better summarization and encoding.
3. **Summarization**: Generate extensive summaries of each chapter using large language models.
4. **Book Quotes Database Creation**: Create a database for specific questions that will need access to quotes from the book.
5. **Vector Store Encoding**: Encode the book content and chapter summaries into vector stores for efficient retrieval.
6. **Question Processing**:
   - Anonymize the question by replacing named entities with variables.
   - Generate a high-level plan to answer the anonymized question.
   - De-anonymize the plan and break it down into retrievable or answerable tasks.
7. **Task Execution**:
   - For each task, decide whether to retrieve information or answer based on context.
   - If retrieving, fetch relevant information from vector stores and distill it.
   - If answering, generate a response using chain-of-thought reasoning.
8. **Verification and Re-planning**:
   - Verify that generated content is grounded in the original context.
   - Re-plan remaining steps based on new information.
9. **Final Answer Generation**: Produce the final answer using accumulated context and chain-of-thought reasoning.

## 📊 Evaluation

The solution is evaluated using `Ragas` metrics:
- Answer Correctness
- Faithfulness
- Answer Relevancy
- Context Recall
- Answer Similarity

## 🔍 Use Case: Harry Potter Book Analysis

The algorithm was tested using the first Harry Potter book, allowing for monitoring of the model's reliance on retrieved information versus pre-trained knowledge. This choice enables us to verify whether the model is using its pre-trained knowledge or strictly relying on the retrieved information from vector stores.

### Example Question
**Q: How did the protagonist defeat the villain's assistant?**

To solve this question, the following steps are necessary:

1. Identify the protagonist of the plot.
2. Identify the villain.
3. Identify the villain's assistant.
4. Search for confrontations or interactions between the protagonist and the villain.
5. Deduce the reason that led the protagonist to defeat the assistant.

The agent's ability to break down and solve such complex queries demonstrates its sophisticated reasoning capabilities.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- API key for your chosen LLM provider

### Installation (without Docker)

1. Clone the repository:
   ```sh
   git clone https://github.com/NirDiamant/Controllable-RAG-Agent.git
   cd Controllable-RAG-Agent
   ```
2. Set up environment variables:
   Create a `.env` file in the root directory with your API key: 
   ```
   OPENAI_API_KEY=
   GROQ_API_KEY=
   ```
   you can look at the `.env.example` file for reference.

## using Docker
3. run the following command to build the docker image
   ```sh
   docker-compose up --build
   ```

## Installation (without Docker)
3. Install required packages:
   ```sh
   pip install -r requirements.txt
   ```

### Usage

1. Explore the step-by-step tutorial: `sophisticated_rag_agent_harry_potter.ipynb`

2. Run real-time agent visualization (no docker):
   ```sh
   streamlit run simulate_agent.py
   ```

3. Run real-time agent visualization (with docker):
   open your browser and go to `http://localhost:8501/`

## 🛠️ Technologies Used

- LangChain
- FAISS Vector Store
- Streamlit (for visualization)
- Ragas (for evaluation)
- Flexible integration with various LLMs (e.g., OpenAI GPT models, Groq, or others of your choice)

## 💡 Heuristics and Techniques

1. Encoding both book content in chunks, chapter summaries generated by LLM, and quotes from the book.<br>
2. Anonymizing the question to create a general plan without biases or pre-trained knowledge of any LLM involved.<br>
3. Breaking down each task from the plan to be executed by custom functions with full control.<br>
4. Distilling retrieved content for better and accurate LLM generations, minimizing hallucinations.<br>
5. Answering a question based on context using a Chain of Thought, which includes both positive and negative examples, to arrive at a well-reasoned answer rather than just a straightforward response.<br>
6. Content verification and hallucination-free verification as suggested in "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" - https://arxiv.org/abs/2310.11511.<br>
7. Utilizing an ongoing updated plan made by an LLM to solve complicated questions. Some ideas are derived from "Plan-and-Solve Prompting" - https://arxiv.org/abs/2305.04091 and the "babyagi" project - https://github.com/yoheinakajima/babyagi.<br>
8. Evaluating the model's performance using `Ragas` metrics like answer correctness, faithfulness, relevancy, recall, and similarity to ensure high-quality answers.<br>

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## 📚 Learn More

- [Lecture Video](https://www.youtube.com/watch?v=orhV7ZMSRyM&t=33s&ab_channel=DiamantAI)
- [Blog Post Article](https://open.substack.com/pub/diamantai/p/controllable-agent-for-complex-rag?r=336pe4&utm_campaign=post&utm_medium=web)

## 🙏 Acknowledgements

Special thanks to Elad Levi for the valuable advice and ideas.

## 📄 License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

---

⭐️ If you find this repository helpful, please consider giving it a star!

Keywords: RAG, Retrieval-Augmented Generation, Agent, Langgraph, NLP, AI, Machine Learning, Information Retrieval, Natural Language Processing, LLM, Embeddings, Semantic Search
## FAQ

### What is the Controllable RAG Agent?

The Controllable RAG Agent is an advanced Retrieval-Augmented Generation (RAG) solution designed to tackle complex questions that simple semantic similarity-based retrieval cannot solve. It showcases a sophisticated deterministic graph acting as the "brain" of a highly controllable autonomous agent capable of answering non-trivial questions from your own data.

### How does it differ from traditional RAG systems?

- **Controllable RAG Agent**: Deterministic graph-based reasoning, multi-step task decomposition, hallucination prevention, adaptive planning, verification and re-planning loops
- **Traditional RAG**: Single-step semantic similarity retrieval, direct answer generation without complex reasoning, no verification mechanisms
- **LangChain RAG**: Chain-based retrieval, simpler orchestration, less sophisticated reasoning

The Controllable RAG Agent handles questions requiring multi-hop reasoning and complex task decomposition.

### What are the key features?

- **Sophisticated Deterministic Graph** — Acts as the "brain" of the agent, enabling complex reasoning
- **Controllable Autonomous Agent** — Capable of answering non-trivial questions from custom datasets
- **Hallucination Prevention** — Ensures answers are solely based on provided data, avoiding AI hallucinations
- **Multi-step Reasoning** — Breaks down complex queries into manageable sub-tasks
- **Adaptive Planning** — Continuously updates its plan based on new information
- **Performance Evaluation** — Uses Ragas metrics for comprehensive quality assessment

### How does the agent work?

1. **PDF Loading and Processing** — Load PDF documents and split into chapters
2. **Text Preprocessing** — Clean and preprocess text for better summarization and encoding
3. **Summarization** — Generate extensive summaries using large language models
4. **Book Quotes Database Creation** — Create database for specific questions needing quotes
5. **Vector Store Encoding** — Encode content and summaries into vector stores for efficient retrieval
6. **Question Processing**:
   - Anonymize question by replacing named entities with variables
   - Generate high-level plan for the anonymized question
   - De-anonymize and break down into retrievable/answerable tasks
7. **Task Execution**:
   - Decide whether to retrieve information or answer based on context
   - If retrieving: fetch from vector stores and distill
   - If answering: generate response using chain-of-thought reasoning
8. **Verification and Re-planning**:
   - Verify content is grounded in original context
   - Re-plan remaining steps based on new information
9. **Final Answer Generation** — Produce answer using accumulated context and chain-of-thought

### What evaluation metrics are used?

The solution is evaluated using Ragas metrics:
- **Answer Correctness** — Measures accuracy of the generated answer
- **Faithfulness** — Ensures answer is grounded in retrieved context
- **Answer Relevancy** — Measures how relevant the answer is to the question
- **Context Recall** — Measures coverage of relevant information
- **Answer Similarity** — Compares generated answer to reference

### How does the agent handle complex questions?

For questions like "How did the protagonist defeat the villain's assistant?", the agent:
1. Identifies the protagonist
2. Identifies the villain
3. Identifies the villain's assistant
4. Searches for confrontations between protagonist and villain
5. Deduces the reason for defeating the assistant

This demonstrates sophisticated reasoning beyond simple retrieval.

### What LLM providers are supported?

The agent is built on LangGraph and supports:
- **OpenAI** — used by default (`gpt-4o`)
- **Groq** — wired up in `functions_for_pipeline.py` for fast inference
- **Any LangChain-compatible provider** — swap the chat model in `functions_for_pipeline.py`

### How do I get started?

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your LLM API keys (OpenAI, Anthropic, etc.)
4. Prepare your PDF documents
5. Run the agent with your custom questions

See the [comprehensive guide on RAG techniques](https://github.com/NirDiamant/RAG_Techniques) for additional context.

### Can I use custom datasets?

Yes! The agent is designed to work with any PDF documents. Load your custom documents, and the agent will:
- Process and split them appropriately
- Create vector stores for efficient retrieval
- Answer complex questions based solely on your data

### How does hallucination prevention work?

The agent ensures answers are solely based on provided data through:
- **Verification loops** — Check generated content against original context
- **Re-planning** — Adjust approach if verification fails
- **Grounded responses** — Only use retrieved information, not pre-trained knowledge

### What is the anonymization technique?

Questions are anonymized by replacing named entities with variables. This helps:
- Generate more general plans
- Avoid bias from pre-trained knowledge about entities
- De-anonymize after planning to apply to specific context

### Where can I find related tutorials?

- **[Agents Towards Production](https://github.com/NirDiamant/agents-towards-production)** — Horizontal, code-first tutorials covering every step in building production-grade GenAI agents
- **[RAG Techniques](https://github.com/NirDiamant/RAG_Techniques)** — Comprehensive guide on RAG techniques
- **[GenAI Agents](https://github.com/NirDiamant/GenAI_Agents)** — Many other AI Agent implementations and tutorials

### What license does this project use?

Apache 2.0 License — open-source, free to use commercially.

### Where can I get help?

- **GitHub Issues**: [github.com/NirDiamant/Controllable-RAG-Agent/issues](https://github.com/NirDiamant/Controllable-RAG-Agent/issues)
- **Discord**: [Join our community](https://discord.gg/cA6Aa4uyDX)
- **Twitter**: [@NirDiamantAI](https://twitter.com/NirDiamantAI)
- **LinkedIn**: [Connect](https://www.linkedin.com/in/nir-diamant-759323134/)
- **Newsletter**: [DiamantAI Substack](https://diamantai.substack.com) — 20,000+ subscribers

