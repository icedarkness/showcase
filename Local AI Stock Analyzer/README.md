# Project 5: Local AI Stock Analyzer

This project demonstrates a powerful, end-to-end application that leverages a locally deployed Large Language Model (LLM) to perform real-time financial data analysis. The entire AI reasoning engine runs on local hardware, ensuring data privacy and offline capability.

The application allows a user to input a stock ticker, and the local LLM provides a qualitative analysis based on the latest market data, such as summarizing market sentiment from recent news or identifying key trends.

![App Screenshot](https://placehold.co/800x500/1f2937/ffffff?text=App+Screenshot+Placeholder)
*(Note: This is a placeholder. You can replace it with a screenshot of your finished application.)*

---

## ✨ Features

* **Local LLM Deployment:** Uses **Ollama** to serve a powerful 7-billion parameter model directly from your local machine.
* **Real-Time Data:** Integrates with the `yfinance` library to pull up-to-the-minute stock prices and news headlines.
* **AI-Powered Analysis:** Leverages the LLM's reasoning capabilities to perform tasks like:
    * **Sentiment Analysis** on financial news.
    * **Trend Identification** from recent price action.
    * **Natural Language Q&A** about a specific stock.
* **Full-Stack Architecture:** Combines a Python (Flask) backend for data fetching and AI logic with a clean, modern web interface.
* **Data Privacy:** All data processing and AI inference happen locally. No data is sent to third-party cloud services.

---

## 🏛️ Architecture

This project consists of three main components that work together:

1.  **The AI Engine (Ollama):**
    * Ollama is a lightweight, powerful tool that serves open-source LLMs (like DeepSeek, Llama 3) on a local API endpoint (`http://localhost:11434`).
    * It handles all the complexity of model management and GPU resource allocation.
    * For this project, we use the `deepseek-llm:7b-chat` model, which offers an excellent balance of performance and resource requirements for general-purpose tasks.

2.  **The Backend (Python/Flask):**
    * A simple Flask web server acts as the bridge between the front-end and the AI engine.
    * It receives requests from the user interface (e.g., "Analyze AAPL").
    * It fetches the required financial data using the `yfinance` library.
    * It constructs a detailed prompt with the fetched data and sends it to the Ollama API.
    * It relays the LLM's response back to the front-end.

3.  **The Frontend (HTML/CSS/JS):**
    * A single-page web interface where the user can interact with the application.
    * It allows the user to input a stock ticker and select an analysis type.
    * It dynamically displays the results received from the backend.

---

## 🚀 Deployment and Setup Instructions

Follow these steps to get the application running on your local machine.

### Prerequisites

* A powerful GPU with at least 8GB of VRAM (NVIDIA 4090 recommended).
* [Python 3.8+](https://www.python.org/downloads/) installed.
* [Ollama](https://ollama.com/) installed on your machine.
* `git` for cloning the repository.

### Part 1: Setting up the Local LLM with Ollama

This is the core of the AI engine. Ollama makes running powerful LLMs on your local machine incredibly simple.

1.  **Download and Install Ollama:**
    * Go to [ollama.com](https://ollama.com/) and download the application for your operating system (Windows, macOS, or Linux).
    * Follow the installation instructions. After installation, Ollama will be running in the background.

2.  **Download the AI Model:**
    * Open your terminal or command prompt.
    * Run the following command to download and run the **DeepSeek 7B Chat** model. This is a powerful general-purpose model that is perfect for this task. The download will be several gigabytes, so it may take some time.

    ```bash
    ollama run deepseek-llm:7b-chat
    ```

3.  **Verify the Model:**
    * After the command completes, you will be in an interactive chat session with the model in your terminal. You can ask it a question like "Hello, who are you?" to confirm it's working.
    * Ollama will now be serving this model on a local API endpoint. You can leave this terminal running or close it; Ollama will continue to serve the model in the background.

### Part 2: Setting up the Python Backend

This server will handle data fetching and communication with the Ollama API.

1.  **Clone the Project Repository:**
    ```bash
    git clone [your-repository-url]
    cd [your-repository-directory]
    ```

2.  **Create a Python Virtual Environment:**
    * It's best practice to create a virtual environment to manage project dependencies.
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    * The required Python libraries are listed in the `requirements.txt` file. Install them using pip.
    ```bash
    pip install -r requirements.txt
    ```
    *(The `requirements.txt` file should contain `Flask`, `requests`, and `yfinance`.)*

### Part 3: Running the Application

1.  **Start the Python Backend:**
    * In your terminal (with the virtual environment activated), run the Flask application.
    ```bash
    python app.py
    ```
    * You should see output indicating that the server is running on `http://127.0.0.1:5000`. Keep this terminal window open.

2.  **Launch the Frontend:**
    * Navigate to the project directory in your file explorer.
    * Open the `index.html` file in a modern web browser (like Google Chrome or Firefox).

You should now see the application interface, ready to accept stock tickers for analysis.
