# 🚀 Optimized Transcription API

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klabacher/your-repo-name/blob/main/V3_Colab_Demo.ipynb)
[![Made with FastAPI](https://img.shields.io/badge/Made%20with-FastAPI-blue.svg)](https://fastapi.tiangolo.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Ler em Português](./README-PT.md)

A high-performance audio transcription API built with FastAPI, focused on providing access to state-of-the-art models like `faster-whisper` and optimized versions from Hugging Face.

The project is designed to be robust and efficient, using isolated worker processes to handle heavy AI processing, ensuring the main API remains responsive at all times.

*(A GIF demonstrating the web interface would be great here!)*

## ✨ Key Features

- **Persistent Workers**: AI models are loaded only once in separate worker processes, eliminating load times on each request and optimizing resource usage (CPU/GPU).
- **Job Queue System**: Transcription tasks are queued and processed asynchronously, allowing the API to handle a high volume of requests.
- **Hardware Detection**: The API automatically detects the presence of a GPU (CUDA) and its capabilities (like FP16 support) to enable the most performant models.
- **Automated Setup**: On the first run, a setup script downloads and caches all necessary AI models, speeding up future initializations.
- **Testing UI**: A simple and functional web interface (`/ui`) to test the API, upload files, track job progress, and view results.
- **Multi-file & .ZIP Processing**: Upload multiple audio files or a single `.zip` file containing audios to create multiple jobs at once.
- **Monitoring and Management**: Endpoints to check the status of specific jobs, cancel tasks in progress, and automatically clean up old jobs.

## 🛠️ Tech Stack & Architectural Choices

- **FastAPI**: For high-performance APIs and automatic documentation.
- **Multiprocessing**: To isolate AI workers from the main API process, ensuring responsiveness.
- **Vanilla JS/HTML/CSS**: For a lightweight, functional, and easy-to-understand UI.
- **AI & Audio**: `faster-whisper`, `transformers`, `torch`, and `soundfile`.

## 🧠 Available Models

| Model ID                 | Implementation   | GPU Required? | Description                                                                    |
| :----------------------- | :--------------- | :-----------: | :----------------------------------------------------------------------------- |
| `distil_large_v3_ptbr`   | Hugging Face     |      ➖ No      | Recommended for local testing. Great quality in PT-BR, light and fast on CPU.  |
| `faster_medium_fp16`     | faster-whisper   |      ✅ Yes     | Excellent balance between speed and quality on GPU.                            |
| `faster_large-v3_fp16`   | faster-whisper   |      ✅ Yes     | Maximum quality and precision in PT-BR. Requires a powerful GPU (VRAM > 8GB).  |
| `faster_large-v3_int8`   | faster-whisper   |      ➖ No      | Large-v3 quality with lower memory usage. Ideal for powerful CPUs or GPUs with limited VRAM. |

## 🚀 Getting Started

You can run this project using Docker (recommended) or by setting up a local Python environment.

### Prerequisites
- Python 3.10+
- Docker and Docker Compose (for the containerized setup).
- (Optional) NVIDIA GPU with CUDA drivers for best performance.

### 🐳 Option 1: Running with Docker (Recommended)
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/klabacher/your-repo-name.git # Replace with your repo URL
    cd your-repo-name
    ```
2.  **Build and run:**
    ```bash
    docker-compose up --build
    ```
    > **Note:** The first launch will take several minutes to download AI models.

3.  **Access the application:**
    - **Web UI**: `http://127.0.0.1:8000/ui`
    - **API Docs**: `http://127.0.0.1:8000/docs`

### 🐍 Option 2: Local Python Environment
1.  **Clone, create a virtual environment, and activate it.**
2.  **Install dependencies:** `pip install -r requirements.txt`
3.  **Start the API:** `uvicorn main:app --reload`
4.  **Access the application** at the same URLs listed above.

## ☁️ Cloud Usage (Google Colab)

For users who want to test the API without a local setup, a Google Colab notebook is provided (`V3_Colab_Demo.ipynb`). This allows you to run the entire application on Google's cloud infrastructure for free.

**How to use it:**

1.  **Open in Google Colab:**
    *   Click the "Open in Colab" badge at the top of this README or manually open `V3_Colab_Demo.ipynb` in [Google Colab](https://colab.research.google.com/) via the GitHub tab.

2.  **Select a GPU Runtime (Recommended):**
    *   In Colab, go to `Runtime` -> `Change runtime type` and select a `T4 GPU`.

3.  **Run the Cells:**
    *   Execute the notebook cells in order. The notebook will guide you through installing dependencies, starting the API, and sending a test job.

## 📡 API Endpoints

- `GET /`: API status and hardware info.
- `GET /models`: List available models.
- `GET /queues`: Monitor job queues.
- `POST /jobs`: Create transcription jobs.
- `GET /jobs/{job_id}`: Check job status.
- `POST /jobs/{job_id}/cancel`: Cancel a job.
- `GET /jobs/{job_id}/download`: Download transcription result.

## 📄 License

This project was created by **[klabacher](https://github.com/klabacher)** and is licensed under the **MIT License**. We ask that you please provide attribution if you use this code.

## ❤️ Credits and Acknowledgements

- **Creator**:
    - **[klabacher](https://github.com/klabacher)**: Initial development and project architecture.
- **AI Models**:
    - Thanks to **OpenAI**, **Guillaume Klein (SYSTRAN)**, and **freds0** on Hugging Face.
- **Tools**:
    - [FastAPI](https://fastapi.tiangolo.com/), [PyTorch](https://pytorch.org/), and [Hugging Face](https://huggingface.co/).

---

> **Disclaimer**: This project was developed with the assistance of Artificial Intelligence.