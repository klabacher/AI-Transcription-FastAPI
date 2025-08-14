# 🚀 API de Transcrição Otimizada

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klabacher/your-repo-name/blob/main/V3_Colab_Demo.ipynb)
[![Feito com FastAPI](https://img.shields.io/badge/Feito%20com-FastAPI-blue.svg)](https://fastapi.tiangolo.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Licença: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Read in English](./README.md)

Uma API de transcrição de áudio de alta performance, construída com FastAPI e focada em oferecer acesso a modelos de *state-of-the-art* como `faster-whisper` e versões otimizadas do Hugging Face.

O projeto foi desenhado para ser robusto e eficiente, utilizando workers em processos isolados para lidar com o processamento pesado de IA, garantindo que a API principal permaneça sempre responsiva.

*(Um GIF demonstrando a interface web seria ótimo aqui!)*

## ✨ Features Principais

- **Workers Persistentes**: Modelos de IA são carregados uma única vez em processos worker separados, eliminando o tempo de carregamento a cada requisição e otimizando o uso de recursos (CPU/GPU).
- **Sistema de Fila de Jobs**: As tarefas de transcrição são enfileiradas e processadas de forma assíncrona, permitindo que a API lide com um grande volume de requisições.
- **Detecção de Hardware**: A API detecta automaticamente a presença de uma GPU (CUDA) e suas capacidades (como suporte a FP16) para ativar os modelos mais performáticos.
- **Setup Automatizado**: Na primeira execução, um script de setup baixa e armazena em cache todos os modelos de IA necessários, agilizando as inicializações futuras.
- **Interface de Testes (UI)**: Uma interface web simples e funcional (`/ui`) para testar a API, enviar arquivos, acompanhar o progresso dos jobs e visualizar os resultados.
- **Processamento de Múltiplos Arquivos e .ZIP**: Envie vários arquivos de áudio ou um único arquivo `.zip` contendo os áudios para criar múltiplos jobs de uma só vez.
- **Monitoramento e Gerenciamento**: Endpoints para verificar o status de jobs específicos, cancelar tarefas em andamento e limpar jobs antigos automaticamente.

## 🛠️ Stack e Escolhas de Arquitetura

- **FastAPI**: Para APIs de alta performance e documentação automática.
- **Multiprocessing**: Para isolar os workers de IA do processo principal da API, garantindo a responsividade.
- **Vanilla JS/HTML/CSS**: Para uma UI leve, funcional e fácil de entender.
- **IA e Áudio**: `faster-whisper`, `transformers`, `torch`, e `soundfile`.

## 🧠 Modelos Disponíveis

| Model ID | Implementação | Requer GPU? | Descrição |
| :--- | :--- | :--- | :--- |
| `distil_large_v3_ptbr` | Hugging Face | ➖ Não | Recomendado para testes locais. Ótima qualidade em PT-BR, leve e rápido em CPU. |
| `faster_medium_fp16` | faster-whisper | ✅ Sim | Excelente equilíbrio entre velocidade e qualidade em GPU. |
| `faster_large_v3_fp16` | faster-whisper | ✅ Sim | Máxima qualidade e precisão em PT-BR. Requer GPU potente (VRAM > 8GB). |
| `faster_large-v3_int8` | faster-whisper | ➖ Não | Qualidade do Large-v3 com menor uso de memória. Ideal para CPUs potentes ou GPUs com VRAM limitada. |

## 🚀 Como Usar

Você pode executar este projeto usando Docker (recomendado) ou configurando um ambiente Python local.

### Pré-requisitos
- Python 3.10+
- Docker e Docker Compose (para a configuração com containers).
- (Opcional) GPU NVIDIA com drivers CUDA para a melhor performance.

### 🐳 Opção 1: Executando com Docker (Recomendado)
1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/klabacher/your-repo-name.git # Substitua pela URL do seu repositório
    cd your-repo-name
    ```
2.  **Construa e execute:**
    ```bash
    docker-compose up --build
    ```
    > **Nota:** O primeiro lançamento levará vários minutos para baixar os modelos de IA.

3.  **Acesse a aplicação:**
    - **UI Web**: `http://127.0.0.1:8000/ui`
    - **Docs da API**: `http://127.0.0.1:8000/docs`

### 🐍 Opção 2: Ambiente Python Local
1.  **Clone, crie um ambiente virtual e ative-o.**
2.  **Instale as dependências:** `pip install -r requirements.txt`
3.  **Inicie a API:** `uvicorn main:app --reload`
4.  **Acesse a aplicação** nas mesmas URLs listadas acima.

## ☁️ Uso na Nuvem (Google Colab)

Para usuários que desejam testar a API sem uma configuração local, um notebook do Google Colab é fornecido (`V3_Colab_Demo.ipynb`). Ele permite que você execute a aplicação completa na infraestrutura de nuvem do Google gratuitamente.

**Como usar:**

1.  **Abra no Google Colab:**
    *   Clique no selo "Open in Colab" no topo deste README ou abra manualmente o `V3_Colab_Demo.ipynb` no [Google Colab](https://colab.research.google.com/) através da aba GitHub.

2.  **Selecione um Ambiente com GPU (Recomendado):**
    *   No Colab, vá em `Ambiente de execução` -> `Alterar o tipo de ambiente de execução` e selecione `T4 GPU`.

3.  **Execute as Células:**
    *   Execute as células do notebook em ordem. O notebook irá guiá-lo na instalação de dependências, inicialização da API e envio de um job de teste.

## 📡 Endpoints da API

- `GET /`: Status da API e informações de hardware.
- `GET /models`: Lista os modelos disponíveis.
- `GET /queues`: Monitora as filas de jobs.
- `POST /jobs`: Cria jobs de transcrição.
- `GET /jobs/{job_id}`: Verifica o status do job.
- `POST /jobs/{job_id}/cancel`: Cancela um job.
- `GET /jobs/{job_id}/download`: Baixa o resultado da transcrição.

## 📄 Licença

Este projeto foi criado por **[klabacher](https://github.com/klabacher)** e está licenciado sob a **Licença MIT**. Pedimos que, por favor, forneça atribuição ao usar este código.

## ❤️ Créditos e Agradecimentos

- **Criador**:
    - **[klabacher](https://github.com/klabacher)**: Desenvolvimento inicial e arquitetura do projeto.
- **Modelos de IA**:
    - Agradecimentos a **OpenAI**, **Guillaume Klein (SYSTRAN)**, e **freds0** no Hugging Face.
- **Ferramentas**:
    - [FastAPI](https://fastapi.tiangolo.com/), [PyTorch](https://pytorch.org/), e [Hugging Face](https://huggingface.co/).

---

> **Aviso**: Este projeto foi desenvolvido com o auxílio de Inteligência Artificial.
