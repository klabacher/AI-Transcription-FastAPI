# 🚀 API de Transcrição Otimizada

[![Feito com FastAPI](https://img.shields.io/badge/Feito%20com-FastAPI-blue.svg)](https://fastapi.tiangolo.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Licença: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

A escolha das tecnologias e da arquitetura foi pensada para criar um sistema desacoplado e escalável.

- **FastAPI**: Escolhido pela sua alta performance, documentação automática (Swagger UI) e sintaxe moderna com `async/await`.
- **Multiprocessing**: A decisão mais crítica da arquitetura. Usamos o módulo `multiprocessing` do Python para isolar os workers de IA do processo principal da API. Isso evita que o consumo intenso de memória e CPU dos modelos de transcrição trave o servidor web, garantindo que a API esteja sempre disponível para receber novas requisições.
- **Gerenciamento de Ciclo de Vida (Lifespan)**: O FastAPI `lifespan` é usado para iniciar os pools de workers e a thread de limpeza (`janitor`) quando a API sobe, e para garantir um desligamento gracioso, finalizando todos os processos de forma segura.
- **Vanilla JS/HTML/CSS**: A interface foi mantida intencionalmente simples, sem frameworks complexos, para ser leve e fácil de entender, focando na funcionalidade.
- **IA e Áudio**:
    - **`faster-whisper`**: Para transcrições otimizadas de alta velocidade em CPU e GPU.
    - **`transformers`**: Para carregar modelos do Hugging Face Hub, como o `distil-whisper`.
    - **`torch`**: A base para todos os modelos de IA.
    - **`soundfile`**: Para ler informações de metadados dos arquivos de áudio, como a duração.

## 🧠 Modelos Disponíveis

A API disponibiliza os seguintes modelos, ativados conforme o hardware detectado:

| Model ID | Implementação | Requer GPU? | Descrição |
| :--- | :--- | :--- | :--- |
| `distil_large_v3_ptbr` | Hugging Face | ➖ Não | Recomendado para testes locais. Ótima qualidade em PT-BR, leve e rápido em CPU. |
| `faster_medium_fp16` | faster-whisper | ✅ Sim | Excelente equilíbrio entre velocidade e qualidade em GPU. |
| `faster_large-v3_fp16` | faster-whisper | ✅ Sim | Máxima qualidade e precisão em PT-BR. Requer GPU potente (VRAM > 8GB). |
| `faster_large-v3_int8` | faster-whisper | ➖ Não | Qualidade do Large-v3 com menor uso de memória. Ideal para CPUs potentes ou GPUs com VRAM limitada. |

## 🚀 Como Usar

### Pré-requisitos
- Python 3.10 ou superior.
- (Opcional, mas recomendado) Uma placa de vídeo NVIDIA com drivers CUDA instalados para performance máxima.

### Instalação e Execução
1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS / Linux
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Inicie a API:**
    ```bash
    uvicorn main:app --reload
    ```
    > ⚠️ **Atenção na Primeira Execução!**
    > Na primeira vez que você iniciar a API, um script de setup será executado para baixar todos os modelos de IA. Este processo pode demorar **vários minutos** e consumir um espaço considerável em disco. As inicializações seguintes serão quase instantâneas.

5.  **Acesse a aplicação:**
    - **Interface Web**: Abra seu navegador em `http://127.0.0.1:8000/ui`
    - **Documentação da API**: Acesse `http://127.0.0.1:8000/docs`

## 📡 Endpoints da API

- `GET /`: Retorna uma mensagem de status e informações sobre o hardware detectado.
- `GET /models`: Lista os modelos de transcrição que estão ativos e compatíveis com o hardware atual.
- `GET /queues`: Monitora o status de todos os jobs na fila.
- `POST /jobs`: Cria um ou mais jobs de transcrição. Recebe `model_id`, `language`, `session_id` e os arquivos de áudio (`files`).
- `GET /jobs/{job_id}`: Verifica o status, progresso e resultado de um job específico.
- `POST /jobs/{job_id}/cancel`: Solicita o cancelamento de um job que está na fila ou em processamento.
- `GET /jobs/{job_id}/download`: Baixa o resultado da transcrição em um arquivo `.txt`.

## ❤️ Créditos e Agradecimentos

Este projeto só é possível graças ao incrível trabalho da comunidade open-source.

- **Modelos de IA**:
    - **Whisper e faster-whisper**: Agradecimentos à [OpenAI](https://openai.com/) pelo modelo Whisper original e a [Guillaume Klein (SYSTRAN)](https://github.com/guillaumekln/faster-whisper) pela implementação otimizada `faster-whisper`.
    - **distil-whisper-large-v3-ptbr**: Obrigado ao usuário [freds0](https://huggingface.co/freds0) do Hugging Face por treinar e disponibilizar a versão destilada para português.

- **Ferramentas**:
    - [FastAPI](https://fastapi.tiangolo.com/) por ser um framework web fantástico.
    - [PyTorch](https://pytorch.org/) e [Hugging Face](https://huggingface.co/) por democratizarem o acesso a modelos de Machine Learning.

---

> **Aviso**: Este projeto foi desenvolvido com o auxílio de Inteligência Artificial para acelerar a criação de código, resolver problemas e gerar documentação.