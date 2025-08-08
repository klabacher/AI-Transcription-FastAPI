# 🚀 API de Transcrição Híbrida (FaaS-like)

Uma API de transcrição de áudio robusta, local e assíncrona, construída em Python com FastAPI. O sistema foi projetado para ser "Function as a Service"-like, permitindo o processamento de áudios em background, monitoramento de filas e gerenciamento automático de cache, com uma interface web simples para testes e uso prático.

> ℹ️ **Nota de Colaboração com IA**
> Este projeto foi desenvolvido em uma colaboração interativa entre um desenvolvedor humano e a inteligência artificial **Gemini**, do Google. A IA foi utilizada como uma parceira de programação para arquitetar a solução, refatorar código, gerar implementações e documentar as funcionalidades.

> ⚠️ **Atribuição e Propriedade dos Modelos**
> Este projeto é uma **ferramenta de orquestração** e não reivindica a propriedade de nenhum dos modelos de machine learning que utiliza. Todo o crédito pertence aos seus respectivos criadores e mantenedores:
> * **Whisper:** Desenvolvido e liberado pela [OpenAI](https://openai.com/research/whisper).
> * **Faster-Whisper:** Uma reimplementação otimizada criada por [Guillaume Luga](https://github.com/guillaumekln/faster-whisper).
> * **Distil-Whisper:** Modelos destilados e otimizados, disponibilizados pela [Hugging Face](https://huggingface.co/distil-whisper) e pela comunidade (ex: [freds0](https://huggingface.co/freds0/distil-whisper-large-v3-ptbr)).
> * **AssemblyAI:** Uma plataforma comercial de Speech-to-Text via API, propriedade da [AssemblyAI, Inc](https://www.assemblyai.com/).

---

## ✨ Features Principais

* **Motor Híbrido:** Suporte nativo para múltiplos backends:
    * **Modelos Locais:** Rode transcrições na sua própria máquina (CPU ou GPU).
    * **Nuvem:** Integre com serviços de ponta como o **AssemblyAI**.
* **Seleção Inteligente de Dispositivo:** Detecção automática de GPU (CUDA) para máxima performance, com a opção de forçar o uso de CPU (`AUTOMATICO`/`GPU`/`CPU`).
* **Processamento Assíncrono:** Envie um ou vários áudios (inclusive `.zip`) e receba um `job_id` imediatamente. A API processa tudo em background, sem travar.
* **Interface Web (HUD):** Um painel de controle simples e funcional em `/ui` para testar todas as funcionalidades da API sem precisar de código.
* **Monitoramento de Filas:** Um endpoint (`/queues`) que permite visualizar todos os jobs em andamento, sua porcentagem e tempo estimado de conclusão (ETA), com filtros por usuário.
* **Sistema de Sessões Descentralizado:** Agrupe jobs por usuário ou cliente através de uma `session_id` anônima, sem necessidade de login ou banco de dados.
* **Limpeza Automática de Cache (Janitor):** Um "faxineiro" roda em background para apagar resultados de jobs antigos, mantendo a memória da API eficiente.
* **Downloads de Resultados:** Baixe as transcrições em múltiplos formatos: `.txt` individuais, um arquivo único concatenado ou um pacote `.zip` completo.
* **Setup Automático:** Na inicialização, a API verifica e baixa automaticamente os modelos locais necessários.

## 🛠️ Tecnologias Utilizadas

* **Backend:** Python 3.10+
* **Framework API:** FastAPI
* **Servidor ASGI:** Uvicorn
* **Machine Learning & Áudio:**
    * PyTorch
    * OpenAI-Whisper
    * Faster-Whisper
    * Transformers (Hugging Face)
* **APIs Externas:** AssemblyAI
* **Interface (HUD):** HTML5, CSS3, JavaScript (Vanilla)

---

## 🚀 Começando

Siga os passos abaixo para ter a API rodando na sua máquina.

### 1. Pré-requisitos

* Python 3.10 ou superior.
* `git` para clonar o repositório.
* (Opcional, mas recomendado para performance) Uma GPU NVIDIA com CUDA instalado para processamento com os modelos locais.

### 2. Instalação

Clone o repositório para a sua máquina local:
```bash
git clone <URL_DO_SEU_REPOSITORIO_GIT>
cd transcription_api
```

Crie e ative um ambiente virtual (recomendado):
```bash
# Criar o ambiente
python -m venv venv

# Ativar no Linux/macOS
source venv/bin/activate

# Ativar no Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

Instale todas as dependências do projeto:
```bash
pip install -r requirements.txt
```
*Nota: A primeira instalação pode demorar, pois baixa bibliotecas pesadas como o PyTorch.*

### 3. Configuração

As configurações principais podem ser ajustadas no arquivo `config.py`:
* `JOB_RETENTION_TIME_SECONDS`: Tempo (em segundos) que um job finalizado fica na memória antes de ser limpo. (Padrão: 3600s = 1 hora)
* `JANITOR_SLEEP_INTERVAL_SECONDS`: Frequência (em segundos) com que o processo de limpeza roda. (Padrão: 300s = 5 minutos)

---

## ▶️ Executando a API

Com o ambiente virtual ativado, inicie o servidor com Uvicorn:

```bash
uvicorn main:app --reload
```
Para debug:
```bash
DEBUG=true uvicorn main:app --reload
```
* O comando `--reload` reinicia o servidor automaticamente sempre que você salvar uma alteração no código.

Na primeira vez que você iniciar, a API vai **baixar e configurar todos os modelos locais**. Isso pode levar vários minutos e consumir alguns gigabytes de espaço em disco. Nas inicializações seguintes, o processo será quase instantâneo.

Quando o servidor estiver pronto, você verá uma mensagem como:
`INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`

## 🕹️ Como Usar

A aplicação oferece três pontos de acesso principais:

### 1. Interface de Usuário (HUD)
A maneira mais fácil de usar e testar.
* **Acesse:** [**http://127.0.0.1:8000/ui**](http://127.0.0.1:8000/ui)
* A HUD permite selecionar o motor, o modelo, enviar arquivos (inclusive `.zip`), configurar opções e ver os resultados e downloads em tempo real.

### 2. Documentação da API (Swagger UI)
Para desenvolvedores que querem interagir diretamente com os endpoints.
* **Acesse:** [**http://127.0.0.1:8000/docs**](http://127.0.0.1:8000/docs)
* Você pode testar cada endpoint, ver os parâmetros necessários e os modelos de resposta.

### 3. Monitoramento de Filas
Para ver o status de todos os jobs no sistema.
* **Acesse:** [**http://127.0.0.1:8000/queues**](http://127.0.0.1:8000/queues)
* Para filtrar por usuários específicos, adicione os `session_id` na URL:
    `http://127.0.0.1:8000/queues?session_ids=ID_SESSAO_1,ID_SESSAO_2`

---

## 🏛️ Arquitetura da Solução

* **API com FastAPI:** Escolhido pela alta performance, tipagem de dados com Pydantic e geração automática de documentação.
* **Jobs em Background:** Para tarefas de longa duração como a transcrição, a API usa `BackgroundTasks` para liberar o cliente imediatamente, evitando timeouts.
* **Gerenciamento de Estado em Memória:** O estado dos jobs e as filas são mantidos em um dicionário Python. Esta é uma solução simples e eficaz para uma aplicação local e de instância única.
* **Janitor com Threading:** Um processo de limpeza é disparado em uma thread separada durante o ciclo de vida da aplicação (`lifespan`) para não bloquear o servidor principal.

## 📜 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.