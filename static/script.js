document.addEventListener('DOMContentLoaded', () => {
    const ui = {
        modelSelect: document.getElementById('model-select'),
        languageSelect: document.getElementById('language-select'),
        transcribeButton: document.getElementById('transcribe-button'),
        clearJobsButton: document.getElementById('clear-jobs-button'),
        downloadSessionZipButton: document.getElementById('download-session-zip-button'),
        audioFilesInput: document.getElementById('audio-files'),
        jobsTableBody: document.getElementById('jobs-table-body'),
    };

    let state = {
        pollInterval: null,
        sessionId: null
    };

    function getOrSetSessionId() {
        state.sessionId = localStorage.getItem('transcription_session_id') || self.crypto.randomUUID();
        localStorage.setItem('transcription_session_id', state.sessionId);
        const headerP = document.querySelector('header p');
        headerP.title = `ID da Sessão: ${state.sessionId}`;
        headerP.textContent = `Interface de testes | Sessão: ${state.sessionId.substring(0, 8)}...`;
    }

    async function populateModels() {
        ui.modelSelect.disabled = true;
        ui.modelSelect.innerHTML = '<option>Carregando...</option>';
        try {
            const response = await fetch('/models');
            if (!response.ok) throw new Error('Falha ao buscar modelos.');
            const data = await response.json();
            
            ui.modelSelect.innerHTML = '';

            if (!data.available_models || data.available_models.length === 0) {
                ui.modelSelect.innerHTML = '<option>Nenhum modelo compatível com seu hardware</option>';
                return;
            }
            data.available_models.forEach(modelId => {
                const option = new Option(modelId, modelId);
                ui.modelSelect.appendChild(option);
            });
        } catch (error) {
            ui.modelSelect.innerHTML = '<option>Erro ao carregar</option>';
            console.error(error);
        } finally {
            ui.modelSelect.disabled = ui.modelSelect.options.length === 0;
        }
    }

    async function startTranscription() {
        if (ui.audioFilesInput.files.length === 0) {
            alert('Por favor, selecione ao menos um arquivo.');
            return;
        }

        setButtonState(true, 'Enviando...');
        const formData = new FormData();
        formData.append('session_id', state.sessionId);
        formData.append('model_id', ui.modelSelect.value);
        formData.append('language', ui.languageSelect.value);

        for (const file of ui.audioFilesInput.files) {
            formData.append('files', file);
        }

        try {
            const response = await fetch('/jobs', { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Erro desconhecido ao criar jobs.');

            data.jobs_created.forEach(job => addJobRowToTable(job));
            if (!state.pollInterval) {
                state.pollInterval = setInterval(pollQueueStatus, 2500);
            }
        } catch (error) {
            alert(`Erro ao criar jobs: ${error.message}`);
        } finally {
            setButtonState(false, '▶️ Iniciar Transcrição');
            ui.audioFilesInput.value = '';
        }
    }

    function addJobRowToTable(job) {
        if (document.getElementById(`job-row-${job.job_id}`)) return;
        const row = ui.jobsTableBody.insertRow(0);
        row.id = `job-row-${job.job_id}`;
        row.innerHTML = `
            <td title="${job.filename}">${job.filename.length > 40 ? '...' + job.filename.slice(-37) : job.filename}</td>
            <td class="job-status status-queued">Na Fila</td>
            <td class="progress-col">
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width: 0%;">0%</div>
                </div>
            </td>
            <td class="job-actions">
                <button class="action-button cancel-button" data-job-id="${job.job_id}">Cancelar</button>
            </td>
        `;
        const detailsRow = ui.jobsTableBody.insertRow(1);
        detailsRow.id = `details-row-${job.job_id}`;
        detailsRow.className = 'details-row';
        detailsRow.innerHTML = `<td colspan="4" class="details-cell"></td>`;
        row.querySelector('.cancel-button').addEventListener('click', (e) => cancelSingleJob(e.target.dataset.jobId));
    }

    async function pollQueueStatus() {
        const activeJobRows = document.querySelectorAll('.job-status.status-queued, .job-status.status-processing');
        if (activeJobRows.length === 0 && state.pollInterval) {
            clearInterval(state.pollInterval);
            state.pollInterval = null;
            return;
        }
        try {
            const response = await fetch(`/queues?session_ids=${state.sessionId}`);
            const jobs = await response.json();
            jobs.forEach(job => {
                const row = document.getElementById(`job-row-${job.job_id}`);
                if (!row) return;
                const statusCell = row.querySelector('.job-status');
                const newStatus = job.status;
                if (statusCell.dataset.status !== newStatus) {
                    statusCell.textContent = getStatusText(newStatus, job.eta_timestamp);
                    statusCell.className = `job-status status-${newStatus}`;
                    statusCell.dataset.status = newStatus;
                }
                const progressBar = row.querySelector('.progress-bar');
                progressBar.style.width = `${job.progress}%`;
                progressBar.textContent = `${job.progress}%`;
                const actionsCell = row.querySelector('.job-actions');
                if (newStatus === 'completed' && !actionsCell.querySelector('.view-result-button')) {
                    actionsCell.innerHTML = `<button class="action-button view-result-button" data-job-id="${job.job_id}">Ver Resultado</button>`;
                    actionsCell.querySelector('button').addEventListener('click', (e) => toggleJobDetails(e.target.dataset.jobId));
                } else if (['failed', 'cancelled'].includes(newStatus)) {
                    actionsCell.innerHTML = `<button class="action-button" disabled>${getStatusText(newStatus)}</button>`;
                }
            });
        } catch (error) {
            console.error("Erro ao atualizar fila:", error);
        }
    }
    
    async function cancelSingleJob(jobId) {
        const button = document.querySelector(`.cancel-button[data-job-id="${jobId}"]`);
        if (button) {
            button.disabled = true;
            button.textContent = 'Cancelando...';
        }
        try {
            await fetch(`/jobs/${jobId}/cancel`, { method: 'POST' });
        } catch(error) {
            alert("Erro ao enviar sinal de cancelamento.");
            if (button) {
                button.disabled = false;
                button.textContent = 'Cancelar';
            }
        }
    }

    async function toggleJobDetails(jobId) {
        const detailsRow = document.getElementById(`details-row-${jobId}`);
        const detailsCell = detailsRow.querySelector('.details-cell');
        
        if (detailsRow.style.display === 'table-row') {
            detailsRow.style.display = 'none';
            detailsCell.innerHTML = '';
        } else {
            detailsCell.innerHTML = '<p>Carregando resultado...</p>';
            detailsRow.style.display = 'table-row';
            try {
                const response = await fetch(`/jobs/${jobId}`);
                if (!response.ok) throw new Error('Job não encontrado no servidor.');
                const job = await response.json();
                if (!job.result) throw new Error("Detalhes do resultado não encontrados no job.");
                detailsCell.innerHTML = createResultDetailsHTML(job.result, jobId);
                addTabListeners(detailsCell);
                addDownloadListeners(detailsCell);
            } catch(error) {
                detailsCell.innerHTML = `<p style="color: var(--error-color);">Erro ao buscar detalhes: ${error.message}</p>`;
            }
        }
    }

    function createResultDetailsHTML(result, jobId) {
        const resultId = `result-content-${jobId}`;
        return `
            <div class="details-content">
                <div class="result-tabs">
                    <button class="tab-button active" data-target="${resultId}-dialogue-md">Diálogo Formatado</button>
                    <button class="tab-button" data-target="${resultId}-raw">Texto Puro</button>
                </div>
                <div class="result-body">
                    <div id="${resultId}-dialogue-md" class="tab-content"><pre>${escapeHtml(result.transcription_dialogue_markdown)}</pre></div>
                    <div id="${resultId}-raw" class="tab-content" style="display: none;"><pre>${escapeHtml(result.transcription_raw)}</pre></div>
                    <div class="download-buttons">
                        <button class="download-btn" data-job="${jobId}" data-type="transcription_dialogue_markdown">Baixar .txt (Formatado)</button>
                        <button class="download-btn" data-job="${jobId}" data-type="transcription_raw">Baixar .txt (Puro)</button>
                    </div>
                </div>
            </div>`;
    }
    
    function addTabListeners(parentElement) {
        parentElement.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', e => {
                const targetId = e.target.dataset.target;
                const contentArea = e.target.closest('.details-content');
                contentArea.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
                e.target.classList.add('active');
                contentArea.querySelectorAll('.tab-content').forEach(content => content.style.display = 'none');
                document.getElementById(targetId).style.display = 'block';
            });
        });
    }

    function addDownloadListeners(parentElement) {
        parentElement.querySelectorAll('.download-btn').forEach(button => {
            button.addEventListener('click', e => {
                const { job, type } = e.target.dataset;
                window.open(`/jobs/${job}/download?text_type=${type}`, '_blank');
            });
        });
    }

    function getStatusText(status, eta) {
        const statusMap = {
            queued: 'Na Fila', processing: 'Processando', completed: 'Concluído',
            failed: 'Falhou', cancelled: 'Cancelado'
        };
        if (status === 'processing' && eta) {
            const etaDate = new Date(eta * 1000);
            const formattedEta = etaDate.toLocaleTimeString('pt-BR');
            return `Processando (ETA: ${formattedEta})`;
        }
        return statusMap[status] || status;
    }
    
    function setButtonState(disabled, text) {
        ui.transcribeButton.disabled = disabled;
        ui.transcribeButton.textContent = text;
    }

    function clearJobs() {
        if (!confirm("Tem certeza que deseja limpar todos os jobs do painel?")) return;
        ui.jobsTableBody.innerHTML = '';
        if (state.pollInterval) {
            clearInterval(state.pollInterval);
            state.pollInterval = null;
        }
    }
    
    function downloadSessionZip() {
        if (!state.sessionId) {
            alert("ID da sessão não encontrado.");
            return;
        }
        const completedJobs = document.querySelectorAll('.job-status.status-completed');
        if (completedJobs.length === 0) {
            alert("Nenhum job foi concluído nesta sessão para ser baixado.");
            return;
        }
        console.log(`Iniciando download do ZIP para a sessão: ${state.sessionId}`);
        window.open(`/jobs/download/session/${state.sessionId}`, '_blank');
    }

    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }

    // --- Inicialização ---
    if (ui.transcribeButton) {
        ui.transcribeButton.addEventListener('click', startTranscription);
    }
    if (ui.clearJobsButton) {
        ui.clearJobsButton.addEventListener('click', clearJobs);
    }
    if (ui.downloadSessionZipButton) {
        ui.downloadSessionZipButton.addEventListener('click', downloadSessionZip);
    }

    getOrSetSessionId();
    populateModels();
});