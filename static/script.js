document.addEventListener('DOMContentLoaded', () => {
    const ui = {
        engineSelect: document.getElementById('engine-select'),
        modelSelect: document.getElementById('model-select'),
        languageSelect: document.getElementById('language-select'),
        deviceChoice: document.getElementById('device-choice'),
        assemblyaiKeyGroup: document.getElementById('assemblyai-key-group'),
        featuresCard: document.getElementById('features-card'),
        deviceChoiceGroup: document.getElementById('device-choice-group'),
        transcribeButton: document.getElementById('transcribe-button'),
        clearJobsButton: document.getElementById('clear-jobs-button'),
        cancelAllButton: document.getElementById('cancel-all-button'),
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
        headerP.textContent = `Interface de testes para a API. | Sessão: ${state.sessionId.substring(0, 8)}...`;
    }

    async function populateModels() {
        ui.modelSelect.disabled = true;
        ui.modelSelect.innerHTML = '<option>Carregando...</option>';
        try {
            const endpoint = `/models`;
            const response = await fetch(endpoint);
            if (!response.ok) throw new Error('Falha ao buscar modelos.');
            const data = await response.json();
            
            ui.modelSelect.innerHTML = '';
            
            const models = data.available_models.filter(m => {
                const isAssembly = m.startsWith('assemblyai');
                return ui.engineSelect.value === 'assemblyai' ? isAssembly : !isAssembly;
            });

            if (models.length === 0) {
                ui.modelSelect.innerHTML = '<option>Nenhum modelo compatível</option>';
                return;
            }
            models.forEach(modelId => {
                const option = new Option(modelId, modelId);
                ui.modelSelect.appendChild(option);
            });
        } catch (error) {
            ui.modelSelect.innerHTML = '<option>Erro ao carregar</option>';
            console.error(error);
        } finally {
            ui.modelSelect.disabled = false;
        }
    }

    function toggleEngineView() {
        const isAssemblyAI = ui.engineSelect.value === 'assemblyai';
        ui.assemblyaiKeyGroup.style.display = isAssemblyAI ? 'block' : 'none';
        ui.featuresCard.style.display = isAssemblyAI ? 'block' : 'none';
        ui.deviceChoiceGroup.style.display = isAssemblyAI ? 'none' : 'block';
        populateModels();
    }

    async function startTranscription() {
        if (ui.audioFilesInput.files.length === 0) return alert('Por favor, selecione ao menos um arquivo.');

        setButtonState(true, 'Enviando...');
        const formData = new FormData();
        formData.append('session_id', state.sessionId);
        formData.append('model_id', ui.modelSelect.value);
        formData.append('language', ui.languageSelect.value);

        for (const file of ui.audioFilesInput.files) formData.append('files', file);

        if (ui.engineSelect.value !== 'local') {
            formData.append('assemblyai_api_key', document.getElementById('assemblyai-key').value);
            formData.append('speaker_labels', document.getElementById('speaker-labels').checked);
            formData.append('entity_detection', document.getElementById('entity-detection').checked);
        } else {
            formData.append('device_choice', ui.deviceChoice.value);
        }

        try {
            const response = await fetch('/jobs', { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Erro desconhecido.');

            data.jobs_created.forEach(job => addJobRowToTable(job));
            if (!state.pollInterval) {
                state.pollInterval = setInterval(pollQueueStatus, 2000);
            }
        } catch (error) {
            alert(`Erro ao criar jobs: ${error.message}`);
        } finally {
            setButtonState(false, '▶️ Iniciar Transcrição');
            ui.audioFilesInput.value = '';
        }
    }

    function addJobRowToTable(job) {
        const row = ui.jobsTableBody.insertRow(0);
        row.id = `job-row-${job.job_id}`;
        row.innerHTML = `
            <td title="${job.filename}">${job.filename.length > 50 ? '...' + job.filename.slice(-47) : job.filename}</td>
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
        const activeJobRows = document.querySelectorAll('.job-status.status-queued, .job-status.status-processing, .job-status.status-cancelling');
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
                statusCell.textContent = getStatusText(job.status, job.eta_timestamp);
                statusCell.className = `job-status status-${job.status}`;

                const progressBar = row.querySelector('.progress-bar');
                progressBar.style.width = `${job.progress}%`;
                progressBar.textContent = `${job.progress}%`;
                
                const actionsCell = row.querySelector('.job-actions');
                if (job.status === 'completed') {
                    actionsCell.innerHTML = `<button class="action-button" data-job-id="${job.job_id}">Ver Resultado</button>`;
                    actionsCell.querySelector('button').addEventListener('click', (e) => toggleJobDetails(e.target.dataset.jobId));
                } else if (job.status === 'failed' || job.status === 'cancelled') {
                    actionsCell.innerHTML = `<button class="action-button" disabled>${getStatusText(job.status)}</button>`;
                }
            });
        } catch (error) {
            console.error("Erro ao atualizar fila:", error);
        }
    }
    
    async function cancelSingleJob(jobId) {
        const row = document.getElementById(`job-row-${jobId}`);
        const button = row.querySelector('.cancel-button');
        if(button) {
            button.disabled = true;
            button.textContent = 'Cancelando...';
        }
        try {
            await fetch(`/jobs/${jobId}/cancel`, { method: 'POST' });
        } catch(error) {
            alert("Erro ao enviar sinal de cancelamento.");
            if(button) {
                button.disabled = false;
                button.textContent = 'Cancelar';
            }
        }
    }

    async function cancelAllSessionJobs() {
        if (!confirm(`Tem certeza que deseja cancelar TODOS os jobs desta sessão?`)) return;
        try {
            const formData = new FormData();
            formData.append('session_id', state.sessionId);
            const response = await fetch(`/queues/cancel-session`, { method: 'POST', body: formData });
            const data = await response.json();
            alert(data.message);
        } catch(error) {
            alert("Erro ao cancelar a sessão.");
        }
    }

    async function toggleJobDetails(jobId) {
        const detailsRow = document.getElementById(`details-row-${jobId}`);
        const detailsCell = detailsRow.querySelector('.details-cell');
        
        if (detailsRow.style.display === 'table-row') {
            detailsRow.style.display = 'none';
            detailsCell.innerHTML = '';
        } else {
            try {
                const response = await fetch(`/jobs/${jobId}`);
                const job = await response.json();
                if (!job.result) throw new Error("Resultado não encontrado.");

                detailsCell.innerHTML = createResultDetailsHTML(job.result, jobId);
                detailsRow.style.display = 'table-row';
                
                addTabListeners(detailsCell);
                addDownloadListeners(detailsCell);

            } catch(error) {
                alert("Erro ao buscar detalhes do job: " + error.message);
            }
        }
    }

    function createResultDetailsHTML(result, jobId) {
        const resultId = `result-content-${jobId}`;
        const entitiesTab = result.entities.length > 0 ? `<button class="tab-button" data-target="${resultId}-entities">Entidades</button>` : '';
        const entitiesContent = result.entities.length > 0 ? `<div id="${resultId}-entities" class="tab-content" style="display: none;"><pre>${JSON.stringify(result.entities, null, 2)}</pre></div>` : '';

        return `
            <div class="details-content">
                <div class="result-tabs">
                    <button class="tab-button active" data-target="${resultId}-dialogue-md">Diálogo (MD)</button>
                    <button class="tab-button" data-target="${resultId}-raw">Texto Puro</button>
                    ${entitiesTab}
                </div>
                <div class="result-body">
                    <div id="${resultId}-dialogue-md" class="tab-content"><pre>${result.transcription_dialogue_markdown}</pre></div>
                    <div id="${resultId}-raw" class="tab-content" style="display: none;"><pre>${result.transcription_raw}</pre></div>
                    ${entitiesContent}
                    <div class="download-buttons">
                        <button class="download-btn" data-job="${jobId}" data-type="transcription_dialogue_markdown">Baixar .txt (MD)</button>
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
            failed: 'Falhou', cancelling: 'Cancelando...', cancelled: 'Cancelado'
        };
        if (status === 'processing' && eta) {
            return `Processando (ETA: ${new Date(eta * 1000).toLocaleTimeString()})`;
        }
        return statusMap[status] || status;
    }
    
    function setButtonState(disabled, text) {
        ui.transcribeButton.disabled = disabled;
        ui.transcribeButton.textContent = text;
    }

    function clearJobs() {
        ui.jobsTableBody.innerHTML = '';
        if (state.pollInterval) {
            clearInterval(state.pollInterval);
            state.pollInterval = null;
        }
    }

    ui.engineSelect.addEventListener('change', toggleEngineView);
    ui.deviceChoice.addEventListener('change', populateModels);
    ui.transcribeButton.addEventListener('click', startTranscription);
    ui.clearJobsButton.addEventListener('click', clearJobs);
    ui.cancelAllButton.addEventListener('click', cancelAllSessionJobs);

    getOrSetSessionId();
    toggleEngineView();
});