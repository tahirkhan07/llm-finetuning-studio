// Tab switching
function showTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
    
    document.getElementById(`tab-${tabId}`).classList.add('active');
    event.currentTarget.classList.add('active');
}

// Helper for loaders
function setLoading(btnId, isLoading) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    if (isLoading) btn.classList.add('loading');
    else btn.classList.remove('loading');
    btn.disabled = isLoading;
}

// 1. Model
function toggleCustomModel() {
    const sel = document.getElementById('model-select').value;
    const cust = document.getElementById('model-id-custom');
    if(sel === 'custom') cust.style.display = 'block';
    else cust.style.display = 'none';
}

async function loadModel() {
    let modelId = document.getElementById('model-select').value;
    if (modelId === 'custom') {
        modelId = document.getElementById('model-id-custom').value;
    }
    const quant = document.getElementById('model-quant').value;
    const status = document.getElementById('model-status');
    const details = document.getElementById('model-details');
    
    setLoading('btn-load-model', true);
    status.innerHTML = 'Loading...';
    if (details) details.innerHTML = '';
    
    try {
        const res = await fetch('/api/models/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model_id: modelId, quantization: quant})
        });
        const data = await res.json();
        if (res.ok) {
            status.innerHTML = `✅ ${data.message} <br><br> Chat Template: ${data.has_chat_template} <br> LoRA Targets: ${data.lora_targets.join(', ')}`;
            
            if (data.properties && details) {
                let tableHtml = '<table style="width:100%; border-collapse: collapse; margin-top: 10px; background: rgba(0,0,0,0.2); border-radius: 6px;">';
                tableHtml += '<tr><th style="text-align:left; border-bottom: 1px solid var(--border); padding: 8px;">Property</th><th style="text-align:left; border-bottom: 1px solid var(--border); padding: 8px;">Value</th></tr>';
                for (const [key, value] of Object.entries(data.properties)) {
                    tableHtml += `<tr><td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">${key}</td><td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">${value}</td></tr>`;
                }
                tableHtml += '</table>';
                details.innerHTML = tableHtml;
            }
        } else {
            status.innerHTML = `❌ Error: ${data.detail}`;
        }
    } catch (e) {
        status.innerHTML = `❌ Error: ${e.message}`;
    }
    setLoading('btn-load-model', false);
}

// 2. Dataset
async function loadDataset() {
    const source = document.getElementById('dataset-source').value;
    const status = document.getElementById('ds-load-status');
    
    setLoading('btn-load-ds', true);
    status.innerHTML = 'Loading...';
    
    try {
        const res = await fetch('/api/datasets/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({source})
        });
        const data = await res.json();
        if (res.ok) {
            status.innerHTML = `✅ ${data.message}`;
            
            if (data.columns) {
                let options = '<option value="">None</option>';
                data.columns.forEach(c => options += `<option value="${c}">${c}</option>`);
                
                const instr = document.getElementById('ds-instr');
                const inp = document.getElementById('ds-inp');
                const out = document.getElementById('ds-out');
                
                instr.innerHTML = options;
                inp.innerHTML = options;
                out.innerHTML = options;
                
                if (data.guesses) {
                    if (data.guesses.instruction) instr.value = data.guesses.instruction;
                    if (data.guesses.input) inp.value = data.guesses.input;
                    if (data.guesses.output) out.value = data.guesses.output;
                }
            }
        } else {
            status.innerHTML = `❌ Error: ${data.detail}`;
        }
    } catch (e) {
        status.innerHTML = `❌ Error: ${e.message}`;
    }
    setLoading('btn-load-ds', false);
}

async function mapDataset() {
    const instr = document.getElementById('ds-instr').value;
    const inp = document.getElementById('ds-inp').value;
    const out = document.getElementById('ds-out').value;
    const sys = document.getElementById('ds-sys').value;
    const status = document.getElementById('ds-map-status');
    
    setLoading('btn-map-ds', true);
    status.innerHTML = 'Mapping...';
    
    try {
        const res = await fetch('/api/datasets/map', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                instruction_col: instr,
                input_col: inp,
                output_col: out,
                system_prompt: sys
            })
        });
        const data = await res.json();
        if (res.ok) {
            status.innerHTML = `✅ ${data.message}`;
        } else {
            status.innerHTML = `❌ Error: ${data.detail}`;
        }
    } catch (e) {
        status.innerHTML = `❌ Error: ${e.message}`;
    }
    setLoading('btn-map-ds', false);
}

// 3. Hardware
async function detectHardware() {
    const status = document.getElementById('hw-status');
    status.innerHTML = '<span class="loader" style="display:inline-block;"></span> Detecting...';
    
    try {
        const res = await fetch('/api/hardware/detect');
        const data = await res.json();
        
        let html = `<div class="hw-card"><div class="hw-icon">💾</div><h4>Free Disk</h4><p>${data.free_disk_gb.toFixed(1)} GB</p></div>`;
        html += `<div class="hw-card"><div class="hw-icon">🧠</div><h4>System RAM</h4><p>${data.cpu_ram_gb.toFixed(1)} GB Total</p></div>`;
        
        if (data.gpu_count > 0) {
            html += `<div class="hw-card border-glow"><div class="hw-icon">🎮</div><h4>GPU: ${data.gpu_name}</h4><p>VRAM: ${data.gpu_vram_gb.toFixed(1)} GB</p></div>`;
        } else {
            html += `<div class="hw-card" style="border-color:var(--danger);"><div class="hw-icon">⚠️</div><h4>No GPUs Detected</h4><p>Training will run on CPU.</p></div>`;
        }
        status.innerHTML = html;
    } catch (e) {
        status.innerHTML = `Error: ${e.message}`;
    }
}

async function recommendConfig() {
    const status = document.getElementById('hw-status');
    status.innerHTML = '<span class="loader" style="display:inline-block;"></span> Getting recommendation...';
    
    try {
        const res = await fetch('/api/hardware/recommend');
        const data = await res.json();
        
        if (data.error) {
            status.innerHTML = `<div class="hw-card" style="border-color:var(--danger);"><p>${data.error}</p></div>`;
            return;
        }
        
        let html = `<div class="hw-card border-glow" style="grid-column: 1 / -1;">
            <div class="hw-icon">✨</div>
            <h4>Recommended Configuration</h4>
            <ul style="margin-top:10px; padding-left:20px; line-height:1.6;">
                <li><strong>Method:</strong> ${data.method}</li>
                <li><strong>Precision:</strong> ${data.precision}</li>
                <li><strong>Batch Size:</strong> ${data.per_device_batch_size}</li>
                <li><strong>Gradient Accumulation:</strong> ${data.gradient_accumulation_steps}</li>
            </ul>
        </div>`;
        status.innerHTML = html;
    } catch (e) {
        status.innerHTML = `Error: ${e.message}`;
    }
}

// 4. Training
let logInterval = null;

async function startTraining() {
    const tSplit = parseFloat(document.getElementById('tr-split-train').value);
    const vSplit = parseFloat(document.getElementById('tr-split-val').value);
    const testSplit = parseFloat(document.getElementById('tr-split-test').value);
    
    if (Math.abs(tSplit + vSplit + testSplit - 100) > 0.1) {
        document.getElementById('tr-logs').innerHTML = `❌ Error: Split ratios must sum to 100%. (Current sum: ${tSplit + vSplit + testSplit})`;
        return;
    }

    const req = {
        method: document.getElementById('tr-method').value,
        epochs: parseInt(document.getElementById('tr-ep').value),
        max_samples: parseInt(document.getElementById('tr-max-s').value),
        batch_size: parseInt(document.getElementById('tr-bs').value),
        grad_acc: parseInt(document.getElementById('tr-ga').value),
        learning_rate: parseFloat(document.getElementById('tr-lr').value),
        lora_rank: parseInt(document.getElementById('tr-lora').value),
        max_seq_length: parseInt(document.getElementById('tr-msl').value),
        train_ratio: tSplit / 100,
        val_ratio: vSplit / 100,
        test_ratio: testSplit / 100,
        eval_steps: parseInt(document.getElementById('tr-eval-steps').value),
        save_steps: parseInt(document.getElementById('tr-save-steps').value),
        early_stopping: document.getElementById('tr-early-stop').checked,
        early_stopping_patience: parseInt(document.getElementById('tr-patience').value)
    };
    
    const statusEl = document.getElementById('tr-status');
    if (statusEl) statusEl.innerHTML = '<span style="color:var(--accent);">⏳ Training in progress...</span>';
    document.getElementById('tr-logs').innerHTML = 'Sending training configuration to server...';
    setLoading('btn-train', true);

    try {
        const res = await fetch('/api/training/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(req)
        });
        const data = await res.json();
        if (res.ok) {
            document.getElementById('tr-logs').innerHTML = '✅ Training started in background!\n\nAllocating VRAM and compiling model...\n(Note: It may take 30-60 seconds for the first log to appear)\n';
            if (logInterval) clearInterval(logInterval);
            logInterval = setInterval(pollLogs, 2000);
        } else {
            document.getElementById('tr-logs').innerHTML = `❌ Error: ${data.detail}`;
            if (statusEl) statusEl.innerHTML = `<span style="color:var(--danger);">❌ Failed to start</span>`;
            setLoading('btn-train', false);
        }
    } catch (e) {
        document.getElementById('tr-logs').innerHTML = `❌ Error: ${e.message}`;
        const statusEl = document.getElementById('tr-status');
        if (statusEl) statusEl.innerHTML = `<span style="color:var(--danger);">❌ Failed to start</span>`;
        setLoading('btn-train', false);
    }
}

async function pollLogs() {
    try {
        const res = await fetch('/api/training/logs');
        const data = await res.json();
        if (data.logs && data.logs.length > 0) {
            let text = "";
            let isComplete = false;
            let hasError = false;
            let errorMsg = "";

            data.logs.slice(-50).forEach(log => {
                if (log.status) {
                    text += `✅ ${log.status}\n`;
                    if (log.status.toLowerCase().includes("complete")) isComplete = true;
                } else if (log.error) {
                    text += `❌ Error: ${log.error}\n`;
                    hasError = true;
                    errorMsg = log.error;
                } else if (log.eval_loss !== undefined) {
                    const step = String(log.step || '?').padEnd(5);
                    const eloss = (log.eval_loss !== null) ? log.eval_loss.toFixed(4) : "N/A   ";
                    const ep = (log.epoch !== undefined && log.epoch !== null) ? log.epoch.toFixed(2) : "N/A ";
                    text += `Step: ${step} | Eval Loss: ${eloss} | Epoch: ${ep}\n`;
                } else {
                    const step = String(log.step || '?').padEnd(5);
                    const loss = (log.loss !== undefined && log.loss !== null) ? log.loss.toFixed(4) : "N/A   ";
                    const lr = (log.learning_rate !== undefined && log.learning_rate !== null) ? log.learning_rate.toExponential(2) : "N/A   ";
                    const ep = (log.epoch !== undefined && log.epoch !== null) ? log.epoch.toFixed(2) : "N/A ";
                    text += `Step: ${step} | Loss: ${loss} | LR: ${lr} | Epoch: ${ep}\n`;
                }
            });
            const logBox = document.getElementById('tr-logs');
            logBox.innerHTML = text;
            logBox.scrollTop = logBox.scrollHeight;

            if (isComplete) {
                if (logInterval) clearInterval(logInterval);
                setLoading('btn-train', false);
                const statusEl = document.getElementById('tr-status');
                if (statusEl) statusEl.innerHTML = '<span style="color:var(--success);">✅ Training Completed Successfully! Model saved!</span>';
            } else if (hasError) {
                if (logInterval) clearInterval(logInterval);
                setLoading('btn-train', false);
                const statusEl = document.getElementById('tr-status');
                if (statusEl) statusEl.innerHTML = `<span style="color:var(--danger);">❌ Training Failed: ${errorMsg}</span>`;
            }
        }
    } catch (e) {}
}

// 5. Evaluation
async function runEval() {
    const status = document.getElementById('eval-results');
    setLoading('btn-eval', true);
    status.innerHTML = 'Running full evaluation. This may take several minutes...';
    
    try {
        const res = await fetch('/api/evaluate/run', {method: 'POST'});
        const data = await res.json();
        if (res.ok) {
            let html = '<table style="width:100%; border-collapse: collapse;"><tr><th style="text-align:left; border-bottom: 1px solid var(--border);">Metric</th><th style="border-bottom: 1px solid var(--border);">Base</th><th style="border-bottom: 1px solid var(--border);">Fine-tuned</th></tr>';
            data.results.forEach(r => {
                html += `<tr><td style="padding: 5px 0;">${r.metric}</td><td>${r.base}</td><td>${r.finetuned}</td></tr>`;
            });
            html += '</table>';
            status.innerHTML = html;
        } else {
            status.innerHTML = `❌ Error: ${data.detail}`;
        }
    } catch (e) {
        status.innerHTML = `❌ Error: ${e.message}`;
    }
    setLoading('btn-eval', false);
}

// 6. Inference
let chatHistory = [];

async function refreshAdapters() {
    const infSelect = document.getElementById('inf-adapter');
    const expSelect = document.getElementById('exp-adapter');
    try {
        const res = await fetch('/api/experiments/');
        const data = await res.json();
        let infOptions = '<option value="">Base Model</option>';
        let expOptions = '<option value="">Select Adapter</option>';
        
        let isFirst = true;
        data.experiments.forEach(e => {
            if (e.Status === 'COMPLETED') {
                const p = e['Adapter Path'] || `outputs/${e['Experiment ID']}/adapter`;
                const label = isFirst ? `${e['Experiment ID']} (Latest)` : e['Experiment ID'];
                infOptions += `<option value="${p}">${label}</option>`;
                expOptions += `<option value="${p}">${label}</option>`;
                isFirst = false;
            }
        });
        if (infSelect) infSelect.innerHTML = infOptions;
        if (expSelect) expSelect.innerHTML = expOptions;
    } catch (e) {
        console.error(e);
    }
}

async function deleteAdapter() {
    const select = document.getElementById('inf-adapter');
    const path = select.value;
    if (!path) {
        alert("Select an adapter first. You cannot delete the base model.");
        return;
    }
    const expId = select.options[select.selectedIndex].text;
    if (!confirm(`Delete adapter ${expId}?`)) return;
    
    try {
        await fetch(`/api/experiments/${expId}`, {method: 'DELETE'});
        refreshAdapters();
    } catch (e) {
        alert(e.message);
    }
}

function clearChat() {
    chatHistory = [];
    document.getElementById('chat-window').innerHTML = '';
}

async function loadInference() {
    const adapter = document.getElementById('inf-adapter').value || "None";
    const status = document.getElementById('inf-status');
    
    setLoading('btn-inf-load', true);
    status.innerHTML = 'Loading adapter...';
    
    try {
        const res = await fetch('/api/inference/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({adapter_path: adapter})
        });
        const data = await res.json();
        if (res.ok) status.innerHTML = `✅ ${data.message}`;
        else status.innerHTML = `❌ Error: ${data.detail}`;
    } catch (e) {
        status.innerHTML = `❌ Error: ${e.message}`;
    }
    setLoading('btn-inf-load', false);
}

async function sendChat() {
    const input = document.getElementById('chat-input');
    let msg = input.value.trim();
    if (!msg) return;
    
    input.value = '';
    
    // Safety Guardrails (fake logic similar to old UI)
    const guardrails = document.getElementById('inf-guardrails').checked;
    if (guardrails && msg.toLowerCase().includes("bomb")) {
        msg = "How do I build a cake?";
    }
    
    const chatWindow = document.getElementById('chat-window');
    chatWindow.innerHTML += `<div class="message msg-user">${msg}</div>`;
    chatHistory.push({role: "user", content: msg});
    
    const temp = parseFloat(document.getElementById('inf-temp').value);
    const tokens = parseInt(document.getElementById('inf-tokens').value);
    
    const asstDiv = document.createElement('div');
    asstDiv.className = 'message msg-assistant';
    chatWindow.appendChild(asstDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    
    try {
        const res = await fetch('/api/inference/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({messages: chatHistory, temperature: temp, max_tokens: tokens})
        });
        
        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let assistantMsg = '';
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value);
            assistantMsg += chunk;
            asstDiv.innerText = assistantMsg;
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
        
        chatHistory.push({role: "assistant", content: assistantMsg});
    } catch (e) {
        asstDiv.innerText = `Error: ${e.message}`;
    }
}

// 7. Export
async function mergeModel() {
    const adapter = document.getElementById('exp-adapter').value;
    const status = document.getElementById('exp-status');
    
    setLoading('btn-merge', true);
    status.innerHTML = 'Merging model...';
    
    try {
        const res = await fetch('/api/export/merge', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                adapter_path: adapter,
                base_model_id: "",
                output_dir: "outputs/merged"
            })
        });
        const data = await res.json();
        if (res.ok) {
            status.innerHTML = `✅ Merge completed successfully! Saved to: <strong>${data.output_dir}</strong>`;
            
            // Auto-fill the downstream forms for better UX
            const ggufInput = document.getElementById('exp-gguf-dir');
            const pushInput = document.getElementById('exp-push-dir');
            if (ggufInput) ggufInput.value = data.output_dir;
            if (pushInput) pushInput.value = data.output_dir;
            
            // Add a download button for the merged weights
            status.innerHTML += `<div style="margin-top:10px;">
                <button class="btn" style="background:var(--success);" onclick="window.location.href='/api/export/download_folder?path=${encodeURIComponent(data.output_dir)}'">📥 Download Merged Weights (ZIP)</button>
            </div>`;
        } else {
            status.innerHTML = `❌ Error: ${data.detail}`;
        }
    } catch (e) {
        status.innerHTML = `❌ Error: ${e.message}`;
    }
    setLoading('btn-merge', false);
}

// 8. Settings
async function hfLogin() {
    const token = document.getElementById('hf-token').value;
    const status = document.getElementById('hf-status');
    setLoading('btn-login', true);
    try {
        const res = await fetch('/api/settings/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({token})
        });
        const data = await res.json();
        status.innerHTML = res.ok ? `✅ ${data.message}` : `❌ Error: ${data.detail}`;
    } catch (e) {
        status.innerHTML = `❌ Error: ${e.message}`;
    }
    setLoading('btn-login', false);
}

async function hfLogout() {
    const status = document.getElementById('hf-status');
    setLoading('btn-logout', true);
    try {
        const res = await fetch('/api/settings/logout', {method: 'POST'});
        const data = await res.json();
        status.innerHTML = res.ok ? `✅ ${data.message}` : `❌ Error: ${data.detail}`;
    } catch (e) {
        status.innerHTML = `❌ Error: ${e.message}`;
    }
    setLoading('btn-logout', false);
}

async function hfCheck() {
    const status = document.getElementById('hf-status');
    try {
        const res = await fetch('/api/settings/status');
        const data = await res.json();
        if (data.success) {
            status.innerHTML = `✅ Logged in as: <strong>${data.user}</strong>`;
        } else {
            status.innerHTML = `ℹ️ ${data.message}`;
        }
    } catch (e) {
        status.innerHTML = `❌ Error: ${e.message}`;
    }
}

// Experiments
async function loadExperiments() {
    const container = document.getElementById('exp-table-container');
    const comp1 = document.getElementById('comp-exp1');
    const comp2 = document.getElementById('comp-exp2');
    container.innerHTML = 'Loading...';
    try {
        const res = await fetch('/api/experiments/');
        const data = await res.json();
        
        let html = '<table style="width:100%; border-collapse: collapse; margin-top: 10px; background: rgba(0,0,0,0.2); border-radius: 6px;">';
        html += '<tr><th style="padding: 8px; text-align:left;">Adapter ID</th><th style="padding: 8px; text-align:left;">Status</th><th style="padding: 8px; text-align:left;">Loss</th><th style="padding: 8px; text-align:left;">Actions</th></tr>';
        
        let options = '<option value="">Select Adapter</option>';
        let isFirst = true;
        
        data.experiments.forEach(e => {
            let actions = '';
            if (e.Status === 'COMPLETED') {
                const p = e['Adapter Path'] || `outputs/${e['Experiment ID']}/adapter`;
                const label = isFirst ? `${e['Experiment ID']} (Latest)` : e['Experiment ID'];
                options += `<option value="${e['Experiment ID']}">${label}</option>`;
                isFirst = false;
                actions = `<button class="btn" style="padding: 4px 10px; font-size: 0.8rem; background: var(--success);" onclick="window.location.href='/api/export/download_folder?path=${encodeURIComponent(p)}'">📥 Download Weights (ZIP)</button>`;
            }
            html += `<tr><td style="padding: 8px; border-bottom: 1px solid var(--border);">${e['Experiment ID']}</td><td style="padding: 8px; border-bottom: 1px solid var(--border);">${e.Status}</td><td style="padding: 8px; border-bottom: 1px solid var(--border);">${e['Final Loss']}</td><td style="padding: 8px; border-bottom: 1px solid var(--border);">${actions}</td></tr>`;
        });
        html += '</table>';
        container.innerHTML = html;
        comp1.innerHTML = options;
        comp2.innerHTML = options;
    } catch (e) {
        container.innerHTML = `Error: ${e.message}`;
    }
}

async function deleteAllExperiments() {
    if (!confirm('Are you sure you want to delete all trained adapters?')) return;
    try {
        await fetch('/api/experiments/', {method: 'DELETE'});
        loadExperiments();
    } catch (e) {
        alert(e.message);
    }
}

async function compareExperiments() {
    const exp1Id = document.getElementById('comp-exp1').value;
    const exp2Id = document.getElementById('comp-exp2').value;
    const resultsEl = document.getElementById('comp-results');
    
    if (!exp1Id || !exp2Id) {
        resultsEl.innerHTML = '<span style="color:var(--danger);">Please select two adapters to compare.</span>';
        return;
    }
    
    resultsEl.innerHTML = 'Comparing...';
    try {
        const res = await fetch('/api/experiments/');
        const data = await res.json();
        
        const exp1 = data.experiments.find(e => e['Experiment ID'] === exp1Id);
        const exp2 = data.experiments.find(e => e['Experiment ID'] === exp2Id);
        
        if (!exp1 || !exp2) {
            resultsEl.innerHTML = '<span style="color:var(--danger);">Could not find one or both adapters.</span>';
            return;
        }
        
        let html = '<table style="width:100%; border-collapse: collapse; background: rgba(0,0,0,0.2); border-radius: 6px;">';
        html += `<tr><th style="padding: 8px; border-bottom: 1px solid var(--border); text-align:left;">Feature</th><th style="padding: 8px; border-bottom: 1px solid var(--border); text-align:left;">${exp1Id}</th><th style="padding: 8px; border-bottom: 1px solid var(--border); text-align:left;">${exp2Id}</th></tr>`;
        
        // Status and Loss
        html += `<tr><td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">Status</td><td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">${exp1.Status}</td><td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">${exp2.Status}</td></tr>`;
        html += `<tr><td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">Final Loss</td><td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">${exp1['Final Loss'] !== null ? exp1['Final Loss'].toFixed(4) : 'N/A'}</td><td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">${exp2['Final Loss'] !== null ? exp2['Final Loss'].toFixed(4) : 'N/A'}</td></tr>`;
        
        // Config Params
        const params = ['learning_rate', 'num_epochs', 'per_device_batch_size', 'gradient_accumulation_steps', 'max_seq_length', 'method'];
        params.forEach(p => {
            const val1 = exp1.Config && exp1.Config[p] !== undefined ? exp1.Config[p] : 'N/A';
            const val2 = exp2.Config && exp2.Config[p] !== undefined ? exp2.Config[p] : 'N/A';
            html += `<tr><td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">${p}</td><td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">${val1}</td><td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">${val2}</td></tr>`;
        });
        
        // LoRA Rank
        const loraRank1 = exp1.Config && exp1.Config.lora ? exp1.Config.lora.rank : 'N/A';
        const loraRank2 = exp2.Config && exp2.Config.lora ? exp2.Config.lora.rank : 'N/A';
        html += `<tr><td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">LoRA Rank</td><td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">${loraRank1}</td><td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">${loraRank2}</td></tr>`;
        
        html += '</table>';
        resultsEl.innerHTML = html;
        
    } catch (e) {
        resultsEl.innerHTML = `<span style="color:var(--danger);">Error: ${e.message}</span>`;
    }
}

// Export extensions
async function convertGguf() {
    const dir = document.getElementById('exp-gguf-dir').value;
    const quant = document.getElementById('exp-gguf-quant').value;
    const status = document.getElementById('gguf-status');
    setLoading('btn-gguf', true);
    status.innerHTML = 'Converting...';
    try {
        const res = await fetch('/api/export/gguf', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({merged_model_dir: dir, quant_type: quant})
        });
        const data = await res.json();
        if (res.ok) {
            const filename = data.gguf_path.split('/').pop() || 'model.gguf';
            const dlUrl = `/api/export/download?file=${encodeURIComponent(data.gguf_path)}`;
            status.innerHTML = `✅ Success: saved to <strong>${data.gguf_path}</strong>
            <div style="margin-top:10px;">
                <button class="btn btn-success" onclick="triggerGgufDownload('${dlUrl}', '${filename}')">📥 Choose Where to Save GGUF</button>
            </div>`;
        } else {
            status.innerHTML = `❌ Error: ${data.detail}`;
        }
    } catch (e) {
        status.innerHTML = `❌ Error: ${e.message}`;
    }
    setLoading('btn-gguf', false);
}

async function pushToHub() {
    const dir = document.getElementById('exp-push-dir').value;
    const repo = document.getElementById('exp-push-repo').value;
    const status = document.getElementById('push-status');
    setLoading('btn-push', true);
    status.innerHTML = 'Pushing...';
    try {
        const res = await fetch('/api/export/push', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({merged_model_dir: dir, repo_id: repo, private: true})
        });
        const data = await res.json();
        if (res.ok) {
            status.innerHTML = `✅ Success: pushed to <a href="${data.url.repo_url}" target="_blank" style="color:var(--accent);">${data.url.repo_url}</a>`;
        } else {
            status.innerHTML = `❌ Error: ${data.detail}`;
        }
    } catch (e) {
        status.innerHTML = `❌ Error: ${e.message}`;
    }
    setLoading('btn-push', false);
}

// Native File System Downloader for GGUF
async function triggerGgufDownload(url, filename) {
    if (window.showSaveFilePicker) {
        try {
            const handle = await window.showSaveFilePicker({
                suggestedName: filename,
                types: [{
                    description: 'GGUF Model',
                    accept: {'application/octet-stream': ['.gguf']}
                }]
            });
            const writable = await handle.createWritable();
            const response = await fetch(url);
            // Stream directly to disk to prevent browser RAM crashes for multi-GB files
            await response.body.pipeTo(writable);
            alert("✅ Model saved successfully to your computer!");
            return;
        } catch (err) {
            if (err.name !== 'AbortError') {
                alert("Error saving file: " + err.message);
            }
            return; // Exit if aborted, do not fallback
        }
    }
    
    // Fallback for browsers that don't support the File System Access API
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
}
