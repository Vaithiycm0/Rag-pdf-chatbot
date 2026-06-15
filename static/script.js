const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('pdf-input');
const uploadText = document.getElementById('upload-text');
const statusIndicator = document.getElementById('status-indicator');
const pdfMetadata = document.getElementById('pdf-metadata');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const chatForm = document.getElementById('chat-form');
const messagesArea = document.getElementById('chat-messages');

let isReady = false;

// Initialize status
async function checkStatus() {
    try {
        const res = await fetch('/status');
        const data = await res.json();
        if (data.ready) {
            updateStatusReady(data.pdf_metadata);
        }
    } catch (err) {
        console.error('Status check failed', err);
    }
}
checkStatus();

// Upload Handling
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
        handleUpload(fileInput.files[0]);
    }
});

async function handleUpload(file) {
    if (file.type !== 'application/pdf') {
        alert('Please upload a PDF file.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    uploadText.textContent = `Processing ${file.name}...`;
    statusIndicator.className = 'status-badge processing';
    statusIndicator.textContent = 'Processing PDF...';

    try {
        const res = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        
        if (res.ok) {
            updateStatusReady(data.result);
            addMessage('System', `✅ PDF processed successfully! You can now ask questions about it.`, 'system-message');
        } else {
            throw new Error(data.detail || 'Upload failed');
        }
    } catch (err) {
        console.error(err);
        uploadText.textContent = 'Click or drag PDF here';
        statusIndicator.className = 'status-badge waiting';
        statusIndicator.textContent = 'Error Processing';
        alert(`Failed to process PDF: ${err.message}`);
    }
}

function updateStatusReady(metadata) {
    isReady = true;
    uploadText.textContent = 'Upload a different PDF';
    statusIndicator.className = 'status-badge ready';
    statusIndicator.textContent = 'Ready';
    
    document.getElementById('meta-filename').textContent = metadata.filename;
    document.getElementById('meta-pages').textContent = metadata.page_count;
    document.getElementById('meta-chunks').textContent = metadata.chunk_count;
    pdfMetadata.style.display = 'block';

    chatInput.disabled = false;
    sendButton.disabled = false;
    chatInput.focus();
}

// Chat Handling
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = chatInput.value.trim();
    if (!question || !isReady) return;

    // Add user message
    addMessage('You', question, 'user');
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;
    sendButton.innerHTML = '<div class="loader"></div>';

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        const data = await res.json();

        if (res.ok) {
            addMessage('Assistant', data.answer, 'assistant', data.sources);
        } else {
            throw new Error(data.detail || 'Chat failed');
        }
    } catch (err) {
        addMessage('Assistant', `⚠️ Error: ${err.message}`, 'assistant');
    } finally {
        chatInput.disabled = false;
        sendButton.disabled = false;
        sendButton.innerHTML = '<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg>';
        chatInput.focus();
    }
});

function addMessage(sender, text, type, sources = null) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${type}`;

    let contentHtml = `<div class="message-bubble">${text.replace(/\n/g, '<br>')}</div>`;
    
    if (sources && sources.length > 0) {
        let sourcesHtml = `<button class="source-toggle">📎 Show Sources</button><div class="source-content">`;
        sources.forEach((src, idx) => {
            sourcesHtml += `<strong>Chunk ${idx + 1} (Page ${src.page})</strong><br>${src.content_preview}<br><br>`;
        });
        sourcesHtml += `</div>`;
        contentHtml += sourcesHtml;
    }

    msgDiv.innerHTML = contentHtml;
    messagesArea.appendChild(msgDiv);
    
    if (sources && sources.length > 0) {
        const toggle = msgDiv.querySelector('.source-toggle');
        const content = msgDiv.querySelector('.source-content');
        toggle.addEventListener('click', () => {
            if (content.style.display === 'block') {
                content.style.display = 'none';
                toggle.textContent = '📎 Show Sources';
            } else {
                content.style.display = 'block';
                toggle.textContent = '📎 Hide Sources';
            }
        });
    }

    messagesArea.scrollTop = messagesArea.scrollHeight;
}
