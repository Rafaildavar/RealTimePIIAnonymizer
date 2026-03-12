const form = document.getElementById('chat-form');
const input = document.getElementById('message');
const output = document.getElementById('output');
const submitButton = form.querySelector('button[type="submit"]');

let currentController = null;

output.innerHTML = '<span class="muted">Ответ LLM появится здесь...</span>';

function appendLine(title, text) {
  const block = document.createElement('div');
  block.textContent = `${title}: ${text}`;
  output.appendChild(block);
  output.scrollTop = output.scrollHeight;
  return block;
}

async function streamChat(message, llmLine) {
  currentController = new AbortController();
  const decoder = new TextDecoder('utf-8');

  const response = await fetch('/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
    signal: currentController.signal,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }

  if (!response.body) {
    const fallback = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });

    if (!fallback.ok) {
      const err = await fallback.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${fallback.status}`);
    }

    const data = await fallback.json();
    llmLine.textContent = `LLM: ${data.answer}`;
    return;
  }

  const reader = response.body.getReader();
  let answer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    answer += decoder.decode(value, { stream: true });
    llmLine.textContent = `LLM: ${answer}`;
    output.scrollTop = output.scrollHeight;
  }

  answer += decoder.decode();
  llmLine.textContent = `LLM: ${answer || 'Пустой ответ'}`;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const message = input.value.trim();
  if (!message) {
    return;
  }

  if (currentController) {
    currentController.abort();
    currentController = null;
  }

  if (output.querySelector('.muted')) {
    output.textContent = '';
  }

  appendLine('Вы', message);
  const llmLine = appendLine('LLM', '...');

  input.value = '';
  input.focus();
  submitButton.disabled = true;

  try {
    await streamChat(message, llmLine);
  } catch (error) {
    if (error.name === 'AbortError') {
      llmLine.textContent = 'LLM: Запрос отменен';
    } else {
      llmLine.textContent = `LLM: Ошибка - ${error.message}`;
    }
  } finally {
    currentController = null;
    submitButton.disabled = false;
  }
});
