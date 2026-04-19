const apiBaseUrlInput = document.getElementById("apiBaseUrl");
const statusBtn = document.getElementById("statusBtn");
const statusText = document.getElementById("statusText");

const pdfFileInput = document.getElementById("pdfFileInput");
const replaceExistingInput = document.getElementById("replaceExisting");
const uploadBtn = document.getElementById("uploadBtn");
const uploadResult = document.getElementById("uploadResult");

const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");
const answerBox = document.getElementById("answerBox");
const sourcesList = document.getElementById("sourcesList");

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderAnswerMarkdown(rawText) {
  const escaped = escapeHtml(rawText || "");

  const withBold = escaped
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__(.+?)__/g, "<strong>$1</strong>");

  const lines = withBold.split(/\r?\n/);
  const renderedLines = [];
  let currentListTag = null;

  function closeListIfNeeded() {
    if (currentListTag) {
      renderedLines.push(`</${currentListTag}>`);
      currentListTag = null;
    }
  }

  for (const line of lines) {
    const bulletMatch = line.match(/^\s*[\*-]\s+(.*)$/);
    const numberedMatch = line.match(/^\s*\d+\.\s+(.*)$/);

    if (bulletMatch) {
      if (currentListTag !== "ul") {
        closeListIfNeeded();
        renderedLines.push("<ul>");
        currentListTag = "ul";
      }
      renderedLines.push(`<li>${bulletMatch[1]}</li>`);
      continue;
    }

    if (numberedMatch) {
      if (currentListTag !== "ol") {
        closeListIfNeeded();
        renderedLines.push("<ol>");
        currentListTag = "ol";
      }
      renderedLines.push(`<li>${numberedMatch[1]}</li>`);
      continue;
    }

    closeListIfNeeded();

    renderedLines.push(line.trim() ? line : "<br>");
  }

  closeListIfNeeded();

  return renderedLines.join("\n").replace(/\n/g, "<br>");
}

function getApiBaseUrl() {
  return apiBaseUrlInput.value.trim().replace(/\/$/, "");
}

function setStatus(message, type = "neutral") {
  statusText.textContent = message;
  statusText.className = "status-pill";
  if (type === "ok") {
    statusText.classList.add("status-ok");
    return;
  }
  if (type === "error") {
    statusText.classList.add("status-error");
    return;
  }
  statusText.classList.add("status-neutral");
}

function formatSourceItem(source, index) {
  const confidence = source.score !== null && source.score !== undefined ? ` | score: ${Number(source.score).toFixed(3)}` : "";
  const chunk = source.chunk_id !== null && source.chunk_id !== undefined ? source.chunk_id : "-";
  const textPreview = (source.text || "").slice(0, 190);
  return `[${index + 1}] ${source.source} | chunk: ${chunk}${confidence}\n${textPreview}`;
}

async function checkStatus() {
  setStatus("Checking status...");

  try {
    const res = await fetch(`${getApiBaseUrl()}/api/query/status`);
    if (!res.ok) {
      throw new Error(`Status endpoint returned ${res.status}`);
    }

    const data = await res.json();
    if (data.ready) {
      setStatus(`Connected | indexed vectors: ${data.vector_count}`, "ok");
    } else {
      setStatus("Connected | no indexed data yet", "neutral");
    }
  } catch (error) {
    setStatus(`Connection failed: ${error.message}`, "error");
  }
}

async function uploadPdf() {
  const file = pdfFileInput.files[0];
  if (!file) {
    uploadResult.textContent = "Please choose a PDF file first.";
    return;
  }

  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading...";
  uploadResult.textContent = "";

  const formData = new FormData();
  formData.append("file", file);
  formData.append("replace_existing", replaceExistingInput.checked ? "true" : "false");

  try {
    const res = await fetch(`${getApiBaseUrl()}/api/upload`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `Upload failed with ${res.status}`);
    }

    uploadResult.textContent = [
      data.message,
      `File: ${data.filename}`,
      `Index mode: ${data.index_mode}`,
      `Chunks created: ${data.chunks_created}`,
      `Vectors added: ${data.vectors_added}`,
      `Total vectors: ${data.total_vectors}`,
    ].join("\n");

    checkStatus();
  } catch (error) {
    uploadResult.textContent = `Upload failed: ${error.message}`;
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Upload and Index";
  }
}

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    answerBox.innerHTML = renderAnswerMarkdown("Please type a question.");
    return;
  }

  askBtn.disabled = true;
  askBtn.textContent = "Thinking...";
  answerBox.innerHTML = renderAnswerMarkdown("Generating answer...");
  sourcesList.innerHTML = "";

  try {
    const res = await fetch(`${getApiBaseUrl()}/api/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `Query failed with ${res.status}`);
    }

    answerBox.innerHTML = renderAnswerMarkdown(data.answer || "No answer returned.");

    if (!Array.isArray(data.sources) || data.sources.length === 0) {
      const li = document.createElement("li");
      li.textContent = "No sources returned.";
      sourcesList.appendChild(li);
      return;
    }

    data.sources.forEach((source, index) => {
      const li = document.createElement("li");
      li.textContent = formatSourceItem(source, index);
      sourcesList.appendChild(li);
    });
  } catch (error) {
    answerBox.innerHTML = renderAnswerMarkdown(`Query failed: ${error.message}`);
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "Get Answer";
  }
}

statusBtn.addEventListener("click", checkStatus);
uploadBtn.addEventListener("click", uploadPdf);
askBtn.addEventListener("click", askQuestion);

checkStatus();
