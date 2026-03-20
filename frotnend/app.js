const classifyEndpoint = "/api/v1/documents/classify";
const healthEndpoint = "/api/health";
const apiKeyStorageKey = "doc-classifier-pro-api-key";
const maxParallelUploads = 2;

const state = {
  documents: [],
  selectedId: null,
  running: false,
  activeTab: "data",
  searchTerm: "",
};

const elements = {
  fileInput: document.getElementById("file-input"),
  dropzone: document.getElementById("dropzone"),
  processButton: document.getElementById("process-button"),
  clearButton: document.getElementById("clear-button"),
  queueGrid: document.getElementById("queue-grid"),
  historyList: document.getElementById("history-list"),
  historySearch: document.getElementById("history-search"),
  historyCount: document.getElementById("history-count"),
  queueSummary: document.getElementById("queue-summary"),
  queueStatus: document.getElementById("queue-status"),
  apiKey: document.getElementById("api-key"),
  runtimeBadge: document.getElementById("runtime-badge"),
  healthBadge: document.getElementById("health-badge"),
  previewSurface: document.getElementById("preview-surface"),
  detailCategory: document.getElementById("detail-category"),
  detailConfidence: document.getElementById("detail-confidence"),
  detailFields: document.getElementById("detail-fields"),
  rawOutput: document.getElementById("raw-output"),
  tabData: document.getElementById("tab-data"),
  tabRaw: document.getElementById("tab-raw"),
  tabPanelData: document.getElementById("tab-panel-data"),
  tabPanelRaw: document.getElementById("tab-panel-raw"),
  historyItemTemplate: document.getElementById("history-item-template"),
  queueCardTemplate: document.getElementById("queue-card-template"),
};

function loadApiKey() {
  elements.apiKey.value = window.localStorage.getItem(apiKeyStorageKey) || "";
}

function saveApiKey() {
  window.localStorage.setItem(apiKeyStorageKey, elements.apiKey.value.trim());
}

function formatBytes(size) {
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function timeAgo(timestamp) {
  const diff = Date.now() - timestamp;
  const minutes = Math.max(1, Math.floor(diff / 60000));
  if (minutes < 60) {
    return `${minutes} min ago`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  }
  const days = Math.floor(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

function createId(file) {
  return `${file.name}-${file.size}-${file.lastModified}-${crypto.randomUUID()}`;
}

function createDocument(file) {
  const previewUrl = file.type.startsWith("image/") ? URL.createObjectURL(file) : null;
  return {
    id: createId(file),
    file,
    name: file.name,
    size: file.size,
    mimeType: file.type || "unknown",
    addedAt: Date.now(),
    status: "queued",
    stageText: "Queued for OCR and classification",
    category: "",
    confidence: null,
    latencyMs: null,
    ocrTextPreview: "",
    response: null,
    previewUrl,
    error: "",
  };
}

function addFiles(fileList) {
  const documents = Array.from(fileList).map(createDocument);
  state.documents.unshift(...documents);
  if (!state.selectedId && documents[0]) {
    state.selectedId = documents[0].id;
  }
  render();
}

function getSelectedDocument() {
  return state.documents.find((documentRecord) => documentRecord.id === state.selectedId) || null;
}

function getFilteredDocuments() {
  const term = state.searchTerm.trim().toLowerCase();
  if (!term) {
    return state.documents;
  }
  return state.documents.filter((documentRecord) => {
    return documentRecord.name.toLowerCase().includes(term) || documentRecord.category.toLowerCase().includes(term);
  });
}

function statusLabel(documentRecord) {
  if (documentRecord.status === "complete") {
    return "Classified";
  }
  if (documentRecord.status === "processing") {
    return "Processing";
  }
  if (documentRecord.status === "error") {
    return "Failed";
  }
  return "Queued";
}

function normalizeCategory(category) {
  return category.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function getCategoryLabel(documentRecord) {
  return documentRecord.category ? normalizeCategory(documentRecord.category) : "Pending";
}

function createFilePreview(documentRecord) {
  if (documentRecord.previewUrl) {
    const image = window.document.createElement("img");
    image.src = documentRecord.previewUrl;
    image.alt = documentRecord.name;
    return image;
  }

  const sheet = window.document.createElement("div");
  sheet.className = "file-sheet";
  const label = window.document.createElement("div");
  label.className = "file-sheet__label";
  const extension = documentRecord.file.name.includes(".") ? documentRecord.file.name.split(".").pop() : documentRecord.mimeType;
  label.textContent = extension || "DOC";
  sheet.appendChild(label);
  return sheet;
}

function renderHistory() {
  const filtered = getFilteredDocuments();
  elements.historyCount.textContent = `${filtered.length} doc${filtered.length === 1 ? "" : "s"}`;
  elements.historyList.innerHTML = "";

  if (filtered.length === 0) {
    const note = window.document.createElement("div");
    note.className = "empty-note";
    note.textContent = "No documents match the current search.";
    elements.historyList.appendChild(note);
    return;
  }

  filtered.forEach((documentRecord) => {
    const fragment = elements.historyItemTemplate.content.cloneNode(true);
    const button = fragment.querySelector(".history-item");
    const thumb = fragment.querySelector(".history-item__thumb");
    const name = fragment.querySelector(".history-item__name");
    const time = fragment.querySelector(".history-item__time");
    const category = fragment.querySelector(".history-item__category");
    const status = fragment.querySelector(".history-item__status");

    button.classList.toggle("history-item--selected", documentRecord.id === state.selectedId);
    button.addEventListener("click", () => {
      state.selectedId = documentRecord.id;
      render();
    });

    thumb.appendChild(createFilePreview(documentRecord));
    name.textContent = documentRecord.name;
    time.textContent = `${timeAgo(documentRecord.addedAt)} · ${formatBytes(documentRecord.size)}`;
    category.textContent = getCategoryLabel(documentRecord);
    status.classList.add(`history-item__status--${documentRecord.status}`);

    elements.historyList.appendChild(fragment);
  });
}

function renderQueue() {
  elements.queueGrid.innerHTML = "";
  const docs = state.documents;
  const queued = docs.filter((documentRecord) => documentRecord.status === "queued").length;
  const processing = docs.filter((documentRecord) => documentRecord.status === "processing").length;
  const complete = docs.filter((documentRecord) => documentRecord.status === "complete").length;
  const failed = docs.filter((documentRecord) => documentRecord.status === "error").length;

  if (docs.length === 0) {
    const note = window.document.createElement("div");
    note.className = "empty-note";
    note.textContent = "Upload documents to populate the processing grid.";
    elements.queueGrid.appendChild(note);
  }

  docs.forEach((documentRecord) => {
    const fragment = elements.queueCardTemplate.content.cloneNode(true);
    const card = fragment.querySelector(".queue-card");
    const preview = fragment.querySelector(".queue-card__preview");
    const name = fragment.querySelector(".queue-card__name");
    const stage = fragment.querySelector(".queue-card__stage");
    const tag = fragment.querySelector(".queue-card__tag");
    const meta = fragment.querySelector(".queue-card__meta");

    card.classList.toggle("queue-card--selected", documentRecord.id === state.selectedId);
    card.addEventListener("click", () => {
      state.selectedId = documentRecord.id;
      render();
    });

    preview.appendChild(createFilePreview(documentRecord));
    name.textContent = documentRecord.name;
    stage.textContent = documentRecord.error || documentRecord.stageText;
    tag.textContent = statusLabel(documentRecord);
    tag.classList.add(`queue-card__tag--${documentRecord.status}`);
    meta.textContent = documentRecord.category ? normalizeCategory(documentRecord.category) : documentRecord.mimeType;

    elements.queueGrid.appendChild(fragment);
  });

  elements.queueSummary.textContent = `${docs.length} file(s) in queue · ${queued} queued · ${complete} complete`;
  elements.queueStatus.textContent = state.running
    ? `Running · ${processing} active`
    : failed > 0
      ? `${failed} failed`
      : "Idle";
  elements.processButton.disabled = state.running || docs.every((documentRecord) => !["queued", "error"].includes(documentRecord.status));
  elements.clearButton.disabled = state.running || docs.length === 0;
}

function parseExtractedFields(text) {
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const keys = ["Name", "ID", "Degree", "Date", "Category", "Institution", "Department", "Address"];
  const found = [];

  keys.forEach((key) => {
    const matcher = new RegExp(`^${key}\\s*[:\\-]\\s*(.+)$`, "i");
    const line = lines.find((candidate) => matcher.test(candidate));
    if (line) {
      found.push({
        label: key,
        value: line.replace(matcher, "$1").trim(),
      });
    }
  });

  if (found.length > 0) {
    return found;
  }

  return lines.slice(0, 5).map((line, index) => ({
    label: `Field ${index + 1}`,
    value: line,
  }));
}

function renderInspector() {
  const selected = getSelectedDocument();
  elements.previewSurface.innerHTML = "";
  elements.detailFields.innerHTML = "";

  if (!selected) {
    const placeholder = window.document.createElement("div");
    placeholder.className = "preview-placeholder";
    placeholder.textContent = "No document selected";
    elements.previewSurface.appendChild(placeholder);
    elements.detailCategory.textContent = "Waiting";
    elements.detailConfidence.textContent = "-";
    elements.rawOutput.textContent = "Select a document to inspect its OCR preview and API response.";
    return;
  }

  elements.previewSurface.appendChild(createFilePreview(selected));
  elements.detailCategory.textContent = getCategoryLabel(selected);
  elements.detailConfidence.textContent = selected.confidence == null ? "-" : selected.confidence.toFixed(2);

  const fields = selected.ocrTextPreview ? parseExtractedFields(selected.ocrTextPreview) : [];
  if (fields.length === 0) {
    const note = window.document.createElement("div");
    note.className = "empty-note";
    note.textContent = "No extracted text available yet.";
    elements.detailFields.appendChild(note);
  } else {
    fields.forEach((field) => {
      const container = window.document.createElement("div");
      container.className = "detail-field";
      const label = window.document.createElement("span");
      const value = window.document.createElement("strong");
      label.textContent = field.label;
      value.textContent = field.value;
      container.append(label, value);
      elements.detailFields.appendChild(container);
    });
  }

  const rawPayload = selected.response || {
    filename: selected.name,
    status: selected.status,
    category: selected.category || null,
    confidence: selected.confidence,
    ocr_text_preview: selected.ocrTextPreview || null,
    error: selected.error || null,
  };
  elements.rawOutput.textContent = JSON.stringify(rawPayload, null, 2);
}

function renderTabs() {
  const showData = state.activeTab === "data";
  elements.tabData.classList.toggle("tab-strip__button--active", showData);
  elements.tabRaw.classList.toggle("tab-strip__button--active", !showData);
  elements.tabPanelData.hidden = !showData;
  elements.tabPanelRaw.hidden = showData;
}

function render() {
  renderHistory();
  renderQueue();
  renderInspector();
  renderTabs();
}

async function fetchHealth() {
  try {
    const response = await fetch(healthEndpoint);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    elements.healthBadge.textContent = `API status: ${payload.status}`;
    const runtime = payload.models?.find((model) => model.type === "ocr");
    elements.runtimeBadge.textContent = runtime?.uses_gpu ? "GPU OCR online" : "CPU OCR online";
  } catch (error) {
    elements.healthBadge.textContent = "API status: unavailable";
    elements.runtimeBadge.textContent = "Runtime: unknown";
  }
}

function updateDocument(documentRecord, patch) {
  Object.assign(documentRecord, patch);
  if (!state.selectedId) {
    state.selectedId = documentRecord.id;
  }
  render();
}

async function classifyDocument(documentRecord) {
  updateDocument(documentRecord, {
    status: "processing",
    stageText: "Uploading and extracting text",
    error: "",
  });

  const formData = new FormData();
  formData.append("file", documentRecord.file);

  const headers = {};
  const apiKey = elements.apiKey.value.trim();
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }

  const response = await fetch(classifyEndpoint, {
    method: "POST",
    body: formData,
    headers,
  });

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
      }
    } catch (error) {
      // Keep fallback detail when response is not JSON.
    }
    throw new Error(detail);
  }

  const payload = await response.json();
  updateDocument(documentRecord, {
    status: "complete",
    stageText: "Classification complete",
    category: payload.classification || "",
    confidence: payload.confidence ?? null,
    latencyMs: payload.latency_ms ?? null,
    ocrTextPreview: payload.ocr_text_preview || "",
    response: payload,
  });
}

async function processQueue() {
  if (state.running) {
    return;
  }

  saveApiKey();
  const pending = state.documents.filter((documentRecord) => ["queued", "error"].includes(documentRecord.status));
  if (pending.length === 0) {
    return;
  }

  state.running = true;
  render();

  const workers = Array.from({ length: Math.min(maxParallelUploads, pending.length) }, async () => {
    while (pending.length > 0) {
      const next = pending.shift();
      if (!next) {
        return;
      }
      try {
        await classifyDocument(next);
      } catch (error) {
        updateDocument(next, {
          status: "error",
          stageText: "Classification failed",
          error: error.message,
          response: {
            filename: next.name,
            error: error.message,
          },
        });
      }
    }
  });

  await Promise.all(workers);
  state.running = false;
  render();
}

function clearQueue() {
  if (state.running) {
    return;
  }
  state.documents.forEach((documentRecord) => {
    if (documentRecord.previewUrl) {
      URL.revokeObjectURL(documentRecord.previewUrl);
    }
  });
  state.documents = [];
  state.selectedId = null;
  render();
}

elements.dropzone.addEventListener("click", () => elements.fileInput.click());
elements.dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.fileInput.click();
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("is-active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (eventName === "drop" && event.dataTransfer?.files?.length) {
      addFiles(event.dataTransfer.files);
    }
    elements.dropzone.classList.remove("is-active");
  });
});

elements.fileInput.addEventListener("change", (event) => {
  if (event.target.files?.length) {
    addFiles(event.target.files);
  }
});

elements.processButton.addEventListener("click", () => {
  processQueue();
});

elements.clearButton.addEventListener("click", () => {
  clearQueue();
});

elements.historySearch.addEventListener("input", (event) => {
  state.searchTerm = event.target.value;
  renderHistory();
});

elements.apiKey.addEventListener("change", saveApiKey);
elements.apiKey.addEventListener("blur", saveApiKey);

elements.tabData.addEventListener("click", () => {
  state.activeTab = "data";
  renderTabs();
});

elements.tabRaw.addEventListener("click", () => {
  state.activeTab = "raw";
  renderTabs();
});

loadApiKey();
render();
fetchHealth();
