const manualState = {
  view: "queue",
  queue: [],
  queuePage: 1,
  queuePages: 1,
  queueTotal: 0,
  currentAlarmId: null,
  currentQueueIndex: -1,
  detail: null,
  annotations: [],
  candidates: [],
  selectedId: null,
  decision: "",
  tool: "select",
  imageKind: "original",
  image: null,
  zoom: 1,
  pointer: null,
  showCandidates: true,
  loading: false,
  dirty: false,
  autosaveTimer: null,
  undoStack: [],
  historyPage: 1,
  historyPages: 1,
};

const manualElements = {
  tabs: [...document.querySelectorAll("[data-review-view]")],
  pages: [...document.querySelectorAll("[data-review-page]")],
  aiFilter: document.querySelector("#manual-ai-filter"),
  humanFilter: document.querySelector("#manual-human-filter"),
  taskFilter: document.querySelector("#manual-task-filter"),
  channelFilter: document.querySelector("#manual-channel-filter"),
  dateFilter: document.querySelector("#manual-date-filter"),
  queueList: document.querySelector("#manual-queue-list"),
  queueEmpty: document.querySelector("#manual-queue-empty"),
  queueTotal: document.querySelector("#manual-queue-total"),
  queuePrevious: document.querySelector("#manual-queue-previous"),
  queueNext: document.querySelector("#manual-queue-next"),
  queuePageSummary: document.querySelector("#manual-queue-page-summary"),
  metricPending: document.querySelector("#manual-metric-pending"),
  metricDraft: document.querySelector("#manual-metric-draft"),
  metricCompleted: document.querySelector("#manual-metric-completed"),
  metricConference: document.querySelector("#manual-metric-conference"),
  metricCorrection: document.querySelector("#manual-metric-correction"),
  currentTitle: document.querySelector("#manual-current-title"),
  currentMeta: document.querySelector("#manual-current-meta"),
  sampleRail: document.querySelector("#manual-sample-rail"),
  canvas: document.querySelector("#manual-annotation-canvas"),
  cropCanvas: document.querySelector("#manual-crop-canvas"),
  cropHint: document.querySelector("#manual-crop-hint"),
  canvasPlaceholder: document.querySelector("#manual-canvas-placeholder"),
  canvasStage: document.querySelector("#manual-canvas-stage"),
  zoomDisplay: document.querySelector("#manual-zoom-display"),
  showCandidates: document.querySelector("#manual-show-candidates"),
  objectList: document.querySelector("#manual-object-list"),
  objectCount: document.querySelector("#manual-object-count"),
  objectProperties: document.querySelector("#manual-object-properties"),
  objectClass: document.querySelector("#manual-object-class"),
  objectTruncated: document.querySelector("#manual-object-truncated"),
  helmetColorField: document.querySelector("#manual-helmet-color-field"),
  objectColor: document.querySelector("#manual-object-color"),
  personHeadStatus: document.querySelector("#manual-person-head-status"),
  personHeadStatusLabel: document.querySelector("#manual-person-head-status-label"),
  personHeadStatusSize: document.querySelector("#manual-person-head-status-size"),
  personHeadStatusReason: document.querySelector("#manual-person-head-status-reason"),
  aiStatus: document.querySelector("#manual-ai-status"),
  aiConfidence: document.querySelector("#manual-ai-confidence"),
  aiQuality: document.querySelector("#manual-ai-quality"),
  aiModel: document.querySelector("#manual-ai-model"),
  aiSummary: document.querySelector("#manual-ai-summary"),
  reasonGrid: document.querySelector("#manual-reason-grid"),
  comment: document.querySelector("#manual-review-comment"),
  reviewer: document.querySelector("#manual-reviewer"),
  decisionGrid: document.querySelector("#manual-decision-grid"),
  saveState: document.querySelector("#manual-save-state"),
  saveDraft: document.querySelector("#manual-save-draft"),
  submitReview: document.querySelector("#manual-submit-review"),
  undoAnnotation: document.querySelector("#manual-undo-annotation"),
  historyDecision: document.querySelector("#manual-history-decision"),
  historyReason: document.querySelector("#manual-history-reason"),
  historyDate: document.querySelector("#manual-history-date"),
  historyBody: document.querySelector("#manual-history-body"),
  historyTotal: document.querySelector("#manual-history-total"),
  historyPrevious: document.querySelector("#manual-history-previous"),
  historyNext: document.querySelector("#manual-history-next"),
  historyPageSummary: document.querySelector("#manual-history-page-summary"),
  toastRegion: document.querySelector("#manual-toast-region"),
};

function manualEscape(value) {
  return String(value ?? "").replace(/[&<>'"]/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

function manualToast(message, type = "") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`.trim();
  toast.textContent = message;
  manualElements.toastRegion.appendChild(toast);
  window.setTimeout(() => toast.remove(), 3200);
}

async function manualApi(url, options = {}) {
  const init = { ...options, headers: { ...(options.headers || {}) } };
  if (Object.prototype.hasOwnProperty.call(options, "body")) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }
  const response = await fetch(url, init);
  let data = {};
  try { data = await response.json(); } catch (_) { data = {}; }
  if (!response.ok) throw new Error(data.detail || `请求失败（${response.status}）`);
  return data;
}

function showManualView(name) {
  manualState.view = name;
  manualElements.tabs.forEach(tab => {
    const active = tab.dataset.reviewView === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  manualElements.pages.forEach(page => page.classList.toggle("hidden", page.dataset.reviewPage !== name));
  if (name === "history") loadManualHistory();
  if (name === "workbench") window.setTimeout(resizeManualCanvas, 0);
}

function formatManualTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).replace("T", " ");
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  }).format(date).replaceAll("/", "-");
}

function aiStatusMarkup(status) {
  const normalized = ["confirmed", "rejected", "uncertain"].includes(status) ? status : "pending-review";
  return `<span class="alarm-review-status ${normalized}"><i></i>${manualEscape(status || "pending")}</span>`;
}

function humanStatusInfo(status, decision) {
  if (status === "completed") {
    return { label: decisionLabel(decision), className: decision === "no_target" || decision === "ignored" ? "excluded" : "completed" };
  }
  if (status === "conference") return { label: "需会审", className: "conference" };
  if (status === "draft") return { label: "草稿", className: "draft" };
  return { label: "待复核", className: "pending" };
}

function decisionLabel(decision) {
  return {
    confirmed: "确认通过",
    corrected: "修改后通过",
    no_target: "无目标",
    uncertain: "暂不确定",
    ignored: "忽略样本",
  }[decision] || "未选择";
}

function reasonLabel(reason) {
  return {
    missing_label: "漏标",
    wrong_class: "类别错误",
    box_offset: "框位置偏移",
    head_assignment_unclear: "人头归属不清",
    poor_image_quality: "图片质量差",
    non_person_target: "非人员目标",
    other: "其他",
  }[reason] || reason;
}

function queueQuery() {
  const params = new URLSearchParams({ page: String(manualState.queuePage), page_size: "20" });
  if (manualElements.aiFilter.value) params.set("ai_status", manualElements.aiFilter.value);
  if (manualElements.humanFilter.value) params.set("human_status", manualElements.humanFilter.value);
  if (manualElements.taskFilter.value) params.set("task_id", manualElements.taskFilter.value);
  if (manualElements.channelFilter.value) params.set("channel_id", manualElements.channelFilter.value);
  if (manualElements.dateFilter.value) params.set("alarm_date", manualElements.dateFilter.value);
  return params.toString();
}

function renderManualQueue(data) {
  manualState.queue = data.items || [];
  manualState.queueTotal = data.total || 0;
  manualState.queuePage = data.page || 1;
  manualState.queuePages = data.pages || 1;
  manualElements.queueTotal.textContent = String(manualState.queueTotal);
  manualElements.queuePageSummary.textContent = `第 ${manualState.queuePage} / ${manualState.queuePages} 页`;
  manualElements.queuePrevious.disabled = manualState.queuePage <= 1;
  manualElements.queueNext.disabled = manualState.queuePage >= manualState.queuePages;
  const metrics = data.metrics || {};
  manualElements.metricPending.textContent = String(metrics.pending || 0);
  manualElements.metricDraft.textContent = String(metrics.draft || 0);
  manualElements.metricCompleted.textContent = String(metrics.completed_today || 0);
  manualElements.metricConference.textContent = String(metrics.conference || 0);
  manualElements.metricCorrection.textContent = `${Math.round((metrics.correction_rate || 0) * 100)}%`;
  manualElements.queueEmpty.classList.toggle("hidden", manualState.queue.length > 0);
  manualElements.queueList.classList.toggle("hidden", manualState.queue.length === 0);
  if (!manualState.queue.length) {
    manualElements.queueList.innerHTML = "";
    return;
  }
  manualElements.queueList.innerHTML = manualState.queue.map((item, index) => {
    const human = humanStatusInfo(item.human_workflow_status, item.human_decision);
    return `
      <button class="manual-queue-row" type="button" data-review-alarm="${item.id}" data-review-index="${index}">
        <img src="${manualEscape(item.image_url)}?v=${encodeURIComponent(item.event_id)}" alt="${manualEscape(item.alarm_type)}告警图">
        <span class="manual-queue-identity"><strong>${manualEscape(item.event_id)}</strong><small>${manualEscape(item.alarm_type)} · ${manualEscape(formatManualTime(item.alarm_time))}</small></span>
        <span><strong>${manualEscape(item.task_code)}</strong><small>${manualEscape(item.channel_name)}</small></span>
        <span>${aiStatusMarkup(item.review_status)}<small title="${manualEscape(item.review_summary)}">${manualEscape(item.review_summary || "暂无模型说明")}</small></span>
        <span class="manual-human-status ${human.className}">${human.label}</span>
        <span class="manual-queue-action">开始复核 →</span>
      </button>`;
  }).join("");
}

async function loadManualQueue(showMessage = false) {
  manualElements.queueList.classList.remove("hidden");
  manualElements.queueEmpty.classList.add("hidden");
  manualElements.queueList.innerHTML = '<div class="manual-review-loading">正在加载复核样本…</div>';
  try {
    const data = await manualApi(`/api/manual-reviews?${queueQuery()}`);
    renderManualQueue(data);
    if (showMessage) manualToast("复核队列已刷新", "success");
  } catch (error) {
    manualElements.queueList.innerHTML = `<div class="manual-review-loading error-text">${manualEscape(error.message)}</div>`;
    manualToast(error.message, "error");
  }
}

async function loadManualFilterOptions() {
  try {
    const [tasks, channels] = await Promise.all([manualApi("/api/video-tasks"), manualApi("/api/video-channels")]);
    manualElements.taskFilter.innerHTML = '<option value="">全部任务</option>' + (tasks.items || []).map(item => `<option value="${item.id}">${manualEscape(item.task_code)}</option>`).join("");
    manualElements.channelFilter.innerHTML = '<option value="">全部通道</option>' + (channels.items || []).map(item => `<option value="${item.id}">${manualEscape(item.name)}</option>`).join("");
  } catch (error) {
    manualToast(`筛选项加载失败：${error.message}`, "error");
  }
}

function cloneAnnotations(items) {
  return (items || []).map((item, index) => ({
    id: String(item.id || `human-${Date.now()}-${index}`),
    class_name: item.class_name,
    box: (item.box || []).map(Number),
    truncated: Boolean(item.truncated),
    color: item.color || "",
    confidence: item.confidence ?? null,
    source: item.source || "human",
    head_status: item.head_status ? { ...item.head_status, reason_codes: [...(item.head_status.reason_codes || [])] } : null,
  }));
}

function manualBoxesRelated(personBox, targetBox) {
  if (!personBox?.length || !targetBox?.length) return false;
  const centerX = (targetBox[0] + targetBox[2]) / 2;
  const centerY = (targetBox[1] + targetBox[3]) / 2;
  return centerX >= personBox[0]
    && centerX <= personBox[2]
    && centerY >= personBox[1] - (personBox[3] - personBox[1]) * .15
    && centerY <= personBox[3];
}

function personHeadStatus(item) {
  if (!item || item.class_name !== "person") return null;
  const associated = manualState.annotations.find(target => ["head", "helmet"].includes(target.class_name) && manualBoxesRelated(item.box, target.box));
  if (associated) {
    if (item.head_status?.code === "labeled") return item.head_status;
    return {
      code: "labeled",
      label: `当前已有 ${associated.class_name}`,
      short_label: `已补 ${associated.class_name}`,
      reason: `当前人工标注中已为该人员关联 ${associated.class_name} 框。`,
      training_short_edge: Math.round(Math.min(associated.box[2] - associated.box[0], associated.box[3] - associated.box[1]) * 10) / 10,
      original_short_edge: null,
      minimum_short_edge: 20,
    };
  }
  return item.head_status || {
    code: item.source === "human" ? "manual_added" : "unavailable",
    label: item.source === "human" ? "人工新增，待确认头部" : "原因不可追溯",
    short_label: item.source === "human" ? "待确认头部" : "原因缺失",
    reason: item.source === "human"
      ? "这是人工新增的 person 框，请确认是否还应标注 head 或 helmet。"
      : "旧版教师预标注未保存与此人员对应的裁剪级状态，需要人工确认。",
    training_short_edge: null,
    original_short_edge: null,
    minimum_short_edge: 20,
  };
}

function personHeadStatusMarkup(item) {
  const status = personHeadStatus(item);
  return status ? `<em class="manual-head-status-badge ${manualEscape(status.code)}">${manualEscape(status.short_label || status.label)}</em>` : "";
}

function renderSampleRail() {
  if (!manualState.queue.length || manualState.currentQueueIndex < 0) {
    manualElements.sampleRail.innerHTML = '<p class="manual-rail-hint">当前样本不在本页队列中</p>';
    return;
  }
  const start = Math.max(0, manualState.currentQueueIndex - 2);
  const end = Math.min(manualState.queue.length, manualState.currentQueueIndex + 3);
  manualElements.sampleRail.innerHTML = manualState.queue.slice(start, end).map((item, offset) => {
    const index = start + offset;
    const active = item.id === manualState.currentAlarmId;
    return `<button class="manual-rail-item ${active ? "active" : ""}" type="button" data-rail-alarm="${item.id}" data-rail-index="${index}"><img src="${manualEscape(item.image_url)}?v=${encodeURIComponent(item.event_id)}" alt=""><span>#${item.id}</span>${aiStatusMarkup(item.review_status)}</button>`;
  }).join("");
}

function setDecision(value) {
  manualState.decision = value || "";
  manualElements.decisionGrid.querySelectorAll("[data-decision]").forEach(button => button.classList.toggle("active", button.dataset.decision === manualState.decision));
  markManualDirty();
}

function updateManualUndoState() {
  manualElements.undoAnnotation.disabled = manualState.undoStack.length === 0;
}

function pushManualUndoSnapshot() {
  if (!manualState.detail) return false;
  const annotations = cloneAnnotations(manualState.annotations);
  const signature = JSON.stringify(annotations);
  const previous = manualState.undoStack.at(-1);
  if (previous?.signature === signature) return false;
  manualState.undoStack.push({ annotations, selectedId: manualState.selectedId, signature });
  if (manualState.undoStack.length > 50) manualState.undoStack.shift();
  updateManualUndoState();
  return true;
}

function discardManualUndoSnapshot() {
  manualState.undoStack.pop();
  updateManualUndoState();
}

function undoManualAnnotation() {
  const snapshot = manualState.undoStack.pop();
  if (!snapshot) return;
  manualState.annotations = cloneAnnotations(snapshot.annotations);
  manualState.selectedId = manualState.annotations.some(item => item.id === snapshot.selectedId)
    ? snapshot.selectedId
    : (manualState.annotations[0]?.id || null);
  updateManualUndoState();
  markManualDirty();
  renderManualObjects();
  drawManualCanvas();
  manualToast("已撤销最近一次标注操作", "success");
}

function populateManualDetail(detail) {
  manualState.detail = detail;
  manualState.candidates = cloneAnnotations(detail.candidate_annotations);
  manualState.annotations = cloneAnnotations(detail.annotations);
  manualState.selectedId = manualState.annotations[0]?.id || null;
  manualState.decision = detail.review?.decision || "";
  manualState.imageKind = "original";
  manualState.zoom = 1;
  manualState.dirty = false;
  manualState.undoStack = [];
  updateManualUndoState();
  const alarm = detail.alarm;
  manualElements.currentTitle.textContent = `${alarm.event_id} · ${alarm.alarm_type}`;
  manualElements.currentMeta.textContent = `${alarm.task_code} · ${alarm.channel_name} · ${formatManualTime(alarm.alarm_time)} · 人工版本 v${detail.review?.version || 0}`;
  manualElements.aiStatus.innerHTML = aiStatusMarkup(alarm.review_status);
  manualElements.aiConfidence.textContent = alarm.review_confidence == null ? "-" : Number(alarm.review_confidence).toFixed(2);
  manualElements.aiQuality.textContent = alarm.review_evidence_quality || "-";
  manualElements.aiModel.textContent = alarm.review_model || "-";
  manualElements.aiSummary.textContent = alarm.review_summary || alarm.review_error || "暂无大模型复核说明。";
  manualElements.comment.value = detail.review?.comment || "";
  manualElements.reviewer.value = detail.review?.reviewer || "本机审核员";
  manualElements.reasonGrid.querySelectorAll("input[type=checkbox]").forEach(input => { input.checked = (detail.review?.reason_codes || []).includes(input.value); });
  manualElements.decisionGrid.querySelectorAll("[data-decision]").forEach(button => button.classList.toggle("active", button.dataset.decision === manualState.decision));
  manualElements.saveState.textContent = detail.review?.workflow_status === "pending" ? "尚未保存人工复核" : `已加载${detail.review.workflow_status === "draft" ? "草稿" : "人工版本"} · ${detail.review.updated_at || ""}`;
  document.querySelectorAll("[data-image-kind]").forEach(button => button.classList.toggle("active", button.dataset.imageKind === "original"));
  renderSampleRail();
  renderManualObjects();
  loadCanvasImage(detail.image.original_url);
}

async function openManualReview(alarmId, queueIndex = -1) {
  if (manualState.loading) return;
  manualState.loading = true;
  manualState.currentAlarmId = Number(alarmId);
  manualState.currentQueueIndex = Number(queueIndex);
  showManualView("workbench");
  manualElements.canvasPlaceholder.classList.remove("hidden");
  manualElements.canvasPlaceholder.textContent = "正在加载原图和标注数据…";
  try {
    const detail = await manualApi(`/api/manual-reviews/${manualState.currentAlarmId}`);
    populateManualDetail(detail);
  } catch (error) {
    manualElements.canvasPlaceholder.textContent = error.message;
    manualToast(error.message, "error");
  } finally {
    manualState.loading = false;
  }
}

function loadCanvasImage(url) {
  const image = new Image();
  manualState.image = image;
  image.onload = () => {
    manualElements.canvas.width = image.naturalWidth;
    manualElements.canvas.height = image.naturalHeight;
    manualElements.canvasPlaceholder.classList.add("hidden");
    resizeManualCanvas();
    drawManualCanvas();
  };
  image.onerror = () => {
    manualElements.canvasPlaceholder.classList.remove("hidden");
    manualElements.canvasPlaceholder.textContent = "复核图片加载失败";
  };
  image.src = `${url}${url.includes("?") ? "&" : "?"}v=${Date.now()}`;
}

function resizeManualCanvas() {
  if (!manualState.image?.complete || !manualState.image.naturalWidth) return;
  const available = Math.max(280, manualElements.canvasStage.clientWidth - 24);
  const ratio = manualState.image.naturalHeight / manualState.image.naturalWidth;
  const width = available * manualState.zoom;
  manualElements.canvas.style.width = `${width}px`;
  manualElements.canvas.style.height = `${width * ratio}px`;
  manualElements.zoomDisplay.textContent = `${Math.round(manualState.zoom * 100)}%`;
}

function setManualZoom(nextZoom, anchor = null) {
  if (!manualState.image?.complete || !manualState.image.naturalWidth) return;
  const stage = manualElements.canvasStage;
  const canvas = manualElements.canvas;
  const stageRect = stage.getBoundingClientRect();
  const canvasRect = canvas.getBoundingClientRect();
  const clientX = anchor?.clientX ?? (stageRect.left + stageRect.width / 2);
  const clientY = anchor?.clientY ?? (stageRect.top + stageRect.height / 2);
  const imageRatioX = canvasRect.width ? Math.max(0, Math.min(1, (clientX - canvasRect.left) / canvasRect.width)) : .5;
  const imageRatioY = canvasRect.height ? Math.max(0, Math.min(1, (clientY - canvasRect.top) / canvasRect.height)) : .5;
  manualState.zoom = Math.max(.5, Math.min(6, Number(nextZoom) || 1));
  resizeManualCanvas();
  const visibleX = clientX - stageRect.left;
  const visibleY = clientY - stageRect.top;
  stage.scrollLeft = Math.max(0, canvas.offsetLeft + imageRatioX * canvas.clientWidth - visibleX);
  stage.scrollTop = Math.max(0, canvas.offsetTop + imageRatioY * canvas.clientHeight - visibleY);
}

function onManualCanvasWheel(event) {
  if (!manualState.detail || !manualState.image?.complete) return;
  event.preventDefault();
  const factor = Math.exp(-event.deltaY * .0015);
  setManualZoom(manualState.zoom * factor, { clientX: event.clientX, clientY: event.clientY });
}

function classColor(className, alpha = 1) {
  const values = {
    person: `rgba(14, 142, 214, ${alpha})`,
    head: `rgba(220, 74, 65, ${alpha})`,
    helmet: `rgba(22, 163, 74, ${alpha})`,
  };
  return values[className] || `rgba(100, 116, 139, ${alpha})`;
}

function drawBox(context, item, options = {}) {
  const [x1, y1, x2, y2] = item.box;
  const selected = item.id === manualState.selectedId && !options.candidate;
  const lineWidth = Math.max(2, manualElements.canvas.width / 900 * (selected ? 2.2 : 1.4));
  context.save();
  context.strokeStyle = options.candidate ? "rgba(245, 158, 11, .95)" : classColor(item.class_name);
  context.fillStyle = context.strokeStyle;
  context.lineWidth = lineWidth;
  context.setLineDash(options.candidate ? [10, 7] : []);
  context.strokeRect(x1, y1, x2 - x1, y2 - y1);
  const personStatus = !options.candidate ? personHeadStatus(item) : null;
  const label = `${options.candidate ? "候选 " : ""}${item.class_name}${personStatus ? ` · ${personStatus.short_label || personStatus.label}` : ""}`;
  context.font = `${Math.max(14, manualElements.canvas.width / 105)}px Microsoft YaHei`;
  const metrics = context.measureText(label);
  const labelHeight = Math.max(20, manualElements.canvas.width / 70);
  const labelY = Math.max(0, y1 - labelHeight);
  context.fillRect(x1, labelY, metrics.width + 12, labelHeight);
  context.fillStyle = "#fff";
  context.fillText(label, x1 + 6, labelY + labelHeight * .73);
  if (selected) {
    context.setLineDash([]);
    const handleSize = Math.max(8, manualElements.canvas.width / 180);
    [
      [x1, y1], [(x1 + x2) / 2, y1], [x2, y1],
      [x2, (y1 + y2) / 2], [x2, y2],
      [(x1 + x2) / 2, y2], [x1, y2], [x1, (y1 + y2) / 2],
    ].forEach(([x, y]) => {
      context.fillStyle = "#fff";
      context.strokeStyle = classColor(item.class_name);
      context.lineWidth = lineWidth;
      context.fillRect(x - handleSize / 2, y - handleSize / 2, handleSize, handleSize);
      context.strokeRect(x - handleSize / 2, y - handleSize / 2, handleSize, handleSize);
    });
  }
  context.restore();
}

function drawManualCanvas() {
  const canvas = manualElements.canvas;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!manualState.image?.complete || !manualState.image.naturalWidth) return;
  context.drawImage(manualState.image, 0, 0, canvas.width, canvas.height);
  if (manualState.showCandidates) manualState.candidates.forEach(item => drawBox(context, item, { candidate: true }));
  manualState.annotations.forEach(item => drawBox(context, item));
  drawManualCrop();
}

function selectedAnnotation() {
  return manualState.annotations.find(item => item.id === manualState.selectedId) || null;
}

function drawManualCrop() {
  const crop = manualElements.cropCanvas;
  const selected = selectedAnnotation();
  if (!manualState.image?.complete || !selected) {
    crop.width = 1;
    crop.height = 1;
    manualElements.cropHint.classList.remove("hidden");
    return;
  }
  manualElements.cropHint.classList.add("hidden");
  crop.width = 320;
  crop.height = 180;
  const context = crop.getContext("2d");
  const [x1, y1, x2, y2] = selected.box;
  const width = x2 - x1;
  const height = y2 - y1;
  let sourceX = Math.max(0, x1 - width * .5);
  let sourceY = Math.max(0, y1 - height * .65);
  let sourceWidth = Math.min(manualState.image.naturalWidth - sourceX, width * 2);
  let sourceHeight = Math.min(manualState.image.naturalHeight - sourceY, height * 2.2);
  if (selected.class_name === "person") {
    sourceY = Math.max(0, y1 - height * .08);
    sourceHeight = Math.min(manualState.image.naturalHeight - sourceY, height * .55);
  }
  context.fillStyle = "#0f172a";
  context.fillRect(0, 0, crop.width, crop.height);
  context.drawImage(manualState.image, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, crop.width, crop.height);
}

function canvasPoint(event) {
  const rect = manualElements.canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * manualElements.canvas.width / rect.width,
    y: (event.clientY - rect.top) * manualElements.canvas.height / rect.height,
  };
}

function hitAnnotation(point) {
  for (let index = manualState.annotations.length - 1; index >= 0; index -= 1) {
    const item = manualState.annotations[index];
    const [x1, y1, x2, y2] = item.box;
    if (point.x >= x1 && point.x <= x2 && point.y >= y1 && point.y <= y2) return item;
  }
  return null;
}

function hitResizeHandle(item, point) {
  if (!item) return null;
  const threshold = Math.max(12, manualElements.canvas.width / 130);
  const [x1, y1, x2, y2] = item.box;
  const handles = {
    nw: [x1, y1], n: [(x1 + x2) / 2, y1], ne: [x2, y1],
    e: [x2, (y1 + y2) / 2], se: [x2, y2], s: [(x1 + x2) / 2, y2],
    sw: [x1, y2], w: [x1, (y1 + y2) / 2],
  };
  return Object.entries(handles).find(([, [x, y]]) => Math.hypot(point.x - x, point.y - y) <= threshold)?.[0] || null;
}

function setManualAnnotationTool(tool) {
  manualState.tool = tool;
  document.querySelectorAll("[data-annotation-tool]").forEach(item => item.classList.toggle("active", item.dataset.annotationTool === tool));
  manualElements.canvas.style.cursor = tool === "select" ? "default" : (tool === "delete" ? "not-allowed" : "crosshair");
}

function resizeCursor(handle) {
  if (["nw", "se"].includes(handle)) return "nwse-resize";
  if (["ne", "sw"].includes(handle)) return "nesw-resize";
  if (["n", "s"].includes(handle)) return "ns-resize";
  if (["e", "w"].includes(handle)) return "ew-resize";
  return "default";
}

function onCanvasPointerDown(event) {
  if (!manualState.detail || !manualState.image?.complete) return;
  event.preventDefault();
  const point = canvasPoint(event);
  manualElements.canvas.setPointerCapture(event.pointerId);
  if (["person", "head", "helmet"].includes(manualState.tool)) {
    const historyAdded = pushManualUndoSnapshot();
    const item = { id: `human-${Date.now()}`, class_name: manualState.tool, box: [point.x, point.y, point.x, point.y], truncated: false, color: "", confidence: null, source: "human" };
    manualState.annotations.push(item);
    manualState.selectedId = item.id;
    manualState.pointer = { mode: "draw", start: point, item, historyAdded };
    renderManualObjects();
    drawManualCanvas();
    return;
  }
  const selected = selectedAnnotation();
  const handle = hitResizeHandle(selected, point);
  if (manualState.tool === "select" && handle) {
    manualState.pointer = { mode: "resize", handle, start: point, item: selected, original: [...selected.box], historyAdded: pushManualUndoSnapshot() };
    return;
  }
  const hit = hitAnnotation(point);
  if (manualState.tool === "delete") {
    if (hit) deleteManualAnnotation(hit.id);
    return;
  }
  manualState.selectedId = hit?.id || null;
  if (hit) manualState.pointer = { mode: "move", start: point, item: hit, original: [...hit.box], historyAdded: pushManualUndoSnapshot() };
  renderManualObjects();
  drawManualCanvas();
}

function onCanvasPointerMove(event) {
  if (!manualState.pointer) {
    if (manualState.tool === "select") {
      const point = canvasPoint(event);
      const selected = selectedAnnotation();
      const handle = hitResizeHandle(selected, point);
      manualElements.canvas.style.cursor = handle ? resizeCursor(handle) : (hitAnnotation(point) ? "move" : "default");
    }
    return;
  }
  event.preventDefault();
  const point = canvasPoint(event);
  const state = manualState.pointer;
  if (state.mode === "draw") {
    state.item.box = [state.start.x, state.start.y, point.x, point.y];
  } else if (state.mode === "move") {
    const dx = point.x - state.start.x;
    const dy = point.y - state.start.y;
    const [x1, y1, x2, y2] = state.original;
    const width = x2 - x1;
    const height = y2 - y1;
    const nextX1 = Math.max(0, Math.min(manualElements.canvas.width - width, x1 + dx));
    const nextY1 = Math.max(0, Math.min(manualElements.canvas.height - height, y1 + dy));
    state.item.box = [nextX1, nextY1, nextX1 + width, nextY1 + height];
  } else if (state.mode === "resize") {
    const [x1, y1, x2, y2] = state.original;
    const box = [x1, y1, x2, y2];
    if (state.handle.includes("w")) box[0] = point.x;
    if (state.handle.includes("e")) box[2] = point.x;
    if (state.handle.includes("n")) box[1] = point.y;
    if (state.handle.includes("s")) box[3] = point.y;
    state.item.box = box;
  }
  drawManualCanvas();
}

function onCanvasPointerUp(event) {
  if (!manualState.pointer) return;
  const pointer = manualState.pointer;
  const item = pointer.item;
  manualState.pointer = null;
  const [rawX1, rawY1, rawX2, rawY2] = item.box;
  item.box = [Math.max(0, Math.min(rawX1, rawX2)), Math.max(0, Math.min(rawY1, rawY2)), Math.min(manualElements.canvas.width, Math.max(rawX1, rawX2)), Math.min(manualElements.canvas.height, Math.max(rawY1, rawY2))];
  if (item.box[2] - item.box[0] < 2 || item.box[3] - item.box[1] < 2) {
    manualState.annotations = manualState.annotations.filter(entry => entry.id !== item.id);
    manualState.selectedId = null;
    if (pointer.historyAdded) discardManualUndoSnapshot();
  } else if (pointer.original && pointer.original.every((value, index) => Math.abs(value - item.box[index]) < .01)) {
    if (pointer.historyAdded) discardManualUndoSnapshot();
  }
  item.truncated = item.box[0] <= 1 || item.box[1] <= 1 || item.box[2] >= manualElements.canvas.width - 1 || item.box[3] >= manualElements.canvas.height - 1;
  if (pointer.mode === "draw" && manualState.annotations.some(entry => entry.id === item.id)) setManualAnnotationTool("select");
  markManualDirty();
  renderManualObjects();
  drawManualCanvas();
  try { manualElements.canvas.releasePointerCapture(event.pointerId); } catch (_) { /* pointer already released */ }
}

function deleteManualAnnotation(id) {
  if (!manualState.annotations.some(item => item.id === id)) return;
  pushManualUndoSnapshot();
  manualState.annotations = manualState.annotations.filter(item => item.id !== id);
  if (manualState.selectedId === id) manualState.selectedId = manualState.annotations[0]?.id || null;
  markManualDirty();
  renderManualObjects();
  drawManualCanvas();
}

function renderManualObjects() {
  manualElements.objectCount.textContent = `${manualState.annotations.length} 个`;
  if (!manualState.annotations.length) {
    manualElements.objectList.innerHTML = "<p>暂无标注框；如图中不存在目标，请选择“无目标”。</p>";
  } else {
    manualElements.objectList.innerHTML = manualState.annotations.map((item, index) => `
      <button class="manual-object-row ${item.id === manualState.selectedId ? "active" : ""}" type="button" data-object-id="${manualEscape(item.id)}">
        <i class="${item.class_name}"></i><span>#${index + 1} ${item.class_name}${personHeadStatusMarkup(item)}</span><small>${Math.round(item.box[2] - item.box[0])}×${Math.round(item.box[3] - item.box[1])}</small><b data-delete-object="${manualEscape(item.id)}">×</b>
      </button>`).join("");
  }
  const selected = selectedAnnotation();
  manualElements.objectProperties.classList.toggle("hidden", !selected);
  if (selected) {
    manualElements.objectClass.value = selected.class_name;
    manualElements.objectTruncated.checked = Boolean(selected.truncated);
    manualElements.objectColor.value = selected.color || "";
    manualElements.helmetColorField.classList.toggle("hidden", selected.class_name !== "helmet");
    const status = personHeadStatus(selected);
    manualElements.personHeadStatus.classList.toggle("hidden", !status);
    if (status) {
      manualElements.personHeadStatusLabel.innerHTML = `<em class="manual-head-status-badge ${manualEscape(status.code)}">${manualEscape(status.label)}</em>`;
      const sizes = [];
      if (status.training_short_edge != null) sizes.push(`裁剪/训练 ${status.training_short_edge}px`);
      if (status.original_short_edge != null) sizes.push(`原图映射 ${status.original_short_edge}px`);
      sizes.push(`阈值 ${status.minimum_short_edge || 20}px`);
      manualElements.personHeadStatusSize.textContent = sizes.join(" · ");
      manualElements.personHeadStatusReason.textContent = status.reason || "未记录具体原因。";
    }
  } else {
    manualElements.personHeadStatus.classList.add("hidden");
  }
  drawManualCrop();
}

function markManualDirty() {
  if (!manualState.currentAlarmId || manualState.loading) return;
  manualState.dirty = true;
  manualElements.saveState.textContent = "有未保存修改";
  window.clearTimeout(manualState.autosaveTimer);
  manualState.autosaveTimer = window.setTimeout(() => saveManualDraft(true), 1800);
}

function manualPayload() {
  return {
    annotations: manualState.annotations,
    decision: manualState.decision,
    reason_codes: [...manualElements.reasonGrid.querySelectorAll("input:checked")].map(input => input.value),
    comment: manualElements.comment.value,
    reviewer: manualElements.reviewer.value,
  };
}

async function saveManualDraft(automatic = false) {
  if (!manualState.currentAlarmId || manualState.loading || (!manualState.dirty && automatic)) return;
  window.clearTimeout(manualState.autosaveTimer);
  try {
    const detail = await manualApi(`/api/manual-reviews/${manualState.currentAlarmId}/draft`, { method: "PUT", body: manualPayload() });
    manualState.detail = detail;
    manualState.dirty = false;
    manualElements.saveState.textContent = `草稿已保存 · ${formatManualTime(detail.review.updated_at)}`;
    if (!automatic) manualToast("人工复核草稿已保存", "success");
  } catch (error) {
    manualElements.saveState.textContent = `草稿保存失败：${error.message}`;
    if (!automatic) manualToast(error.message, "error");
  }
}

async function submitManualReview() {
  if (!manualState.currentAlarmId || manualState.loading) return;
  if (!manualState.decision) {
    manualToast("请先选择人工结论", "error");
    return;
  }
  window.clearTimeout(manualState.autosaveTimer);
  manualElements.submitReview.disabled = true;
  manualElements.submitReview.textContent = "正在提交…";
  try {
    const detail = await manualApi(`/api/manual-reviews/${manualState.currentAlarmId}/submit`, { method: "POST", body: manualPayload() });
    manualState.detail = detail;
    manualState.dirty = false;
    manualElements.saveState.textContent = `已提交人工版本 v${detail.review.version} · ${decisionLabel(detail.review.decision)}`;
    manualToast("人工复核已提交，AI 原始结果保持不变", "success");
    await loadManualQueue();
    const next = manualState.queue[Math.max(0, manualState.currentQueueIndex)] || manualState.queue[0];
    if (next) await openManualReview(next.id, manualState.queue.indexOf(next));
    else showManualView("queue");
  } catch (error) {
    manualToast(error.message, "error");
    manualElements.saveState.textContent = `提交失败：${error.message}`;
  } finally {
    manualElements.submitReview.disabled = false;
    manualElements.submitReview.textContent = "提交并下一条";
  }
}

function historyQuery() {
  const params = new URLSearchParams({ page: String(manualState.historyPage), page_size: "20" });
  if (manualElements.historyDecision.value) params.set("decision", manualElements.historyDecision.value);
  if (manualElements.historyReason.value) params.set("reason_code", manualElements.historyReason.value);
  if (manualElements.historyDate.value) params.set("review_date", manualElements.historyDate.value);
  return params.toString();
}

function renderManualHistory(data) {
  const items = data.items || [];
  manualState.historyPage = data.page || 1;
  manualState.historyPages = data.pages || 1;
  manualElements.historyTotal.textContent = String(data.total || 0);
  manualElements.historyPageSummary.textContent = `第 ${manualState.historyPage} / ${manualState.historyPages} 页`;
  manualElements.historyPrevious.disabled = manualState.historyPage <= 1;
  manualElements.historyNext.disabled = manualState.historyPage >= manualState.historyPages;
  if (!items.length) {
    manualElements.historyBody.innerHTML = '<tr><td colspan="9" class="manual-review-loading">暂无审核记录</td></tr>';
    return;
  }
  manualElements.historyBody.innerHTML = items.map(item => {
    const counts = item.annotation_counts || {};
    const training = { positive: "可入训练集", negative: "负样本", excluded: "不入训练集" }[item.training_eligibility] || "-";
    return `<tr>
      <td><strong>${manualEscape(item.event_id)}</strong><small>${manualEscape(item.task_code)} · ${manualEscape(item.channel_name)}</small></td>
      <td>${aiStatusMarkup(item.ai_review_status)}</td>
      <td><span class="manual-human-status completed">${decisionLabel(item.decision)}</span></td>
      <td>person ${counts.person || 0} / head ${counts.head || 0} / helmet ${counts.helmet || 0}</td>
      <td title="${manualEscape(item.comment)}">${manualEscape((item.reason_codes || []).map(reasonLabel).join("、") || "无")}</td>
      <td>${manualEscape(item.reviewer)}</td>
      <td>${manualEscape(formatManualTime(item.completed_at))}</td>
      <td><span class="manual-training-status ${item.training_eligibility}">${training}</span></td>
      <td><button class="button secondary compact" type="button" data-history-alarm="${item.alarm_id}">查看/修订</button></td>
    </tr>`;
  }).join("");
}

async function loadManualHistory() {
  manualElements.historyBody.innerHTML = '<tr><td colspan="9" class="manual-review-loading">正在加载审核记录…</td></tr>';
  try {
    renderManualHistory(await manualApi(`/api/manual-reviews/history?${historyQuery()}`));
  } catch (error) {
    manualElements.historyBody.innerHTML = `<tr><td colspan="9" class="manual-review-loading error-text">${manualEscape(error.message)}</td></tr>`;
  }
}

function moveQueueSample(offset) {
  if (!manualState.queue.length) return;
  const nextIndex = Math.max(0, Math.min(manualState.queue.length - 1, manualState.currentQueueIndex + offset));
  const item = manualState.queue[nextIndex];
  if (item) openManualReview(item.id, nextIndex);
}

manualElements.tabs.forEach(tab => tab.addEventListener("click", () => showManualView(tab.dataset.reviewView)));
document.querySelector("#manual-query-button").addEventListener("click", () => { manualState.queuePage = 1; loadManualQueue(); });
document.querySelector("#manual-reset-button").addEventListener("click", () => {
  manualElements.aiFilter.value = "";
  manualElements.humanFilter.value = "pending";
  manualElements.taskFilter.value = "";
  manualElements.channelFilter.value = "";
  manualElements.dateFilter.value = "";
  manualState.queuePage = 1;
  loadManualQueue();
});
manualElements.queuePrevious.addEventListener("click", () => { if (manualState.queuePage > 1) { manualState.queuePage -= 1; loadManualQueue(); } });
manualElements.queueNext.addEventListener("click", () => { if (manualState.queuePage < manualState.queuePages) { manualState.queuePage += 1; loadManualQueue(); } });
manualElements.queueList.addEventListener("click", event => {
  const button = event.target.closest("[data-review-alarm]");
  if (button) openManualReview(button.dataset.reviewAlarm, button.dataset.reviewIndex);
});
manualElements.sampleRail.addEventListener("click", event => {
  const button = event.target.closest("[data-rail-alarm]");
  if (button) openManualReview(button.dataset.railAlarm, button.dataset.railIndex);
});
document.querySelector("#manual-back-queue").addEventListener("click", () => showManualView("queue"));
document.querySelector("#manual-previous-sample").addEventListener("click", () => moveQueueSample(-1));
document.querySelector("#manual-next-sample").addEventListener("click", () => moveQueueSample(1));

document.querySelectorAll("[data-annotation-tool]").forEach(button => button.addEventListener("click", () => {
  setManualAnnotationTool(button.dataset.annotationTool);
}));
document.querySelectorAll("[data-image-kind]").forEach(button => button.addEventListener("click", () => {
  if (!manualState.detail) return;
  manualState.imageKind = button.dataset.imageKind;
  document.querySelectorAll("[data-image-kind]").forEach(item => item.classList.toggle("active", item === button));
  loadCanvasImage(manualState.imageKind === "alarm" ? manualState.detail.image.alarm_url : manualState.detail.image.original_url);
}));
manualElements.showCandidates.addEventListener("change", () => {
  manualState.showCandidates = manualElements.showCandidates.checked;
  drawManualCanvas();
});
document.querySelector("#manual-reset-annotations").addEventListener("click", () => {
  if (!manualState.detail) return;
  if (!window.confirm("确定用模型候选框替换当前人工修改吗？")) return;
  pushManualUndoSnapshot();
  manualState.annotations = cloneAnnotations(manualState.candidates);
  manualState.selectedId = manualState.annotations[0]?.id || null;
  markManualDirty();
  renderManualObjects();
  drawManualCanvas();
});
manualElements.undoAnnotation.addEventListener("click", undoManualAnnotation);

manualElements.canvas.addEventListener("pointerdown", onCanvasPointerDown);
manualElements.canvas.addEventListener("pointermove", onCanvasPointerMove);
manualElements.canvas.addEventListener("pointerup", onCanvasPointerUp);
manualElements.canvas.addEventListener("pointercancel", onCanvasPointerUp);
manualElements.canvasStage.addEventListener("wheel", onManualCanvasWheel, { passive: false });
manualElements.zoomDisplay.addEventListener("click", () => setManualZoom(1));
window.addEventListener("resize", resizeManualCanvas);

manualElements.objectList.addEventListener("click", event => {
  const deleteButton = event.target.closest("[data-delete-object]");
  if (deleteButton) {
    event.stopPropagation();
    deleteManualAnnotation(deleteButton.dataset.deleteObject);
    return;
  }
  const row = event.target.closest("[data-object-id]");
  if (!row) return;
  manualState.selectedId = row.dataset.objectId;
  renderManualObjects();
  drawManualCanvas();
});
manualElements.objectClass.addEventListener("change", () => {
  const selected = selectedAnnotation();
  if (!selected) return;
  pushManualUndoSnapshot();
  selected.class_name = manualElements.objectClass.value;
  if (selected.class_name !== "helmet") selected.color = "";
  markManualDirty();
  renderManualObjects();
  drawManualCanvas();
});
manualElements.objectTruncated.addEventListener("change", () => {
  const selected = selectedAnnotation();
  if (!selected) return;
  pushManualUndoSnapshot();
  selected.truncated = manualElements.objectTruncated.checked;
  markManualDirty();
});
manualElements.objectColor.addEventListener("input", () => {
  const selected = selectedAnnotation();
  if (!selected || selected.class_name !== "helmet") return;
  pushManualUndoSnapshot();
  selected.color = manualElements.objectColor.value;
  markManualDirty();
});

manualElements.reasonGrid.addEventListener("change", markManualDirty);
manualElements.comment.addEventListener("input", markManualDirty);
manualElements.reviewer.addEventListener("input", markManualDirty);
manualElements.decisionGrid.addEventListener("click", event => {
  const button = event.target.closest("[data-decision]");
  if (button) setDecision(button.dataset.decision);
});
manualElements.saveDraft.addEventListener("click", () => saveManualDraft(false));
manualElements.submitReview.addEventListener("click", submitManualReview);

document.querySelector("#manual-history-query").addEventListener("click", () => { manualState.historyPage = 1; loadManualHistory(); });
manualElements.historyPrevious.addEventListener("click", () => { if (manualState.historyPage > 1) { manualState.historyPage -= 1; loadManualHistory(); } });
manualElements.historyNext.addEventListener("click", () => { if (manualState.historyPage < manualState.historyPages) { manualState.historyPage += 1; loadManualHistory(); } });
manualElements.historyBody.addEventListener("click", event => {
  const button = event.target.closest("[data-history-alarm]");
  if (button) openManualReview(button.dataset.historyAlarm, -1);
});

document.addEventListener("keydown", event => {
  if (manualState.view !== "workbench" || !manualState.currentAlarmId) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveManualDraft(false);
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) {
    event.preventDefault();
    undoManualAnnotation();
  }
  if (event.key === "Delete" && manualState.selectedId && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) {
    deleteManualAnnotation(manualState.selectedId);
  }
});

Promise.all([loadManualFilterOptions(), loadManualQueue()]);
