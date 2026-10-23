"use strict";

const alarmVisibilityStorageKey = "videoAlarms.showUnverified";
let alarmRequestVersion = 0;

const alarmState = {
  items: [],
  tasks: [],
  channels: [],
  selected: new Set(),
  page: 1,
  pages: 1,
  pageSize: 12,
  total: 0,
};

const alarmElements = {
  grid: document.querySelector("#alarm-card-grid"),
  empty: document.querySelector("#alarm-empty"),
  taskFilter: document.querySelector("#alarm-task-filter"),
  channelFilter: document.querySelector("#alarm-channel-filter"),
  dateFilter: document.querySelector("#alarm-date-filter"),
  reviewFilter: document.querySelector("#alarm-review-filter"),
  showUnverified: document.querySelector("#alarm-show-unverified"),
  visibilityHint: document.querySelector("#alarm-visibility-hint"),
  total: document.querySelector("#alarm-total"),
  capacity: document.querySelector("#alarm-capacity"),
  deleteSelected: document.querySelector("#delete-selected-alarms"),
  selectedCount: document.querySelector("#selected-alarm-count"),
  clear: document.querySelector("#clear-alarms"),
  previous: document.querySelector("#alarm-previous"),
  next: document.querySelector("#alarm-next"),
  pageNumbers: document.querySelector("#alarm-page-numbers"),
  pageSummary: document.querySelector("#alarm-page-summary"),
  imageModal: document.querySelector("#alarm-image-modal"),
  imageTitle: document.querySelector("#alarm-image-title"),
  imagePreview: document.querySelector("#alarm-image-preview"),
  toastRegion: document.querySelector("#toast-region"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function alarmApi(url, options = {}) {
  const request = { cache: "no-store", ...options, headers: { ...(options.headers || {}) } };
  if (options.body && typeof options.body !== "string") {
    request.body = JSON.stringify(options.body);
    request.headers["Content-Type"] = "application/json";
  }
  const response = await fetch(url, request);
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(data?.detail || `请求失败（${response.status}）`);
  return data;
}

function alarmToast(message, type = "") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`.trim();
  toast.textContent = message;
  alarmElements.toastRegion.append(toast);
  setTimeout(() => toast.remove(), 3400);
}

function formatAlarmTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value || "—";
  return date.toLocaleString("zh-CN", { hour12: false });
}

function reportStatus(status) {
  return {
    local_only: { label: "仅本地保存", className: "local" },
    pending: { label: "待上报", className: "pending" },
    reporting: { label: "正在上报", className: "pending" },
    success: { label: "上报成功", className: "success" },
    failed: { label: "上报失败", className: "failed" },
    review_blocked: { label: "复核未通过，未上报", className: "review-blocked" },
  }[status] || { label: "未知", className: "local" };
}

function reviewStatus(status) {
  return {
    confirmed: { label: "confirmed", className: "confirmed" },
    rejected: { label: "rejected", className: "rejected" },
    uncertain: { label: "uncertain", className: "uncertain" },
    pending: { label: "pending", className: "pending-review" },
    reviewing: { label: "reviewing", className: "pending-review" },
  }[status] || { label: "uncertain", className: "uncertain" };
}

function manualReportButton(item) {
  if (item.review_status !== "uncertain") return "";
  let label = "手动上报";
  let disabled = "";
  let title = "人工确认后绕过复核门禁并上报告警";
  if (!item.report_url) {
    label = "未配置上报地址";
    disabled = "disabled";
    title = "该告警没有配置上报地址";
  } else if (item.report_status === "success") {
    label = "已上报";
    disabled = "disabled";
    title = "该告警已经上报成功";
  } else if (item.report_status === "reporting") {
    label = "正在上报";
    disabled = "disabled";
  } else if (!item.can_manual_report) {
    disabled = "disabled";
  }
  return `<div class="alarm-card-actions"><button class="button alarm-manual-report" type="button" data-manual-report="${item.id}" title="${escapeHtml(title)}" ${disabled}>${label}</button></div>`;
}

function renderAlarmCards() {
  alarmElements.total.textContent = String(alarmState.total);
  alarmElements.empty.classList.toggle("hidden", alarmState.items.length > 0);
  alarmElements.grid.classList.toggle("hidden", alarmState.items.length === 0);
  if (!alarmState.items.length) {
    alarmElements.grid.innerHTML = "";
    return;
  }
  alarmElements.grid.innerHTML = alarmState.items.map(item => {
    const status = reportStatus(item.report_status);
    const review = reviewStatus(item.review_status);
    return `
      <article class="alarm-card review-${review.className}" data-alarm-id="${item.id}">
        <header class="alarm-card-header">
          <label class="alarm-check"><input type="checkbox" data-select-alarm="${item.id}" aria-label="选择告警 ${escapeHtml(item.task_code)}" ${alarmState.selected.has(item.id) ? "checked" : ""}><i></i></label>
          <strong>告警任务：${escapeHtml(item.task_code)}</strong>
        </header>
        <button class="alarm-card-image" type="button" data-preview-alarm="${item.id}" aria-label="查看 ${escapeHtml(item.task_code)} 告警图片">
          <img src="${escapeHtml(item.image_url)}?v=${encodeURIComponent(item.event_id)}" alt="${escapeHtml(item.alarm_type)}">
          <span>${escapeHtml(item.alarm_type)}</span>
        </button>
        <dl class="alarm-card-details">
          <div><dt>视频通道</dt><dd>${escapeHtml(item.channel_name)}</dd></div>
          <div><dt>告警日期</dt><dd>${escapeHtml(formatAlarmTime(item.alarm_time))}</dd></div>
          <div><dt>上报地址</dt><dd title="${escapeHtml(item.report_url)}">${escapeHtml(item.report_url || "未配置，仅本地保存")}</dd></div>
          <div><dt>上报状态</dt><dd><span class="alarm-report-status ${status.className}"><i></i>${status.label}</span></dd></div>
          <div><dt>复核结果</dt><dd title="${escapeHtml(item.review_summary || item.review_error)}"><span class="alarm-review-status ${review.className}"><i></i>${review.label}</span></dd></div>
          <div class="alarm-content-row"><dt>告警内容</dt><dd>${escapeHtml(item.alarm_content)}</dd></div>
        </dl>
        ${manualReportButton(item)}
      </article>`;
  }).join("");
}

function updateSelectedState() {
  alarmElements.selectedCount.textContent = String(alarmState.selected.size);
  alarmElements.deleteSelected.disabled = alarmState.selected.size === 0;
}

function renderPagination() {
  alarmElements.previous.disabled = alarmState.page <= 1;
  alarmElements.next.disabled = alarmState.page >= alarmState.pages;
  alarmElements.pageSummary.textContent = `第 ${alarmState.page} / ${alarmState.pages} 页`;
  const start = Math.max(1, alarmState.page - 2);
  const end = Math.min(alarmState.pages, start + 4);
  alarmElements.pageNumbers.innerHTML = Array.from({ length: end - start + 1 }, (_, index) => {
    const page = start + index;
    return `<button class="page-number ${page === alarmState.page ? "active" : ""}" type="button" data-alarm-page="${page}">${page}</button>`;
  }).join("");
}

function filterQuery() {
  const params = new URLSearchParams({ page: String(alarmState.page), page_size: String(alarmState.pageSize) });
  if (alarmElements.taskFilter.value) params.set("task_id", alarmElements.taskFilter.value);
  if (alarmElements.channelFilter.value) params.set("channel_id", alarmElements.channelFilter.value);
  if (alarmElements.dateFilter.value) params.set("alarm_date", alarmElements.dateFilter.value);
  const review = alarmElements.showUnverified.checked ? alarmElements.reviewFilter.value : "confirmed";
  if (review) params.set("review_status", review);
  return params.toString();
}

async function loadAlarms(showMessage = false) {
  const requestVersion = ++alarmRequestVersion;
  alarmElements.grid.innerHTML = '<div class="alarm-loading">正在加载告警记录…</div>';
  alarmElements.grid.classList.remove("hidden");
  alarmElements.empty.classList.add("hidden");
  try {
    const data = await alarmApi(`/api/video-alarms?${filterQuery()}`);
    if (requestVersion !== alarmRequestVersion) return;
    alarmState.items = data.items || [];
    alarmState.total = data.total || 0;
    alarmState.page = data.page || 1;
    alarmState.pages = data.pages || 1;
    alarmElements.capacity.textContent = `${data.max_records || 2000} 条`;
    renderAlarmCards();
    renderPagination();
    updateSelectedState();
    if (showMessage) alarmToast("告警记录已刷新", "success");
  } catch (error) {
    if (requestVersion !== alarmRequestVersion) return;
    alarmElements.grid.innerHTML = `<div class="alarm-loading error-text">${escapeHtml(error.message)}</div>`;
    alarmToast(error.message, "error");
  }
}

async function loadFilterOptions() {
  try {
    const [tasks, channels] = await Promise.all([alarmApi("/api/video-tasks"), alarmApi("/api/video-channels")]);
    alarmState.tasks = tasks.items || [];
    alarmState.channels = channels.items || [];
    alarmElements.taskFilter.innerHTML = '<option value="">全部任务</option>' + alarmState.tasks
      .map(task => `<option value="${task.id}">${escapeHtml(task.task_code)}</option>`).join("");
    alarmElements.channelFilter.innerHTML = '<option value="">全部通道</option>' + alarmState.channels
      .map(channel => `<option value="${channel.id}">${escapeHtml(channel.name)}</option>`).join("");
  } catch (error) {
    alarmToast(`筛选项加载失败：${error.message}`, "error");
  }
}

function resetFilters() {
  alarmElements.taskFilter.value = "";
  alarmElements.channelFilter.value = "";
  alarmElements.dateFilter.value = "";
  alarmElements.reviewFilter.value = alarmElements.showUnverified.checked ? "" : "confirmed";
  alarmState.page = 1;
  alarmState.selected.clear();
  loadAlarms();
}

async function deleteSelectedAlarms() {
  if (!alarmState.selected.size) return;
  const confirmed = window.confirm(`确定删除选中的 ${alarmState.selected.size} 条告警记录及其本地事件文件吗？\n删除后无法恢复。`);
  if (!confirmed) return;
  try {
    const result = await alarmApi("/api/video-alarms/delete", { method: "POST", body: { ids: [...alarmState.selected] } });
    alarmState.selected.clear();
    await loadAlarms();
    alarmToast(`已删除 ${result.deleted} 条告警记录`, "success");
  } catch (error) {
    alarmToast(error.message, "error");
  }
}

async function clearAlarmRecords() {
  if (!alarmState.total) return;
  const confirmed = window.confirm(`确定清空所有告警记录及其本地事件文件吗？包括被开关隐藏及不符合当前筛选条件的记录。\n该操作无法恢复。`);
  if (!confirmed) return;
  try {
    const result = await alarmApi("/api/video-alarms", { method: "DELETE" });
    alarmState.selected.clear();
    alarmState.page = 1;
    await loadAlarms();
    alarmToast(`已清空 ${result.deleted} 条告警记录`, "success");
  } catch (error) {
    alarmToast(error.message, "error");
  }
}

function openAlarmImage(item) {
  alarmElements.imageTitle.textContent = `${item.task_code} · ${item.alarm_type}`;
  alarmElements.imagePreview.src = `${item.image_url}?v=${encodeURIComponent(item.event_id)}`;
  alarmElements.imageModal.classList.remove("hidden");
}

function closeAlarmImage() {
  alarmElements.imageModal.classList.add("hidden");
  alarmElements.imagePreview.removeAttribute("src");
}

async function manualReportAlarm(item, button) {
  const confirmed = window.confirm(
    `确定手动上报这条告警吗？\n\n告警任务：${item.task_code}\n视频通道：${item.channel_name}\n告警时间：${formatAlarmTime(item.alarm_time)}\n复核结果：uncertain\n\n确认后将绕过大模型复核门禁，并向已配置地址发送告警。`
  );
  if (!confirmed) return;
  button.disabled = true;
  button.textContent = "正在上报…";
  try {
    const result = await alarmApi(`/api/video-alarms/${item.id}/manual-report`, {
      method: "POST",
      body: { confirmed: true },
    });
    if (!result.success) throw new Error(result.error || "手动上报失败");
    await loadAlarms();
    alarmToast("手动上报成功", "success");
  } catch (error) {
    button.disabled = false;
    button.textContent = "重新手动上报";
    alarmToast(error.message, "error");
  }
}

alarmElements.grid.addEventListener("change", event => {
  const checkbox = event.target.closest("[data-select-alarm]");
  if (!checkbox) return;
  const alarmId = Number(checkbox.dataset.selectAlarm);
  if (checkbox.checked) alarmState.selected.add(alarmId);
  else alarmState.selected.delete(alarmId);
  updateSelectedState();
});

alarmElements.grid.addEventListener("click", event => {
  const manualButton = event.target.closest("[data-manual-report]");
  if (manualButton) {
    const item = alarmState.items.find(alarm => alarm.id === Number(manualButton.dataset.manualReport));
    if (item) manualReportAlarm(item, manualButton);
    return;
  }
  const button = event.target.closest("[data-preview-alarm]");
  if (!button) return;
  const item = alarmState.items.find(alarm => alarm.id === Number(button.dataset.previewAlarm));
  if (item) openAlarmImage(item);
});

alarmElements.pageNumbers.addEventListener("click", event => {
  const button = event.target.closest("[data-alarm-page]");
  if (!button) return;
  alarmState.page = Number(button.dataset.alarmPage);
  loadAlarms();
});
alarmElements.previous.addEventListener("click", () => { if (alarmState.page > 1) { alarmState.page -= 1; loadAlarms(); } });
alarmElements.next.addEventListener("click", () => { if (alarmState.page < alarmState.pages) { alarmState.page += 1; loadAlarms(); } });
document.querySelector("#alarm-query-button").addEventListener("click", () => { alarmState.page = 1; loadAlarms(); });
document.querySelector("#alarm-reset-button").addEventListener("click", resetFilters);
alarmElements.deleteSelected.addEventListener("click", deleteSelectedAlarms);
alarmElements.clear.addEventListener("click", clearAlarmRecords);
document.querySelectorAll("[data-close-alarm-image]").forEach(item => item.addEventListener("click", closeAlarmImage));
document.addEventListener("keydown", event => { if (event.key === "Escape") closeAlarmImage(); });

function syncAlarmVisibility() {
  const showAll = alarmElements.showUnverified.checked;
  alarmElements.reviewFilter.disabled = !showAll;
  alarmElements.reviewFilter.value = showAll ? "" : "confirmed";
  alarmElements.visibilityHint.textContent = showAll
    ? "已开放全部复核状态（含待复核和复核中），可使用复核结果筛选"
    : "仅展示复核已确认的告警；存疑、已拒绝及尚未完成复核的记录已隐藏";
}

try {
  alarmElements.showUnverified.checked = localStorage.getItem(alarmVisibilityStorageKey) === "true";
} catch (_) {
  alarmElements.showUnverified.checked = false;
}
syncAlarmVisibility();
alarmElements.showUnverified.addEventListener("change", () => {
  try { localStorage.setItem(alarmVisibilityStorageKey, String(alarmElements.showUnverified.checked)); } catch (_) {}
  syncAlarmVisibility();
  alarmState.page = 1;
  alarmState.selected.clear();
  updateSelectedState();
  closeAlarmImage();
  loadAlarms();
});

Promise.all([loadFilterOptions(), loadAlarms()]);
