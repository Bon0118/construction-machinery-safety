"use strict";

const taskState = {
  tasks: [],
  channels: [],
  meta: { default_report_url: "", worker_connected: false, algorithms: [] },
  editingId: null,
  selectedChannelId: null,
  openMenuTaskId: null,
};

const taskElements = {
  tableBody: document.querySelector("#task-table-body"),
  tableWrap: document.querySelector(".task-table-wrap"),
  empty: document.querySelector("#task-empty-state"),
  search: document.querySelector("#task-search"),
  total: document.querySelector("#task-total"),
  running: document.querySelector("#task-running"),
  drawer: document.querySelector("#task-drawer"),
  drawerTitle: document.querySelector("#task-drawer-title"),
  form: document.querySelector("#task-form"),
  formError: document.querySelector("#task-form-error"),
  saveButton: document.querySelector("#save-task-button"),
  sourceDrawer: document.querySelector("#source-drawer"),
  sourceList: document.querySelector("#source-choice-list"),
  sourceSearch: document.querySelector("#source-search"),
  selectedSourceText: document.querySelector("#selected-source-text"),
  algorithmList: document.querySelector("#algorithm-option-list"),
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

function errorMessage(data, fallback = "操作失败") {
  if (!data) return fallback;
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) return data.detail.map(item => item.msg || String(item)).join("；");
  return data.message || fallback;
}

async function taskApi(url, options = {}) {
  const requestOptions = { cache: "no-store", ...options };
  requestOptions.headers = { ...(options.headers || {}) };
  if (options.body && typeof options.body !== "string") {
    requestOptions.body = JSON.stringify(options.body);
    requestOptions.headers["Content-Type"] = "application/json";
  }
  const response = await fetch(url, requestOptions);
  if (response.status === 204) return null;
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) throw new Error(errorMessage(data, `请求失败（${response.status}）`));
  return data;
}

function taskToast(message, type = "") {
  const item = document.createElement("div");
  item.className = `toast ${type}`.trim();
  item.textContent = message;
  taskElements.toastRegion.append(item);
  setTimeout(() => item.remove(), 3400);
}

function sourceTypeText(sourceType) {
  return {
    camera_index_code: "cameraIndexCode",
    rtsp: "直接 RTSP",
    local_file: "本地视频",
  }[sourceType] || "未知类型";
}

function algorithmMarkup(task) {
  const algorithms = task.algorithms || [];
  const intervals = task.algorithm_alarm_intervals || {};
  if (!algorithms.length) return '<span class="task-muted">未选择算法</span>';
  return algorithms
    .map(item => `<span class="task-algorithm-tag" title="告警间隔 ${escapeHtml(formatAlarmInterval(intervals[item.code]))} 秒"><i>✓</i> ${escapeHtml(item.name)} · ${escapeHtml(formatAlarmInterval(intervals[item.code]))}秒</span>`)
    .join(" ");
}

function formatAlarmInterval(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "30";
  return Number.isInteger(number) ? String(number) : String(Number(number.toFixed(3)));
}

function alarmIntervalInput(algorithmCode) {
  return Array.from(document.querySelectorAll("[data-algorithm-interval]"))
    .find(input => input.dataset.algorithmInterval === algorithmCode) || null;
}

function algorithmParameterMarkup(item, parameterValues, checked) {
  const definitions = item.parameters || [];
  if (!definitions.length) return "";
  return `
    <div class="algorithm-parameter-grid">
      ${definitions.map(definition => {
        const value = parameterValues[definition.key] ?? definition.default;
        const maximum = definition.max === null || definition.max === undefined
          ? ""
          : `max="${escapeHtml(definition.max)}"`;
        return `
          <label class="algorithm-parameter-field">
            <span>${escapeHtml(definition.name)}</span>
            <span class="algorithm-parameter-input">
              <input type="number" required
                min="${escapeHtml(definition.min)}" ${maximum}
                step="${escapeHtml(definition.step)}"
                data-algorithm-parameter-code="${escapeHtml(item.code)}"
                data-algorithm-parameter-key="${escapeHtml(definition.key)}"
                value="${escapeHtml(formatAlarmInterval(value))}" ${checked ? "" : "disabled"}>
              ${definition.unit ? `<b>${escapeHtml(definition.unit)}</b>` : ""}
            </span>
            ${definition.help ? `<small>${escapeHtml(definition.help)}</small>` : ""}
          </label>`;
      }).join("")}
    </div>`;
}

function renderAlgorithmOptions(
  selectedCodes = ["machinery_hazard"],
  alarmIntervals = {},
  algorithmParameters = {},
) {
  const selected = new Set(selectedCodes || []);
  const algorithms = taskState.meta.algorithms || [];
  if (!algorithms.length) {
    taskElements.algorithmList.innerHTML = '<div class="algorithm-options-loading">暂无可用算法</div>';
    return;
  }
  taskElements.algorithmList.innerHTML = algorithms.map((item, index) => {
    const checked = selected.has(item.code);
    const interval = alarmIntervals[item.code] ?? item.default_alarm_interval_seconds ?? 30;
    const parameterValues = algorithmParameters[item.code] || {};
    return `
      <div class="algorithm-option">
        <label class="algorithm-option-choice">
          <input type="checkbox" data-algorithm-code="${escapeHtml(item.code)}" ${checked ? "checked" : ""}>
          <i></i>
          <span>
            <strong>${escapeHtml(item.name)}</strong>
            <small>${escapeHtml(item.description || "目标检测算法")}</small>
          </span>
          <em>${index === 0 ? "风险算法" : "YOLOv5"}</em>
        </label>
        <label class="algorithm-interval-field">
          <span>告警间隔</span>
          <span class="algorithm-interval-input">
            <input type="number" min="0" step="1" required
              data-algorithm-interval="${escapeHtml(item.code)}"
              value="${escapeHtml(formatAlarmInterval(interval))}" ${checked ? "" : "disabled"}>
            <b>秒</b>
          </span>
        </label>
        ${algorithmParameterMarkup(item, parameterValues, checked)}
      </div>`;
  }).join("");
}

function renderTaskTable() {
  const query = taskElements.search.value.trim().toLocaleLowerCase();
  const filtered = taskState.tasks.filter(task => {
    const haystack = [task.task_code, task.channel_name, task.report_url].join(" ").toLocaleLowerCase();
    return !query || haystack.includes(query);
  });

  taskElements.total.textContent = String(taskState.tasks.length);
  taskElements.running.textContent = String(taskState.tasks.filter(task => task.status === "running").length);
  taskElements.empty.classList.toggle("hidden", taskState.tasks.length > 0);
  taskElements.tableWrap.classList.toggle("hidden", taskState.tasks.length === 0);

  if (!filtered.length && taskState.tasks.length) {
    taskElements.tableBody.innerHTML = '<tr><td class="loading-cell" colspan="6">没有符合搜索条件的视频任务。</td></tr>';
    return;
  }

  taskElements.tableBody.innerHTML = filtered.map(task => {
    const running = task.status === "running";
    const runtimeState = task.runtime_state || (running ? "starting" : "stopped");
    const runtimeLabels = {
      starting: "启动中",
      running: "检测中",
      event_running: "事件检测中",
      scheduled_idle: "时段外已断流",
      waiting_event: "等待事件",
      stopping: "停止中",
      completed: "已完成",
      failed: "运行失败",
      stopped: "未运行",
    };
    const runtimeActive = ["starting", "running", "event_running", "stopping"].includes(runtimeState);
    const runtimeClass = runtimeState === "failed" ? "failed" : (runtimeActive ? "running" : "stopped");
    const runtimeTitle = task.runtime_error || task.runtime_stop_reason || "";
    const channelMissing = task.channel_id === null;
    const menuOpen = taskState.openMenuTaskId === task.id;
    return `
      <tr data-task-id="${task.id}">
        <td><span class="task-code">${escapeHtml(task.task_code)}</span><span class="task-row-id">#${task.id}</span></td>
        <td>
          <span class="task-source-name ${channelMissing ? "missing" : ""}">${escapeHtml(task.channel_name)}</span>
          <span class="task-source-meta">${escapeHtml(sourceTypeText(task.channel_source_type))}${!task.channel_enabled && !channelMissing ? " · 已停用" : ""}</span>
        </td>
        <td>${task.report_url
          ? `<span class="task-report-url" title="${escapeHtml(task.report_url)}">${escapeHtml(task.report_url)}</span>`
          : '<span class="task-local-only">仅本地保存</span>'}</td>
        <td>${algorithmMarkup(task)}</td>
        <td><span class="task-status ${runtimeClass}" title="${escapeHtml(runtimeTitle)}"><i></i>${runtimeLabels[runtimeState] || runtimeState}</span></td>
        <td>
          <div class="task-actions">
            <button class="task-text-action" type="button" data-action="edit" ${running ? "disabled title=\"请先停止任务后再编辑\"" : ""}>编辑</button>
            <button class="task-text-action ${running ? "stop" : "start"}" type="button" data-action="toggle">${running ? "停止" : "启动"}</button>
            <div class="task-options">
              <button class="task-text-action" type="button" data-action="options">选项 <span>⌄</span></button>
              <div class="task-options-menu ${menuOpen ? "" : "hidden"}">
                <button type="button" data-action="delete"><span>×</span> 删除任务</button>
              </div>
            </div>
          </div>
        </td>
      </tr>`;
  }).join("");
}

function selectedChannel() {
  return taskState.channels.find(channel => channel.id === taskState.selectedChannelId) || null;
}

function generateTaskCode() {
  const now = new Date();
  const part = value => String(value).padStart(2, "0");
  return `TASK-${now.getFullYear()}${part(now.getMonth() + 1)}${part(now.getDate())}-${part(now.getHours())}${part(now.getMinutes())}${part(now.getSeconds())}`;
}

function updateSelectedSource() {
  const channel = selectedChannel();
  taskElements.selectedSourceText.innerHTML = channel
    ? `<strong>${escapeHtml(channel.name)}</strong><small>${escapeHtml(sourceTypeText(channel.source_type))}${channel.enabled ? "" : " · 已停用"}</small>`
    : "点击选择已配置的视频通道";
}

function setMapping(mapping = {}) {
  document.querySelectorAll("[data-mapping]").forEach(input => {
    input.value = mapping[input.dataset.mapping] || "";
  });
}

function openTaskDrawer(task = null) {
  taskState.editingId = task?.id || null;
  taskState.selectedChannelId = task?.channel_id || null;
  taskElements.drawerTitle.textContent = task ? "编辑视频任务" : "新增视频任务";
  document.querySelector("#task-id").value = task?.id || "";
  document.querySelector("#task-code").value = task?.task_code || generateTaskCode();
  document.querySelector("#task-report-url").value = task
    ? (task.report_url ?? "")
    : (taskState.meta.default_report_url || "");
  renderAlgorithmOptions(
    task ? (task.algorithm_codes || []) : ["machinery_hazard"],
    task?.algorithm_alarm_intervals || {},
    task?.algorithm_parameters || {},
  );
  const policy = task?.runtime_policy || {};
  document.querySelector("#policy-schedule-enabled").checked = Boolean(policy.schedule_enabled);
  document.querySelector("#policy-schedule-start").value = policy.schedule_start || "00:00";
  document.querySelector("#policy-schedule-end").value = policy.schedule_end || "23:59";
  document.querySelector("#policy-event-wakeup-enabled").checked = Boolean(policy.event_wakeup_enabled);
  document.querySelector("#policy-event-wakeup-seconds").value = policy.event_wakeup_seconds || 300;
  document.querySelector("#policy-main-confirm-enabled").checked = Boolean(policy.main_stream_confirmation_enabled);
  document.querySelector("#policy-small-area-ratio").value = policy.small_target_area_ratio || 0.02;
  document.querySelector("#policy-confirm-timeout").value = policy.confirmation_timeout_seconds || 10;
  setMapping(task?.report_mapping || {});
  updateSelectedSource();
  taskElements.formError.classList.add("hidden");
  taskElements.formError.textContent = "";
  taskElements.drawer.classList.remove("hidden");
  document.querySelector("#task-code").focus();
}

function closeSourceDrawer() {
  taskElements.sourceDrawer.classList.add("hidden");
  taskElements.sourceSearch.value = "";
}

function closeTaskDrawer() {
  closeSourceDrawer();
  taskElements.drawer.classList.add("hidden");
  taskElements.form.reset();
  taskState.editingId = null;
  taskState.selectedChannelId = null;
}

function renderSourceChoices() {
  const query = taskElements.sourceSearch.value.trim().toLocaleLowerCase();
  const channels = taskState.channels.filter(channel => {
    const haystack = [channel.name, channel.camera_index_code, channel.camera_id].join(" ").toLocaleLowerCase();
    return !query || haystack.includes(query);
  });
  if (!channels.length) {
    taskElements.sourceList.innerHTML = '<div class="source-choice-empty">没有可选择的视频通道</div>';
    return;
  }
  taskElements.sourceList.innerHTML = channels.map(channel => `
    <button class="source-choice ${channel.id === taskState.selectedChannelId ? "selected" : ""}" type="button" data-channel-id="${channel.id}">
      <span class="source-choice-radio"></span>
      <span class="source-choice-main"><strong>${escapeHtml(channel.name)}</strong><small>${escapeHtml(sourceTypeText(channel.source_type))}</small></span>
      <span class="source-choice-state ${channel.enabled ? "enabled" : "disabled"}"><i></i>${channel.enabled ? "已启用" : "已停用"}</span>
    </button>`).join("");
}

function openSourceDrawer() {
  renderSourceChoices();
  taskElements.sourceDrawer.classList.remove("hidden");
  taskElements.sourceSearch.focus();
}

function taskPayload() {
  const mapping = {};
  document.querySelectorAll("[data-mapping]").forEach(input => {
    mapping[input.dataset.mapping] = input.value.trim();
  });
  const algorithmCodes = Array.from(document.querySelectorAll("[data-algorithm-code]:checked"))
    .map(input => input.dataset.algorithmCode);
  const alarmIntervals = {};
  const algorithmParameters = {};
  algorithmCodes.forEach(code => {
    const input = alarmIntervalInput(code);
    alarmIntervals[code] = Number(input.value);
    const parameters = {};
    document.querySelectorAll(`[data-algorithm-parameter-code="${code}"]`).forEach(parameterInput => {
      parameters[parameterInput.dataset.algorithmParameterKey] = Number(parameterInput.value);
    });
    if (Object.keys(parameters).length) algorithmParameters[code] = parameters;
  });
  return {
    task_code: document.querySelector("#task-code").value.trim(),
    channel_id: taskState.selectedChannelId,
    report_url: document.querySelector("#task-report-url").value.trim(),
    algorithm_codes: algorithmCodes,
    algorithm_alarm_intervals: alarmIntervals,
    algorithm_parameters: algorithmParameters,
    report_mapping: mapping,
    runtime_policy: {
      schedule_enabled: document.querySelector("#policy-schedule-enabled").checked,
      schedule_start: document.querySelector("#policy-schedule-start").value || "00:00",
      schedule_end: document.querySelector("#policy-schedule-end").value || "23:59",
      event_wakeup_enabled: document.querySelector("#policy-event-wakeup-enabled").checked,
      event_wakeup_seconds: Number(document.querySelector("#policy-event-wakeup-seconds").value || 300),
      main_stream_confirmation_enabled: document.querySelector("#policy-main-confirm-enabled").checked,
      small_target_area_ratio: Number(document.querySelector("#policy-small-area-ratio").value || 0.02),
      confirmation_timeout_seconds: Number(document.querySelector("#policy-confirm-timeout").value || 10),
    },
  };
}

async function loadTaskData(showMessage = false) {
  try {
    const [tasks, channels, meta] = await Promise.all([
      taskApi("/api/video-tasks"),
      taskApi("/api/video-channels"),
      taskApi("/api/video-tasks/meta"),
    ]);
    taskState.tasks = tasks.items || [];
    taskState.channels = channels.items || [];
    taskState.meta = meta;
    if (taskElements.drawer.classList.contains("hidden")) {
      renderAlgorithmOptions(["machinery_hazard"]);
    }
    renderTaskTable();
    if (showMessage) taskToast("视频任务已刷新", "success");
  } catch (error) {
    taskElements.tableBody.innerHTML = `<tr><td class="loading-cell" colspan="6">${escapeHtml(error.message)}</td></tr>`;
    taskToast(error.message, "error");
  }
}

async function saveTask(event) {
  event.preventDefault();
  taskElements.formError.classList.add("hidden");
  if (!taskState.selectedChannelId) {
    taskElements.formError.textContent = "请选择视频源";
    taskElements.formError.classList.remove("hidden");
    return;
  }
  const editing = taskState.editingId !== null;
  taskElements.saveButton.disabled = true;
  taskElements.saveButton.textContent = "正在保存…";
  try {
    await taskApi(editing ? `/api/video-tasks/${taskState.editingId}` : "/api/video-tasks", {
      method: editing ? "PUT" : "POST",
      body: taskPayload(),
    });
    closeTaskDrawer();
    await loadTaskData();
    taskToast(editing ? "视频任务已更新" : "视频任务已创建", "success");
  } catch (error) {
    taskElements.formError.textContent = error.message;
    taskElements.formError.classList.remove("hidden");
  } finally {
    taskElements.saveButton.disabled = false;
    taskElements.saveButton.textContent = "保存任务";
  }
}

function taskFromTarget(target) {
  const row = target.closest("tr[data-task-id]");
  if (!row) return null;
  return taskState.tasks.find(task => task.id === Number(row.dataset.taskId)) || null;
}

async function toggleTask(task, button) {
  const nextStatus = task.status === "running" ? "stopped" : "running";
  button.disabled = true;
  try {
    await taskApi(`/api/video-tasks/${task.id}/status`, { method: "PATCH", body: { status: nextStatus } });
    await loadTaskData();
    taskToast(nextStatus === "running" ? "任务已启动，正在加载视频流和检测算法" : "任务已停止", "success");
  } catch (error) {
    taskToast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function deleteTask(task, button) {
  const confirmed = window.confirm(`确定删除视频任务“${task.task_code}”吗？\n删除后无法恢复。`);
  if (!confirmed) return;
  button.disabled = true;
  try {
    await taskApi(`/api/video-tasks/${task.id}`, { method: "DELETE" });
    taskState.openMenuTaskId = null;
    await loadTaskData();
    taskToast("视频任务已删除", "success");
  } catch (error) {
    taskToast(error.message, "error");
  }
}

taskElements.tableBody.addEventListener("click", event => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const task = taskFromTarget(button);
  if (!task) return;
  const action = button.dataset.action;
  if (action === "edit") openTaskDrawer(task);
  if (action === "toggle") toggleTask(task, button);
  if (action === "options") {
    taskState.openMenuTaskId = taskState.openMenuTaskId === task.id ? null : task.id;
    renderTaskTable();
  }
  if (action === "delete") deleteTask(task, button);
});

taskElements.sourceList.addEventListener("click", event => {
  const button = event.target.closest("[data-channel-id]");
  if (!button) return;
  taskState.selectedChannelId = Number(button.dataset.channelId);
  updateSelectedSource();
  closeSourceDrawer();
});

taskElements.algorithmList.addEventListener("change", event => {
  const checkbox = event.target.closest("[data-algorithm-code]");
  if (!checkbox) return;
  const interval = alarmIntervalInput(checkbox.dataset.algorithmCode);
  if (interval) interval.disabled = !checkbox.checked;
  taskElements.algorithmList
    .querySelectorAll(`[data-algorithm-parameter-code="${checkbox.dataset.algorithmCode}"]`)
    .forEach(input => { input.disabled = !checkbox.checked; });
});

document.addEventListener("click", event => {
  if (event.target.closest(".task-options")) return;
  if (taskState.openMenuTaskId !== null) {
    taskState.openMenuTaskId = null;
    renderTaskTable();
  }
});

document.querySelector("#add-task-button").addEventListener("click", () => openTaskDrawer());
document.querySelector("#empty-add-task-button").addEventListener("click", () => openTaskDrawer());
document.querySelector("#refresh-task-button").addEventListener("click", () => loadTaskData(true));
document.querySelector("#open-source-drawer").addEventListener("click", openSourceDrawer);
document.querySelector("#close-source-drawer").addEventListener("click", closeSourceDrawer);
document.querySelectorAll("[data-close-task]").forEach(item => item.addEventListener("click", closeTaskDrawer));
taskElements.sourceSearch.addEventListener("input", renderSourceChoices);
taskElements.search.addEventListener("input", renderTaskTable);
taskElements.form.addEventListener("submit", saveTask);
document.addEventListener("keydown", event => {
  if (event.key !== "Escape") return;
  if (!taskElements.sourceDrawer.classList.contains("hidden")) closeSourceDrawer();
  else if (!taskElements.drawer.classList.contains("hidden")) closeTaskDrawer();
});

loadTaskData();
window.setInterval(() => loadTaskData(), 5000);
