"use strict";

const state = { channels: [], editingId: null, snapshotUrl: null };

const elements = {
  tableBody: document.querySelector("#channel-table-body"),
  emptyState: document.querySelector("#empty-state"),
  search: document.querySelector("#channel-search"),
  modal: document.querySelector("#channel-modal"),
  modalTitle: document.querySelector("#channel-modal-title"),
  form: document.querySelector("#channel-form"),
  formError: document.querySelector("#form-error"),
  saveButton: document.querySelector("#save-channel-button"),
  sourceType: document.querySelector("#source-type"),
  cameraCodeField: document.querySelector("#camera-code-field"),
  streamProfileField: document.querySelector("#stream-profile-field"),
  streamProfile: document.querySelector("#stream-profile"),
  sourceUrlField: document.querySelector("#source-url-field"),
  sourceUrlLabel: document.querySelector("#source-url-label"),
  sourceUrlHelp: document.querySelector("#source-url-help"),
  snapshotModal: document.querySelector("#snapshot-modal"),
  snapshotTitle: document.querySelector("#snapshot-modal-title"),
  snapshotImage: document.querySelector("#snapshot-image"),
  snapshotStage: document.querySelector("#snapshot-stage"),
  snapshotMessage: document.querySelector("#snapshot-message"),
  summaryTotal: document.querySelector("#summary-total"),
  summaryEnabled: document.querySelector("#summary-enabled"),
  summaryCameraCode: document.querySelector("#summary-camera-code"),
  summaryOnline: document.querySelector("#summary-online"),
  summaryBitrate: document.querySelector("#summary-bitrate"),
  summaryTraffic: document.querySelector("#summary-traffic"),
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

async function api(url, options = {}) {
  const requestOptions = { ...options };
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

function toast(message, type = "default") {
  const item = document.createElement("div");
  item.className = `toast ${type}`;
  item.textContent = message;
  elements.toastRegion.appendChild(item);
  window.setTimeout(() => item.remove(), 3200);
}

function sourceTypeText(sourceType) {
  return {
    camera_index_code: "cameraIndexCode",
    rtsp: "直接 RTSP",
    local_file: "本地视频",
  }[sourceType] || sourceType;
}

function sourceTypeBadge(sourceType) {
  return sourceType === "camera_index_code" ? "blue" : sourceType === "rtsp" ? "purple" : "gray";
}

function channelSource(channel) {
  return channel.source_type === "camera_index_code" ? channel.camera_index_code : channel.source_url;
}

function statusView(channel) {
  const labels = { online: "在线", offline: "离线", untested: "未测试" };
  const status = labels[channel.last_status] ? channel.last_status : "untested";
  const detail = channel.last_error
    ? `<span class="status-detail" title="${escapeHtml(channel.last_error)}">${escapeHtml(channel.last_error)}</span>`
    : channel.last_tested_at
      ? `<span class="status-detail">${escapeHtml(channel.last_tested_at)}</span>`
      : "";
  return `<span class="badge ${status}">${labels[status]}</span>${detail}`;
}

function formatBitrate(value) {
  const bitrate = Number(value || 0);
  if (bitrate <= 0) return "等待统计";
  if (bitrate >= 1000) return `${(bitrate / 1000).toFixed(2)} Mbps`;
  return `${Math.round(bitrate)} Kbps`;
}

function formatTrafficGb(value) {
  const amount = Number(value || 0);
  if (amount < 0.001) return "0 GB";
  return `${amount.toFixed(amount >= 10 ? 1 : 2)} GB`;
}

function streamProfileText(channel) {
  if (channel.source_type !== "camera_index_code") return "由地址决定";
  return channel.stream_profile === "sub" ? "子码流 · streamType=1" : "主码流 · streamType=0";
}

function trafficView(channel) {
  const traffic = channel.traffic || {};
  const shared = traffic.shared_connection;
  const subscribers = Number(traffic.subscriber_count || 0);
  return `
    <span class="traffic-value">${escapeHtml(streamProfileText(channel))}</span>
    <span class="traffic-detail">${shared ? `${formatBitrate(traffic.bitrate_kbps)} · ${subscribers} 个订阅` : "共享流未连接"}</span>
    <span class="traffic-detail">今日约 ${formatTrafficGb(traffic.estimated_gb_today)} · 全天预计 ${formatTrafficGb(traffic.projected_gb_per_day)}</span>
  `;
}

function renderSummary() {
  elements.summaryTotal.textContent = state.channels.length;
  elements.summaryEnabled.textContent = state.channels.filter(item => item.enabled).length;
  elements.summaryCameraCode.textContent = state.channels.filter(item => item.source_type === "camera_index_code").length;
  elements.summaryOnline.textContent = state.channels.filter(item => item.last_status === "online").length;
  const totalBitrate = state.channels.reduce((total, item) => total + Number(item.traffic?.bitrate_kbps || 0), 0);
  const projectedTraffic = state.channels.reduce((total, item) => total + Number(item.traffic?.projected_gb_per_day || 0), 0);
  elements.summaryBitrate.textContent = formatBitrate(totalBitrate);
  elements.summaryTraffic.textContent = `预计全天 ${formatTrafficGb(projectedTraffic)}`;
}

function renderTable() {
  const query = elements.search.value.trim().toLowerCase();
  const channels = state.channels.filter(channel => {
    const haystack = [channel.name, channel.camera_index_code, channel.source_url, channel.camera_id]
      .join(" ").toLowerCase();
    return !query || haystack.includes(query);
  });

  elements.emptyState.classList.toggle("hidden", state.channels.length !== 0);
  document.querySelector(".table-wrap").classList.toggle("hidden", state.channels.length === 0);

  if (!channels.length && state.channels.length) {
    elements.tableBody.innerHTML = '<tr><td class="loading-cell" colspan="8">没有符合搜索条件的视频通道。</td></tr>';
    return;
  }

  elements.tableBody.innerHTML = channels.map(channel => `
    <tr data-channel-id="${channel.id}">
      <td>
        <span class="channel-name">${escapeHtml(channel.name)}</span>
        <span class="channel-meta">#${channel.id}${channel.reconnect_enabled ? " · 断线重连" : ""}</span>
      </td>
      <td><span class="badge ${sourceTypeBadge(channel.source_type)}">${escapeHtml(sourceTypeText(channel.source_type))}</span></td>
      <td>${trafficView(channel)}</td>
      <td><span class="source-value" title="${escapeHtml(channelSource(channel))}">${escapeHtml(channelSource(channel))}</span></td>
      <td>${escapeHtml(channel.camera_id || "—")}</td>
      <td>${statusView(channel)}</td>
      <td>
        <label class="mini-switch" title="${channel.enabled ? "点击停用" : "点击启用"}">
          <input type="checkbox" data-action="toggle" ${channel.enabled ? "checked" : ""}>
          <i></i>
        </label>
      </td>
      <td>
        <div class="actions">
          <button class="button secondary compact" type="button" data-action="test">测试</button>
          <button class="button secondary compact" type="button" data-action="snapshot">抓拍</button>
          <button class="button secondary compact" type="button" data-action="edit">编辑</button>
          <button class="button danger compact" type="button" data-action="delete">删除</button>
        </div>
      </td>
    </tr>
  `).join("");
}

async function loadChannels(showToast = false) {
  try {
    const data = await api("/api/video-channels");
    state.channels = data.items || [];
    renderSummary();
    renderTable();
    if (showToast) toast("视频通道已刷新", "success");
  } catch (error) {
    elements.tableBody.innerHTML = `<tr><td class="loading-cell" colspan="8">${escapeHtml(error.message)}</td></tr>`;
    toast(error.message, "error");
  }
}

function syncSourceFields() {
  const type = elements.sourceType.value;
  const isCameraCode = type === "camera_index_code";
  elements.cameraCodeField.classList.toggle("hidden", !isCameraCode);
  elements.streamProfileField.classList.toggle("hidden", !isCameraCode);
  elements.sourceUrlField.classList.toggle("hidden", isCameraCode);
  if (type === "local_file") {
    elements.sourceUrlLabel.innerHTML = "本地视频文件路径 <b>*</b>";
    elements.sourceUrlHelp.textContent = "填写服务器本机上的绝对路径或项目相对路径。";
    document.querySelector("#source-url").placeholder = "例如：E:\\videos\\test.mp4";
  } else {
    elements.sourceUrlLabel.innerHTML = "RTSP 地址 <b>*</b>";
    elements.sourceUrlHelp.textContent = "包含密码时，列表和日志只显示脱敏地址。";
    document.querySelector("#source-url").placeholder = "rtsp://user:password@host/path";
  }
}

function openChannelModal(channel = null) {
  state.editingId = channel?.id || null;
  elements.modalTitle.textContent = channel ? "编辑视频通道" : "新增视频通道";
  document.querySelector("#channel-id").value = channel?.id || "";
  document.querySelector("#channel-name").value = channel?.name || "";
  elements.sourceType.value = channel?.source_type || "camera_index_code";
  document.querySelector("#camera-index-code").value = channel?.camera_index_code || "";
  elements.streamProfile.value = channel?.stream_profile || "sub";
  document.querySelector("#source-url").value = channel?.source_url || "";
  document.querySelector("#camera-id").value = channel?.camera_id || "";
  document.querySelector("#channel-enabled").checked = channel ? channel.enabled : true;
  elements.formError.classList.add("hidden");
  elements.formError.textContent = "";
  syncSourceFields();
  elements.modal.classList.remove("hidden");
  document.querySelector("#channel-name").focus();
}

function closeChannelModal() {
  elements.modal.classList.add("hidden");
  elements.form.reset();
  state.editingId = null;
}

function channelPayload() {
  return {
    name: document.querySelector("#channel-name").value.trim(),
    source_type: elements.sourceType.value,
    camera_index_code: document.querySelector("#camera-index-code").value.trim(),
    stream_profile: elements.streamProfile.value,
    source_url: document.querySelector("#source-url").value.trim(),
    camera_id: document.querySelector("#camera-id").value.trim(),
    enabled: document.querySelector("#channel-enabled").checked,
  };
}

async function saveChannel(event) {
  event.preventDefault();
  const wasEditing = state.editingId !== null;
  elements.formError.classList.add("hidden");
  elements.saveButton.disabled = true;
  elements.saveButton.textContent = "正在保存…";
  try {
    const url = state.editingId ? `/api/video-channels/${state.editingId}` : "/api/video-channels";
    const method = state.editingId ? "PUT" : "POST";
    await api(url, { method, body: channelPayload() });
    closeChannelModal();
    await loadChannels();
    toast(wasEditing ? "视频通道已更新" : "视频通道已创建", "success");
  } catch (error) {
    elements.formError.textContent = error.message;
    elements.formError.classList.remove("hidden");
  } finally {
    elements.saveButton.disabled = false;
    elements.saveButton.textContent = "保存通道";
  }
}

function channelFromRow(target) {
  const row = target.closest("tr[data-channel-id]");
  if (!row) return null;
  return state.channels.find(item => item.id === Number(row.dataset.channelId)) || null;
}

async function toggleChannel(channel, checkbox) {
  checkbox.disabled = true;
  try {
    await api(`/api/video-channels/${channel.id}/enabled`, {
      method: "PATCH",
      body: { enabled: checkbox.checked },
    });
    await loadChannels();
    toast(checkbox.checked ? "视频通道已启用" : "视频通道已停用", "success");
  } catch (error) {
    checkbox.checked = !checkbox.checked;
    toast(error.message, "error");
  } finally {
    checkbox.disabled = false;
  }
}

async function testChannel(channel, button) {
  button.disabled = true;
  button.textContent = "测试中…";
  toast(`${channel.name}：正在连接视频源`);
  try {
    const result = await api(`/api/video-channels/${channel.id}/test`, { method: "POST" });
    await loadChannels();
    toast(`连接成功：${result.width}×${result.height}，${result.elapsed_seconds} 秒`, "success");
  } catch (error) {
    await loadChannels();
    toast(`连接失败：${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = "测试";
  }
}

function closeSnapshotModal() {
  elements.snapshotModal.classList.add("hidden");
  if (state.snapshotUrl) URL.revokeObjectURL(state.snapshotUrl);
  state.snapshotUrl = null;
  elements.snapshotImage.removeAttribute("src");
}

async function snapshotChannel(channel, button) {
  button.disabled = true;
  elements.snapshotTitle.textContent = `${channel.name} · 抓拍预览`;
  elements.snapshotImage.classList.add("hidden");
  elements.snapshotStage.querySelector(".snapshot-loading").classList.remove("hidden");
  elements.snapshotStage.querySelector(".snapshot-loading").textContent = "正在连接并抓取画面…";
  elements.snapshotMessage.textContent = "连接超时为 60 秒";
  elements.snapshotModal.classList.remove("hidden");
  try {
    const response = await fetch(`/api/video-channels/${channel.id}/snapshot`, { method: "POST" });
    if (!response.ok) {
      const data = await response.json().catch(() => null);
      throw new Error(errorMessage(data, `抓拍失败（${response.status}）`));
    }
    const blob = await response.blob();
    if (state.snapshotUrl) URL.revokeObjectURL(state.snapshotUrl);
    state.snapshotUrl = URL.createObjectURL(blob);
    elements.snapshotImage.src = state.snapshotUrl;
    elements.snapshotImage.classList.remove("hidden");
    elements.snapshotStage.querySelector(".snapshot-loading").classList.add("hidden");
    elements.snapshotMessage.textContent = `抓拍时间：${new Date().toLocaleString()}`;
    await loadChannels();
  } catch (error) {
    elements.snapshotStage.querySelector(".snapshot-loading").textContent = error.message;
    elements.snapshotMessage.textContent = "请检查地址、网络和摄像头状态";
    await loadChannels();
  } finally {
    button.disabled = false;
  }
}

async function deleteChannel(channel, button) {
  const confirmed = window.confirm(`确定删除视频通道“${channel.name}”吗？\n删除后无法恢复。`);
  if (!confirmed) return;
  button.disabled = true;
  try {
    await api(`/api/video-channels/${channel.id}`, { method: "DELETE" });
    await loadChannels();
    toast("视频通道已删除", "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

elements.tableBody.addEventListener("click", event => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const channel = channelFromRow(button);
  if (!channel) return;
  const action = button.dataset.action;
  if (action === "edit") openChannelModal(channel);
  if (action === "test") testChannel(channel, button);
  if (action === "snapshot") snapshotChannel(channel, button);
  if (action === "delete") deleteChannel(channel, button);
});

elements.tableBody.addEventListener("change", event => {
  const checkbox = event.target.closest('input[data-action="toggle"]');
  if (!checkbox) return;
  const channel = channelFromRow(checkbox);
  if (channel) toggleChannel(channel, checkbox);
});

document.querySelector("#add-channel-button").addEventListener("click", () => openChannelModal());
document.querySelector("#empty-add-button").addEventListener("click", () => openChannelModal());
document.querySelector("#refresh-button").addEventListener("click", () => loadChannels(true));
elements.search.addEventListener("input", renderTable);
elements.sourceType.addEventListener("change", syncSourceFields);
elements.form.addEventListener("submit", saveChannel);
document.querySelectorAll("[data-close-modal]").forEach(item => item.addEventListener("click", closeChannelModal));
document.querySelectorAll("[data-close-snapshot]").forEach(item => item.addEventListener("click", closeSnapshotModal));
document.addEventListener("keydown", event => {
  if (event.key !== "Escape") return;
  if (!elements.snapshotModal.classList.contains("hidden")) closeSnapshotModal();
  else if (!elements.modal.classList.contains("hidden")) closeChannelModal();
});

loadChannels();
window.setInterval(() => {
  if (document.visibilityState === "visible") loadChannels();
}, 5000);
