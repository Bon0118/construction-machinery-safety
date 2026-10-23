const previewState = {
  channels: [],
  layout: 4,
  page: 1,
  visibleIds: [],
  statuses: new Map(),
  search: "",
  statusTimer: null,
  clientId: globalThis.crypto?.randomUUID?.() || `preview-${Date.now()}-${Math.random().toString(16).slice(2)}`,
};

const previewElements = {
  console: document.querySelector("#preview-console"),
  workspace: document.querySelector("#preview-workspace"),
  grid: document.querySelector("#preview-grid"),
  empty: document.querySelector("#preview-empty"),
  pagination: document.querySelector("#preview-pagination"),
  pageNumbers: document.querySelector("#preview-page-numbers"),
  previous: document.querySelector("#preview-previous"),
  next: document.querySelector("#preview-next"),
  pageSummary: document.querySelector("#preview-page-summary"),
  refresh: document.querySelector("#preview-refresh"),
  refreshTime: document.querySelector("#preview-refresh-time"),
  fullscreen: document.querySelector("#preview-fullscreen"),
  protocol: document.querySelector("#preview-protocol"),
  channelTree: document.querySelector("#channel-tree"),
  channelSearch: document.querySelector("#channel-search"),
  channelCount: document.querySelector("#preview-channel-count"),
  onlineCount: document.querySelector("#preview-online-count"),
  totalOnline: document.querySelector("#preview-total-online"),
  totalOther: document.querySelector("#preview-total-other"),
  workspaceTitle: document.querySelector("#workspace-title"),
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

function showToast(message, kind = "") {
  const toast = document.createElement("div");
  toast.className = `toast ${kind}`.trim();
  toast.textContent = message;
  previewElements.toastRegion.append(toast);
  setTimeout(() => toast.remove(), 3200);
}

async function apiRequest(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch (_error) {
      // Keep the HTTP status message.
    }
    throw new Error(message);
  }
  return response.json();
}

function totalPages() {
  return Math.max(1, Math.ceil(previewState.channels.length / previewState.layout));
}

function sourceLabel(channel) {
  if (channel.source_type === "camera_index_code") return "cameraIndexCode";
  if (channel.source_type === "local_file") return "本地视频";
  return "直接 RTSP";
}

function groupLabel(sourceType) {
  if (sourceType === "camera_index_code") return "平台接入通道";
  if (sourceType === "local_file") return "本地视频文件";
  return "RTSP 通道";
}

function stateText(state) {
  if (state === "online") return "在线";
  if (state === "reconnecting") return "正在重连";
  if (state === "connecting") return "正在连接";
  if (state === "disabled") return "已停用";
  if (state === "offline") return "离线";
  return "待连接";
}

function channelState(channel) {
  return previewState.statuses.get(channel.id)?.state || channel.last_status || "idle";
}

function tileMarkup(channel) {
  const streamUrl = `/api/live-preview/${channel.id}/stream?client_id=${encodeURIComponent(previewState.clientId)}&v=${Date.now()}`;
  return `
    <article class="preview-tile" data-channel-id="${channel.id}">
      <div class="preview-video-stage">
        <img src="${streamUrl}" alt="${escapeHtml(channel.name)} 实时画面">
        <div class="preview-overlay" data-preview-overlay>
          <span class="preview-spinner"></span>
          <strong data-preview-state>正在连接</strong>
          <small data-preview-error>等待视频首帧…</small>
        </div>
        <div class="preview-live-badge"><i></i><span data-live-label>连接中</span></div>
        <div class="preview-tile-actions">
          <button class="tile-action-button" type="button" data-capture-channel="${channel.id}" title="保存当前画面" aria-label="保存当前画面">⌁</button>
          <button class="tile-action-button focus-button" type="button" data-focus-channel="${channel.id}" title="切换到单画面" aria-label="切换到单画面">⛶</button>
        </div>
      </div>
      <footer class="preview-tile-footer">
        <div>
          <strong>${escapeHtml(channel.name)}</strong>
          <span>${escapeHtml(sourceLabel(channel))}</span>
        </div>
        <div class="preview-metrics">
          <span data-preview-resolution>—</span>
          <span data-preview-fps>— FPS</span>
        </div>
      </footer>
    </article>`;
}

function emptyTileMarkup() {
  return `
    <div class="preview-tile preview-placeholder" aria-hidden="true">
      <span>空闲画面</span>
    </div>`;
}

function renderChannelTree() {
  const term = previewState.search.trim().toLocaleLowerCase();
  const filtered = previewState.channels.filter((channel) => channel.name.toLocaleLowerCase().includes(term));
  const groups = new Map();
  filtered.forEach((channel) => {
    if (!groups.has(channel.source_type)) groups.set(channel.source_type, []);
    groups.get(channel.source_type).push(channel);
  });

  if (!filtered.length) {
    previewElements.channelTree.innerHTML = `<div class="channel-tree-empty">${previewState.channels.length ? "没有匹配的通道" : "暂无已启用通道"}</div>`;
    return;
  }

  previewElements.channelTree.innerHTML = [...groups.entries()].map(([sourceType, channels]) => `
    <details class="channel-tree-group" open>
      <summary>
        <span><i class="tree-chevron">›</i>${escapeHtml(groupLabel(sourceType))}</span>
        <b>${channels.length}</b>
      </summary>
      <div class="channel-tree-nodes">
        ${channels.map((channel) => {
          const state = channelState(channel);
          const isVisible = previewState.visibleIds.includes(channel.id);
          return `
            <button class="channel-tree-node ${isVisible ? "active" : ""}" type="button" data-select-channel="${channel.id}" title="单画面查看 ${escapeHtml(channel.name)}">
              <i class="tree-state-dot ${escapeHtml(state)}"></i>
              <span><strong>${escapeHtml(channel.name)}</strong><small>${escapeHtml(sourceLabel(channel))}</small></span>
              <em>${escapeHtml(stateText(state))}</em>
            </button>`;
        }).join("")}
      </div>
    </details>`).join("");

  const online = previewState.channels.filter((channel) => channelState(channel) === "online").length;
  previewElements.totalOnline.textContent = String(online);
  previewElements.totalOther.textContent = String(Math.max(0, previewState.channels.length - online));
}

function renderPagination() {
  const pages = totalPages();
  previewState.page = Math.min(Math.max(1, previewState.page), pages);
  previewElements.pagination.classList.toggle("hidden", previewState.channels.length <= previewState.layout);
  previewElements.previous.disabled = previewState.page <= 1;
  previewElements.next.disabled = previewState.page >= pages;
  previewElements.pageSummary.textContent = `第 ${previewState.page} / ${pages} 页`;
  previewElements.pageNumbers.innerHTML = Array.from({ length: pages }, (_, index) => {
    const page = index + 1;
    return `<button class="page-number ${page === previewState.page ? "active" : ""}" type="button" data-page="${page}">${page}</button>`;
  }).join("");
}

function stopVisibleStreams() {
  previewElements.grid.querySelectorAll(".preview-video-stage > img").forEach((image) => {
    image.removeAttribute("src");
    image.src = "data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=";
  });
}

async function sendHeartbeat(channelIds = previewState.visibleIds) {
  try {
    await apiRequest("/api/live-preview/heartbeat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_id: previewState.clientId, channel_ids: channelIds }),
      keepalive: true,
    });
  } catch (_error) {
    // The next scheduled heartbeat will retry.
  }
}

function updateWorkspaceTitle(visible) {
  if (previewState.layout === 1 && visible.length) {
    previewElements.workspaceTitle.textContent = visible[0].name;
    return;
  }
  previewElements.workspaceTitle.textContent = `${previewState.layout} 画面预览`;
}

function renderPreview() {
  stopVisibleStreams();
  previewElements.channelCount.textContent = String(previewState.channels.length);
  previewElements.grid.dataset.layout = String(previewState.layout);
  previewElements.empty.classList.toggle("hidden", previewState.channels.length > 0);
  previewElements.grid.classList.toggle("hidden", previewState.channels.length === 0);
  if (!previewState.channels.length) {
    previewState.visibleIds = [];
    previewElements.pagination.classList.add("hidden");
    previewElements.workspaceTitle.textContent = "通道预览";
    renderChannelTree();
    sendHeartbeat([]);
    return;
  }

  const pages = totalPages();
  previewState.page = Math.min(previewState.page, pages);
  const start = (previewState.page - 1) * previewState.layout;
  const visible = previewState.channels.slice(start, start + previewState.layout);
  previewState.visibleIds = visible.map((channel) => channel.id);
  const placeholders = Math.max(0, previewState.layout - visible.length);
  previewElements.grid.innerHTML = visible.map(tileMarkup).join("")
    + Array.from({ length: placeholders }, emptyTileMarkup).join("");
  updateWorkspaceTitle(visible);
  renderPagination();
  renderChannelTree();
  sendHeartbeat();
  refreshStatuses();
}

function applyStatus(status) {
  previewState.statuses.set(status.channel_id, status);
  const tile = previewElements.grid.querySelector(`[data-channel-id="${status.channel_id}"]`);
  if (!tile) return;
  const online = status.state === "online";
  tile.dataset.state = status.state || "idle";
  tile.querySelector("[data-preview-overlay]").classList.toggle("hidden", online);
  tile.querySelector("[data-preview-state]").textContent = stateText(status.state);
  tile.querySelector("[data-preview-error]").textContent = status.error || (online ? "" : "等待视频首帧…");
  tile.querySelector("[data-live-label]").textContent = online ? "LIVE" : stateText(status.state);
  tile.querySelector("[data-preview-resolution]").textContent = status.width && status.height
    ? `${status.width}×${status.height}`
    : "—";
  tile.querySelector("[data-preview-fps]").textContent = status.fps
    ? `${Number(status.fps).toFixed(1)} FPS`
    : "— FPS";
}

async function refreshStatuses() {
  if (!previewState.channels.length) return;
  try {
    if (previewState.visibleIds.length) await sendHeartbeat();
    const payload = await apiRequest("/api/live-preview/status");
    payload.items.forEach(applyStatus);
    const onlineCount = previewState.visibleIds.filter((id) => previewState.statuses.get(id)?.state === "online").length;
    previewElements.onlineCount.textContent = String(onlineCount);
    renderChannelTree();
  } catch (_error) {
    // A temporary status request failure must not interrupt the MJPEG streams.
  }
}

async function loadChannels(showMessage = false) {
  previewElements.refresh.disabled = true;
  try {
    const payload = await apiRequest("/api/video-channels");
    previewState.channels = payload.items.filter((channel) => channel.enabled);
    previewState.page = Math.min(previewState.page, totalPages());
    previewElements.refreshTime.textContent = `更新于 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`;
    renderPreview();
    if (showMessage) showToast("视频通道已刷新", "success");
  } catch (error) {
    previewElements.grid.innerHTML = `<div class="preview-loading error-text">${escapeHtml(error.message)}</div>`;
    showToast(error.message, "error");
  } finally {
    previewElements.refresh.disabled = false;
  }
}

function selectChannel(channelId) {
  const index = previewState.channels.findIndex((channel) => channel.id === channelId);
  if (index < 0) return;
  previewState.layout = 1;
  previewState.page = index + 1;
  document.querySelectorAll("[data-layout]").forEach((item) => {
    item.classList.toggle("active", item.dataset.layout === "1");
  });
  renderPreview();
}

function captureCurrentFrame(channelId) {
  const image = previewElements.grid.querySelector(`[data-channel-id="${channelId}"] img`);
  if (!image || !image.complete || !image.naturalWidth) {
    showToast("视频首帧尚未到达，暂时无法抓拍", "error");
    return;
  }
  try {
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    canvas.getContext("2d").drawImage(image, 0, 0);
    const channel = previewState.channels.find((item) => item.id === channelId);
    const link = document.createElement("a");
    link.download = `${(channel?.name || `channel-${channelId}`).replace(/[\\/:*?\"<>|]/g, "_")}-${Date.now()}.jpg`;
    link.href = canvas.toDataURL("image/jpeg", 0.92);
    link.click();
    showToast("当前画面已保存", "success");
  } catch (_error) {
    showToast("当前浏览器无法保存该画面", "error");
  }
}

document.querySelectorAll("[data-layout]").forEach((button) => {
  button.addEventListener("click", () => {
    previewState.layout = Number(button.dataset.layout);
    previewState.page = 1;
    document.querySelectorAll("[data-layout]").forEach((item) => item.classList.toggle("active", item === button));
    renderPreview();
  });
});

previewElements.previous.addEventListener("click", () => {
  if (previewState.page > 1) {
    previewState.page -= 1;
    renderPreview();
  }
});

previewElements.next.addEventListener("click", () => {
  if (previewState.page < totalPages()) {
    previewState.page += 1;
    renderPreview();
  }
});

previewElements.pageNumbers.addEventListener("click", (event) => {
  const button = event.target.closest("[data-page]");
  if (!button) return;
  previewState.page = Number(button.dataset.page);
  renderPreview();
});

previewElements.channelTree.addEventListener("click", (event) => {
  const button = event.target.closest("[data-select-channel]");
  if (button) selectChannel(Number(button.dataset.selectChannel));
});

previewElements.grid.addEventListener("click", (event) => {
  const focusButton = event.target.closest("[data-focus-channel]");
  if (focusButton) {
    selectChannel(Number(focusButton.dataset.focusChannel));
    return;
  }
  const captureButton = event.target.closest("[data-capture-channel]");
  if (captureButton) captureCurrentFrame(Number(captureButton.dataset.captureChannel));
});

previewElements.channelSearch.addEventListener("input", () => {
  previewState.search = previewElements.channelSearch.value;
  renderChannelTree();
});

previewElements.protocol.addEventListener("change", () => {
  localStorage.setItem("preview-protocol", previewElements.protocol.value);
});

previewElements.fullscreen.addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await previewElements.workspace.requestFullscreen();
    }
  } catch (_error) {
    showToast("当前浏览器不支持全屏预览", "error");
  }
});

document.addEventListener("fullscreenchange", () => {
  previewElements.fullscreen.textContent = document.fullscreenElement ? "×" : "⛶";
  previewElements.fullscreen.title = document.fullscreenElement ? "退出全屏" : "全屏预览";
});

previewElements.refresh.addEventListener("click", () => loadChannels(true));

window.addEventListener("beforeunload", () => {
  stopVisibleStreams();
  fetch("/api/live-preview/heartbeat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ client_id: previewState.clientId, channel_ids: [] }),
    keepalive: true,
  }).catch(() => {});
});

previewState.statusTimer = setInterval(refreshStatuses, 2000);
loadChannels();
