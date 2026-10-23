const trainingState = {
  overview: null,
  activeJobId: null,
  selectedJobId: null,
  loading: false,
  pollTimer: null,
};

const trainingElements = {
  envName: document.querySelector("#training-env-name"),
  envPython: document.querySelector("#training-env-python"),
  envGpu: document.querySelector("#training-env-gpu"),
  envCuda: document.querySelector("#training-env-cuda"),
  envWeights: document.querySelector("#training-env-weights"),
  envClasses: document.querySelector("#training-env-classes"),
  envStatus: document.querySelector("#training-env-status"),
  envError: document.querySelector("#training-env-error"),
  refreshEnvironment: document.querySelector("#training-refresh-environment"),
  dataset: document.querySelector("#training-dataset"),
  runName: document.querySelector("#training-run-name"),
  imageSize: document.querySelector("#training-image-size"),
  epochs: document.querySelector("#training-epochs"),
  batchSize: document.querySelector("#training-batch-size"),
  workers: document.querySelector("#training-workers"),
  patience: document.querySelector("#training-patience"),
  freezeBackbone: document.querySelector("#training-freeze-backbone"),
  runtimeHint: document.querySelector("#training-runtime-hint"),
  datasetFile: document.querySelector("#training-dataset-file"),
  uploadDataset: document.querySelector("#training-upload-dataset"),
  uploadName: document.querySelector("#training-upload-name"),
  start: document.querySelector("#training-start"),
  datasetCount: document.querySelector("#training-dataset-count"),
  datasetHint: document.querySelector("#training-dataset-hint"),
  classPerson: document.querySelector("#training-class-person"),
  classHead: document.querySelector("#training-class-head"),
  classHelmet: document.querySelector("#training-class-helmet"),
  datasetSummary: document.querySelector("#training-dataset-summary"),
  liveCard: document.querySelector("#training-live-card"),
  liveName: document.querySelector("#training-live-name"),
  liveStatus: document.querySelector("#training-live-status"),
  liveEpoch: document.querySelector("#training-live-epoch"),
  livePercent: document.querySelector("#training-live-percent"),
  liveProgress: document.querySelector("#training-live-progress"),
  metricPrecision: document.querySelector("#training-metric-precision"),
  metricRecall: document.querySelector("#training-metric-recall"),
  metricMap50: document.querySelector("#training-metric-map50"),
  metricMap5095: document.querySelector("#training-metric-map5095"),
  log: document.querySelector("#training-log"),
  stop: document.querySelector("#training-stop"),
  jobBody: document.querySelector("#training-job-body"),
  toastRegion: document.querySelector("#training-toast-region"),
};

function trainingEscape(value) {
  return String(value ?? "").replace(/[&<>'"]/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

function trainingToast(message, type = "") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`.trim();
  toast.textContent = message;
  trainingElements.toastRegion.appendChild(toast);
  window.setTimeout(() => toast.remove(), 3500);
}

async function trainingApi(url, options = {}) {
  const request = { ...options, headers: { ...(options.headers || {}) } };
  if (Object.prototype.hasOwnProperty.call(options, "body")) {
    request.headers["Content-Type"] = "application/json";
    request.body = JSON.stringify(options.body);
  }
  const response = await fetch(url, request);
  let data = {};
  try { data = await response.json(); } catch (_) { data = {}; }
  if (!response.ok) throw new Error(data.detail || `请求失败（${response.status}）`);
  return data;
}

function shortPath(value) {
  const parts = String(value || "").replaceAll("/", "\\").split("\\");
  return parts.length > 3 ? `…\\${parts.slice(-3).join("\\")}` : String(value || "-");
}

function formatTrainingTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value).replace("T", " ") : new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  }).format(date).replaceAll("/", "-");
}

function statusLabel(status) {
  return ({
    starting: "正在启动", running: "训练中", stopping: "正在停止", stopped: "已停止",
    completed: "已完成", failed: "失败", interrupted: "已中断",
  })[status] || status || "未知";
}

function metric(value) {
  return value == null ? "-" : Number(value).toFixed(4);
}

function renderEnvironment(environment) {
  trainingElements.envName.textContent = "Conda · yolov5";
  trainingElements.envPython.textContent = shortPath(environment.python);
  trainingElements.envGpu.textContent = environment.gpu || "未检测到 GPU";
  trainingElements.envCuda.textContent = environment.cuda_available
    ? `CUDA ${environment.cuda_version} · ${environment.gpu_memory_gb} GB`
    : "CUDA 不可用";
  trainingElements.envWeights.textContent = String(environment.weights || "").split(/[\\/]/).pop() || "model1_best.pt";
  trainingElements.envClasses.textContent = (environment.weight_names || ["person", "head", "helmet"]).join(" / ");
  trainingElements.envStatus.textContent = environment.ready ? "可以训练" : "暂不可训练";
  trainingElements.envError.textContent = environment.ready ? `Torch ${environment.torch} · YOLOv5 兼容检查通过` : (environment.error || "环境检查失败");
  const article = trainingElements.envStatus.closest("article");
  article.classList.toggle("ready", Boolean(environment.ready));
  article.classList.toggle("error", !environment.ready);
}

function renderDatasets(datasets) {
  const selected = trainingElements.dataset.value;
  trainingElements.dataset.innerHTML = '<option value="">请选择数据集</option>' + datasets.map(item => {
    const ready = item.training_ready ? "可训练" : "不完整";
    return `<option value="${trainingEscape(item.dataset_id)}">${trainingEscape(item.original_name || item.dataset_id)} · ${item.image_count || 0}图 · ${ready}</option>`;
  }).join("");
  if (datasets.some(item => item.dataset_id === selected)) trainingElements.dataset.value = selected;
  else if (datasets[0]) trainingElements.dataset.value = datasets[0].dataset_id;
  trainingElements.datasetCount.textContent = String(datasets.length);
  renderSelectedDataset();
}

function renderSelectedDataset() {
  const datasets = trainingState.overview?.datasets || [];
  const selected = datasets.find(item => item.dataset_id === trainingElements.dataset.value);
  if (!selected) {
    trainingElements.datasetSummary.textContent = "尚未上传可训练数据集";
    trainingElements.datasetHint.textContent = "选择本地 ZIP 文件后，平台会复制、规范化并校验训练数据。";
    trainingElements.classPerson.textContent = "0";
    trainingElements.classHead.textContent = "0";
    trainingElements.classHelmet.textContent = "0";
    return;
  }
  const counts = selected.class_counts || {};
  const warning = (selected.warnings || []).join("；");
  trainingElements.datasetSummary.innerHTML = `<strong>${trainingEscape(selected.training_ready ? "数据集检查通过" : "数据集暂不可训练")}</strong><br>`
    + `${selected.split_counts?.train || 0} 张训练图 · ${selected.split_counts?.val || 0} 张验证图<br>`
    + `person ${counts.person || 0} · head ${counts.head || 0} · helmet ${counts.helmet || 0}`
    + (warning ? `<br><span>${trainingEscape(warning)}</span>` : "");
  trainingElements.datasetHint.textContent = `${selected.original_name || selected.dataset_id}：${selected.image_count || 0} 张图片，其中 ${selected.labeled_images || 0} 张有标注、${selected.negative_images || 0} 张为负样本。`;
  trainingElements.classPerson.textContent = String(counts.person || 0);
  trainingElements.classHead.textContent = String(counts.head || 0);
  trainingElements.classHelmet.textContent = String(counts.helmet || 0);
}

function updateStartAvailability() {
  const overview = trainingState.overview || {};
  const datasets = overview.datasets || [];
  const selected = datasets.find(item => item.dataset_id === trainingElements.dataset.value);
  const activeTraining = (overview.jobs || []).find(job => ["starting", "running", "stopping"].includes(job.status));
  const activeVideoTasks = Number(overview.active_video_tasks || 0);
  let reason = "训练输出进入独立目录，不会覆盖当前生产模型，也不会自动部署。";
  if (!overview.environment?.ready) reason = "训练环境尚未通过检查，请先查看上方环境状态。";
  else if (!selected) reason = "请先上传并选择一个检查通过的本地训练数据集。";
  else if (!selected.training_ready) reason = "当前数据集类别或训练/验证划分不完整，不能启动训练。";
  else if (activeTraining) reason = `训练任务 ${activeTraining.run_name} 正在运行，同一时间只允许一个训练任务。`;
  else if (activeVideoTasks > 0) reason = `当前有 ${activeVideoTasks} 个视频检测任务正在运行；请先在“视频任务”中停止它们，释放 RTX 4060 显存。`;
  else reason = "环境、数据集和 GPU 资源检查通过，可以启动训练；候选模型不会自动部署。";
  trainingElements.runtimeHint.textContent = reason;
  trainingElements.start.disabled = !overview.environment?.ready || !selected?.training_ready || Boolean(activeTraining) || activeVideoTasks > 0;
}

function renderJobs(jobs) {
  if (!jobs.length) {
    trainingElements.jobBody.innerHTML = '<tr><td colspan="8">暂无训练记录；上传并选择检查通过的本地数据集后即可创建首个训练任务。</td></tr>';
    return;
  }
  trainingElements.jobBody.innerHTML = jobs.map(job => {
    const progress = Math.round(Number(job.metrics?.progress || 0) * 100);
    const active = ["starting", "running", "stopping"].includes(job.status);
    const model = job.best_model_path
      ? `<a class="button secondary compact" href="/api/training/jobs/${job.id}/best-model">下载 best.pt</a>`
      : "-";
    return `<tr>
      <td><strong>${trainingEscape(job.run_name)}</strong><small>${trainingEscape(job.config?.image_size || 640)}px · batch ${trainingEscape(job.config?.batch_size || "-")}</small></td>
      <td>${trainingEscape(job.dataset_id)}</td>
      <td><span class="training-job-status ${trainingEscape(job.status)}">${trainingEscape(statusLabel(job.status))}</span>${job.error ? `<small title="${trainingEscape(job.error)}">${trainingEscape(job.error)}</small>` : ""}</td>
      <td>${job.metrics?.epoch || 0}/${job.config?.epochs || 0}<small>${progress}%</small></td>
      <td>${metric(job.metrics?.map50)}</td>
      <td>${formatTrainingTime(job.started_at || job.created_at)}</td>
      <td>${model}</td>
      <td><div class="training-row-actions"><button class="button secondary compact" type="button" data-training-view="${job.id}">日志</button>${active ? `<button class="button danger compact" type="button" data-training-stop="${job.id}">停止</button>` : ""}</div></td>
    </tr>`;
  }).join("");
}

async function renderLiveJob(job) {
  if (!job) {
    trainingElements.liveCard.classList.add("hidden");
    return;
  }
  trainingElements.liveCard.classList.remove("hidden");
  trainingElements.liveName.textContent = job.run_name;
  trainingElements.liveStatus.textContent = statusLabel(job.status);
  const epochs = Number(job.config?.epochs || 0);
  const current = Number(job.metrics?.epoch || 0);
  const progress = Math.round(Number(job.metrics?.progress || 0) * 100);
  trainingElements.liveEpoch.textContent = `${current} / ${epochs}`;
  trainingElements.livePercent.textContent = `${progress}%`;
  trainingElements.liveProgress.style.width = `${progress}%`;
  trainingElements.metricPrecision.textContent = metric(job.metrics?.precision);
  trainingElements.metricRecall.textContent = metric(job.metrics?.recall);
  trainingElements.metricMap50.textContent = metric(job.metrics?.map50);
  trainingElements.metricMap5095.textContent = metric(job.metrics?.map5095);
  trainingElements.stop.disabled = !["starting", "running", "stopping"].includes(job.status);
  trainingElements.stop.dataset.jobId = String(job.id);
  try {
    const data = await trainingApi(`/api/training/jobs/${job.id}/log?lines=240`);
    trainingElements.log.textContent = (data.lines || []).join("\n") || "训练进程已创建，等待日志输出…";
    trainingElements.log.scrollTop = trainingElements.log.scrollHeight;
  } catch (error) {
    trainingElements.log.textContent = error.message;
  }
}

async function loadTrainingOverview(forceEnvironment = false, quiet = false) {
  if (trainingState.loading) return;
  trainingState.loading = true;
  try {
    const overview = await trainingApi(`/api/training/overview${forceEnvironment ? "?refresh_environment=true" : ""}`);
    trainingState.overview = overview;
    const defaults = overview.defaults || {};
    if (!trainingElements.epochs.dataset.initialized) {
      trainingElements.epochs.value = defaults.epochs || 100;
      trainingElements.batchSize.value = defaults.batch_size || 2;
      trainingElements.imageSize.value = defaults.image_size || 640;
      trainingElements.workers.value = defaults.workers || 2;
      trainingElements.patience.value = defaults.patience || 20;
      trainingElements.freezeBackbone.checked = defaults.freeze_backbone !== false;
      trainingElements.epochs.dataset.initialized = "1";
    }
    renderEnvironment(overview.environment || {});
    renderDatasets(overview.datasets || []);
    renderJobs(overview.jobs || []);
    const active = (overview.jobs || []).find(job => ["starting", "running", "stopping"].includes(job.status));
    trainingState.activeJobId = active?.id || null;
    const selected = (overview.jobs || []).find(job => job.id === trainingState.selectedJobId) || active;
    await renderLiveJob(selected || null);
    updateStartAvailability();
    if (!quiet && forceEnvironment) trainingToast(overview.environment?.ready ? "训练环境检查通过" : "训练环境尚未就绪", overview.environment?.ready ? "success" : "error");
  } catch (error) {
    if (!quiet) trainingToast(error.message, "error");
  } finally {
    trainingState.loading = false;
  }
}

async function uploadTrainingDataset() {
  const file = trainingElements.datasetFile.files?.[0];
  if (!file) return trainingToast("请先选择 ZIP 数据集", "error");
  if (!file.name.toLowerCase().endsWith(".zip")) return trainingToast("数据集必须为 ZIP 压缩包", "error");
  trainingElements.uploadDataset.disabled = true;
  trainingElements.uploadDataset.textContent = "正在上传并校验…";
  try {
    const response = await fetch("/api/training/datasets/upload", {
      method: "POST",
      headers: { "Content-Type": "application/zip", "X-Dataset-Filename": encodeURIComponent(file.name) },
      body: file,
    });
    let dataset = {};
    try { dataset = await response.json(); } catch (_) { dataset = {}; }
    if (!response.ok) throw new Error(dataset.detail || `上传失败（${response.status}）`);
    trainingToast(`数据集 ${file.name} 已上传并完成校验`, dataset.training_ready ? "success" : "error");
    await loadTrainingOverview(false, true);
    trainingElements.dataset.value = dataset.dataset_id;
    renderSelectedDataset();
    updateStartAvailability();
  } catch (error) {
    trainingToast(error.message, "error");
  } finally {
    trainingElements.uploadDataset.textContent = "上传并校验";
    trainingElements.uploadDataset.disabled = false;
  }
}

async function startTraining() {
  if (!trainingElements.dataset.value) return trainingToast("请先选择已准备数据集", "error");
  if (!window.confirm("训练会独占较多 GPU 显存。确认当前视频检测任务已停止，并启动本次训练吗？")) return;
  trainingElements.start.disabled = true;
  trainingElements.start.textContent = "正在启动…";
  try {
    const job = await trainingApi("/api/training/jobs", {
      method: "POST",
      body: {
        dataset_id: trainingElements.dataset.value,
        run_name: trainingElements.runName.value,
        image_size: Number(trainingElements.imageSize.value),
        epochs: Number(trainingElements.epochs.value),
        batch_size: Number(trainingElements.batchSize.value),
        workers: Number(trainingElements.workers.value),
        patience: Number(trainingElements.patience.value),
        freeze_backbone: trainingElements.freezeBackbone.checked,
      },
    });
    trainingState.selectedJobId = job.id;
    trainingToast(`训练任务 ${job.run_name} 已启动`, "success");
    await loadTrainingOverview(false, true);
  } catch (error) {
    trainingToast(error.message, "error");
  } finally {
    trainingElements.start.textContent = "启动训练";
    updateStartAvailability();
  }
}

async function stopTraining(jobId) {
  if (!jobId || !window.confirm("确认停止该训练任务吗？已完成的轮次和当前权重文件会保留。")) return;
  try {
    await trainingApi(`/api/training/jobs/${jobId}/stop`, { method: "POST" });
    trainingToast("已发送停止训练指令", "success");
    await loadTrainingOverview(false, true);
  } catch (error) {
    trainingToast(error.message, "error");
  }
}

trainingElements.refreshEnvironment.addEventListener("click", () => loadTrainingOverview(true));
trainingElements.dataset.addEventListener("change", () => { renderSelectedDataset(); updateStartAvailability(); });
trainingElements.datasetFile.addEventListener("change", () => {
  const file = trainingElements.datasetFile.files?.[0];
  trainingElements.uploadName.textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB` : "尚未选择数据集";
});
trainingElements.uploadDataset.addEventListener("click", uploadTrainingDataset);
trainingElements.start.addEventListener("click", startTraining);
trainingElements.stop.addEventListener("click", () => stopTraining(Number(trainingElements.stop.dataset.jobId || 0)));
trainingElements.jobBody.addEventListener("click", event => {
  const view = event.target.closest("[data-training-view]");
  const stop = event.target.closest("[data-training-stop]");
  if (view) {
    trainingState.selectedJobId = Number(view.dataset.trainingView);
    const job = trainingState.overview?.jobs?.find(item => item.id === trainingState.selectedJobId);
    renderLiveJob(job || null);
    trainingElements.liveCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  if (stop) stopTraining(Number(stop.dataset.trainingStop));
});

loadTrainingOverview();
trainingState.pollTimer = window.setInterval(() => loadTrainingOverview(false, true), 3000);
