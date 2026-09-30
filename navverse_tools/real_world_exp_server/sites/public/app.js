import {
  ImuDistanceIntegrator,
  headingFromOrientation,
  normalizeDegrees,
  screenAngleDegrees,
} from "./imu.mjs";
import {
  nativeCapabilities,
  nativeCaptureAvailable,
  requestNativeCapture,
} from "./native_bridge.mjs";

const elements = Object.fromEntries(
  [
    "connectionBadge", "connectionText", "runsButton", "setupPanel", "episodeForm", "episodeId",
    "serverUrl", "agentSelect", "agentDescription", "instruction", "hfov", "tokenLabel", "accessToken", "startButton", "secureContextNote",
    "runner", "camera", "nativePreview", "captureCanvas", "activeEpisodeId", "activeInstruction",
    "activeAgentName", "sensorStatus", "stepValue", "headingValue", "distanceValue", "wakeLockButton", "guidanceCard", "guidanceHint", "compass",
    "targetArrow", "actionCode", "actionLabel", "actionDetail", "alignment", "alignmentFill",
    "alignmentText", "nextActionButton", "nextActionText", "retryButton", "requestStatus",
    "finishedPanel", "finishedTitle", "finishedSummary", "viewRunButton", "newEpisodeButton",
    "runsPanel", "runsBackButton", "runsRefreshButton", "runsStatus", "runsList", "runDetail",
    "runDetailBackButton", "runDetailMethod", "runDetailInstruction", "runDetailOutcome",
    "runDetailMeta", "runVideo", "runVideoMissing", "toast",
  ].map((id) => [id, document.getElementById(id)]),
);

const state = {
  token: localStorage.getItem("realWorldExpServerToken") || "",
  serverUrl: localStorage.getItem("realWorldExpServerServerUrl") || "",
  agents: [],
  runs: [],
  episode: null,
  stream: null,
  heading: null,
  orientation: {},
  motion: {},
  action: null,
  actionStartHeading: null,
  targetHeading: null,
  pendingRequest: null,
  wakeLock: null,
  busy: false,
  resumeNeedsAccess: false,
  deviceAccessReady: false,
  sensorListenersAttached: false,
  lastAbsoluteOrientationAt: null,
  openRunId: null,
  runsRefreshTimer: null,
};
const imuIntegrator = new ImuDistanceIntegrator();
const usingNativeCapture = nativeCaptureAvailable();
if (usingNativeCapture) document.documentElement.classList.add("native-arkit");

elements.accessToken.value = state.token;
const queryServer = new URLSearchParams(window.location.search).get("server");
if (queryServer) state.serverUrl = queryServer;
elements.serverUrl.value = state.serverUrl;

function authHeaders(extra = {}) {
  const headers = { ...extra };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  return headers;
}

async function api(path, options = {}) {
  const response = await fetch(apiUrl(path), {
    cache: "no-store",
    ...options,
    headers: authHeaders({ "Content-Type": "application/json", ...(options.headers || {}) }),
  });
  let payload = {};
  try { payload = await response.json(); } catch (_) { /* handled below */ }
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function apiUrl(path) {
  return state.serverUrl ? `${state.serverUrl.replace(/\/$/, "")}${path}` : path;
}

function setConnection(online, label) {
  elements.connectionBadge.classList.toggle("online", online);
  elements.connectionText.textContent = label;
}

async function checkHealth() {
  try {
    const response = await fetch(apiUrl("/api/health"), { cache: "no-store" });
    const health = await response.json();
    setConnection(true, "Server online");
    elements.tokenLabel.hidden = !health.auth_required;
    elements.secureContextNote.hidden = window.isSecureContext;
    await loadAgents();
    await resumeEpisode();
  } catch (_) {
    setConnection(false, "Server offline");
  }
}

async function loadAgents() {
  try {
    const selected = localStorage.getItem("realWorldExpServerAgent") || "internvla-n1";
    let agents;
    if (state.serverUrl) {
      agents = [{
        id: "direct-server",
        label: "Direct server agent",
        description: `Handled directly by ${state.serverUrl} without the Sites bridge.`,
        capabilities: { observation: "rgb" },
      }];
    } else {
      ({ agents } = await api("/api/agents"));
    }
    state.agents = agents;
    elements.agentSelect.replaceChildren();
    if (!agents.length) {
      elements.agentSelect.add(new Option("No online local agent", ""));
      elements.agentSelect.disabled = true;
      elements.startButton.disabled = true;
      elements.agentDescription.textContent = "Run bridge on the model machine; this page will refresh automatically.";
      return;
    }
    const selectableAgents = agents.filter((agent) => agent.capabilities?.selectable !== false);
    for (const agent of agents) {
      const observation = String(agent.capabilities?.observation || "rgb").toUpperCase();
      const status = agentStateLabel(agent.capabilities?.state);
      const option = new Option(`${agent.label} · ${observation} · ${status}`, agent.id);
      option.disabled = agent.capabilities?.selectable === false;
      elements.agentSelect.add(option);
    }
    const selectedAgent = selectableAgents.find((agent) => agent.id === selected) || selectableAgents[0];
    elements.agentSelect.value = selectedAgent?.id || "";
    elements.agentSelect.disabled = !selectableAgents.length;
    elements.startButton.disabled = !selectableAgents.length;
    updateAgentDescription();
  } catch (_) {
    state.agents = [];
    elements.agentSelect.replaceChildren(new Option("Agent list unavailable", ""));
    elements.agentSelect.disabled = true;
    elements.startButton.disabled = true;
    elements.agentDescription.textContent = "Failed to read local agent status.";
  }
}

function updateAgentDescription() {
  const agent = state.agents.find((item) => item.id === elements.agentSelect.value);
  const detail = agent?.capabilities?.detail;
  elements.agentDescription.textContent = agent
    ? `${agent.description}${detail ? ` ${detail}` : ""}`
    : "Please select a local agent.";
  if (agent) localStorage.setItem("realWorldExpServerAgent", agent.id);
}

function agentStateLabel(stateName) {
  return ({
    stopped: "Ready to start",
    starting: "Starting",
    ready: "Ready",
    error: "Start failed",
    blocked: "Input unsupported",
  })[stateName] || "Connected";
}

async function loadRuns(openEpisodeId = null, attempt = 0) {
  elements.runsStatus.hidden = false;
  elements.runsStatus.textContent = "Loading completed runs...";
  elements.runsList.replaceChildren();
  try {
    const { runs, bridge_online: bridgeOnline, last_synced_at: lastSyncedAt } = await api("/api/runs");
    state.runs = runs;
    if (!runs.length) {
      elements.runsStatus.textContent = bridgeOnline
        ? "No completed runs in the local runs folder."
        : "Local bridge offline; no live run index is available.";
      return;
    }
    elements.runsStatus.hidden = bridgeOnline;
    if (!bridgeOnline) {
      const synced = lastSyncedAt ? formatRunTime(lastSyncedAt) : "never";
      elements.runsStatus.textContent = `Local bridge offline · last folder sync ${synced}`;
    }
    for (const run of runs) elements.runsList.append(buildRunCard(run));
    if (openEpisodeId) {
      const run = runs.find((item) => item.episode_id === openEpisodeId);
      if (run) {
        openRun(run);
      } else if (attempt < 20 && !elements.runsPanel.hidden) {
        elements.runsStatus.hidden = false;
        elements.runsStatus.textContent = "Waiting for the local runs folder to update...";
        window.setTimeout(() => loadRuns(openEpisodeId, attempt + 1), 1500);
      }
    }
  } catch (error) {
    elements.runsStatus.textContent = `Unable to load runs: ${error.message}`;
  }
}

function buildRunCard(run) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "run-card";
  button.addEventListener("click", () => openRun(run));

  const header = document.createElement("span");
  header.className = "run-card-header";
  const method = document.createElement("strong");
  method.textContent = run.agent?.label || run.agent?.id || "Agent";
  const time = document.createElement("time");
  time.dateTime = run.finished_at;
  time.textContent = formatRunTime(run.finished_at);
  header.append(method, time);

  const instruction = document.createElement("span");
  instruction.className = "run-card-instruction";
  instruction.textContent = run.instruction;

  const footer = document.createElement("span");
  footer.className = "run-card-footer";
  const outcome = document.createElement("span");
  outcome.className = run.termination?.success ? "success" : "";
  outcome.textContent = runOutcome(run);
  const video = document.createElement("span");
  video.textContent = run.video?.cached
    ? "Ready to play  →"
    : run.video?.available ? "Fetch video  →" : "No video";
  footer.append(outcome, video);
  button.append(header, instruction, footer);
  return button;
}

function showRuns(openEpisodeId = null) {
  elements.setupPanel.hidden = true;
  elements.finishedPanel.hidden = true;
  elements.runsPanel.hidden = false;
  elements.runDetail.hidden = true;
  elements.runsList.hidden = false;
  loadRuns(openEpisodeId);
  window.clearInterval(state.runsRefreshTimer);
  state.runsRefreshTimer = window.setInterval(() => {
    if (!elements.runsPanel.hidden && elements.runDetail.hidden) loadRuns();
  }, 5000);
}

function hideRuns() {
  window.clearInterval(state.runsRefreshTimer);
  state.runsRefreshTimer = null;
  state.openRunId = null;
  elements.runVideo.pause();
  elements.runVideo.removeAttribute("src");
  elements.runVideo.load();
  elements.runsPanel.hidden = true;
  elements.runDetail.hidden = true;
  elements.setupPanel.hidden = false;
  state.episode = null;
  loadAgents();
}

async function openRun(run) {
  state.openRunId = run.episode_id;
  elements.runsList.hidden = true;
  elements.runsStatus.hidden = true;
  elements.runDetail.hidden = false;
  elements.runDetailMethod.textContent = run.agent?.label || run.agent?.id || "Agent";
  elements.runDetailInstruction.textContent = run.instruction;
  elements.runDetailOutcome.textContent = runOutcome(run);
  elements.runDetailOutcome.classList.toggle("success", Boolean(run.termination?.success));
  elements.runDetailMeta.textContent = `${formatRunTime(run.finished_at)} · ${run.num_steps} actions · ${run.episode_id}`;
  const hasVideo = Boolean(run.video?.cached && run.video?.url);
  elements.runVideo.hidden = !hasVideo;
  elements.runVideoMissing.hidden = hasVideo;
  if (hasVideo) {
    elements.runVideo.src = apiUrl(run.video.url);
    elements.runVideo.load();
  } else if (run.video?.available) {
    elements.runVideoMissing.textContent = "Requesting video from the local runs folder...";
    await requestRunVideo(run.episode_id);
  } else {
    elements.runVideoMissing.textContent = "This local run has no replay video.";
  }
}

async function requestRunVideo(episodeId) {
  try {
    const requested = await api(`/api/runs/${encodeURIComponent(episodeId)}/video/request`, {
      method: "POST",
      body: "{}",
    });
    if (requested.status === "ready") {
      showRunVideo(episodeId, requested.url);
      return;
    }
    for (let attempt = 0; attempt < 180 && state.openRunId === episodeId; attempt += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
      const result = await api(`/api/runs/${encodeURIComponent(episodeId)}/video/status`);
      if (result.status === "ready") {
        showRunVideo(episodeId, result.url);
        return;
      }
      if (result.status === "failed") throw new Error(result.error || "Local video transfer failed");
      elements.runVideoMissing.textContent = result.status === "processing"
        ? "Transferring video from the local model machine..."
        : "Waiting for the local bridge...";
    }
  } catch (error) {
    if (state.openRunId === episodeId) {
      elements.runVideoMissing.textContent = `Unable to fetch video: ${error.message}`;
    }
  }
}

function showRunVideo(episodeId, url) {
  if (state.openRunId !== episodeId) return;
  const run = state.runs.find((item) => item.episode_id === episodeId);
  if (run?.video) {
    run.video.cached = true;
    run.video.url = url;
  }
  elements.runVideoMissing.hidden = true;
  elements.runVideo.hidden = false;
  elements.runVideo.src = apiUrl(url);
  elements.runVideo.load();
}

function closeRun() {
  state.openRunId = null;
  elements.runVideo.pause();
  elements.runVideo.removeAttribute("src");
  elements.runVideo.load();
  elements.runDetail.hidden = true;
  elements.runsList.hidden = false;
  loadRuns();
}

function formatRunTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown time";
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  }).format(date);
}

function runOutcome(run) {
  if (run.termination?.success) return "Success";
  return String(run.termination?.reason || "Finished").replaceAll("_", " ");
}

elements.runsButton.addEventListener("click", () => showRuns());
elements.runsBackButton.addEventListener("click", hideRuns);
elements.runsRefreshButton.addEventListener("click", () => loadRuns());
elements.runDetailBackButton.addEventListener("click", closeRun);

async function refreshActiveAgentStatus() {
  if (!state.episode?.agent?.id || state.serverUrl) return;
  try {
    const { agents } = await api("/api/agents");
    const agent = agents.find((item) => item.id === state.episode.agent.id);
    if (!agent) return;
    const status = agentStateLabel(agent.capabilities?.state);
    elements.activeAgentName.textContent = `${agent.label} · ${status}`;
    if (agent.capabilities?.state === "starting") {
      elements.requestStatus.textContent = "Starting local model on GPU 0; first load may take a few minutes...";
    } else if (agent.capabilities?.state === "error") {
    elements.requestStatus.textContent = agent.capabilities?.detail || "Local model failed to start";
    }
  } catch (_) { /* the action request remains authoritative */ }
}

elements.agentSelect.addEventListener("change", updateAgentDescription);

async function resumeEpisode() {
  let episodeId = localStorage.getItem("realWorldExpServerActiveEpisode");
  try {
    if (!episodeId) {
      const { episodes } = await api("/api/episodes");
      episodeId = episodes.find((episode) => episode.status === "active")?.episode_id;
    }
    if (!episodeId) return false;
    const { episode } = await api(`/api/episodes/${encodeURIComponent(episodeId)}`);
    if (episode.status === "active") {
      state.episode = episode;
      localStorage.setItem("realWorldExpServerActiveEpisode", episode.episode_id);
      showRunner();
      elements.sensorStatus.textContent = "Tap the button below to grant permissions";
      if (episode.steps.length) {
        presentAction(episode.steps.at(-1));
        state.resumeNeedsAccess = true;
        elements.nextActionText.textContent = "Resume camera and IMU";
        elements.guidanceHint.textContent = "Restore permissions first, then continue the last received action.";
      }
      return true;
    } else {
      localStorage.removeItem("realWorldExpServerActiveEpisode");
    }
  } catch (_) {
    return false;
  }
  return false;
}

elements.episodeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy) return;
  state.busy = true;
  elements.startButton.disabled = true;
  state.token = elements.accessToken.value.trim();
  state.serverUrl = elements.serverUrl.value.trim().replace(/\/$/, "");
  localStorage.setItem("realWorldExpServerToken", state.token);
  localStorage.setItem("realWorldExpServerServerUrl", state.serverUrl);
  try {
    requestLandscape();
    const deviceAccess = requestDeviceAccess();
    const resumed = await resumeEpisode();
    if (resumed) {
      await deviceAccess;
      showToast("Recovered unfinished Episode from server");
      return;
    }
    const { episode } = await api("/api/episodes", {
      method: "POST",
      body: JSON.stringify({
        episode_id: elements.episodeId.value,
        agent_id: elements.agentSelect.value,
        instruction: elements.instruction.value,
        hfov: Number(elements.hfov.value),
        client_metadata: {
          user_agent: navigator.userAgent,
          platform: navigator.platform,
          screen: { width: screen.width, height: screen.height, pixel_ratio: devicePixelRatio },
          capture_mode: usingNativeCapture ? "arkit_rgbd_pose" : "web_rgb_imu",
          native_capabilities: nativeCapabilities(),
        },
      }),
    });
    state.episode = episode;
    localStorage.setItem("realWorldExpServerActiveEpisode", episode.episode_id);
    showRunner();
    elements.requestStatus.textContent = "Episode created; starting local model on GPU 0...";
    refreshActiveAgentStatus();
    await deviceAccess;
  } catch (error) {
    showToast(error.message);
  } finally {
    state.busy = false;
    elements.startButton.disabled = false;
  }
});

async function requestDeviceAccess() {
  if (state.deviceAccessReady && (state.stream || usingNativeCapture)) return;
  let sensorGranted = false;
  const permissions = [];
  for (const eventClass of [window.DeviceOrientationEvent, window.DeviceMotionEvent]) {
    if (eventClass && typeof eventClass.requestPermission === "function") {
      permissions.push(eventClass.requestPermission());
    }
  }
  if (permissions.length) {
    const outcomes = await Promise.all(permissions);
    sensorGranted = outcomes.every((result) => result === "granted");
  } else {
    sensorGranted = Boolean(window.DeviceOrientationEvent || window.DeviceMotionEvent);
  }
  if (!state.stream && !usingNativeCapture) {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera API is unavailable in this browser; use a trusted HTTPS origin");
    }
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    elements.camera.srcObject = state.stream;
    await elements.camera.play();
  }
  if (!state.sensorListenersAttached) {
    window.addEventListener("deviceorientation", onOrientation, true);
    window.addEventListener("deviceorientationabsolute", onOrientation, true);
    window.addEventListener("devicemotion", onMotion, true);
    state.sensorListenersAttached = true;
  }
  state.deviceAccessReady = true;
  if (usingNativeCapture) {
    const capabilities = nativeCapabilities();
    elements.sensorStatus.textContent = capabilities?.scene_depth ? "ARKit RGB-D" : "ARKit RGB";
  } else {
    elements.sensorStatus.textContent = sensorGranted ? "Enabled" : "Unavailable (manual execution)";
  }
}

function showRunner() {
  document.body.classList.add("episode-running");
  elements.setupPanel.hidden = true;
  elements.finishedPanel.hidden = true;
  elements.runner.hidden = false;
  elements.activeEpisodeId.textContent = state.episode.episode_id;
  elements.activeInstruction.textContent = state.episode.instruction;
  elements.activeAgentName.textContent = state.episode.agent?.label
    || state.agents.find((agent) => agent.id === elements.agentSelect.value)?.label
    || "Local agent";
  elements.stepValue.textContent = state.episode.steps.length;
  if (state.episode.steps.length) {
    imuIntegrator.restore(state.episode.steps.at(-1)?.sensor?.imu_odometry);
  } else {
    imuIntegrator.reset();
    resetGuidance();
  }
  updateDistanceDisplay();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function onOrientation(event) {
  const now = performance.now();
  const isAbsolute = Boolean(event.absolute) || Number.isFinite(event.webkitCompassHeading);
  if (isAbsolute) state.lastAbsoluteOrientationAt = now;
  else if (state.lastAbsoluteOrientationAt !== null && now - state.lastAbsoluteOrientationAt < 1500) return;
  const screenAngle = screenAngleDegrees();
  const heading = headingFromOrientation(event, screenAngle);
  state.heading = heading;
  state.orientation = {
    heading_deg: heading,
    alpha: finiteOrNull(event.alpha),
    beta: finiteOrNull(event.beta),
    gamma: finiteOrNull(event.gamma),
    absolute: Boolean(event.absolute),
    screen_angle: screenAngle,
    compass_source: Number.isFinite(event.webkitCompassHeading)
      ? "webkitCompassHeading + tilt compensation"
      : `${event.absolute ? "absolute" : "relative"} alpha/beta/gamma camera axis`,
  };
  elements.headingValue.textContent = heading === null ? "—" : `${Math.round(heading)}°`;
  if (state.action && state.targetHeading === null && heading !== null) anchorCurrentAction();
  updateAlignment();
}

function onMotion(event) {
  state.motion = {
    interval_ms: finiteOrNull(event.interval),
    acceleration: vector(event.acceleration),
    acceleration_including_gravity: vector(event.accelerationIncludingGravity),
    rotation_rate: vector(event.rotationRate),
  };
  imuIntegrator.update(
    event.acceleration && vector(event.acceleration),
    state.orientation,
    event.interval,
    event.timeStamp,
    event.rotationRate,
    event.accelerationIncludingGravity && vector(event.accelerationIncludingGravity),
  );
  updateDistanceDisplay();
}

function vector(value) {
  if (!value) return null;
  return {
    x: finiteOrNull(value.x), y: finiteOrNull(value.y), z: finiteOrNull(value.z),
    alpha: finiteOrNull(value.alpha), beta: finiteOrNull(value.beta), gamma: finiteOrNull(value.gamma),
  };
}

function sensorSnapshot() {
  return {
    orientation: state.orientation,
    motion: state.motion,
    imu_odometry: imuIntegrator.snapshot(),
  };
}

function updateDistanceDisplay() {
  const estimate = imuIntegrator.snapshot();
  elements.distanceValue.textContent = estimate.available
    ? `${estimate.path_distance_m.toFixed(2)} m`
    : "—";
}

elements.nextActionButton.addEventListener("click", async () => {
  if (!state.resumeNeedsAccess) {
    requestAction(false);
    return;
  }
  if (state.busy) return;
  state.busy = true;
  elements.nextActionButton.disabled = true;
  try {
    await requestDeviceAccess();
    state.resumeNeedsAccess = false;
    anchorCurrentAction();
    elements.nextActionText.textContent = "Action complete, capture next frame";
    elements.guidanceHint.textContent = "Permissions restored. Continue from the last action.";
  } catch (error) {
    showToast(error.message);
  } finally {
    state.busy = false;
    elements.nextActionButton.disabled = false;
  }
});
elements.retryButton.addEventListener("click", () => requestAction(true));

async function requestAction(retry) {
  if (state.busy || !state.episode) return;
  state.busy = true;
  elements.nextActionButton.disabled = true;
  elements.retryButton.hidden = true;
  elements.requestStatus.textContent = "Capturing frame and waiting for agent...";
  try {
    await requestDeviceAccess();
    if (!retry || !state.pendingRequest) state.pendingRequest = await buildStepPayload();
    let response = await api(`/api/episodes/${encodeURIComponent(state.episode.episode_id)}/step`, {
      method: "POST",
      body: JSON.stringify(state.pendingRequest),
    });
    if (response.pending_job_id) response = await waitForStep(response.pending_job_id);
    const { step } = response;
    state.pendingRequest = null;
    state.episode.steps.push(step);
    elements.stepValue.textContent = state.episode.steps.length;
    presentAction(step);
    elements.requestStatus.textContent = step.latency_ms ? `Agent latency ${step.latency_ms} ms` : "Action returned and saved";
    navigator.vibrate?.([70, 45, 70]);
  } catch (error) {
    elements.requestStatus.textContent = `Request failed: ${error.message}`;
    elements.retryButton.hidden = false;
    showToast(error.message);
  } finally {
    state.busy = false;
    elements.nextActionButton.disabled = false;
  }
}

async function waitForStep(requestId) {
  const deadline = Date.now() + 20 * 60 * 1000;
  elements.requestStatus.textContent = "Request delivered, waiting for local GPU agent...";
  let polls = 0;
  while (Date.now() < deadline) {
    await new Promise((resolve) => window.setTimeout(resolve, 750));
    const result = await api(`/api/steps/${encodeURIComponent(requestId)}`);
    if (result.step) return result;
    if (result.error) throw new Error(result.error);
    polls += 1;
    if (polls % 4 === 0) await refreshActiveAgentStatus();
  }
  throw new Error("Timeout waiting for local agent; you can keep current frame and retry");
}

async function buildStepPayload() {
  const requestId = crypto.randomUUID?.() || `${Date.now()}-${Math.random()}`;
  if (usingNativeCapture) {
    const observation = await requestNativeCapture(requestId);
    if (!observation?.rgb_jpeg_base64 || !observation?.metadata) {
      throw new Error("ARKit returned an incomplete observation");
    }
    const image = `data:image/jpeg;base64,${observation.rgb_jpeg_base64}`;
    elements.nativePreview.src = image;
    const attachments = {};
    if (observation.depth_png_base64) {
      attachments.depth = `data:image/png;base64,${observation.depth_png_base64}`;
    }
    if (observation.confidence_png_base64) {
      attachments.confidence = `data:image/png;base64,${observation.confidence_png_base64}`;
    }
    return {
      request_id: requestId,
      captured_at: observation.captured_at || new Date().toISOString(),
      image,
      attachments,
      sensor: { ...sensorSnapshot(), arkit: observation.metadata },
    };
  }
  if (elements.camera.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    await new Promise((resolve) => elements.camera.addEventListener("loadeddata", resolve, { once: true }));
  }
  const sourceWidth = elements.camera.videoWidth;
  const sourceHeight = elements.camera.videoHeight;
  const scale = Math.min(1, 960 / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  elements.captureCanvas.width = width;
  elements.captureCanvas.height = height;
  elements.captureCanvas.getContext("2d").drawImage(elements.camera, 0, 0, width, height);
  return {
    request_id: requestId,
    captured_at: new Date().toISOString(),
    image: elements.captureCanvas.toDataURL("image/jpeg", 0.82),
    sensor: sensorSnapshot(),
  };
}

function presentAction(step) {
  state.action = step;
  anchorCurrentAction();
  elements.actionCode.textContent = step.action;
  elements.actionLabel.textContent = step.label;
  elements.guidanceHint.textContent = step.action === "STOP"
    ? "Agent requested stop; choose a termination reason based on current scene conditions."
    : "Hold the phone and execute the action, align direction, then capture the next frame.";
  elements.actionDetail.textContent = actionDetail(step);
  elements.alignment.hidden = step.action === "STOP" || state.heading === null;
  elements.nextActionButton.hidden = step.action === "STOP";
  elements.nextActionText.textContent = "Action complete, capture next frame";
  elements.compass.style.setProperty("--arrow-angle", `${step.target_delta_deg}deg`);
  elements.compass.classList.toggle("aligned", step.action === "STOP");
  updateAlignment();
}

function anchorCurrentAction() {
  state.actionStartHeading = state.heading;
  state.targetHeading = state.heading === null || !state.action
    ? null
    : normalizeDegrees(state.heading + state.action.target_delta_deg);
}

function actionDetail(step) {
  if (step.action === "STOP") {
    return `Do not move further; choose "Reached goal" or another true termination reason below.`;
  }
  const distance = Number(step.distance_m || 0);
  if (step.action === "TURN_LEFT" || step.action === "TURN_RIGHT") {
    return `Turn approximately ${Math.abs(step.target_delta_deg)}°, then stop when the arrow is aligned.`;
  }
  if (step.action === "FOLLOW_TRAJECTORY") {
    return `Relative to current heading ${signedDegrees(step.target_delta_deg)}, move ${distance.toFixed(2)} m.`;
  }
  return `${step.label} about ${distance.toFixed(2)} m. IMU only calibrates heading; distance is confirmed by human execution.`;
}

function updateAlignment() {
  if (!state.action || state.action.action === "STOP") return;
  if (state.heading === null || state.targetHeading === null) {
    elements.alignment.hidden = true;
    return;
  }
  const error = shortestDelta(state.heading, state.targetHeading);
  const tolerance = 8;
  const score = Math.max(0, 1 - Math.abs(error) / 90);
  elements.targetArrow.style.setProperty("--arrow-angle", `${error}deg`);
  elements.alignmentFill.style.width = `${Math.round(score * 100)}%`;
  const aligned = Math.abs(error) <= tolerance;
  elements.compass.classList.toggle("aligned", aligned);
  elements.alignmentText.textContent = aligned
    ? "Heading is aligned, you can execute movement"
    : `Need to turn ${error < 0 ? "left" : "right"} ${Math.round(Math.abs(error))}°`;
}

document.querySelectorAll("[data-reason]").forEach((button) => {
  button.addEventListener("click", () => finishEpisode(button.dataset.reason, button.textContent.trim()));
});

async function finishEpisode(reason, label) {
  if (state.busy || !state.episode) return;
  if (!window.confirm(`Confirm ending this episode as "${label}"?`)) return;
  state.busy = true;
  try {
    const { episode } = await api(`/api/episodes/${encodeURIComponent(state.episode.episode_id)}/finish`, {
      method: "POST",
      body: JSON.stringify({
        reason,
        notes: "",
        client_time: new Date().toISOString(),
        sensor: sensorSnapshot(),
      }),
    });
    localStorage.removeItem("realWorldExpServerActiveEpisode");
    const finalDistance = imuIntegrator.snapshot();
    stopCamera();
    document.body.classList.remove("episode-running");
    try { screen.orientation?.unlock?.(); } catch (_) { /* optional capability */ }
    elements.runner.hidden = true;
    elements.finishedPanel.hidden = false;
    elements.finishedTitle.textContent = reason === "success" ? "Episode completed successfully" : "Episode ended";
    const distanceText = finalDistance.available ? ` · IMU about ${finalDistance.path_distance_m.toFixed(2)} m` : "";
    elements.finishedSummary.textContent = `${episode.episode_id} · ${episode.steps.length} actions · ${label}${distanceText}`;
  } catch (error) {
    showToast(error.message);
  } finally {
    state.busy = false;
  }
}

elements.newEpisodeButton.addEventListener("click", () => {
  state.episode = null;
  state.action = null;
  state.resumeNeedsAccess = false;
  elements.finishedPanel.hidden = true;
  elements.setupPanel.hidden = false;
  elements.episodeId.value = "";
  loadAgents();
});

elements.viewRunButton.addEventListener("click", () => {
  showRuns(state.episode?.episode_id || null);
});

elements.wakeLockButton.addEventListener("click", async () => {
  try {
    if (state.wakeLock) {
      await state.wakeLock.release();
      state.wakeLock = null;
    } else {
      state.wakeLock = await navigator.wakeLock.request("screen");
      state.wakeLock.addEventListener("release", () => {
        state.wakeLock = null;
        elements.wakeLockButton.classList.remove("active");
      });
    }
    elements.wakeLockButton.classList.toggle("active", Boolean(state.wakeLock));
  } catch (error) { showToast(`Unable to keep screen awake: ${error.message}`); }
});

function stopCamera() {
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  state.deviceAccessReady = false;
  elements.camera.srcObject = null;
  elements.nativePreview.removeAttribute("src");
  if (state.sensorListenersAttached) {
    window.removeEventListener("deviceorientation", onOrientation, true);
    window.removeEventListener("deviceorientationabsolute", onOrientation, true);
    window.removeEventListener("devicemotion", onMotion, true);
    state.sensorListenersAttached = false;
  }
  state.lastAbsoluteOrientationAt = null;
}

function resetGuidance() {
  state.action = null;
  state.actionStartHeading = null;
  state.targetHeading = null;
  state.pendingRequest = null;
  elements.actionCode.textContent = "READY";
  elements.actionLabel.textContent = "Waiting for first frame";
  elements.actionDetail.textContent = "Point the phone toward the movement direction and tap the button below to start.";
  elements.guidanceHint.textContent = "Capture one frame first, then let the agent choose the next step.";
  elements.nextActionButton.hidden = false;
  elements.nextActionButton.disabled = false;
  elements.nextActionText.textContent = "Capture and request action";
  elements.retryButton.hidden = true;
  elements.alignment.hidden = true;
  elements.compass.classList.remove("aligned");
  elements.targetArrow.style.setProperty("--arrow-angle", "0deg");
  elements.requestStatus.textContent = "";
}

function requestLandscape() {
  try {
    const lock = screen.orientation?.lock?.("landscape");
    lock?.catch?.(() => {});
  } catch (_) { /* manual rotation remains available */ }
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => { elements.toast.hidden = true; }, 5000);
}

function finiteOrNull(value) { return Number.isFinite(value) ? value : null; }
function shortestDelta(from, to) { return ((to - from + 540) % 360) - 180; }
function signedDegrees(value) { return `${value < 0 ? "Left" : "Right"} ${Math.abs(value).toFixed(0)}°`; }

document.addEventListener("visibilitychange", async () => {
  if (document.visibilityState === "visible" && state.episode && !state.wakeLock && navigator.wakeLock) {
    try {
      state.wakeLock = await navigator.wakeLock.request("screen");
      elements.wakeLockButton.classList.add("active");
    } catch (_) { /* optional capability */ }
  }
});

window.addEventListener("beforeunload", stopCamera);
checkHealth();
window.setInterval(() => {
  if (!state.episode && !elements.setupPanel.hidden) loadAgents();
  else if (state.episode) refreshActiveAgentStatus();
}, 15_000);
