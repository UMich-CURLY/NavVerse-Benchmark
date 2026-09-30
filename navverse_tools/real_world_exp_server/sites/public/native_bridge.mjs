const pendingCaptures = new Map();

export function nativeCaptureAvailable(target = globalThis.window) {
  return Boolean(target?.webkit?.messageHandlers?.capture?.postMessage);
}

export function nativeCapabilities(target = globalThis.window) {
  return target?.navverseNativeCapabilities || null;
}

export function requestNativeCapture(requestId, timeoutMs = 20_000, target = globalThis.window) {
  if (!nativeCaptureAvailable(target)) return Promise.reject(new Error("ARKit capture bridge is unavailable"));
  return new Promise((resolve, reject) => {
    const timer = target.setTimeout(() => {
      pendingCaptures.delete(requestId);
      reject(new Error("ARKit capture timed out"));
    }, timeoutMs);
    pendingCaptures.set(requestId, { resolve, reject, timer, target });
    target.webkit.messageHandlers.capture.postMessage({ request_id: requestId });
  });
}

export function receiveNativeCapture(message) {
  const requestId = String(message?.request_id || "");
  const pending = pendingCaptures.get(requestId);
  if (!pending) return false;
  pendingCaptures.delete(requestId);
  pending.target.clearTimeout(pending.timer);
  if (message?.ok !== true) {
    pending.reject(new Error(String(message?.error || "ARKit capture failed")));
  } else {
    pending.resolve(message.observation);
  }
  return true;
}

if (typeof window !== "undefined") {
  window.navverseNativeCaptureResult = receiveNativeCapture;
}
