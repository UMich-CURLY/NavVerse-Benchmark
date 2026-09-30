import assert from "node:assert/strict";
import test from "node:test";

import {
  nativeCaptureAvailable,
  receiveNativeCapture,
  requestNativeCapture,
} from "../public/native_bridge.mjs";

test("native capture resolves the matching request", async () => {
  const messages = [];
  const target = {
    setTimeout,
    clearTimeout,
    webkit: { messageHandlers: { capture: { postMessage: (message) => messages.push(message) } } },
  };
  assert.equal(nativeCaptureAvailable(target), true);
  const capture = requestNativeCapture("frame-1", 1000, target);
  assert.deepEqual(messages, [{ request_id: "frame-1" }]);
  assert.equal(receiveNativeCapture({ request_id: "frame-1", ok: true, observation: { rgb_jpeg_base64: "abc" } }), true);
  assert.deepEqual(await capture, { rgb_jpeg_base64: "abc" });
});

test("native capture rejects an explicit native error", async () => {
  const target = {
    setTimeout,
    clearTimeout,
    webkit: { messageHandlers: { capture: { postMessage() {} } } },
  };
  const capture = requestNativeCapture("frame-2", 1000, target);
  receiveNativeCapture({ request_id: "frame-2", ok: false, error: "no depth" });
  await assert.rejects(capture, /no depth/);
});
