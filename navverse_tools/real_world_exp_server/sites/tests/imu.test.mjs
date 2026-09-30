import assert from "node:assert/strict";
import test from "node:test";

import {
  ImuDistanceIntegrator,
  headingFromOrientation,
  rotateDeviceToWorld,
  worldLinearAcceleration,
} from "../public/imu.mjs";

test("heading follows compass convention and screen rotation", () => {
  assert.equal(headingFromOrientation({ alpha: 30 }, 0), 330);
  assert.equal(headingFromOrientation({ alpha: 30 }, 90), 60);
  assert.equal(headingFromOrientation({ webkitCompassHeading: 100 }, 90), 190);
});

test("heading follows the rear camera axis while the phone is upright", () => {
  assert.ok(Math.abs(headingFromOrientation({ alpha: 230, beta: 0, gamma: 90 }, 90) - 40) < 1e-10);
  assert.ok(Math.abs(headingFromOrientation({ alpha: 30, beta: 90, gamma: 0 }, 0) - 330) < 1e-10);
});

test("device acceleration rotates into the earth frame", () => {
  assert.equal(rotateDeviceToWorld(
    { x: 1, y: 0, z: 0 },
    { alpha: null, beta: 0, gamma: 0 },
  ), null);
  assert.deepEqual(rotateDeviceToWorld(
    { x: 1, y: 0, z: 0 },
    { alpha: 0, beta: 0, gamma: 0 },
  ), { x: 1, y: 0, z: 0 });
  const rotated = rotateDeviceToWorld(
    { x: 1, y: 0, z: 0 },
    { alpha: 90, beta: 0, gamma: 0 },
  );
  assert.ok(Math.abs(rotated.x) < 1e-10);
  assert.ok(Math.abs(rotated.y - 1) < 1e-10);
});

test("double integration estimates displacement", () => {
  const integrator = new ImuDistanceIntegrator({
    deadbandMss: 0,
    stationaryThresholdMss: 0,
    velocityDampingPerS: 0,
  });
  for (let index = 0; index < 10; index += 1) {
    integrator.update(
      { x: 1, y: 0, z: 0 },
      { alpha: 0, beta: 0, gamma: 0 },
      100,
      index * 100,
    );
  }
  const estimate = integrator.snapshot();
  assert.equal(estimate.displacement_m, 0.5);
  assert.equal(estimate.path_distance_m, 0.5);
});

test("gravity-including acceleration falls back to world-frame gravity removal", () => {
  const resolved = worldLinearAcceleration(
    null,
    { x: 1, y: 0, z: 9.80665 },
    { alpha: 0, beta: 0, gamma: 0 },
  );
  assert.equal(resolved.source, "accelerationIncludingGravity - gravity");
  assert.ok(Math.abs(resolved.vector.x - 1) < 1e-10);
  assert.ok(Math.abs(resolved.vector.y) < 1e-10);
  assert.ok(Math.abs(resolved.vector.z) < 1e-10);
});
