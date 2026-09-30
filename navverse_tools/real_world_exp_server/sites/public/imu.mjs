const DEG_TO_RAD = Math.PI / 180;
const STANDARD_GRAVITY_MSS = 9.80665;

export function normalizeDegrees(value) {
  return ((value % 360) + 360) % 360;
}

export function screenAngleDegrees() {
  const value = screen.orientation?.angle ?? window.orientation ?? 0;
  return Number.isFinite(value) ? Number(value) : 0;
}

export function headingFromOrientation(event, screenAngle = 0) {
  const webkitHeading = Number.isFinite(event.webkitCompassHeading)
    ? event.webkitCompassHeading
    : Number.NaN;
  const alpha = Number.isFinite(webkitHeading)
    ? normalizeDegrees(360 - webkitHeading)
    : (Number.isFinite(event.alpha) ? event.alpha : Number.NaN);
  const orientation = {
    alpha,
    beta: Number.isFinite(event.beta) ? event.beta : Number.NaN,
    gamma: Number.isFinite(event.gamma) ? event.gamma : Number.NaN,
  };
  const cameraVector = rotateDeviceToWorld({ x: 0, y: 0, z: -1 }, orientation);
  if (cameraVector && Math.hypot(cameraVector.x, cameraVector.y) > 1e-6) {
    return normalizeDegrees(Math.atan2(cameraVector.x, cameraVector.y) / DEG_TO_RAD);
  }
  if (Number.isFinite(webkitHeading)) return normalizeDegrees(webkitHeading + screenAngle);
  return Number.isFinite(alpha) ? normalizeDegrees(360 - alpha + screenAngle) : null;
}

export function rotateDeviceToWorld(vector, orientation) {
  const values = [orientation.alpha, orientation.beta, orientation.gamma, vector.x, vector.y, vector.z];
  if (!values.every(Number.isFinite)) return null;
  const alpha = orientation.alpha * DEG_TO_RAD;
  const beta = orientation.beta * DEG_TO_RAD;
  const gamma = orientation.gamma * DEG_TO_RAD;

  const ca = Math.cos(alpha);
  const sa = Math.sin(alpha);
  const cb = Math.cos(beta);
  const sb = Math.sin(beta);
  const cg = Math.cos(gamma);
  const sg = Math.sin(gamma);
  return {
    x: (ca * cg - sa * sb * sg) * vector.x - sa * cb * vector.y + (ca * sg + sa * sb * cg) * vector.z,
    y: (sa * cg + ca * sb * sg) * vector.x + ca * cb * vector.y + (sa * sg - ca * sb * cg) * vector.z,
    z: -cb * sg * vector.x + sb * vector.y + cb * cg * vector.z,
  };
}

export function worldLinearAcceleration(acceleration, accelerationIncludingGravity, orientation) {
  if (acceleration) {
    const vector = rotateDeviceToWorld(acceleration, orientation);
    if (vector) return { vector, source: "acceleration" };
  }
  if (!accelerationIncludingGravity) return null;
  const vector = rotateDeviceToWorld(accelerationIncludingGravity, orientation);
  if (!vector) return null;
  return {
    vector: { ...vector, z: vector.z - STANDARD_GRAVITY_MSS },
    source: "accelerationIncludingGravity - gravity",
  };
}

export class ImuDistanceIntegrator {
  constructor(options = {}) {
    this.deadbandMss = options.deadbandMss ?? 0.08;
    this.stationaryThresholdMss = options.stationaryThresholdMss ?? 0.18;
    this.stationaryHoldS = options.stationaryHoldS ?? 0.3;
    this.velocityDampingPerS = options.velocityDampingPerS ?? 0.7;
    this.biasLearningRate = options.biasLearningRate ?? 0.015;
    this.reset();
  }

  reset() {
    this.position = { x: 0, y: 0, z: 0 };
    this.velocity = { x: 0, y: 0, z: 0 };
    this.bias = { x: 0, y: 0, z: 0 };
    this.pathDistanceM = 0;
    this.stationaryForS = 0;
    this.lastTimestampMs = null;
    this.sampleCount = 0;
    this.available = false;
    this.accelerationSource = null;
  }

  restore(snapshot) {
    this.reset();
    if (!snapshot) return;
    for (const key of ["x", "y", "z"]) {
      if (Number.isFinite(snapshot.position_m?.[key])) this.position[key] = snapshot.position_m[key];
      if (Number.isFinite(snapshot.velocity_m_s?.[key])) this.velocity[key] = snapshot.velocity_m_s[key];
    }
    if (Number.isFinite(snapshot.path_distance_m)) this.pathDistanceM = snapshot.path_distance_m;
    if (Number.isFinite(snapshot.sample_count)) this.sampleCount = snapshot.sample_count;
    this.available = Boolean(snapshot.available);
    this.accelerationSource = snapshot.acceleration_source || null;
  }

  update(
    acceleration,
    orientation,
    intervalMs,
    timestampMs = performance.now(),
    rotationRate = null,
    accelerationIncludingGravity = null,
  ) {
    const resolved = worldLinearAcceleration(acceleration, accelerationIncludingGravity, orientation);
    if (!resolved) return this.snapshot();
    const { vector: world, source } = resolved;
    let dt = Number(intervalMs) / 1000;
    if (!Number.isFinite(dt) || dt <= 0) {
      dt = this.lastTimestampMs === null ? 0 : (timestampMs - this.lastTimestampMs) / 1000;
    }
    this.lastTimestampMs = timestampMs;
    if (!Number.isFinite(dt) || dt <= 0 || dt > 0.2) return this.snapshot();

    this.available = true;
    this.accelerationSource = source;
    this.sampleCount += 1;
    const horizontalMagnitude = Math.hypot(world.x - this.bias.x, world.y - this.bias.y);
    const rotationMagnitude = rotationRate
      ? Math.hypot(rotationRate.alpha || 0, rotationRate.beta || 0, rotationRate.gamma || 0)
      : 0;
    const stationary = horizontalMagnitude < this.stationaryThresholdMss && rotationMagnitude < 5;
    this.stationaryForS = stationary ? this.stationaryForS + dt : 0;
    if (stationary) {
      for (const key of ["x", "y", "z"]) {
        this.bias[key] += this.biasLearningRate * (world[key] - this.bias[key]);
      }
    }

    const corrected = {};
    for (const key of ["x", "y", "z"]) {
      const value = world[key] - this.bias[key];
      corrected[key] = Math.abs(value) < this.deadbandMss ? 0 : value;
    }
    const previousVelocity = { ...this.velocity };
    const damping = Math.exp(-this.velocityDampingPerS * dt);
    for (const key of ["x", "y", "z"]) {
      this.velocity[key] = (this.velocity[key] + corrected[key] * dt) * damping;
    }
    if (this.stationaryForS >= this.stationaryHoldS) {
      this.velocity = { x: 0, y: 0, z: 0 };
    }

    const delta = {};
    for (const key of ["x", "y", "z"]) {
      delta[key] = (previousVelocity[key] + this.velocity[key]) * 0.5 * dt;
      this.position[key] += delta[key];
    }
    this.pathDistanceM += Math.hypot(delta.x, delta.y);
    return this.snapshot();
  }

  snapshot() {
    return {
      available: this.available,
      displacement_m: round(Math.hypot(this.position.x, this.position.y), 3),
      path_distance_m: round(this.pathDistanceM, 3),
      position_m: roundedVector(this.position),
      velocity_m_s: roundedVector(this.velocity),
      sample_count: this.sampleCount,
      acceleration_source: this.accelerationSource,
      method: "device-to-world acceleration rotation + horizontal double integration",
      drift_control: "bias learning, acceleration deadband, velocity damping, stationary zero-velocity update",
    };
  }
}

function roundedVector(vector) {
  return Object.fromEntries(Object.entries(vector).map(([key, value]) => [key, round(value, 4)]));
}

function round(value, digits) {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}
