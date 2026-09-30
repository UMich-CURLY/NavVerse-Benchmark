import { env } from "cloudflare:workers";

export type RelayEnv = {
  DB: D1Database;
  FRAMES: R2Bucket;
  BRIDGE_TOKEN?: string;
};

export function relayEnv(): RelayEnv {
  return env as unknown as RelayEnv;
}

export function nowIso(): string {
  return new Date().toISOString();
}

export function parseJson<T>(value: string | null, fallback: T): T {
  if (!value) return fallback;
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

export function safeEpisodeId(input: unknown): string {
  const raw = String(input ?? "").trim();
  const cleaned = raw.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^[._-]+|[._-]+$/g, "");
  return (cleaned || `phone_${new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)}`).slice(0, 120);
}

export function bridgeAuthorized(request: Request): boolean {
  const expected = relayEnv().BRIDGE_TOKEN;
  return Boolean(expected) && request.headers.get("Authorization") === `Bearer ${expected}`;
}

export function unauthorized(): Response {
  return Response.json({ error: "Invalid bridge token" }, { status: 401 });
}

export function decodeJpeg(value: unknown): Uint8Array {
  if (typeof value !== "string") throw new Error("image must be a base64 string or data URL");
  const encoded = value.startsWith("data:") && value.includes(",") ? value.split(",", 2)[1] : value;
  const binary = atob(encoded);
  if (!binary.length || binary.length > 12 * 1024 * 1024) throw new Error("image size is invalid");
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8) throw new Error("image must be JPEG");
  return bytes;
}

export function decodePng(value: unknown): Uint8Array {
  if (typeof value !== "string") throw new Error("attachment must be a base64 string or data URL");
  const encoded = value.startsWith("data:") && value.includes(",") ? value.split(",", 2)[1] : value;
  const binary = atob(encoded);
  if (!binary.length || binary.length > 16 * 1024 * 1024) throw new Error("attachment size is invalid");
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (!signature.every((value, index) => bytes[index] === value)) throw new Error("attachment must be PNG");
  return bytes;
}

export function attachmentKey(frameKey: string, name: "depth" | "confidence"): string {
  if (!frameKey.endsWith(".jpg")) throw new Error("invalid frame key");
  return `${frameKey.slice(0, -4)}.${name}.png`;
}

export function runVideoKey(episodeId: string): string {
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$/.test(episodeId)) {
    throw new Error("invalid episode id");
  }
  return `runs/${episodeId}/episode.mp4`;
}

export async function episodeJson(episodeId: string) {
  const row = await relayEnv().DB.prepare(
    `SELECT id, instruction, agent_id, agent_label, status, hfov, client_metadata_json, started_at, updated_at,
            termination_json
       FROM episodes WHERE id = ?`,
  )
    .bind(episodeId)
    .first<Record<string, unknown>>();
  if (!row) return null;
  const steps = await relayEnv().DB.prepare(
    `SELECT request_id, captured_at, sensor_json, result_json, created_at, updated_at
       FROM jobs
      WHERE episode_id = ? AND kind = 'step' AND status = 'complete'
      ORDER BY id`,
  )
    .bind(episodeId)
    .all<Record<string, unknown>>();
  return {
    schema_version: 1,
    episode_id: row.id,
    instruction: row.instruction,
    agent: { id: row.agent_id, label: row.agent_label },
    status: row.status,
    hfov: row.hfov,
    client_metadata: parseJson(String(row.client_metadata_json ?? "{}"), {}),
    started_at: row.started_at,
    updated_at: row.updated_at,
    termination: parseJson(String(row.termination_json ?? ""), null),
    steps: steps.results.map((step, index) => ({
      step_index: index,
      request_id: step.request_id,
      captured_at: step.captured_at,
      received_at: step.created_at,
      sensor: parseJson(String(step.sensor_json ?? "{}"), {}),
      ...parseJson(String(step.result_json ?? "{}"), {}),
    })),
  };
}

export async function stepJson(requestId: string) {
  const row = await relayEnv().DB.prepare(
    `SELECT request_id, episode_id, status, captured_at, sensor_json, result_json,
            created_at, updated_at, error
       FROM jobs WHERE request_id = ? AND kind = 'step'`,
  )
    .bind(requestId)
    .first<Record<string, unknown>>();
  if (!row) return null;
  if (row.status !== "complete") {
    return { pending: true, status: row.status, error: row.error };
  }
  const count = await relayEnv().DB.prepare(
    `SELECT COUNT(*) AS count FROM jobs
      WHERE episode_id = ? AND kind = 'step' AND status = 'complete' AND id <=
            (SELECT id FROM jobs WHERE request_id = ?)`,
  )
    .bind(row.episode_id, requestId)
    .first<{ count: number }>();
  return {
    step: {
      step_index: Math.max(0, Number(count?.count ?? 1) - 1),
      request_id: row.request_id,
      captured_at: row.captured_at,
      received_at: row.created_at,
      sensor: parseJson(String(row.sensor_json ?? "{}"), {}),
      ...parseJson(String(row.result_json ?? "{}"), {}),
    },
  };
}
