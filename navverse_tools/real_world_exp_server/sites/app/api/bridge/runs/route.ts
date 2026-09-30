import { bridgeAuthorized, nowIso, relayEnv, unauthorized } from "@/lib/relay";

type LocalRun = {
  episode_id?: unknown;
  instruction?: unknown;
  agent?: { id?: unknown; label?: unknown };
  status?: unknown;
  started_at?: unknown;
  updated_at?: unknown;
  num_steps?: unknown;
  termination?: unknown;
  video?: { available?: unknown; size_bytes?: unknown };
};

const EPISODE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$/;

export async function POST(request: Request) {
  if (!bridgeAuthorized(request)) return unauthorized();
  const payload = (await request.json()) as { runs?: LocalRun[] };
  const runs = Array.isArray(payload.runs) ? payload.runs.slice(0, 100) : [];
  const snapshotId = crypto.randomUUID();
  const indexedAt = nowIso();
  const statements = [];

  for (const run of runs) {
    const episodeId = String(run.episode_id ?? "").trim();
    const instruction = String(run.instruction ?? "").trim().slice(0, 2000);
    const status = String(run.status ?? "").trim();
    const startedAt = String(run.started_at ?? "").trim();
    const updatedAt = String(run.updated_at ?? "").trim();
    const numSteps = Number(run.num_steps ?? 0);
    if (
      !EPISODE_ID.test(episodeId) || !instruction ||
      !["active", "finished"].includes(status) ||
      !Number.isFinite(Date.parse(startedAt)) || !Number.isFinite(Date.parse(updatedAt)) ||
      !Number.isInteger(numSteps) || numSteps < 0
    ) {
      return Response.json({ error: "invalid local run snapshot" }, { status: 400 });
    }
    const videoSize = Number(run.video?.size_bytes ?? 0);
    statements.push(relayEnv().DB.prepare(
      `INSERT INTO local_runs
         (episode_id, instruction, agent_id, agent_label, status, started_at, updated_at,
          num_steps, termination_json, video_available, video_size_bytes, snapshot_id, indexed_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(episode_id) DO UPDATE SET
         instruction = excluded.instruction,
         agent_id = excluded.agent_id,
         agent_label = excluded.agent_label,
         status = excluded.status,
         started_at = excluded.started_at,
         updated_at = excluded.updated_at,
         num_steps = excluded.num_steps,
         termination_json = excluded.termination_json,
         video_available = excluded.video_available,
         video_size_bytes = excluded.video_size_bytes,
         snapshot_id = excluded.snapshot_id,
         indexed_at = excluded.indexed_at`,
    ).bind(
      episodeId,
      instruction,
      String(run.agent?.id ?? "unknown").slice(0, 64),
      String(run.agent?.label ?? "Unknown agent").slice(0, 100),
      status,
      startedAt,
      updatedAt,
      numSteps,
      run.termination == null ? null : JSON.stringify(run.termination),
      run.video?.available === true ? 1 : 0,
      videoSize > 0 && Number.isSafeInteger(videoSize) ? videoSize : null,
      snapshotId,
      indexedAt,
    ));
  }
  statements.push(relayEnv().DB.prepare("DELETE FROM local_runs WHERE snapshot_id != ?").bind(snapshotId));
  await relayEnv().DB.batch(statements);
  return Response.json({ ok: true, indexed: runs.length, indexed_at: indexedAt });
}
