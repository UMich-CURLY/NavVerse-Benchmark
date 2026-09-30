import { nowIso, relayEnv, runVideoKey } from "@/lib/relay";

export async function POST(
  _request: Request,
  context: { params: Promise<{ episodeId: string }> },
) {
  const { episodeId } = await context.params;
  const cached = await relayEnv().FRAMES.head(runVideoKey(episodeId));
  if (cached) {
    return Response.json({ status: "ready", url: `/api/runs/${encodeURIComponent(episodeId)}/video` });
  }
  const localRun = await relayEnv().DB.prepare(
    "SELECT video_available FROM local_runs WHERE episode_id = ? AND status = 'finished'",
  )
    .bind(episodeId)
    .first<{ video_available: number }>();
  if (!localRun?.video_available) {
    return Response.json({ error: "local run video is unavailable" }, { status: 404 });
  }
  const now = nowIso();
  await relayEnv().DB.prepare(
    `INSERT INTO video_requests (episode_id, status, error, lease_until, created_at, updated_at)
     VALUES (?, 'pending', NULL, NULL, ?, ?)
     ON CONFLICT(episode_id) DO UPDATE SET
       status = CASE WHEN video_requests.status = 'failed' THEN 'pending' ELSE video_requests.status END,
       error = CASE WHEN video_requests.status = 'failed' THEN NULL ELSE video_requests.error END,
       lease_until = CASE WHEN video_requests.status = 'failed' THEN NULL ELSE video_requests.lease_until END,
       updated_at = excluded.updated_at`,
  )
    .bind(episodeId, now, now)
    .run();
  return Response.json({ status: "pending" }, { status: 202 });
}
