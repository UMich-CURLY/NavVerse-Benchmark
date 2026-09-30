import { bridgeAuthorized, nowIso, relayEnv, unauthorized } from "@/lib/relay";

export async function GET(request: Request) {
  if (!bridgeAuthorized(request)) return unauthorized();
  const now = nowIso();
  const leaseUntil = new Date(Date.now() + 10 * 60_000).toISOString();
  const row = await relayEnv().DB.prepare(
    `SELECT id, episode_id FROM video_requests
      WHERE status = 'pending' OR (status = 'processing' AND lease_until < ?)
      ORDER BY id LIMIT 1`,
  )
    .bind(now)
    .first<{ id: number; episode_id: string }>();
  if (!row) return new Response(null, { status: 204 });
  const claimed = await relayEnv().DB.prepare(
    `UPDATE video_requests SET status = 'processing', lease_until = ?, error = NULL, updated_at = ?
      WHERE id = ? AND (status = 'pending' OR (status = 'processing' AND lease_until < ?))`,
  )
    .bind(leaseUntil, now, row.id, now)
    .run();
  if (!claimed.meta.changes) return new Response(null, { status: 204 });
  return Response.json({ request: { id: row.id, episode_id: row.episode_id } });
}
