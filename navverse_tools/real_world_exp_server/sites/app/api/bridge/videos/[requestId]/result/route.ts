import { bridgeAuthorized, nowIso, relayEnv, unauthorized } from "@/lib/relay";

export async function POST(
  request: Request,
  context: { params: Promise<{ requestId: string }> },
) {
  if (!bridgeAuthorized(request)) return unauthorized();
  const { requestId } = await context.params;
  const payload = (await request.json()) as { error?: unknown };
  const error = String(payload.error ?? "").trim().slice(0, 1000);
  if (!error) return Response.json({ error: "error is required" }, { status: 400 });
  const result = await relayEnv().DB.prepare(
    `UPDATE video_requests
        SET status = 'failed', error = ?, lease_until = NULL, updated_at = ?
      WHERE id = ?`,
  )
    .bind(error, nowIso(), requestId)
    .run();
  if (!result.meta.changes) return Response.json({ error: "video request does not exist" }, { status: 404 });
  return Response.json({ ok: true, failed: true });
}
