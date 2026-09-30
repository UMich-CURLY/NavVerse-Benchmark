import { attachmentKey, bridgeAuthorized, nowIso, relayEnv, unauthorized } from "@/lib/relay";

async function deleteObservation(frameKey: string | null) {
  if (!frameKey) return;
  await Promise.all([
    relayEnv().FRAMES.delete(frameKey),
    relayEnv().FRAMES.delete(attachmentKey(frameKey, "depth")),
    relayEnv().FRAMES.delete(attachmentKey(frameKey, "confidence")),
  ]);
}

export async function POST(request: Request, context: { params: Promise<{ jobId: string }> }) {
  if (!bridgeAuthorized(request)) return unauthorized();
  const { jobId } = await context.params;
  const payload = (await request.json()) as Record<string, unknown>;
  const row = await relayEnv().DB.prepare("SELECT frame_key, kind FROM jobs WHERE id = ?")
    .bind(jobId)
    .first<{ frame_key: string | null; kind: string }>();
  if (!row) return Response.json({ error: "job does not exist" }, { status: 404 });
  const failure = typeof payload.error === "string" ? payload.error.slice(0, 1000) : "";
  if (failure) {
    const now = nowIso();
    await relayEnv().DB.prepare(
      "UPDATE jobs SET status = 'failed', error = ?, lease_until = NULL, updated_at = ? WHERE id = ?",
    )
      .bind(failure, now, jobId)
      .run();
    await deleteObservation(row.frame_key);
    return Response.json({ ok: true, failed: true });
  }
  if (row.kind === "step" && typeof payload.action !== "string") {
    return Response.json({ error: "step result requires an action" }, { status: 400 });
  }
  const now = nowIso();
  await relayEnv().DB.prepare(
    "UPDATE jobs SET status = 'complete', result_json = ?, lease_until = NULL, updated_at = ? WHERE id = ?",
  )
    .bind(JSON.stringify(payload), now, jobId)
    .run();
  await deleteObservation(row.frame_key);
  return Response.json({ ok: true });
}
