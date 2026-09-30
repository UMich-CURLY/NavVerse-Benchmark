import { attachmentKey, decodeJpeg, decodePng, episodeJson, nowIso, relayEnv, stepJson } from "@/lib/relay";

export async function POST(request: Request, context: { params: Promise<{ episodeId: string }> }) {
  const { episodeId } = await context.params;
  const episode = await episodeJson(episodeId);
  if (!episode) return Response.json({ error: "episode does not exist" }, { status: 404 });
  if (episode.status !== "active") return Response.json({ error: "episode is already finished" }, { status: 409 });
  const payload = (await request.json()) as Record<string, unknown>;
  const requestId = String(payload.request_id ?? "").trim();
  if (!requestId || requestId.length > 160) {
    return Response.json({ error: "request_id is required" }, { status: 400 });
  }
  const existing = await stepJson(requestId);
  if (existing) {
    return Response.json(existing, { status: "step" in existing ? 200 : 202 });
  }
  let image: Uint8Array;
  let depth: Uint8Array | null = null;
  let confidence: Uint8Array | null = null;
  try {
    image = decodeJpeg(payload.image);
    const attachments = payload.attachments as Record<string, unknown> | undefined;
    if (attachments?.depth) depth = decodePng(attachments.depth);
    if (attachments?.confidence) confidence = decodePng(attachments.confidence);
  } catch (error) {
    return Response.json({ error: error instanceof Error ? error.message : "invalid image" }, { status: 400 });
  }
  const frameKey = `pending/${episodeId}/${requestId}.jpg`;
  const sensor = payload.sensor && typeof payload.sensor === "object"
    ? { ...(payload.sensor as Record<string, unknown>) }
    : {};
  if (sensor.arkit && typeof sensor.arkit === "object") {
    sensor.arkit = {
      ...(sensor.arkit as Record<string, unknown>),
      assets: { depth: Boolean(depth), confidence: Boolean(confidence) },
    };
  } else if (depth || confidence) {
    return Response.json({ error: "ARKit metadata is required for RGB-D attachments" }, { status: 400 });
  }
  const sensorJson = JSON.stringify(sensor);
  if (sensorJson.length > 128 * 1024) {
    return Response.json({ error: "sensor metadata is too large" }, { status: 400 });
  }
  const storedKeys = [frameKey];
  const now = nowIso();
  try {
    await relayEnv().FRAMES.put(frameKey, image, { httpMetadata: { contentType: "image/jpeg" } });
    if (depth) {
      const key = attachmentKey(frameKey, "depth");
      storedKeys.push(key);
      await relayEnv().FRAMES.put(key, depth, { httpMetadata: { contentType: "image/png" } });
    }
    if (confidence) {
      const key = attachmentKey(frameKey, "confidence");
      storedKeys.push(key);
      await relayEnv().FRAMES.put(key, confidence, { httpMetadata: { contentType: "image/png" } });
    }
    await relayEnv().DB.prepare(
      `INSERT INTO jobs
         (request_id, episode_id, kind, status, frame_key, captured_at, sensor_json, created_at, updated_at)
       VALUES (?, ?, 'step', 'pending', ?, ?, ?, ?, ?)`,
    )
      .bind(requestId, episodeId, frameKey, payload.captured_at ?? null, sensorJson, now, now)
      .run();
    await relayEnv().DB.prepare("UPDATE episodes SET updated_at = ? WHERE id = ?").bind(now, episodeId).run();
  } catch (error) {
    await Promise.all(storedKeys.map((key) => relayEnv().FRAMES.delete(key)));
    return Response.json({ error: error instanceof Error ? error.message : "job creation failed" }, { status: 409 });
  }
  return Response.json({ pending: true, pending_job_id: requestId }, { status: 202 });
}
