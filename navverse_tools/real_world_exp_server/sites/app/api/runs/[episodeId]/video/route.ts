import { relayEnv, runVideoKey } from "@/lib/relay";

async function videoObject(episodeId: string) {
  try {
    return await relayEnv().FRAMES.get(runVideoKey(episodeId));
  } catch {
    return null;
  }
}

export async function GET(
  _request: Request,
  context: { params: Promise<{ episodeId: string }> },
) {
  const { episodeId } = await context.params;
  const object = await videoObject(episodeId);
  if (!object) return Response.json({ error: "run video is unavailable" }, { status: 404 });
  return new Response(object.body, {
    headers: {
      "Content-Type": "video/mp4",
      "Content-Length": String(object.size),
      "Cache-Control": "public, max-age=3600",
      "Content-Disposition": `inline; filename="${episodeId}.mp4"`,
    },
  });
}

export async function HEAD(
  _request: Request,
  context: { params: Promise<{ episodeId: string }> },
) {
  const { episodeId } = await context.params;
  let object;
  try {
    object = await relayEnv().FRAMES.head(runVideoKey(episodeId));
  } catch {
    object = null;
  }
  if (!object) return new Response(null, { status: 404 });
  return new Response(null, {
    headers: {
      "Content-Type": "video/mp4",
      "Content-Length": String(object.size),
      "Cache-Control": "public, max-age=3600",
    },
  });
}
