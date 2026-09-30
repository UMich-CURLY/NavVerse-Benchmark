import { stepJson } from "@/lib/relay";

export async function GET(_request: Request, context: { params: Promise<{ requestId: string }> }) {
  const { requestId } = await context.params;
  const result = await stepJson(requestId);
  if (!result) return Response.json({ error: "step does not exist" }, { status: 404 });
  return Response.json(result, { status: "step" in result ? 200 : 202 });
}
