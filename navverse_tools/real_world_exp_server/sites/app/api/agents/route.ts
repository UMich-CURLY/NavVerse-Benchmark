import { parseJson, relayEnv } from "@/lib/relay";

export async function GET() {
  const onlineAfter = new Date(Date.now() - 60_000).toISOString();
  const rows = await relayEnv().DB.prepare(
    `SELECT id, label, description, capabilities_json, updated_at
       FROM agents WHERE updated_at >= ? ORDER BY label`,
  )
    .bind(onlineAfter)
    .all<Record<string, unknown>>();
  return Response.json({
    agents: rows.results.map((row) => ({
      id: row.id,
      label: row.label,
      description: row.description,
      capabilities: parseJson(String(row.capabilities_json ?? "{}"), {}),
      updated_at: row.updated_at,
    })),
  });
}
