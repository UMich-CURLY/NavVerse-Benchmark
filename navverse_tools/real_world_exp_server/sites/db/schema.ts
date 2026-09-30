import { index, integer, real, sqliteTable, text, uniqueIndex } from "drizzle-orm/sqlite-core";

export const episodes = sqliteTable(
  "episodes",
  {
    id: text("id").primaryKey(),
    instruction: text("instruction").notNull(),
    agentId: text("agent_id").notNull().default("demo"),
    agentLabel: text("agent_label").notNull().default("Demo agent"),
    status: text("status").notNull().default("active"),
    hfov: real("hfov").notNull().default(79),
    clientMetadataJson: text("client_metadata_json").notNull().default("{}"),
    startedAt: text("started_at").notNull(),
    updatedAt: text("updated_at").notNull(),
    terminationJson: text("termination_json"),
  },
  (table) => [index("idx_episodes_status_updated").on(table.status, table.updatedAt)],
);

export const agents = sqliteTable(
  "agents",
  {
    id: text("id").primaryKey(),
    label: text("label").notNull(),
    description: text("description").notNull().default(""),
    capabilitiesJson: text("capabilities_json").notNull().default("{}"),
    updatedAt: text("updated_at").notNull(),
  },
  (table) => [index("idx_agents_updated").on(table.updatedAt)],
);

export const jobs = sqliteTable(
  "jobs",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    requestId: text("request_id").notNull(),
    episodeId: text("episode_id").notNull(),
    kind: text("kind").notNull().default("step"),
    status: text("status").notNull().default("pending"),
    frameKey: text("frame_key"),
    capturedAt: text("captured_at"),
    sensorJson: text("sensor_json").notNull().default("{}"),
    resultJson: text("result_json"),
    error: text("error"),
    leaseUntil: text("lease_until"),
    createdAt: text("created_at").notNull(),
    updatedAt: text("updated_at").notNull(),
  },
  (table) => [
    uniqueIndex("idx_jobs_request_id").on(table.requestId),
    index("idx_jobs_status_id").on(table.status, table.id),
    index("idx_jobs_episode_id").on(table.episodeId, table.id),
  ],
);

export const localRuns = sqliteTable(
  "local_runs",
  {
    episodeId: text("episode_id").primaryKey(),
    instruction: text("instruction").notNull(),
    agentId: text("agent_id").notNull().default("unknown"),
    agentLabel: text("agent_label").notNull().default("Unknown agent"),
    status: text("status").notNull(),
    startedAt: text("started_at").notNull(),
    updatedAt: text("updated_at").notNull(),
    numSteps: integer("num_steps").notNull().default(0),
    terminationJson: text("termination_json"),
    videoAvailable: integer("video_available", { mode: "boolean" }).notNull().default(false),
    videoSizeBytes: integer("video_size_bytes"),
    snapshotId: text("snapshot_id").notNull(),
    indexedAt: text("indexed_at").notNull(),
  },
  (table) => [index("idx_local_runs_status_updated").on(table.status, table.updatedAt)],
);

export const videoRequests = sqliteTable(
  "video_requests",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    episodeId: text("episode_id").notNull(),
    status: text("status").notNull().default("pending"),
    error: text("error"),
    leaseUntil: text("lease_until"),
    createdAt: text("created_at").notNull(),
    updatedAt: text("updated_at").notNull(),
  },
  (table) => [
    uniqueIndex("idx_video_requests_episode_id").on(table.episodeId),
    index("idx_video_requests_status_id").on(table.status, table.id),
  ],
);
