CREATE TABLE `local_runs` (
	`episode_id` text PRIMARY KEY NOT NULL,
	`instruction` text NOT NULL,
	`agent_id` text DEFAULT 'unknown' NOT NULL,
	`agent_label` text DEFAULT 'Unknown agent' NOT NULL,
	`status` text NOT NULL,
	`started_at` text NOT NULL,
	`updated_at` text NOT NULL,
	`num_steps` integer DEFAULT 0 NOT NULL,
	`termination_json` text,
	`video_available` integer DEFAULT false NOT NULL,
	`video_size_bytes` integer,
	`snapshot_id` text NOT NULL,
	`indexed_at` text NOT NULL
);
--> statement-breakpoint
CREATE INDEX `idx_local_runs_status_updated` ON `local_runs` (`status`,`updated_at`);--> statement-breakpoint
CREATE TABLE `video_requests` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`episode_id` text NOT NULL,
	`status` text DEFAULT 'pending' NOT NULL,
	`error` text,
	`lease_until` text,
	`created_at` text NOT NULL,
	`updated_at` text NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `idx_video_requests_episode_id` ON `video_requests` (`episode_id`);--> statement-breakpoint
CREATE INDEX `idx_video_requests_status_id` ON `video_requests` (`status`,`id`);--> statement-breakpoint
PRAGMA optimize;
