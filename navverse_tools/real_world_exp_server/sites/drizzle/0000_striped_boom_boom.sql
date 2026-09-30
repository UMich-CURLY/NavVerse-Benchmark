CREATE TABLE `episodes` (
	`id` text PRIMARY KEY NOT NULL,
	`instruction` text NOT NULL,
	`status` text DEFAULT 'active' NOT NULL,
	`hfov` real DEFAULT 79 NOT NULL,
	`client_metadata_json` text DEFAULT '{}' NOT NULL,
	`started_at` text NOT NULL,
	`updated_at` text NOT NULL,
	`termination_json` text
);
--> statement-breakpoint
CREATE INDEX `idx_episodes_status_updated` ON `episodes` (`status`,`updated_at`);--> statement-breakpoint
CREATE TABLE `jobs` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`request_id` text NOT NULL,
	`episode_id` text NOT NULL,
	`kind` text DEFAULT 'step' NOT NULL,
	`status` text DEFAULT 'pending' NOT NULL,
	`frame_key` text,
	`captured_at` text,
	`sensor_json` text DEFAULT '{}' NOT NULL,
	`result_json` text,
	`error` text,
	`lease_until` text,
	`created_at` text NOT NULL,
	`updated_at` text NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `idx_jobs_request_id` ON `jobs` (`request_id`);--> statement-breakpoint
CREATE INDEX `idx_jobs_status_id` ON `jobs` (`status`,`id`);--> statement-breakpoint
CREATE INDEX `idx_jobs_episode_id` ON `jobs` (`episode_id`,`id`);