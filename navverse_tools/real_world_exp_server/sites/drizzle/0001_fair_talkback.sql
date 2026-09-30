CREATE TABLE `agents` (
	`id` text PRIMARY KEY NOT NULL,
	`label` text NOT NULL,
	`description` text DEFAULT '' NOT NULL,
	`capabilities_json` text DEFAULT '{}' NOT NULL,
	`updated_at` text NOT NULL
);
--> statement-breakpoint
CREATE INDEX `idx_agents_updated` ON `agents` (`updated_at`);--> statement-breakpoint
ALTER TABLE `episodes` ADD `agent_id` text DEFAULT 'demo' NOT NULL;--> statement-breakpoint
ALTER TABLE `episodes` ADD `agent_label` text DEFAULT 'Demo agent' NOT NULL;