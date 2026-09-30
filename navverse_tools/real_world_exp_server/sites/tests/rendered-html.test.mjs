import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("server renders the phone executor shell", async () => {
  const [page, layout] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
  ]);
  assert.match(layout, /const title = ["']NavVerse Real-World Experiment Server["']/);
  assert.match(layout, /summary_large_image/);
  assert.match(page, /src=["']\/phone\.html["']/);
  assert.match(page, /allow=["']camera; accelerometer; gyroscope; magnetometer; screen-wake-lock["']/);
  assert.doesNotMatch(`${page}\n${layout}`, /codex-preview|loading skeleton|Starter Project/i);
});

test("packages the landscape phone runner and relay bindings", async () => {
  const [phoneHtml, phoneScript, imuScript, nativeBridge, phoneStyles, episodeRoute, assetRoute, runsRoute, runIndexRoute, videoRoute, videoRequestRoute, videoStatusRoute, uploadRoute, nextVideoRoute, hostingConfig, initialMigration, agentMigration, runMigration] = await Promise.all([
    readFile(new URL("../dist/client/phone.html", import.meta.url), "utf8"),
    readFile(new URL("../dist/client/app.js", import.meta.url), "utf8"),
    readFile(new URL("../dist/client/imu.mjs", import.meta.url), "utf8"),
    readFile(new URL("../dist/client/native_bridge.mjs", import.meta.url), "utf8"),
    readFile(new URL("../dist/client/styles.css", import.meta.url), "utf8"),
    readFile(new URL("../app/api/episodes/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/bridge/jobs/[jobId]/asset/[assetName]/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/runs/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/bridge/runs/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/runs/[episodeId]/video/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/runs/[episodeId]/video/request/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/runs/[episodeId]/video/status/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/bridge/runs/[episodeId]/video/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/bridge/videos/next/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../dist/.openai/hosting.json", import.meta.url), "utf8"),
    readFile(new URL("../dist/.openai/drizzle/0000_striped_boom_boom.sql", import.meta.url), "utf8"),
    readFile(new URL("../dist/.openai/drizzle/0001_fair_talkback.sql", import.meta.url), "utf8"),
    readFile(new URL("../dist/.openai/drizzle/0002_legal_unus.sql", import.meta.url), "utf8"),
  ]);
  assert.match(phoneHtml, /id="camera"/);
  assert.match(phoneHtml, /id="agentSelect"/);
  assert.match(phoneHtml, /id="distanceValue"/);
  assert.match(phoneHtml, /id="runsPanel"/);
  assert.match(phoneHtml, /id="runVideo"/);
  assert.match(phoneHtml, /type="module" src="\/app\.js"/);
  assert.match(phoneScript, /waitForStep/);
  assert.match(phoneScript, /headingFromOrientation/);
  assert.match(phoneScript, /imu_odometry/);
  assert.match(phoneScript, /resetGuidance\(\)/);
  assert.match(phoneScript, /accelerationIncludingGravity/);
  assert.match(phoneScript, /requestNativeCapture/);
  assert.match(imuScript, /class ImuDistanceIntegrator/);
  assert.match(nativeBridge, /messageHandlers\?\.capture/);
  assert.match(phoneScript, /agent_id:\s*elements\.agentSelect\.value/);
  assert.match(phoneScript, /loadRuns/);
  assert.match(phoneScript, /requestRunVideo/);
  assert.match(phoneScript, /runsRefreshTimer/);
  assert.match(phoneScript, /formatRunTime/);
  assert.match(phoneScript, /Starting/);
  const episodeCreate = phoneScript.indexOf("const { episode } = await api");
  const runnerShown = phoneScript.indexOf("showRunner();", episodeCreate);
  const cameraAwait = phoneScript.indexOf("await deviceAccess;", runnerShown);
  assert.ok(episodeCreate >= 0 && episodeCreate < runnerShown && runnerShown < cameraAwait,
    "new episodes must trigger model warmup before waiting for camera access");
  assert.match(episodeRoute, /'warmup', 'pending'/);
  assert.match(assetRoute, /attachmentKey/);
  assert.match(runsRoute, /FROM local_runs/);
  assert.match(runsRoute, /source: "local_bridge"/);
  assert.match(runIndexRoute, /DELETE FROM local_runs/);
  assert.match(videoRoute, /Content-Type.*video\/mp4/s);
  assert.match(videoRequestRoute, /INSERT INTO video_requests/);
  assert.match(videoStatusRoute, /status: "ready"/);
  assert.match(uploadRoute, /FRAMES\.put/);
  assert.match(nextVideoRoute, /status = 'pending'/);
  assert.match(phoneStyles, /overflow:\s*hidden/);
  assert.match(phoneStyles, /height:\s*100dvh/);
  assert.match(phoneStyles, /orientation:\s*landscape/);
  assert.match(phoneStyles, /body\.episode-running \.topbar/);
  const hosting = JSON.parse(hostingConfig);
  assert.equal(hosting.project_id, undefined);
  assert.equal(hosting.d1, "DB");
  assert.equal(hosting.r2, "FRAMES");
  assert.match(initialMigration, /CREATE TABLE `episodes`/);
  assert.match(initialMigration, /CREATE TABLE `jobs`/);
  assert.match(agentMigration, /CREATE TABLE `agents`/);
  assert.match(agentMigration, /ADD `agent_id`/);
  assert.match(runMigration, /CREATE TABLE `local_runs`/);
  assert.match(runMigration, /CREATE TABLE `video_requests`/);
  assert.match(runMigration, /PRAGMA optimize/);
});
