import { access, copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sourceRoot = resolve(projectRoot, "..", "static");
const publicRoot = resolve(projectRoot, "public");

const assets = [
  ["index.html", "phone.html"],
  ["app.js", "app.js"],
  ["imu.mjs", "imu.mjs"],
  ["native_bridge.mjs", "native_bridge.mjs"],
  ["styles.css", "styles.css"],
  ["manifest.webmanifest", "manifest.webmanifest"],
];

await mkdir(publicRoot, { recursive: true });
const hasSource = await access(sourceRoot).then(() => true, () => false);
if (hasSource) {
  await Promise.all(assets.map(([source, destination]) =>
    copyFile(resolve(sourceRoot, source), resolve(publicRoot, destination))));
} else {
  await Promise.all(assets.map(([, destination]) => access(resolve(publicRoot, destination))));
}
