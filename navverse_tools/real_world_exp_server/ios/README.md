# NavVerse Real-World Experiment Server iOS host

This target is a thin native host for the existing Sites UI. It owns one `ARSession`, opens the deployed phone page in `WKWebView`, and handles `window.webkit.messageHandlers.capture` requests from the page.

Each capture is derived from one `ARFrame` and returns:

- sensor-native RGB as JPEG;
- raw `sceneDepth` when available, otherwise `smoothedSceneDepth`, encoded as UInt16 millimeter PNG;
- the matching UInt8 confidence map;
- RGB and scaled depth intrinsics;
- `camera_to_world`, ARKit timestamp, tracking state, and image sizes.

The webpage performs the authenticated Sites upload. The native host never copies browser cookies into `URLSession` and does not upload data independently.

## Generate and build

Set `NAVVERSE_SITE_URL` in `Info.plist` to your own HTTPS deployment before running on a device. Install XcodeGen on the Mac, then run:

```bash
cd navverse_tools/real_world_exp_server/ios
xcodegen generate
xcodebuild \
  -project NavVerseRealWorldExpServer.xcodeproj \
  -scheme NavVerseRealWorldExpServer \
  -sdk iphoneos \
  -configuration Debug \
  CODE_SIGNING_ALLOWED=NO \
  build
```

Open the generated project in Xcode, select a development team, and run on a physical iPhone or iPad. Simulator builds can type-check the WebKit code but cannot produce ARKit scene depth. A device without LiDAR still captures synchronized RGB, intrinsics, and pose; `scene_depth` is reported as unavailable at runtime.

`camera_to_world` is the unmodified ARKit `world_from_camera` transform in ARKit's right-handed coordinate system. A downstream robot body frame must provide a calibrated fixed `T_body_camera`; the host deliberately does not invent one.
