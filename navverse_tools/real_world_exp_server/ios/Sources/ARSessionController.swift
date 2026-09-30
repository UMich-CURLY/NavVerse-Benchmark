import ARKit
import Foundation

@MainActor
final class ARSessionController: NSObject, ObservableObject {
    let session = ARSession()
    private let encoder = ARFrameEncoder()
    private(set) var sceneDepthEnabled = false
    private(set) var smoothedSceneDepthEnabled = false

    override init() {
        super.init()
        session.delegate = self
    }

    var capabilities: [String: Any] {
        [
            "arkit": ARWorldTrackingConfiguration.isSupported,
            "scene_depth": ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth),
            "smoothed_scene_depth": ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth),
            "rgb": true,
            "camera_pose": true,
            "intrinsics": true,
        ]
    }

    func start() {
        guard ARWorldTrackingConfiguration.isSupported else { return }
        let configuration = ARWorldTrackingConfiguration()
        configuration.worldAlignment = .gravity
        let bothDepthModes: ARConfiguration.FrameSemantics = [.sceneDepth, .smoothedSceneDepth]
        if ARWorldTrackingConfiguration.supportsFrameSemantics(bothDepthModes) {
            configuration.frameSemantics.formUnion(bothDepthModes)
            sceneDepthEnabled = true
            smoothedSceneDepthEnabled = true
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics.insert(.sceneDepth)
            sceneDepthEnabled = true
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            configuration.frameSemantics.insert(.smoothedSceneDepth)
            smoothedSceneDepthEnabled = true
        }
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }

    func pause() {
        session.pause()
    }

    func capture(completion: @escaping (Result<[String: Any], Error>) -> Void) {
        guard let frame = session.currentFrame else {
            completion(.failure(CaptureError.frameUnavailable))
            return
        }
        encoder.encode(frame: frame, completion: completion)
    }
}

extension ARSessionController: ARSessionDelegate {}

enum CaptureError: LocalizedError {
    case frameUnavailable
    case rgbEncodingFailed
    case depthEncodingFailed
    case malformedRequest

    var errorDescription: String? {
        switch self {
        case .frameUnavailable: "ARKit has not produced a frame yet."
        case .rgbEncodingFailed: "The ARKit RGB buffer could not be encoded."
        case .depthEncodingFailed: "The ARKit depth buffer could not be encoded."
        case .malformedRequest: "The webpage sent an invalid capture request."
        }
    }
}
