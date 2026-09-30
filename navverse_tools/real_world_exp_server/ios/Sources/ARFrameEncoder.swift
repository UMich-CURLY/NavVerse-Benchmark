import ARKit
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers

final class ARFrameEncoder {
    private let queue = DispatchQueue(label: "org.navverse.real-world-exp-server.frame-encoder", qos: .userInitiated)
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

    func encode(frame: ARFrame, completion: @escaping (Result<[String: Any], Error>) -> Void) {
        queue.async {
            do {
                completion(.success(try self.encode(frame: frame)))
            } catch {
                completion(.failure(error))
            }
        }
    }

    private func encode(frame: ARFrame) throws -> [String: Any] {
        let rgbBuffer = frame.capturedImage
        let rgbImage = CIImage(cvPixelBuffer: rgbBuffer)
        guard let rgbData = context.jpegRepresentation(
            of: rgbImage,
            colorSpace: colorSpace,
            options: [.lossyCompressionQuality: 0.88]
        ) else {
            throw CaptureError.rgbEncodingFailed
        }

        let rawDepth = frame.sceneDepth
        let selectedDepth = rawDepth ?? frame.smoothedSceneDepth
        let depthSource = rawDepth == nil ? "smoothedSceneDepth" : "sceneDepth"
        var depthData: Data?
        var confidenceData: Data?
        var depthMetadata: [String: Any] = ["available": false]
        var confidenceMetadata: [String: Any] = ["available": false]
        var depthIntrinsics: [[Double]]?

        if let selectedDepth {
            let depthBuffer = selectedDepth.depthMap
            depthData = try encodeDepthMillimeters(depthBuffer)
            let depthWidth = CVPixelBufferGetWidth(depthBuffer)
            let depthHeight = CVPixelBufferGetHeight(depthBuffer)
            let rgbWidth = CVPixelBufferGetWidth(rgbBuffer)
            let rgbHeight = CVPixelBufferGetHeight(rgbBuffer)
            depthIntrinsics = scaledIntrinsics(
                frame.camera.intrinsics,
                scaleX: Float(depthWidth) / Float(rgbWidth),
                scaleY: Float(depthHeight) / Float(rgbHeight)
            )
            depthMetadata = [
                "available": true,
                "source": depthSource,
                "encoding": "png",
                "pixel_format": "uint16",
                "unit": "millimeter",
                "scale_m": 0.001,
                "width": depthWidth,
                "height": depthHeight,
            ]
            if let confidence = selectedDepth.confidenceMap {
                confidenceData = try encodeConfidence(confidence)
                confidenceMetadata = [
                    "available": true,
                    "source": depthSource,
                    "encoding": "png",
                    "pixel_format": "uint8",
                    "width": CVPixelBufferGetWidth(confidence),
                    "height": CVPixelBufferGetHeight(confidence),
                    "values": ["low": 0, "medium": 1, "high": 2],
                ]
            }
        }

        let frameDate = Date(timeIntervalSinceNow: frame.timestamp - ProcessInfo.processInfo.systemUptime)
        let rgbIntrinsics = matrix3Rows(frame.camera.intrinsics)
        let depthIntrinsicsValue: Any = depthIntrinsics.map { $0 as Any } ?? NSNull()
        let depthResolutionValue: Any = selectedDepth.map {
            [CVPixelBufferGetWidth($0.depthMap), CVPixelBufferGetHeight($0.depthMap)] as Any
        } ?? NSNull()
        let metadata: [String: Any] = [
            "schema_version": 1,
            "source": "arkit",
            "frame_timestamp_s": frame.timestamp,
            "rgb": [
                "encoding": "jpeg",
                "pixel_format": "ycbcr_sensor_capture_to_jpeg",
                "orientation": "sensor_native",
                "width": CVPixelBufferGetWidth(rgbBuffer),
                "height": CVPixelBufferGetHeight(rgbBuffer),
            ],
            "depth": depthMetadata,
            "confidence": confidenceMetadata,
            "intrinsics": [
                "rgb": rgbIntrinsics,
                "depth": depthIntrinsicsValue,
                "matrix_layout": "row-major",
                "rgb_resolution": [Int(frame.camera.imageResolution.width), Int(frame.camera.imageResolution.height)],
                "depth_resolution": depthResolutionValue,
            ],
            "pose": [
                "camera_to_world": matrix4Rows(frame.camera.transform),
                "matrix_layout": "row-major",
                "transform_semantics": "world_from_camera",
                "coordinate_system": "ARKit right-handed; x right, y up, camera looks along -z",
                "world_alignment": "gravity",
                "body_pose_available": false,
            ],
            "tracking_state": trackingState(frame.camera.trackingState),
            "world_mapping_status": worldMappingStatus(frame.worldMappingStatus),
        ]

        return [
            "captured_at": ISO8601DateFormatter().string(from: frameDate),
            "rgb_jpeg_base64": rgbData.base64EncodedString(),
            "depth_png_base64": depthData.map { $0.base64EncodedString() as Any } ?? NSNull(),
            "confidence_png_base64": confidenceData.map { $0.base64EncodedString() as Any } ?? NSNull(),
            "metadata": metadata,
        ]
    }

    private func encodeDepthMillimeters(_ buffer: CVPixelBuffer) throws -> Data {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            throw CaptureError.depthEncodingFailed
        }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceStride = CVPixelBufferGetBytesPerRow(buffer) / MemoryLayout<Float32>.stride
        let source = baseAddress.assumingMemoryBound(to: Float32.self)
        var millimeters = [UInt16](repeating: 0, count: width * height)
        for row in 0..<height {
            for column in 0..<width {
                let meters = source[row * sourceStride + column]
                if meters.isFinite && meters > 0 {
                    millimeters[row * width + column] = UInt16(clamping: Int((meters * 1000).rounded()))
                }
            }
        }
        return try encodeGrayPNG(
            bytes: millimeters.withUnsafeBytes { Data($0) },
            width: width,
            height: height,
            bitsPerComponent: 16,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue).union(.byteOrder16Little)
        )
    }

    private func encodeConfidence(_ buffer: CVPixelBuffer) throws -> Data {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            throw CaptureError.depthEncodingFailed
        }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceStride = CVPixelBufferGetBytesPerRow(buffer)
        let source = baseAddress.assumingMemoryBound(to: UInt8.self)
        var values = [UInt8](repeating: 0, count: width * height)
        for row in 0..<height {
            values.replaceSubrange(
                (row * width)..<((row + 1) * width),
                with: UnsafeBufferPointer(start: source + row * sourceStride, count: width)
            )
        }
        return try encodeGrayPNG(
            bytes: Data(values),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        )
    }

    private func encodeGrayPNG(
        bytes: Data,
        width: Int,
        height: Int,
        bitsPerComponent: Int,
        bitmapInfo: CGBitmapInfo
    ) throws -> Data {
        guard
            let provider = CGDataProvider(data: bytes as CFData),
            let image = CGImage(
                width: width,
                height: height,
                bitsPerComponent: bitsPerComponent,
                bitsPerPixel: bitsPerComponent,
                bytesPerRow: width * bitsPerComponent / 8,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: bitmapInfo,
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
            )
        else {
            throw CaptureError.depthEncodingFailed
        }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            output,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw CaptureError.depthEncodingFailed
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw CaptureError.depthEncodingFailed
        }
        return output as Data
    }

    private func scaledIntrinsics(_ matrix: simd_float3x3, scaleX: Float, scaleY: Float) -> [[Double]] {
        var scaled = matrix
        scaled.columns.0.x *= scaleX
        scaled.columns.1.y *= scaleY
        scaled.columns.2.x *= scaleX
        scaled.columns.2.y *= scaleY
        return matrix3Rows(scaled)
    }

    private func matrix3Rows(_ matrix: simd_float3x3) -> [[Double]] {
        (0..<3).map { row in (0..<3).map { column in Double(matrix[column][row]) } }
    }

    private func matrix4Rows(_ matrix: simd_float4x4) -> [[Double]] {
        (0..<4).map { row in (0..<4).map { column in Double(matrix[column][row]) } }
    }

    private func trackingState(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal: "normal"
        case .notAvailable: "not_available"
        case .limited(let reason): "limited_\(String(describing: reason))"
        }
    }

    private func worldMappingStatus(_ status: ARFrame.WorldMappingStatus) -> String {
        switch status {
        case .notAvailable: "not_available"
        case .limited: "limited"
        case .extending: "extending"
        case .mapped: "mapped"
        @unknown default: "unknown"
        }
    }
}
