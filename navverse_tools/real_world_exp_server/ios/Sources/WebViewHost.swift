import Foundation
import SwiftUI
import WebKit

struct WebViewHost: UIViewRepresentable {
    let arSession: ARSessionController

    func makeCoordinator() -> Coordinator {
        Coordinator(arSession: arSession, allowedOrigin: siteURL)
    }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.allowsInlineMediaPlayback = true
        configuration.mediaTypesRequiringUserActionForPlayback = []
        let controller = configuration.userContentController
        controller.add(context.coordinator, name: "capture")
        let capabilities = jsonString(arSession.capabilities)
        controller.addUserScript(WKUserScript(
            source: "window.navverseNativeCapabilities = \(capabilities);",
            injectionTime: .atDocumentStart,
            forMainFrameOnly: false
        ))
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        webView.scrollView.isScrollEnabled = false
        context.coordinator.webView = webView
        webView.load(URLRequest(url: siteURL))
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {}

    static func dismantleUIView(_ webView: WKWebView, coordinator: Coordinator) {
        webView.configuration.userContentController.removeScriptMessageHandler(forName: "capture")
        coordinator.arSession.pause()
    }

    private var siteURL: URL {
        guard
            let value = Bundle.main.object(forInfoDictionaryKey: "NAVVERSE_SITE_URL") as? String,
            let url = URL(string: value),
            url.scheme == "https"
        else {
            preconditionFailure("NAVVERSE_SITE_URL must be an HTTPS URL")
        }
        return url
    }

    private func jsonString(_ object: Any) -> String {
        guard
            JSONSerialization.isValidJSONObject(object),
            let data = try? JSONSerialization.data(withJSONObject: object),
            let value = String(data: data, encoding: .utf8)
        else { return "{}" }
        return value
    }

    @MainActor
    final class Coordinator: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
        let arSession: ARSessionController
        let allowedOrigin: URL
        weak var webView: WKWebView?
        private var captureInFlight = false

        init(arSession: ARSessionController, allowedOrigin: URL) {
            self.arSession = arSession
            self.allowedOrigin = allowedOrigin
        }

        func userContentController(
            _ userContentController: WKUserContentController,
            didReceive message: WKScriptMessage
        ) {
            guard
                message.name == "capture",
                isAllowedOrigin(message.frameInfo.request.url),
                let body = message.body as? [String: Any],
                let requestID = body["request_id"] as? String,
                !requestID.isEmpty
            else { return }
            let frameInfo = message.frameInfo
            guard !captureInFlight else {
                deliver([
                    "request_id": requestID,
                    "ok": false,
                    "error": "An ARKit capture is already being encoded.",
                ], to: frameInfo)
                return
            }
            captureInFlight = true
            arSession.capture { [weak self] result in
                DispatchQueue.main.async {
                    self?.captureInFlight = false
                    let response: [String: Any]
                    switch result {
                    case .success(let observation):
                        response = [
                            "request_id": requestID,
                            "ok": true,
                            "observation": observation,
                        ]
                    case .failure(let error):
                        response = [
                            "request_id": requestID,
                            "ok": false,
                            "error": error.localizedDescription,
                        ]
                    }
                    self?.deliver(response, to: frameInfo)
                }
            }
        }

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
        ) {
            guard navigationAction.targetFrame?.isMainFrame != false else {
                decisionHandler(.allow)
                return
            }
            decisionHandler(isAllowedOrigin(navigationAction.request.url) ? .allow : .cancel)
        }

        private func isAllowedOrigin(_ url: URL?) -> Bool {
            guard
                let url,
                url.scheme == allowedOrigin.scheme,
                url.host == allowedOrigin.host
            else { return false }
            return effectivePort(url) == effectivePort(allowedOrigin)
        }

        private func effectivePort(_ url: URL) -> Int? {
            url.port ?? (url.scheme == "https" ? 443 : nil)
        }

        private func deliver(_ response: [String: Any], to frameInfo: WKFrameInfo) {
            webView?.callAsyncJavaScript(
                "window.navverseNativeCaptureResult(message)",
                arguments: ["message": response],
                in: frameInfo,
                in: .page
            ) { result in
                if case .failure(let error) = result {
                    print("Failed to deliver ARKit capture: \(error.localizedDescription)")
                }
            }
        }
    }
}
