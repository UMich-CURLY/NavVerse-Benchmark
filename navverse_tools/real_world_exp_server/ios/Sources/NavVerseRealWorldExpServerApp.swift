import SwiftUI

@main
@MainActor
struct NavVerseRealWorldExpServerApp: App {
    @StateObject private var arSession = ARSessionController()

    var body: some Scene {
        WindowGroup {
            WebViewHost(arSession: arSession)
                .ignoresSafeArea()
                .onAppear { arSession.start() }
        }
    }
}
