/// SystemAudioHelper — Captures macOS system audio via ScreenCaptureKit
/// and writes raw PCM (int16, mono) to stdout for consumption by a parent process.
///
/// Usage:
///   SystemAudioHelper [--sample-rate 16000] [--exclude-pid 1234]
///
/// Output format: raw PCM int16 little-endian mono at the specified sample rate.
/// Logs go to stderr. Audio data goes to stdout.
///
/// Uses ScreenCaptureKit (macOS 13+) to capture all system audio output.
/// ScreenCaptureKit natively supports sample rate and channel count configuration,
/// so no manual AudioConverter is needed.

import Foundation
import ScreenCaptureKit
import CoreMedia

// MARK: - Configuration

struct Config {
    var sampleRate: Float64 = 16000
    var excludePIDs: [pid_t] = []
}

func parseArgs() -> Config {
    var config = Config()
    let args = CommandLine.arguments
    var i = 1
    while i < args.count {
        switch args[i] {
        case "--sample-rate":
            i += 1
            if i < args.count, let sr = Float64(args[i]) {
                config.sampleRate = sr
            }
        case "--exclude-pid":
            i += 1
            if i < args.count, let pid = pid_t(args[i]) {
                config.excludePIDs.append(pid)
            }
        default:
            break
        }
        i += 1
    }
    return config
}

func log(_ message: String) {
    FileHandle.standardError.write(Data("[SystemAudioHelper] \(message)\n".utf8))
}

// MARK: - System Audio Capture via ScreenCaptureKit

@available(macOS 13, *)
class SystemAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    let config: Config
    private var stream: SCStream?

    /// Track initial callbacks to detect silent-permission-denial (all-zero audio).
    /// ScreenCaptureKit may deliver zeros when Screen Recording permission is missing.
    private var callbackCount = 0
    private var nonZeroDetected = false
    private let permCheckCallbacks = 100  // ~2s worth of callbacks

    init(config: Config) {
        self.config = config
    }

    func start() -> Bool {
        let semaphore = DispatchSemaphore(value: 0)
        var success = false

        SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: false) { [self] content, error in
            guard let content = content else {
                log("Failed to get shareable content: \(error?.localizedDescription ?? "unknown")")
                semaphore.signal()
                return
            }

            guard let display = content.displays.first else {
                log("No displays found")
                semaphore.signal()
                return
            }

            log("Found \(content.displays.count) display(s), \(content.applications.count) app(s)")

            // Build exclusion list for specified PIDs
            let excludeApps = content.applications.filter { app in
                self.config.excludePIDs.contains(app.processID)
            }
            if !excludeApps.isEmpty {
                log("Excluding \(excludeApps.count) app(s) by PID")
            }

            let filter = SCContentFilter(
                display: display,
                excludingApplications: excludeApps,
                exceptingWindows: []
            )

            let streamConfig = SCStreamConfiguration()
            streamConfig.capturesAudio = true
            streamConfig.excludesCurrentProcessAudio = true
            streamConfig.channelCount = 1
            streamConfig.sampleRate = Int(self.config.sampleRate)

            // Minimize video overhead (we only want audio)
            streamConfig.width = 2
            streamConfig.height = 2
            streamConfig.minimumFrameInterval = CMTime(value: 1, timescale: 1)

            let scStream = SCStream(filter: filter, configuration: streamConfig, delegate: self)
            self.stream = scStream

            do {
                try scStream.addStreamOutput(self, type: .audio, sampleHandlerQueue: DispatchQueue(label: "audio-capture"))
                scStream.startCapture { startError in
                    if let startError = startError {
                        log("Failed to start capture: \(startError)")
                    } else {
                        log("Capturing system audio at \(Int(self.config.sampleRate)) Hz, mono, int16")
                        success = true
                    }
                    semaphore.signal()
                }
            } catch {
                log("Failed to configure stream: \(error)")
                semaphore.signal()
            }
        }

        semaphore.wait()
        return success
    }

    func stop() {
        stream?.stopCapture { _ in }
        stream = nil
    }

    // MARK: - SCStreamOutput

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio else { return }

        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }

        var length = 0
        var dataPointer: UnsafeMutablePointer<Int8>?
        let status = CMBlockBufferGetDataPointer(
            blockBuffer, atOffset: 0, lengthAtOffsetOut: nil,
            totalLengthOut: &length, dataPointerOut: &dataPointer
        )
        guard status == noErr, let ptr = dataPointer, length > 0 else { return }

        // ScreenCaptureKit delivers float32 PCM at the requested sample rate
        let sampleCount = length / 4

        // Detect silent-permission-denial
        if !nonZeroDetected {
            callbackCount += 1
            ptr.withMemoryRebound(to: Float32.self, capacity: sampleCount) { fp in
                for i in 0..<sampleCount {
                    if fp[i] != 0 {
                        nonZeroDetected = true
                        return
                    }
                }
            }
            if callbackCount == permCheckCallbacks && !nonZeroDetected {
                log("WARNING: \(permCheckCallbacks) callbacks received but all audio data is zero.")
                log("WARNING: This usually means Screen Recording permission is not granted.")
                log("WARNING: Grant permission in System Settings → Privacy & Security → Screen Recording, then restart the app.")
            }
        }

        // Convert float32 → int16 and write to stdout
        ptr.withMemoryRebound(to: Float32.self, capacity: sampleCount) { floatPtr in
            var int16Samples = [Int16](repeating: 0, count: sampleCount)
            for i in 0..<sampleCount {
                let clamped = max(-1.0, min(1.0, floatPtr[i]))
                int16Samples[i] = Int16(clamped * 32767)
            }
            int16Samples.withUnsafeBufferPointer { bufPtr in
                FileHandle.standardOutput.write(Data(buffer: bufPtr))
            }
        }
    }

    // MARK: - SCStreamDelegate

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("Stream stopped with error: \(error.localizedDescription)")
    }
}

// MARK: - Main

let config = parseArgs()
log("Starting with sample rate: \(Int(config.sampleRate)) Hz")

if #available(macOS 13, *) {
    let capture = SystemAudioCapture(config: config)

    signal(SIGTERM) { _ in
        log("Received SIGTERM, exiting...")
        exit(0)
    }
    signal(SIGINT) { _ in
        log("Received SIGINT, exiting...")
        exit(0)
    }

    guard capture.start() else {
        log("Failed to start capture")
        exit(1)
    }

    withExtendedLifetime(capture) {
        CFRunLoopRun()
    }
} else {
    log("macOS 13 or later is required for ScreenCaptureKit")
    exit(1)
}
