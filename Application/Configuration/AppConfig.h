// AppConfig
// General runtime configuration (server port, timeouts, internal service addresses).
//
// All values are read from environment variables with safe local-dev
// defaults, matching Config/global_config.yaml's stated purpose ("منافذ،
// عناوين خدمات داخلية مشتركة"). Never hardcode secrets here.

#pragma once

#include <cstdint>
#include <string>

namespace Application::Configuration {

struct AppConfig {
    // --- Server ---
    uint16_t serverPort = 8080;
    int serverThreads = 4;

    // --- Orchestration (Application -> Orchestration bridge) ---
    // Base URL of Orchestration/main.py's internal endpoint.
    std::string orchestrationBaseUrl = "http://127.0.0.1:8000";
    int orchestrationTimeoutSeconds = 300; // Unstructured + Qwen-VL fallback can exceed 2 minutes

    // --- Chat channel temp file handling ---
    // Root directory for per-request temp files (see ChatService). This is
    // intentionally NOT Storage/files/uploads: the chat channel is stateless
    // and never writes a Document row before/around processing.
    // Shared temp root readable by both Application (C++) and Orchestration (Python).
    // Override with APP_TEMP_UPLOAD_ROOT_DIR on deployment.
    std::string tempUploadRootDir = "Storage/files/temp_processing";

    // Mirrors Presentation/config/chatConfig.ts CHAT_CONFIG.upload so both
    // layers agree on limits (Application still re-validates independently;
    // never trust client-side limits alone).
    size_t maxUploadSizeBytes = 10 * 1024 * 1024; // 10MB

    // Loads config from environment variables, falling back to the defaults
    // above when a variable is unset. Safe to call once at startup.
    static AppConfig loadFromEnv();
};

} // namespace Application::Configuration
