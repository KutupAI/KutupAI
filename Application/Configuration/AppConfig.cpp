#include "AppConfig.h"

#include <cstdlib>

namespace Application::Configuration {

namespace {

std::string envOr(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? std::string(v) : fallback;
}

int envIntOr(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    try {
        return std::stoi(v);
    } catch (...) {
        return fallback;
    }
}

} // namespace

AppConfig AppConfig::loadFromEnv() {
    AppConfig cfg;

    cfg.serverPort = static_cast<uint16_t>(envIntOr("APP_SERVER_PORT", cfg.serverPort));
    cfg.serverThreads = envIntOr("APP_SERVER_THREADS", cfg.serverThreads);

    cfg.orchestrationBaseUrl = envOr("ORCHESTRATION_BASE_URL", cfg.orchestrationBaseUrl);
    cfg.orchestrationTimeoutSeconds =
        envIntOr("ORCHESTRATION_TIMEOUT_SECONDS", cfg.orchestrationTimeoutSeconds);

    const std::string tempDir = envOr("APP_TEMP_UPLOAD_ROOT_DIR", cfg.tempUploadRootDir);
    // Docs sometimes show a truncated placeholder "C:\...\..." — ignore it.
    if (tempDir.find("...") == std::string::npos) {
        cfg.tempUploadRootDir = tempDir;
    }

    const int maxMb = envIntOr("APP_MAX_UPLOAD_SIZE_MB", -1);
    if (maxMb > 0) {
        cfg.maxUploadSizeBytes = static_cast<size_t>(maxMb) * 1024 * 1024;
    }

    return cfg;
}

} // namespace Application::Configuration
