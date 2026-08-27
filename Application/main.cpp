// Application entry point (Drogon server bootstrap).
//
// Wiring order matters: Configuration -> Validators -> Services -> Clients
// -> Controllers -> Routes -> run(). Only the Chat channel is fully wired
// here; Document/Auth/User/Status controllers are pre-existing stubs left
// for a follow-up change.

#include "Configuration/AppConfig.h"
#include "Controllers/ChatController.h"
#include "Routes/ApiRoutes.h"
#include "Services/ChatService.h"
#include "Services/OrchestrationClient.h"
#include "Validators/DocumentValidator.h"

#include <drogon/drogon.h>

#include <memory>

int main() {
    using Application::Configuration::AppConfig;
    using Application::Controllers::ChatController;
    using Application::Services::ChatService;
    using Application::Services::OrchestrationClient;
    using Application::Validators::DocumentValidator;

    const AppConfig config = AppConfig::loadFromEnv();

    DocumentValidator validator(config);
    OrchestrationClient orchestrationClient(config.orchestrationBaseUrl,
                                             config.orchestrationTimeoutSeconds);
    auto chatService = std::make_shared<ChatService>(config, validator, orchestrationClient);
    auto chatController = std::make_shared<ChatController>(chatService);

    Application::Routes::registerAllRoutes(chatController);

    LOG_INFO << "SmartGovernmentAI Application layer starting on port " << config.serverPort
             << " (Orchestration at " << config.orchestrationBaseUrl
             << ", temp uploads at " << config.tempUploadRootDir << ")";

    // Drogon default client_max_body_size is 1MB. Chat uploads send Base64 in
    // JSON (~4/3 of file size + envelope), so allow headroom above APP_MAX_UPLOAD_SIZE_MB.
    const size_t maxHttpBody =
        config.maxUploadSizeBytes + (config.maxUploadSizeBytes / 3) + (2 * 1024 * 1024);

    drogon::app()
        .addListener("0.0.0.0", config.serverPort)
        .setThreadNum(config.serverThreads)
        .setClientMaxBodySize(maxHttpBody)
        .setClientMaxMemoryBodySize(maxHttpBody)
        .run();

    return 0;
}
