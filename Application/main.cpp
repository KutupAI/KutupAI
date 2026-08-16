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

    drogon::app()
        .addListener("0.0.0.0", config.serverPort)
        .setThreadNum(config.serverThreads)
        .run();

    return 0;
}
