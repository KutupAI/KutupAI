// ChatRoutes
// Binds the Chat channel endpoint to ChatController::sendMessage.
// Route: POST api/Chat/SendMessage (see DTOs/ChatDTO.h for why this exact
// path/casing is non-negotiable — it comes from the frozen Presentation layer).

#include "ChatRoutes.h"
#include "../Controllers/ChatController.h"

#include <drogon/HttpAppFramework.h>

namespace Application::Routes {

void registerChatRoutes(const std::shared_ptr<Application::Controllers::ChatController>& controller) {
    drogon::app().registerHandler(
        "/api/Chat/SendMessage",
        [controller](const drogon::HttpRequestPtr& req,
                     std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            controller->sendMessage(req, std::move(callback));
        },
        {drogon::Post});
}

} // namespace Application::Routes
