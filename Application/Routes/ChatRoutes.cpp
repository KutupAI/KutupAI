// ChatRoutes
// Binds the Chat channel endpoint to ChatController::sendMessage.
// Route: POST api/Chat/SendMessage (see DTOs/ChatDTO.h for why this exact
// path/casing is non-negotiable — it comes from the frozen Presentation layer).

#include "ChatRoutes.h"
#include "../Controllers/ChatController.h"

#include <drogon/HttpAppFramework.h>

namespace Application::Routes {

void registerChatRoutes(const std::shared_ptr<Application::Controllers::ChatController>& controller) {
    // CORS preflight for browser clients that call Application directly.
    drogon::app().registerHandler(
        "/api/Chat/SendMessage",
        [](const drogon::HttpRequestPtr&,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k200OK);
            resp->addHeader("Access-Control-Allow-Origin", "*");
            resp->addHeader("Access-Control-Allow-Headers", "Content-Type");
            resp->addHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
            callback(resp);
        },
        {drogon::Options});

    drogon::app().registerHandler(
        "/api/Chat/SendMessage",
        [controller](const drogon::HttpRequestPtr& req,
                     std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            controller->sendMessage(req, std::move(callback));
        },
        {drogon::Post});
}

} // namespace Application::Routes
