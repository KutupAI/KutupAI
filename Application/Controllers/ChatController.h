// ChatController
// Receives chat requests (question + optional file) from Presentation and
// delegates to ChatService. Route: POST api/Chat/SendMessage — matches
// Presentation/config/chatConfig.ts CHAT_CONFIG.endpoints.sendMessage
// exactly; Presentation is frozen, this controller adapts to it.
//
// Hard rule (Application/README.md): no AI logic here — only request
// intake, validation delegation, and returning a ready ApiResponseDTO.

#pragma once

#include "../DTOs/ChatDTO.h"
#include "../Services/ChatService.h"

#include <drogon/HttpRequest.h>
#include <drogon/HttpResponse.h>

#include <functional>
#include <memory>

namespace Application::Controllers {

// Plain, dependency-injected class (ChatService is passed in, not
// default-constructed) — deliberately NOT a drogon::HttpController with
// auto-registration macros, so route wiring stays explicit and visible in
// Routes/ChatRoutes.cpp, consistent with how DocumentRoutes.cpp/AuthRoutes.cpp
// are documented to bind endpoints to controller actions.
class ChatController {
public:
    explicit ChatController(std::shared_ptr<Application::Services::ChatService> chatService);

    void sendMessage(const drogon::HttpRequestPtr& req,
                      std::function<void(const drogon::HttpResponsePtr&)>&& callback);

private:
    std::shared_ptr<Application::Services::ChatService> chatService_;
};

} // namespace Application::Controllers
