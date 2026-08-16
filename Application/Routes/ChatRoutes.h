#pragma once

#include <memory>

namespace Application::Controllers {
class ChatController;
}

namespace Application::Routes {

void registerChatRoutes(const std::shared_ptr<Application::Controllers::ChatController>& controller);

} // namespace Application::Routes
