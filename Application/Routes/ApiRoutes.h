// ApiRoutes
// Aggregates all Routes under api/[controller]/[action].
//
// Only the Chat channel is wired here (this change's scope). DocumentRoutes/
// AuthRoutes are pre-existing stubs and are intentionally left untouched —
// hook registerDocumentRoutes()/registerAuthRoutes() in here the same way
// once those controllers are implemented.

#pragma once

#include "ChatRoutes.h"
#include "../Controllers/ChatController.h"

#include <memory>

namespace Application::Routes {

inline void registerAllRoutes(const std::shared_ptr<Application::Controllers::ChatController>& chatController) {
    registerChatRoutes(chatController);
    // registerDocumentRoutes(...);  // TODO: wire once DocumentController is implemented
    // registerAuthRoutes(...);      // TODO: wire once AuthController is implemented
}

} // namespace Application::Routes
