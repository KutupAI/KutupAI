#include "ChatController.h"
#include "../DTOs/ApiResponseDTO.h"

#include <trantor/utils/Logger.h>

using Application::DTOs::ApiResponseDTO;
using Application::DTOs::ErrorStage;
using Application::DTOs::MalformedRequestError;
using Application::DTOs::SendMessageRequestDTO;

namespace Application::Controllers {

ChatController::ChatController(std::shared_ptr<Application::Services::ChatService> chatService)
    : chatService_(std::move(chatService)) {}

void ChatController::sendMessage(const drogon::HttpRequestPtr& req,
                                  std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    // --- 1) Parse body (structural only) ---
    auto jsonBody = req->getJsonObject();
    if (!jsonBody) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            ApiResponseDTO::emptyDocumentEnvelope(false));
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    SendMessageRequestDTO request;
    try {
        request = SendMessageRequestDTO::fromJson(*jsonBody);
    } catch (const MalformedRequestError& e) {
        LOG_WARN << "ChatController: malformed request: " << e.what();
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            ApiResponseDTO::emptyDocumentEnvelope(false));
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    // --- 2) Delegate to ChatService (validation + Orchestration call live there) ---
    // NOTE: ChatService::sendMessage is currently synchronous/blocking
    // (OrchestrationClient blocks internally on the Orchestration round
    // trip). This is acceptable for now given llama.cpp/OCR turnaround
    // times, but should move to Drogon's coroutine handlers
    // (drogon::Task<...>) if concurrent chat volume grows.
    try {
        const auto result = chatService_->sendMessage(request);

        if (!result.success) {
            LOG_ERROR << "ChatController: " << result.errorStage << " " << result.errorMessage;
            auto resp = drogon::HttpResponse::newHttpJsonResponse(result.data.toJson());
            // Validation/malformed-input failures are client errors; anything
            // past that boundary (temp storage, Orchestration) is a server-side
            // condition from the client's point of view.
            resp->setStatusCode(result.errorStage == ErrorStage::ValidationFailed
                                     ? drogon::k400BadRequest
                                     : drogon::k502BadGateway);
            callback(resp);
            return;
        }

        auto resp = drogon::HttpResponse::newHttpJsonResponse(result.data.toJson());
        resp->setStatusCode(drogon::k200OK);
        callback(resp);
    } catch (const std::exception& e) {
        // Nothing below this should be able to throw past ChatService, but
        // this backstop guarantees Presentation always gets a well-formed
        // ApiResponse instead of a raw 500/connection reset, per the
        // "every hop reports success/failure with a traceable stage" requirement.
        LOG_ERROR << "ChatController::sendMessage unexpected exception: " << e.what();
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            ApiResponseDTO::emptyDocumentEnvelope(false));
        resp->setStatusCode(drogon::k500InternalServerError);
        callback(resp);
    }
}

} // namespace Application::Controllers
