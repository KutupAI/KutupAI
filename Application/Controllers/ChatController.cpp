#include "ChatController.h"
#include "../DTOs/ApiResponseDTO.h"

#include <trantor/utils/Logger.h>

using Application::DTOs::ApiResponseDTO;
using Application::DTOs::ErrorStage;
using Application::DTOs::MalformedRequestError;
using Application::DTOs::SendMessageRequestDTO;

namespace Application::Controllers {

namespace {

drogon::HttpResponsePtr jsonResponse(const Json::Value& body, drogon::HttpStatusCode status) {
    auto resp = drogon::HttpResponse::newHttpJsonResponse(body);
    resp->setStatusCode(status);
    resp->addHeader("Access-Control-Allow-Origin", "*");
    resp->addHeader("Access-Control-Allow-Headers", "Content-Type");
    resp->addHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    return resp;
}

drogon::HttpResponsePtr jsonBodyResponse(const std::string& body, drogon::HttpStatusCode status) {
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(status);
    resp->setBody(body);
    resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    resp->addHeader("Access-Control-Allow-Origin", "*");
    resp->addHeader("Access-Control-Allow-Headers", "Content-Type");
    resp->addHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    return resp;
}

} // namespace

ChatController::ChatController(std::shared_ptr<Application::Services::ChatService> chatService)
    : chatService_(std::move(chatService)) {}

void ChatController::sendMessage(const drogon::HttpRequestPtr& req,
                                  std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    auto jsonBody = req->getJsonObject();
    if (!jsonBody) {
        callback(jsonResponse(
            ApiResponseDTO::failure("Request body must be JSON", ErrorStage::MalformedRequest),
            drogon::k400BadRequest));
        return;
    }

    SendMessageRequestDTO request;
    try {
        request = SendMessageRequestDTO::fromJson(*jsonBody);
    } catch (const MalformedRequestError& e) {
        LOG_WARN << "ChatController: malformed request: " << e.what();
        callback(jsonResponse(ApiResponseDTO::failure(e.what(), ErrorStage::MalformedRequest),
                              drogon::k400BadRequest));
        return;
    }

    try {
        const auto result = chatService_->sendMessage(request);

        if (!result.success) {
            LOG_ERROR << "ChatController: " << result.errorStage << " " << result.errorMessage;
            const auto status = result.errorStage == ErrorStage::ValidationFailed
                                    ? drogon::k400BadRequest
                                    : drogon::k502BadGateway;
            callback(jsonResponse(
                ApiResponseDTO::failure(result.errorMessage, result.errorStage), status));
            return;
        }

        // Ordered wire format: Success first, then State in pipeline stage order.
        callback(jsonBodyResponse(result.data.toOrderedJsonString(), drogon::k200OK));
    } catch (const std::exception& e) {
        LOG_ERROR << "ChatController::sendMessage unexpected exception: " << e.what();
        callback(jsonResponse(
            ApiResponseDTO::failure(e.what(), ErrorStage::InternalError),
            drogon::k500InternalServerError));
    }
}

} // namespace Application::Controllers
