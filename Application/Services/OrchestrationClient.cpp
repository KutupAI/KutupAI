#include "OrchestrationClient.h"
#include "../DTOs/ApiResponseDTO.h"

#include <drogon/HttpClient.h>
#include <drogon/HttpTypes.h>
#include <json/json.h>

#include <chrono>
#include <future>

using Application::DTOs::ApiResponseDTO;
using Application::DTOs::ErrorStage;

namespace Application::Services {

OrchestrationClient::OrchestrationClient(std::string baseUrl, int timeoutSeconds)
    : baseUrl_(std::move(baseUrl)), timeoutSeconds_(timeoutSeconds) {}

OrchestrationResult OrchestrationClient::process(const OrchestrationRequest& request) const {
    OrchestrationResult result;
    result.contract = ApiResponseDTO::emptyDocumentEnvelope(false);

    // Primary payload: unified pipeline envelope from LayerStateDTO.
    Json::Value payload = request.layerState.toJson();

    // Flat wire fields kept for Orchestration/main.py ProcessRequest and
    // process_service path checks (document_path must be absolute on disk).
    payload["document_id"] = request.requestId;
    payload["question"] = request.question;
    payload["text"] = request.question;
    payload["accompanying_text"] = request.question;
    payload["document_path"] = request.documentPath.has_value()
                                    ? Json::Value(*request.documentPath)
                                    : Json::Value(Json::nullValue);

    auto client = drogon::HttpClient::newHttpClient(baseUrl_);
    auto req = drogon::HttpRequest::newHttpJsonRequest(payload);
    req->setMethod(drogon::Post);
    req->setPath("/process");

    auto responsePromise = std::make_shared<std::promise<drogon::HttpResponsePtr>>();
    auto responseFuture = responsePromise->get_future();
    auto reqErr = std::make_shared<drogon::ReqResult>(drogon::ReqResult::Ok);

    client->sendRequest(
        req,
        [responsePromise, reqErr](drogon::ReqResult r, const drogon::HttpResponsePtr& resp) {
            *reqErr = r;
            responsePromise->set_value(resp);
        },
        static_cast<double>(timeoutSeconds_));

    if (responseFuture.wait_for(std::chrono::seconds(timeoutSeconds_ + 5)) ==
        std::future_status::timeout) {
        result.reachable = false;
        result.errorStage = ErrorStage::OrchestrationUnreachable;
        result.errorMessage = "Orchestration did not respond within " +
                               std::to_string(timeoutSeconds_) + "s (request_id=" +
                               request.requestId + ")";
        return result;
    }

    if (*reqErr != drogon::ReqResult::Ok) {
        result.reachable = false;
        result.errorStage = ErrorStage::OrchestrationUnreachable;
        result.errorMessage = "Failed to reach Orchestration at " + baseUrl_ +
                               "/process (request_id=" + request.requestId + ")";
        return result;
    }

    auto resp = responseFuture.get();
    if (!resp || resp->getStatusCode() != drogon::k200OK) {
        result.reachable = false;
        result.errorStage = ErrorStage::OrchestrationUnreachable;
        result.errorMessage = "Orchestration returned status " +
                               std::to_string(resp ? static_cast<int>(resp->getStatusCode()) : -1) +
                               " (request_id=" + request.requestId + ")";
        return result;
    }

    auto body = resp->getJsonObject();
    if (!body) {
        result.reachable = false;
        result.errorStage = ErrorStage::OrchestrationUnreachable;
        result.errorMessage = "Orchestration returned a non-JSON body (request_id=" +
                               request.requestId + ")";
        return result;
    }

    if (!(*body).isMember("Success") || !(*body).isMember("Data") || !(*body)["Data"].isArray()) {
        result.reachable = false;
        result.errorStage = ErrorStage::OrchestrationError;
        result.errorMessage = "Orchestration returned a body that is not the unified contract "
                               "(request_id=" +
                               request.requestId + ")";
        return result;
    }

    result.reachable = true;
    result.contract = *body;
    return result;
}

} // namespace Application::Services
