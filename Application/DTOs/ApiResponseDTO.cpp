#include "ApiResponseDTO.h"

namespace Application::DTOs {

Json::Value ApiResponseDTO::success(const Json::Value& additionalData,
                                     const std::string& message) {
    Json::Value root(Json::objectValue);
    root["Success"] = true;
    if (!message.empty()) {
        root["Message"] = message;
    }
    root["AdditionalData"] = additionalData;
    return root;
}

Json::Value ApiResponseDTO::failure(const std::string& message,
                                     const std::string& code) {
    Json::Value root(Json::objectValue);
    root["Success"] = false;
    root["Message"] = message;
    root["Code"] = code;
    return root;
}

Json::Value ApiResponseDTO::documentEnvelope(bool success, const Json::Value& data) {
    Json::Value root(Json::objectValue);
    root["Success"] = success;
    if (data.isArray()) {
        root["Data"] = data;
    } else {
        root["Data"] = Json::Value(Json::arrayValue);
        if (!data.isNull() && data.isObject()) {
            root["Data"].append(data);
        }
    }
    return root;
}

Json::Value ApiResponseDTO::emptyDocumentEnvelope(bool success) {
    return documentEnvelope(success, Json::Value(Json::arrayValue));
}

} // namespace Application::DTOs
