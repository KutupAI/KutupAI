#include "ChatDTO.h"

namespace Application::DTOs {

namespace {

std::string requireString(const Json::Value& body, const char* field) {
    if (!body.isMember(field) || !body[field].isString()) {
        throw MalformedRequestError(std::string("Missing or non-string field: ") + field);
    }
    return body[field].asString();
}

} // namespace

SendMessageRequestDTO SendMessageRequestDTO::fromJson(const Json::Value& body) {
    if (!body.isObject()) {
        throw MalformedRequestError("Request body must be a JSON object");
    }

    SendMessageRequestDTO dto;

    // ChatId: nullable string
    if (body.isMember("ChatId") && !body["ChatId"].isNull()) {
        if (!body["ChatId"].isString()) {
            throw MalformedRequestError("ChatId must be a string or null");
        }
        dto.chatId = body["ChatId"].asString();
    }

    dto.question = requireString(body, "Question");

    // File: nullable object { FileName, FileType, FileBase64 }
    if (body.isMember("File") && !body["File"].isNull()) {
        const Json::Value& fileJson = body["File"];
        if (!fileJson.isObject()) {
            throw MalformedRequestError("File must be an object or null");
        }
        ChatFileDTO file;
        file.fileName = requireString(fileJson, "FileName");
        file.fileType = requireString(fileJson, "FileType");
        file.fileBase64 = requireString(fileJson, "FileBase64");
        dto.file = std::move(file);
    }

    return dto;
}

Json::Value SendMessageResponseData::toJson() const {
    if (contract.isObject() && contract.isMember("Success") && contract.isMember("Data")) {
        return contract;
    }
    Json::Value root(Json::objectValue);
    root["Success"] = false;
    root["Data"] = Json::Value(Json::arrayValue);
    return root;
}

} // namespace Application::DTOs
