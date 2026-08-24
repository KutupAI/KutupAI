#include "ChatDTO.h"
#include "ApiResponseDTO.h"

#include <sstream>

namespace Application::DTOs {

namespace {

std::string requireString(const Json::Value& body, const char* field) {
    if (!body.isMember(field) || !body[field].isString()) {
        throw MalformedRequestError(std::string("Missing or non-string field: ") + field);
    }
    return body[field].asString();
}

Json::Value emptyObject() {
    return Json::Value(Json::objectValue);
}

Json::Value copyObjectOrEmpty(const Json::Value& parent, const char* key) {
    if (parent.isObject() && parent.isMember(key) && parent[key].isObject()) {
        return parent[key];
    }
    return emptyObject();
}

std::string extensionOf(const std::string& fileName) {
    const auto dot = fileName.find_last_of('.');
    if (dot == std::string::npos || dot + 1 >= fileName.size()) {
        return "";
    }
    return fileName.substr(dot + 1);
}

// Canonical pipeline key order (matches Presentation / Orchestration contract).
constexpr const char* kPipelineKeys[] = {
    "request", "ocr", "classification", "extraction", "validation",
    "rag", "summary", "routing", "writing",
};

std::string jsonWrite(const Json::Value& value) {
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    builder["emitUTF8"] = true;
    return Json::writeString(builder, value);
}

std::string jsonQuote(const std::string& raw) {
    return jsonWrite(Json::Value(raw));
}

std::string writeOrderedState(const Json::Value& state) {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const char* key : kPipelineKeys) {
        if (!first) {
            oss << ",";
        }
        first = false;
        const Json::Value& section =
            (state.isObject() && state.isMember(key) && state[key].isObject())
                ? state[key]
                : emptyObject();
        oss << jsonQuote(key) << ":" << jsonWrite(section);
    }
    oss << "}";
    return oss.str();
}

} // namespace

SendMessageRequestDTO SendMessageRequestDTO::fromJson(const Json::Value& body) {
    if (!body.isObject()) {
        throw MalformedRequestError("Request body must be a JSON object");
    }

    SendMessageRequestDTO dto;

    if (body.isMember("ChatId") && !body["ChatId"].isNull()) {
        if (!body["ChatId"].isString()) {
            throw MalformedRequestError("ChatId must be a string or null");
        }
        dto.chatId = body["ChatId"].asString();
    }

    if (body.isMember("Question") && body["Question"].isNull()) {
        dto.question = "";
    } else {
        dto.question = requireString(body, "Question");
    }

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

Json::Value SendMessageResponseData::emptyPipelineState() {
    Json::Value root(Json::objectValue);
    for (const char* key : kPipelineKeys) {
        root[key] = emptyObject();
    }
    return root;
}

Json::Value SendMessageResponseData::pipelineStateFromOrchestrationDoc(
    const Json::Value& doc,
    const std::string& chatId,
    const std::string& question,
    const std::optional<ChatFileDTO>& file) {
    Json::Value state(Json::objectValue);

    const std::string documentId =
        doc.isMember("document_id") && doc["document_id"].isString()
            ? doc["document_id"].asString()
            : chatId;

    std::string fileName =
        doc.isMember("file_name") && doc["file_name"].isString() ? doc["file_name"].asString()
                                                                  : "";
    std::string fileType =
        doc.isMember("file_type") && doc["file_type"].isString() ? doc["file_type"].asString()
                                                                  : "";
    if (fileName.empty() && file.has_value()) {
        fileName = file->fileName;
    }
    if (fileType.empty() && !fileName.empty()) {
        fileType = extensionOf(fileName);
    }

    const std::string resolvedQuestion =
        doc.isMember("question") && doc["question"].isString() && !doc["question"].asString().empty()
            ? doc["question"].asString()
            : question;

    Json::Value request(Json::objectValue);
    request["success"] = true;
    request["question"] = resolvedQuestion;
    Json::Value document(Json::objectValue);
    document["document_id"] = documentId;
    document["file_name"] = fileName;
    document["file_type"] = fileType;
    request["document"] = document;
    state["request"] = request;

    state["ocr"] = copyObjectOrEmpty(doc, "ocr");
    state["classification"] = copyObjectOrEmpty(doc, "classification");
    state["extraction"] = copyObjectOrEmpty(doc, "extraction");
    state["validation"] = copyObjectOrEmpty(doc, "validation");
    state["rag"] = copyObjectOrEmpty(doc, "rag");

    if (doc.isMember("summary") && doc["summary"].isObject()) {
        state["summary"] = doc["summary"];
    } else if (doc.isMember("summary") && doc["summary"].isString() &&
               !doc["summary"].asString().empty()) {
        Json::Value summary(Json::objectValue);
        summary["success"] = true;
        summary["rag_summary_text"] = doc["summary"].asString();
        state["summary"] = summary;
    } else {
        state["summary"] = emptyObject();
    }

    state["routing"] = copyObjectOrEmpty(doc, "routing");
    state["writing"] = copyObjectOrEmpty(doc, "writing");

    if ((!state["writing"].isMember("answer") || !state["writing"]["answer"].isString() ||
         state["writing"]["answer"].asString().empty()) &&
        doc.isMember("answer") && doc["answer"].isString() && !doc["answer"].asString().empty()) {
        Json::Value writing = state["writing"].isObject() ? state["writing"] : emptyObject();
        writing["success"] = true;
        writing["answer"] = doc["answer"].asString();
        state["writing"] = writing;
    }

    return state;
}

Json::Value SendMessageResponseData::toJson() const {
    Json::Value additional(Json::objectValue);
    additional["ChatId"] = chatId;
    additional["State"] = state.isObject() ? state : emptyPipelineState();
    return ApiResponseDTO::success(additional);
}

std::string SendMessageResponseData::toOrderedJsonString() const {
    const Json::Value& pipeline = state.isObject() ? state : emptyPipelineState();
    std::ostringstream oss;
    // Explicit order: Success first (JsonCpp map would sort AdditionalData first).
    oss << "{\"Success\":true,"
        << "\"AdditionalData\":{"
        << "\"ChatId\":" << jsonQuote(chatId) << ","
        << "\"State\":" << writeOrderedState(pipeline)
        << "}}";
    return oss.str();
}

} // namespace Application::DTOs
