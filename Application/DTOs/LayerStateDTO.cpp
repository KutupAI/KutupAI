#include "LayerStateDTO.h"

#include <algorithm>
#include <cctype>

namespace Application::DTOs {

namespace {

Json::Value emptyObject() {
    return Json::Value(Json::objectValue);
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string extensionOf(const std::string& fileName) {
    const auto slash = fileName.find_last_of("\\/");
    const std::string base =
        slash == std::string::npos ? fileName : fileName.substr(slash + 1);
    const auto dot = base.find_last_of('.');
    if (dot == std::string::npos || dot + 1 >= base.size()) {
        return "";
    }
    return toLower(base.substr(dot + 1));
}

std::string mimeToExtension(const std::string& mime) {
    const std::string lower = toLower(mime);
    if (lower == "application/pdf") return "pdf";
    if (lower == "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        return "docx";
    if (lower == "application/vnd.openxmlformats-officedocument.presentationml.presentation")
        return "pptx";
    if (lower == "text/plain") return "txt";
    if (lower.rfind("image/", 0) == 0) {
        const auto slash = lower.find('/');
        if (slash != std::string::npos && slash + 1 < lower.size()) {
            std::string subtype = lower.substr(slash + 1);
            if (subtype == "jpeg" || subtype == "pjpeg" || subtype == "jpg") return "jpg";
            if (subtype == "x-ms-bmp") return "bmp";
            if (subtype == "x-icon" || subtype == "vnd.microsoft.icon") return "ico";
            return subtype;
        }
    }
    return "";
}

} // namespace

LayerStateDTO LayerStateDTO::initial(LayerDocumentDTO document, std::string question, bool success) {
    LayerStateDTO state;
    state.success = success;
    state.question = std::move(question);
    state.document = std::move(document);
    return state;
}

std::string LayerStateDTO::normalizeFileType(const std::string& fileName,
                                              const std::string& mimeOrType) {
    const std::string fromName = extensionOf(fileName);
    if (!fromName.empty()) {
        return fromName;
    }
    const std::string fromMime = mimeToExtension(mimeOrType);
    if (!fromMime.empty()) {
        return fromMime;
    }
    // Already an extension-like token (e.g. "pdf") — keep lowercase.
    if (!mimeOrType.empty() && mimeOrType.find('/') == std::string::npos) {
        return toLower(mimeOrType);
    }
    return "";
}

Json::Value LayerStateDTO::requestJson() const {
    Json::Value request(Json::objectValue);
    request["success"] = success;
    request["question"] = question;

    Json::Value document(Json::objectValue);
    document["document_id"] = this->document.documentId;
    document["file_name"] = this->document.fileName;
    document["file_type"] = this->document.fileType;
    if (this->document.documentPath.has_value() && !this->document.documentPath->empty()) {
        document["document_path"] = *this->document.documentPath;
    }
    request["document"] = document;
    return request;
}

Json::Value LayerStateDTO::toJson() const {
    Json::Value root(Json::objectValue);
    root["request"] = requestJson();
    root["ocr"] = emptyObject();
    root["classification"] = emptyObject();
    root["extraction"] = emptyObject();
    root["validation"] = emptyObject();
    root["rag"] = emptyObject();
    root["summary"] = emptyObject();
    root["routing"] = emptyObject();
    root["writing"] = emptyObject();
    return root;
}

} // namespace Application::DTOs
