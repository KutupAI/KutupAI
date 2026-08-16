#include "DocumentValidator.h"

#include <algorithm>
#include <cctype>

namespace Application::Validators {

// Mirrors Presentation/config/chatConfig.ts CHAT_CONFIG.upload.allowedMimeTypes.
// Kept in sync manually; if the two ever drift, Presentation still enforces
// its own allow-list client-side, so this is a server-side backstop, not the
// single source of truth for what the UI offers.
const std::vector<std::string> DocumentValidator::kAllowedMimeTypes = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "image/jpeg",
    "image/jpg",
    "image/pjpeg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/x-ms-bmp",
    "image/tiff",
    "image/webp",
    "image/x-icon",
    "image/vnd.microsoft.icon",
    "text/plain",
};

bool DocumentValidator::isAllowedType(const Application::DTOs::ChatFileDTO& file) {
    if (!file.fileType.empty()) {
        std::string mime = file.fileType;
        std::transform(mime.begin(), mime.end(), mime.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (mime.rfind("image/", 0) == 0) return true;
        const bool typeAllowed = std::find(kAllowedMimeTypes.begin(), kAllowedMimeTypes.end(),
                                            mime) != kAllowedMimeTypes.end();
        if (typeAllowed) return true;
    }
    // Some browsers send an empty MIME type; fall back to the filename extension.
    std::string name = file.fileName;
    auto slash = name.find_last_of("\\/");
    if (slash != std::string::npos) name = name.substr(slash + 1);
    auto dot = name.find_last_of('.');
    if (dot == std::string::npos || dot + 1 >= name.size()) return false;
    std::string ext = name.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    static const std::vector<std::string> kAllowedExtensions = {
        "pdf",  "docx", "pptx", "txt",  "jpg",  "jpeg", "jpe", "jfif", "png",
        "gif",  "bmp",  "dib",  "tif",  "tiff", "webp", "ico", "ppm",  "pgm",
        "pbm",  "heic", "heif", "avif", "apng",
    };
    return std::find(kAllowedExtensions.begin(), kAllowedExtensions.end(), ext) !=
           kAllowedExtensions.end();
}

DocumentValidator::DocumentValidator(Application::Configuration::AppConfig config)
    : config_(std::move(config)) {}

ValidationResult DocumentValidator::validateQuestion(const std::string& question) const {
    const bool isBlank = std::all_of(question.begin(), question.end(), [](unsigned char c) {
        return std::isspace(c);
    });
    if (question.empty() || isBlank) {
        return {false, "Question must not be empty."};
    }
    return {true, ""};
}

ValidationResult DocumentValidator::validateFile(const Application::DTOs::ChatFileDTO& file) const {
    const bool typeAllowed = isAllowedType(file);
    if (!typeAllowed) {
        return {false, "Unsupported file type: " + file.fileType};
    }

    // Estimate decoded size from base64 length (avoids decoding twice: once
    // here, once in ChatService). Padding chars ('=') don't encode data.
    const std::string& b64 = file.fileBase64;
    if (b64.empty()) {
        return {false, "FileBase64 must not be empty."};
    }
    size_t padding = 0;
    if (b64.size() >= 1 && b64[b64.size() - 1] == '=') padding++;
    if (b64.size() >= 2 && b64[b64.size() - 2] == '=') padding++;
    const size_t approxDecodedBytes = (b64.size() / 4) * 3 - padding;

    if (approxDecodedBytes > config_.maxUploadSizeBytes) {
        return {false, "File exceeds maximum allowed size of " +
                            std::to_string(config_.maxUploadSizeBytes / (1024 * 1024)) + "MB."};
    }

    return {true, ""};
}

} // namespace Application::Validators
