// ChatDTO
// Request shape for POST api/Chat/SendMessage (Presentation inbound).
// Response uses the unified layer contract { Success, Data }.

#pragma once

#include <json/json.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace Application::DTOs {

// Thrown by fromJson() for structural problems only (missing/wrong-typed
// fields). Business validation (file size/type, empty question) is a
// separate concern -> Validators/DocumentValidator.
struct MalformedRequestError : public std::runtime_error {
    explicit MalformedRequestError(const std::string& msg) : std::runtime_error(msg) {}
};

// Presentation: SendMessageRequestModel.File
struct ChatFileDTO {
    std::string fileName;    // JSON: FileName
    std::string fileType;    // JSON: FileType (MIME type)
    std::string fileBase64;  // JSON: FileBase64 — no "data:" prefix, per ChatMessage.ts comment
};

// Presentation: SendMessageRequestModel
struct SendMessageRequestDTO {
    std::optional<std::string> chatId;  // JSON: ChatId (nullable — first message in a new chat)
    std::string question;               // JSON: Question (optional when File is present)
    std::optional<ChatFileDTO> file;    // JSON: File (nullable — question-only turns)

    // Structural parse only. Throws MalformedRequestError on missing/invalid
    // JSON shape (e.g. Question not a string). Does not enforce business
    // rules (empty question, unsupported file type, etc).
    static SendMessageRequestDTO fromJson(const Json::Value& body);
};

// Unified processing contract forwarded to Presentation.
struct SendMessageResponseData {
    Json::Value contract;  // { "Success": bool, "Data": [ document, ... ] }

    Json::Value toJson() const;
};

} // namespace Application::DTOs
