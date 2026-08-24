// ChatDTO
// Request shape for POST api/Chat/SendMessage (Presentation inbound).
// Response shape matches Presentation ApiResponse:
//   { Success, Message?, Code?, AdditionalData: { ChatId, State } }
// where State is the 9-key pipeline envelope
//   { request, ocr, classification, extraction, validation, rag, summary, routing, writing }

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

// Presentation: SendMessageResponseData inside AdditionalData.
struct SendMessageResponseData {
    std::string chatId;
    Json::Value state;  // 9-key pipeline envelope for the UI

    // Builds { Success, AdditionalData: { ChatId, State } }.
    // Note: JsonCpp sorts object keys alphabetically, so prefer
    // toOrderedJsonString() for the Presentation wire format.
    Json::Value toJson() const;

    // Same payload as toJson(), but key order is stable for the UI contract:
    //   Success → AdditionalData → ChatId → State
    //   State: request → ocr → classification → extraction → validation
    //          → rag → summary → routing → writing
    std::string toOrderedJsonString() const;

    // Map Orchestration { Success, Data: [doc] } (+ request identity) into
    // the Presentation pipeline State.
    static Json::Value pipelineStateFromOrchestrationDoc(
        const Json::Value& doc,
        const std::string& chatId,
        const std::string& question,
        const std::optional<ChatFileDTO>& file);

    static Json::Value emptyPipelineState();
};

} // namespace Application::DTOs
