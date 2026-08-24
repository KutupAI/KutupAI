// ChatService
// Bridges the Chat channel (Presentation -> POST api/Chat/SendMessage) to
// Orchestration.
//
// Deliberately stateless: this channel does NOT write to Storage (no
// Document row, no status="Received") because the source file already lives
// on the user's device and is only needed for the lifetime of this single
// request/response cycle — see the earlier decision to keep the chat path
// decoupled from the Storage-backed upload channel. architecture.md's
// Single-Writer Rule ("لا Agent يتصل بـ Storage مباشرة") is untouched: this
// path simply never calls a Repository at all, from either side.
//
// For the persistent, Storage-backed upload flow, see DocumentProcessingService
// (unchanged by this file).

#pragma once

#include "../Configuration/AppConfig.h"
#include "../DTOs/ChatDTO.h"
#include "../Validators/DocumentValidator.h"
#include "OrchestrationClient.h"

#include <stdexcept>
#include <string>

namespace Application::Services {

// Every failure this service can produce is reported through this struct
// (never thrown past this class) so ChatController can return Presentation's
// ApiResponse { Success, AdditionalData: { ChatId, State } } and log the
// failed boundary.
struct ChatServiceResult {
    bool success = false;
    Application::DTOs::SendMessageResponseData data;
    std::string errorStage;                            // ApiResponseDTO::ErrorStage value
    std::string errorMessage;
};

class ChatService {
public:
    ChatService(Application::Configuration::AppConfig config,
                Application::Validators::DocumentValidator validator,
                OrchestrationClient orchestrationClient);

    // Full flow: validate -> (optional) persist temp file -> call
    // Orchestration -> cleanup temp file -> map to ChatServiceResult.
    // Temp file cleanup happens regardless of the Orchestration outcome.
    ChatServiceResult sendMessage(const Application::DTOs::SendMessageRequestDTO& request) const;

private:
    Application::Configuration::AppConfig config_;
    Application::Validators::DocumentValidator validator_;
    OrchestrationClient orchestrationClient_;

    // Decodes File.FileBase64 and writes it under
    // {resolvedTempRoot()}/{requestId}/{FileName}. Returns the absolute path.
    // Throws TempStorageError on decode/disk failure.
    std::string persistTempFile(const Application::DTOs::ChatFileDTO& file,
                                 const std::string& requestId) const;

    // Best-effort recursive delete of {resolvedTempRoot()}/{requestId}/.
    // Never throws — cleanup failure is logged by the caller, not fatal to
    // the response already produced.
    void cleanupTempDir(const std::string& requestId) const;

    std::string resolvedTempRoot() const;
};

struct TempStorageError : public std::runtime_error {
    explicit TempStorageError(const std::string& msg) : std::runtime_error(msg) {}
};

} // namespace Application::Services
