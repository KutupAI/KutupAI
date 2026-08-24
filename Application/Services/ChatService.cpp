#include "ChatService.h"
#include "../DTOs/ApiResponseDTO.h"
#include "../DTOs/LayerStateDTO.h"

#include <drogon/utils/Utilities.h>
#include <json/json.h>
#include <trantor/utils/Logger.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>

using Application::DTOs::ApiResponseDTO;
using Application::DTOs::ErrorStage;
using Application::DTOs::LayerDocumentDTO;
using Application::DTOs::LayerStateDTO;

namespace Application::Services {

namespace {

// Lightweight request-id generator (UUIDv4-ish). Good enough to correlate
// logs across Application <-> Orchestration for a single chat turn; not used
// as a security token anywhere.
std::string generateRequestId() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist;
    std::ostringstream oss;
    oss << std::hex << dist(rng) << dist(rng);
    return "req-" + oss.str();
}

Json::Value firstOrchestrationDoc(const Json::Value& contract) {
    if (contract.isObject() && contract.isMember("Data") && contract["Data"].isArray() &&
        !contract["Data"].empty() && contract["Data"][0].isObject()) {
        return contract["Data"][0];
    }
    return Json::Value(Json::objectValue);
}

ChatServiceResult failed(const std::string& stage,
                          const std::string& message,
                          const std::string& chatId = "") {
    ChatServiceResult result;
    result.success = false;
    result.errorStage = stage;
    result.errorMessage = message;
    result.data.chatId = chatId;
    result.data.state = Application::DTOs::SendMessageResponseData::emptyPipelineState();
    return result;
}

} // namespace

ChatService::ChatService(Application::Configuration::AppConfig config,
                          Application::Validators::DocumentValidator validator,
                          OrchestrationClient orchestrationClient)
    : config_(std::move(config)),
      validator_(std::move(validator)),
      orchestrationClient_(std::move(orchestrationClient)) {}

std::string ChatService::persistTempFile(const Application::DTOs::ChatFileDTO& file,
                                          const std::string& requestId) const {
    namespace fs = std::filesystem;

    std::string decoded;
    try {
        decoded = drogon::utils::base64Decode(file.fileBase64);
    } catch (const std::exception& e) {
        throw TempStorageError(std::string("Invalid base64 payload: ") + e.what());
    }
    if (decoded.empty()) {
        throw TempStorageError("Decoded file content is empty.");
    }

    const fs::path dir = fs::path(resolvedTempRoot()) / requestId;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        throw TempStorageError("Failed to create temp directory " + dir.string() + ": " +
                                ec.message());
    }

    // FileName comes from the client — never trust it as a path. Keep only
    // the filename component to prevent path traversal.
    const std::string safeName = fs::path(file.fileName).filename().string();
    const fs::path filePath = dir / (safeName.empty() ? "upload.bin" : safeName);

    std::ofstream out(filePath, std::ios::binary);
    if (!out.is_open()) {
        throw TempStorageError("Failed to open temp file for writing: " + filePath.string());
    }
    out.write(decoded.data(), static_cast<std::streamsize>(decoded.size()));
    if (!out.good()) {
        throw TempStorageError("Failed to write temp file: " + filePath.string());
    }
    out.close();

    return filePath.string();
}

std::string ChatService::resolvedTempRoot() const {
    namespace fs = std::filesystem;

    auto usable = [](const fs::path& candidate) -> std::optional<fs::path> {
        if (candidate.empty()) return std::nullopt;
        const std::string asString = candidate.string();
        if (asString.find("...") != std::string::npos) return std::nullopt;
        std::error_code ec;
        fs::create_directories(candidate, ec);
        if (ec) return std::nullopt;
        const fs::path absolute = fs::absolute(candidate, ec);
        if (ec) return candidate;
        return absolute;
    };

    if (auto ok = usable(fs::path(config_.tempUploadRootDir))) {
        return ok->string();
    }

    const fs::path cwd = fs::current_path();
    if (auto ok = usable(cwd / ".." / "Storage" / "files" / "temp_processing")) {
        return ok->string();
    }
    if (auto ok = usable(cwd / "Storage" / "files" / "temp_processing")) {
        return ok->string();
    }

    std::error_code ec;
    const fs::path sysTemp = fs::temp_directory_path(ec) / "SmartGovernmentAI" / "temp_processing";
    if (!ec) {
        if (auto ok = usable(sysTemp)) {
            return ok->string();
        }
    }

    throw TempStorageError("No writable temp directory for uploaded files.");
}

void ChatService::cleanupTempDir(const std::string& requestId) const {
    namespace fs = std::filesystem;
    const fs::path dir = fs::path(resolvedTempRoot()) / requestId;
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (ec) {
        LOG_WARN << "ChatService: failed to clean up temp dir " << dir.string() << ": "
                 << ec.message();
    }
}

ChatServiceResult ChatService::sendMessage(
    const Application::DTOs::SendMessageRequestDTO& request) const {
    ChatServiceResult result;

    // Question is required only when no file is attached.
    // File + optional accompanying text is the OCR document upload path.
    if (!request.file.has_value()) {
        const auto questionCheck = validator_.validateQuestion(request.question);
        if (!questionCheck.valid) {
            return failed(ErrorStage::ValidationFailed, questionCheck.errorMessage);
        }
    }

    if (request.file.has_value()) {
        const auto fileCheck = validator_.validateFile(*request.file);
        if (!fileCheck.valid) {
            return failed(ErrorStage::ValidationFailed, fileCheck.errorMessage);
        }
    } else if (request.question.empty()) {
        return failed(ErrorStage::ValidationFailed, "Either Question or File is required.");
    }

    // requestId doubles as: temp dir name, Orchestration document_id, and
    // (when the client didn't send one) the new ChatId returned to Presentation.
    const std::string requestId = request.chatId.has_value() && !request.chatId->empty()
                                       ? *request.chatId
                                       : generateRequestId();

    // --- 2) Optional temp file persistence ---
    std::optional<std::string> tempPath;
    if (request.file.has_value()) {
        try {
            tempPath = persistTempFile(*request.file, requestId);
        } catch (const TempStorageError& e) {
            return failed(ErrorStage::TempStorageFailed, e.what());
        }
    }

    // --- 3) Build unified pipeline envelope, then delegate to Orchestration ---
    LayerDocumentDTO document;
    document.documentId = requestId;
    if (request.file.has_value()) {
        document.fileName = std::filesystem::path(request.file->fileName).filename().string();
        document.fileType =
            LayerStateDTO::normalizeFileType(document.fileName, request.file->fileType);
    }
    if (tempPath.has_value()) {
        document.documentPath = tempPath;
    }

    OrchestrationRequest orchestrationRequest;
    orchestrationRequest.requestId = requestId;
    orchestrationRequest.question = request.question;
    orchestrationRequest.documentPath = tempPath;
    orchestrationRequest.layerState =
        LayerStateDTO::initial(std::move(document), request.question, /*success=*/true);

    const OrchestrationResult orchestrationResult = orchestrationClient_.process(orchestrationRequest);

    // --- 4) Cleanup happens regardless of outcome ---
    if (tempPath.has_value()) {
        cleanupTempDir(requestId);
    }

    if (!orchestrationResult.reachable) {
        LOG_ERROR << "ChatService: Orchestration unreachable: " << orchestrationResult.errorMessage;
        return failed(orchestrationResult.errorStage, orchestrationResult.errorMessage, requestId);
    }

    const Json::Value& contract = orchestrationResult.contract;
    const bool orchSuccess =
        contract.isMember("Success") && contract["Success"].isBool() && contract["Success"].asBool();

    result.success = orchSuccess;
    result.data.chatId = requestId;
    result.data.state = Application::DTOs::SendMessageResponseData::pipelineStateFromOrchestrationDoc(
        firstOrchestrationDoc(contract), requestId, request.question, request.file);

    if (!orchSuccess) {
        result.errorStage = ErrorStage::OrchestrationError;
        result.errorMessage = "Orchestration returned Success=false";
    }
    return result;
}

} // namespace Application::Services
