// ApiResponseDTO
// Shared response envelope for ALL Application endpoints.
//
// Mirrors Presentation/models/ApiResponse.ts field-for-field:
//   Success / Message / Code / AdditionalData / CarryOnData
// Presentation/models/ApiResponse.ts explicitly forbids inventing a new
// response shape ("لا تُنشئ أي شكل استجابة جديد") — every controller in this
// layer must build its JSON response through this helper instead of hand
// rolling Json::Value objects inline.
//
// `Code` is used as a cross-layer trace marker (see ErrorStage below), not a
// generic HTTP status — it lets Presentation/support show *which* layer/step
// failed without leaking stack traces to the client.

#pragma once

#include <json/json.h>
#include <string>

namespace Application::DTOs {

// Stable, greppable stage identifiers. Every failure returned to Presentation
// carries one of these in the `Code` field so a failure can be traced back to
// the exact boundary it happened at (Application validation vs. the
// Application<->Orchestration hop vs. inside Orchestration/Agents).
struct ErrorStage {
    static constexpr const char* ValidationFailed        = "APPLICATION_VALIDATION_FAILED";
    static constexpr const char* MalformedRequest         = "APPLICATION_MALFORMED_REQUEST";
    static constexpr const char* TempStorageFailed         = "APPLICATION_TEMP_STORAGE_FAILED";
    static constexpr const char* OrchestrationUnreachable = "APPLICATION_ORCHESTRATION_UNREACHABLE";
    static constexpr const char* OrchestrationError        = "ORCHESTRATION_ERROR";
    static constexpr const char* InternalError              = "APPLICATION_INTERNAL_ERROR";
};

class ApiResponseDTO {
public:
    // Success == true, AdditionalData == data. Message is optional context
    // (e.g. a warning that doesn't block success).
    static Json::Value success(const Json::Value& additionalData,
                                const std::string& message = "");

    // Success == false. `code` should be one of ErrorStage's constants so
    // failures stay traceable across layers.
    static Json::Value failure(const std::string& message,
                                const std::string& code);

    // Unified processing contract between Presentation, Application,
    // Orchestration, and OCR Agent: { "Success": bool, "Data": [ ... ] }
    static Json::Value documentEnvelope(bool success, const Json::Value& data);
    static Json::Value emptyDocumentEnvelope(bool success = false);
};

} // namespace Application::DTOs
