// OrchestrationClient
// The single internal bridge from Application to the Orchestration Layer's
// REST endpoint (Orchestration/main.py -> POST /process).
//
// Forwards the unified pipeline envelope built by LayerStateDTO:
//   { request, ocr:{}, classification:{}, extraction:{}, validation:{},
//     rag:{}, summary:{}, routing:{}, writing:{} }
// plus flat wire fields (document_id / document_path / question) so older
// Orchestration handlers keep working. Returns { Success, Data } unchanged.

#pragma once

#include "../DTOs/LayerStateDTO.h"

#include <json/json.h>
#include <optional>
#include <string>

namespace Application::Services {

struct OrchestrationRequest {
    std::string requestId;                   // correlates Application <-> Orchestration logs
    std::string question;                    // optional accompanying text (may be empty)
    std::optional<std::string> documentPath; // temp file path; unset when no file was sent
    Application::DTOs::LayerStateDTO layerState; // unified envelope (request + empty stages)
};

struct OrchestrationResult {
    bool reachable = false;
    Json::Value contract;      // { Success, Data } when reachable
    std::string errorStage;    // ApiResponseDTO::ErrorStage value when not reachable
    std::string errorMessage;  // human-readable detail, safe to log
};

class OrchestrationClient {
public:
    OrchestrationClient(std::string baseUrl, int timeoutSeconds);

    // Calls POST {baseUrl}/process with the LayerStateDTO envelope (+ flat
    // document_id / document_path / question for wire compatibility) and
    // expects:
    //   { "Success": bool, "Data": [ document, ... ] }
    //
    // Never throws — network errors, timeouts, non-2xx responses and
    // malformed JSON are all captured in the returned OrchestrationResult.
    OrchestrationResult process(const OrchestrationRequest& request) const;

private:
    std::string baseUrl_;
    int timeoutSeconds_;
};

} // namespace Application::Services
