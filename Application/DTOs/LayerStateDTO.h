// LayerStateDTO
// Unified pipeline envelope owned by Application and forwarded to
// Orchestration. Matches the Agents contract:
//
//   {
//     "request": {
//       "success": true,
//       "question": "...",
//       "document": {
//         "document_id": "...",
//         "file_name": "...",
//         "file_type": "pdf"
//       }
//     },
//     "ocr": {},
//     "classification": {},
//     "extraction": {},
//     "validation": {},
//     "rag": {},
//     "summary": {},
//     "routing": {},
//     "writing": {}
//   }
//
// Application fills `request` only; every stage section starts as {}.
// `document_path` (optional) is Application-only so Orchestration/OCR can
// resolve the temp file on shared disk — Agents ignore unknown keys.

#pragma once

#include <json/json.h>
#include <optional>
#include <string>

namespace Application::DTOs {

struct LayerDocumentDTO {
    std::string documentId;
    std::string fileName;
    std::string fileType;                          // extension e.g. "pdf", not MIME
    std::optional<std::string> documentPath;       // temp absolute path when a file was uploaded
};

struct LayerStateDTO {
    bool success = true;
    std::string question;
    LayerDocumentDTO document;

    // Build the initial empty envelope Application hands to Orchestration.
    static LayerStateDTO initial(LayerDocumentDTO document, std::string question, bool success = true);

    // Derive file_type from a filename extension (fallback: MIME subtype).
    static std::string normalizeFileType(const std::string& fileName, const std::string& mimeOrType);

    // Full envelope JSON (request + empty stage sections).
    Json::Value toJson() const;

    // request section only (useful for logs / partial checks).
    Json::Value requestJson() const;
};

} // namespace Application::DTOs
