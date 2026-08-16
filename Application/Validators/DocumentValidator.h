// DocumentValidator
// Validates file format/size/type before processing.
//
// Pure business-rule validation, no I/O. Shared by every channel that
// accepts a file (DocumentController's upload channel and ChatController's
// chat channel), so limits stay consistent across entry points.

#pragma once

#include "../Configuration/AppConfig.h"
#include "../DTOs/ChatDTO.h"
#include <optional>
#include <string>
#include <vector>

namespace Application::Validators {

struct ValidationResult {
    bool valid = true;
    std::string errorMessage; // empty when valid == true
};

class DocumentValidator {
public:
    explicit DocumentValidator(Application::Configuration::AppConfig config);

    // Question must be non-empty after trimming.
    ValidationResult validateQuestion(const std::string& question) const;

    // Validates MIME type against the allow-list and decoded size against
    // maxUploadSizeBytes. Does not touch disk.
    ValidationResult validateFile(const Application::DTOs::ChatFileDTO& file) const;

private:
    Application::Configuration::AppConfig config_;
    static const std::vector<std::string> kAllowedMimeTypes; // mirrors chatConfig.ts
    static bool isAllowedType(const Application::DTOs::ChatFileDTO& file);
};

} // namespace Application::Validators
