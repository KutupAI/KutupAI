cmake_minimum_required(VERSION 3.16)
project(SmartGovernmentAI_Application CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Drogon CONFIG REQUIRED)

add_executable(${PROJECT_NAME}
    main.cpp

    Configuration/AppConfig.cpp

    DTOs/ApiResponseDTO.cpp
    DTOs/ChatDTO.cpp

    Validators/DocumentValidator.cpp

    Services/OrchestrationClient.cpp
    Services/ChatService.cpp

    Controllers/ChatController.cpp

    Routes/ChatRoutes.cpp

    # --- Pre-existing stubs (Document/Auth/User/Status) ---
    # Kept in the build so the project keeps compiling once these are
    # implemented; currently header-only comments with no symbols.
    # Controllers/DocumentController.cpp
    # Controllers/AuthController.cpp
    # Controllers/UserController.cpp
    # Controllers/StatusController.cpp
    # Services/DocumentProcessingService.cpp
    # Services/AuthService.cpp
    # Services/UserService.cpp
    # Routes/DocumentRoutes.cpp
    # Routes/AuthRoutes.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE Drogon::Drogon)
