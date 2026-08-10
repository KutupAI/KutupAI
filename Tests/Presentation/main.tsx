import React from "react";
import ReactDOM from "react-dom/client";
import ChatPage from "../../Presentation/pages/ChatPage";
import "../../Presentation/styles/theme.css";
import "./preview.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ChatPage />
  </React.StrictMode>
);
