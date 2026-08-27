import React from "react";
import ChatPage from "./pages/ChatPage";

/**
 * App root — currently mounts the chat surface (other pages are stubs).
 *
 * `.kutup-skyfield` is a fixed, decorative "night sky" backdrop (defined in
 * styles/theme.css) rendered once here so it sits behind every page without
 * each page having to know about it. Purely visual — aria-hidden, z-index
 * below content, no pointer events.
 */
const App: React.FC = () => (
  <>
    <div className="kutup-skyfield" aria-hidden="true" />
    <ChatPage />
  </>
);

export default App;
