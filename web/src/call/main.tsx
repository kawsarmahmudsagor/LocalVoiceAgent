import { createRoot } from "react-dom/client";
import CallApp from "./CallApp";
import "./call.css";

// No StrictMode: its double-mounted effects would open two WebSockets in dev, and
// the single-client server would answer the second with session_busy.
createRoot(document.getElementById("root")!).render(<CallApp />);
