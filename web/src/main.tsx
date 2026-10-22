import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./auth";
import { Auth } from "./pages/Auth";
import { FolderDetail, Folders } from "./pages/Folders";
import { History } from "./pages/History";
import { Home } from "./pages/Home";
import { Result } from "./pages/Result";
import "./styles/app.css";
import "./styles/product.css";

function Routed() {
  const { user, checking } = useAuth();
  if (checking) return <div className="boot" aria-busy="true" />;
  if (!user) return <Auth />;
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/results/:analysisId" element={<Result />} />
      <Route path="/history" element={<History />} />
      <Route path="/folders" element={<Folders />} />
      <Route path="/folders/:folderId" element={<FolderDetail />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <HashRouter>
      <AuthProvider>
        <Routed />
      </AuthProvider>
    </HashRouter>
  </StrictMode>,
);
