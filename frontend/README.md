# AIMD Frontend

This is a beautiful, modern frontend for the AIMD AI Media Detection & Digital Forensics Platform, built with:

- **React 18** + **TypeScript**
- **Vite** for fast development and builds
- **React Query** for data fetching and state management
- **Three.js** for 3D visualizations
- **Taste-skill** (via `npx skills add Leonxlnx/taste-skill`) for design guidance
- Additional skills: soft-skill, redesign-skill, image-to-code-skill (installed via the taste-skill bundle)

## Features

- Media upload with progress tracking
- Real-time analysis job status and event streaming
- Tabbed results view:
  - Heatmap visualization
  - Detected regions list
  - Analysis timeline (frame-by-frame scores)
  - Provenance graph
  - Interactive 3D view (using Three.js)
- Responsive design with dark theme
- Type-safe API endpoints
- Mock service support (extendable)

## Getting Started

1. **Start the backend** (see backend README):
   ```bash
   cd backend
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   ```

3. **Configure environment**:
   Create a `.env` file (or use the provided example) with:
   ```
   VITE_API_BASE_URL=http://localhost:8000/api/v1
   ```

4. **Start the development server**:
   ```bash
   npm run dev
   ```
   The app will be available at `http://localhost:5173`.

5. **Build for production**:
   ```bash
   npm run build
   ```
   Output will be in the `dist/` directory.

## Design & Skills

This frontend was generated with the assistance of the **taste-skill** package, which provides guidelines for layout, typography, motion, and density to avoid generic, boring UIs. Additional skills from the taste-skill bundle (soft-skill, redesign-skill, etc.) were also incorporated to enhance the visual experience.

The 3D view uses Three.js to render interactive visualizations of analysis results (e.g., provenance graphs as node-link diagrams, feature embeddings in 3D space, etc.).

## Extending

- Add more visualization tabs (e.g., 3D scatter of perceptual hashes, network graphs)
- Implement WebSocket connections for real-time updates instead of polling
- Integrate charting libraries (Chart.js, D3, Recharts) for richer graphs
- Add authentication flows (login/register)
- Write unit and integration tests using Vitest and React Testing Library

---

*Built with ❤️ for forensic excellence.*
