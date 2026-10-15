import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  useHeatmap,
  useRegions,
  useTimeline,
  useProvenanceGraph,
} from '../services';
import * as THREE from 'three';

const AnalysisViewer: React.FC = () => {
  const { analysisId } = useParams<{ analysisId: string }>();
  const [tab, setTab] = useState<'heatmap' | 'regions' | 'timeline' | 'provenance' | '3d'>(
    'heatmap'
  );

  const { data: heatmap, isLoading: heatmapLoading } = useHeatmap(analysisId ?? '');
  const { data: regions, isLoading: regionsLoading } = useRegions(analysisId ?? '');
  const { data: timeline, isLoading: timelineLoading } = useTimeline(analysisId ?? '');
  const { data: graph, isLoading: graphLoading } = useProvenanceGraph(analysisId ?? '');

  // Simple 3D scene using three.js (placeholder)
  useEffect(() => {
    if (tab !== '3d') return;
    const container = document.getElementById('three-container');
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    const camera = new THREE.PerspectiveCamera(
      75,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 5;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Add a rotating cube
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshStandardMaterial({ color: 0x00ff88 });
    const cube = new THREE.Mesh(geometry, material);
    scene.add(cube);

    // Light
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(5, 5, 5);
    scene.add(light);

    const animate = () => {
      requestAnimationFrame(animate);
      cube.rotation.x += 0.01;
      cube.rotation.y += 0.01;
      renderer.render(scene, camera);
    };
    animate();

    // Cleanup
    return () => {
      container.removeChild(renderer.domElement);
      // dispose geometries etc.
    };
  }, [tab, analysisId]);

  return (
    <div className=\"analysis-viewer\">
      <div className=\"tabs\">
        <button
          className={tab === 'heatmap' ? 'active' : ''}
          onClick={() => setTab('heatmap')}
          disabled={heatmapLoading}
        >
          Heatmap
        </button>
        <button
          className={tab === 'regions' ? 'active' : ''}
          onClick={() => setTab('regions')}
          disabled={regionsLoading}
        >
          Regions
        </button>
        <button
          className={tab === 'timeline' ? 'active' : ''}
          onClick={() => setTab('timeline')}
          disabled={timelineLoading}
        >
          Timeline
        </button>
        <button
          className={tab === 'provenance' ? 'active' : ''}
          onClick={() => setTab('provenance')}
          disabled={graphLoading}
        >
          Provenance
        </button>
        <button
          className={tab === '3d' ? 'active' : ''}
          onClick={() => setTab('3d')}
        >
          3D View
        </button>
      </div>

      <div className=\"tab-content\">
        {tab === 'heatmap' && heatmapLoading && <p>Loading heatmap...</p>}
        {tab === 'heatmap' && !heatmapLoading && heatmap && (
          <div>
            <h3>Manipulation Heatmap</h3>
            {/* Assuming heatmap.heatmap_url is a base64 image or URL */}
            <img
              src={heatmap.heatmap_url.startsWith('http')
                ? heatmap.heatmap_url
                : `data:image/png;base64,${heatmap.heatmap_url}`}
              alt=\"Heatmap\"
              style={{ maxWidth: '100%', border: '1px solid #ccc' }}
            />
          </div>
        )}

        {tab === 'regions' && regionsLoading && <p>Loading regions...</p>}
        {tab === 'regions' && !regionsLoading && regions && (
          <div>
            <h3>Detected Regions ({regions.regions.length})</h3>
            <ul>
              {regions.regions
                .filter((r): r is DetectionResult & { region: DetectionRegion } => r.region !== undefined)
                .map((r, idx) => (
                  <li key={idx}>
                    Confidence: {(r.confidence * 100).toFixed(1)}% &nbsp;|&nbsp;
                    {r.region.x}, {r.region.y}, {r.region.width}×{r.region.height}
                    {r.region.frame_number !== undefined && (
                      ` (frame: ${r.region.frame_number})`
                    )}
                    {r.region.timestamp !== undefined && (
                      ` (time: ${r.region.timestamp}s)`
                    )}
                  </li>
                ))}
            </ul>
          </div>
        )}

        {tab === 'timeline' && timelineLoading && <p>Loading timeline...</p>}
        {tab === 'timeline' && !timelineLoading && timeline && (
          <div>
            <h3>Analysis Timeline</h3>
            <p>
              Frame scores: {timeline.frame_scores.length} frames
            </p>
            {/* Simple bar chart using divs */}
            <div style={{ height: '200px', overflow: 'auto' }}>
              {timeline.frame_scores.map((fs) => (
                <div
                  key={fs.frame_number}
                  style={{
                    display: 'inline-block',
                    width: '2px',
                    height: `${fs.score * 100}px`,
                    backgroundColor: fs.score > 0.7 ? '#ff4444' : '#44ff44',
                    marginRight: '1px',
                    verticalAlign: 'bottom',
                  }}
                />
              ))}
            </div>
          </div>
        )}

        {tab === 'provenance' && graphLoading && <p>Loading provenance graph...</p>}
        {tab === 'provenance' && !graphLoading && graph && (
          <div>
            <h3>Provenance Graph</h3>
            <p>
              Nodes: {graph.nodes.length} &nbsp;|&nbsp; Edges:{' '}
              {graph.edges.length}
            </p>
            <ul>
              {graph.nodes.map((n) => (
                <li key={n.id}>
                  {n.type} ({n.media_id}) - Confidence: {
                    (n.confidence * 100).toFixed(1)
                  }%
                </li>
              ))}
            </ul>
          </div>
        )}

        {tab === '3d' && (
          <div id=\"three-container\" style={{ width: '100%', height: '500px' }} />\n        )}
      </div>
    </div>
  );
};

export default AnalysisViewer;
