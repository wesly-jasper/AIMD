import { useState, type RefObject } from "react";
import { useBlobUrl } from "../hooks";
import type { Detection, MediaType } from "../types";

export interface DrawnRegion {
  detectionId: string;
  bbox: [number, number, number, number];
  /** True only when the detector that produced it observed an anomaly. */
  anomaly: boolean;
}

/**
 * The evidence itself, with every localised region drawn in the one data hue.
 *
 * Regions are drawn in the image's own pixel space (the SVG's viewBox is the
 * natural size), so they land where the detector put them at any display size.
 * A region covering effectively the whole frame localises nothing and is not
 * drawn; it is still counted in the caption.
 */
export function EvidenceView({
  mediaId, mediaType, regions, highlight, alt, videoRef,
}: {
  mediaId: string;
  mediaType: MediaType;
  regions: DrawnRegion[];
  highlight: Set<string> | null;
  alt: string;
  /** Lets the keyframe timeline seek the player. */
  videoRef?: RefObject<HTMLVideoElement | null>;
}) {
  const { url, error } = useBlobUrl(`/media/${mediaId}/download`);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);

  if (error) return <div className="evidence-missing">The evidence bytes could not be retrieved from storage.</div>;
  if (!url) return <div className="evidence-frame loading" aria-busy="true" />;

  if (mediaType === "video") {
    return <video ref={videoRef} className="evidence-media" src={url} controls muted preload="metadata" />;
  }
  if (mediaType === "audio") {
    return <audio className="evidence-audio" src={url} controls preload="metadata" />;
  }

  // The hue means "a detector localised an anomaly here" and nothing else. A
  // region from a detector that found nothing (a face box, a clean mask) is
  // context: drawn dashed in ink, and only while its detector is pointed at.
  const drawable = size
    ? regions.filter((r) =>
        r.bbox[2] * r.bbox[3] < 0.95 * size.w * size.h &&
        (r.anomaly || (highlight !== null && highlight.has(r.detectionId))))
    : [];

  return (
    <div className="evidence-frame">
      {/* The inner box is exactly the image's rendered size, so the overlay's
          coordinates match the pixels even when the frame letterboxes it. */}
      <span className="evidence-inner">
      <img
        src={url}
        alt={alt}
        onLoad={(e) => setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
      />
      {size && drawable.length > 0 && (
        <svg className="regions" viewBox={`0 0 ${size.w} ${size.h}`} preserveAspectRatio="none" aria-hidden="true">
          {drawable.map((r, i) => {
            const dim = highlight !== null && !highlight.has(r.detectionId);
            return (
              <rect
                key={i}
                x={r.bbox[0]} y={r.bbox[1]} width={r.bbox[2]} height={r.bbox[3]}
                className={"region" + (r.anomaly ? "" : " context") + (dim ? " dim" : "")}
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
        </svg>
      )}
      </span>
    </div>
  );
}

export function regionsOf(detections: Detection[], anomalous: Set<string>): DrawnRegion[] {
  return detections.flatMap((d) =>
    d.regions
      .filter((r) => Array.isArray(r.bbox) && r.bbox.length === 4 && r.bbox[2] > 0 && r.bbox[3] > 0)
      .map((r) => ({
        detectionId: d.id,
        bbox: r.bbox as [number, number, number, number],
        anomaly: anomalous.has(d.id),
      })),
  );
}

/** A small square thumbnail for evidence rows. */
export function Thumb({ mediaId, mediaType }: { mediaId: string; mediaType: MediaType }) {
  const { url } = useBlobUrl(mediaType === "image" ? `/media/${mediaId}/download` : null);
  if (mediaType !== "image") {
    return <span className="thumb kind" aria-hidden="true">{mediaType === "video" ? "Video" : "Audio"}</span>;
  }
  return url ? <img className="thumb" src={url} alt="" /> : <span className="thumb" aria-hidden="true" />;
}
