/**
 * Derives a stable HSL color string from a request id.
 * The same id always produces the same hue, making it easy to
 * cross-reference the same request across BlockGrid and QueueLanes.
 */
export function requestHue(requestId: number): number {
  return (requestId * 57) % 360;
}

export function requestColor(requestId: number, alpha = 1): string {
  const hue = requestHue(requestId);
  return `hsla(${hue}, 70%, 55%, ${alpha})`;
}

export function requestColorLight(requestId: number, alpha = 0.18): string {
  const hue = requestHue(requestId);
  return `hsla(${hue}, 70%, 55%, ${alpha})`;
}
