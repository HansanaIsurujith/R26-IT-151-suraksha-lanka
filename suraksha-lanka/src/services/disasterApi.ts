/**
 * Disaster Prediction API Service
 * React Native — connect to FastAPI backend
 *
 * Usage:
 *   import { predictDisaster, predictBatch } from './disasterApi';
 */

// ─────────────────────────────────────────────
// CONFIG — change this to your server IP/URL
// ─────────────────────────────────────────────
const API_BASE_URL = "http://10.26.3.24:8000"; 
// Local testing:  "http://192.168.1.x:8000"  (ඔයාගේ PC IP එක)
// Production:     "https://your-domain.com"

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────
export type WeatherInput = {
  rainfall:    number;  // mm
  rain_3d:     number;  // mm cumulative 3 days
  rain_7d:     number;  // mm cumulative 7 days
  temperature: number;  // Celsius
  humidity:    number;  // 0-100
  latitude:    number;
  longitude:   number;
  elevation:   number;  // meters
  date?:       string;  // "YYYY-MM-DD" — optional, defaults to today
};

export type PredictionResult = {
  event:         string;   // "Flood" | "Landslide" | "Fog" | "Warning" | "No Risk"
  event_label:   number;   // 0-4
  confidence:    number;   // 0-100 percentage
  risk_level:    string;   // "LOW" | "MEDIUM" | "MODERATE" | "HIGH"
  message:       string;
  probabilities: Record<string, number>;
};

export type BatchLocation = WeatherInput;

export type BatchResult = {
  count: number;
  predictions: Array<{
    latitude:   number;
    longitude:  number;
    event:      string;
    risk_level: string;
    confidence: number;
    message:    string;
  }>;
};

// ─────────────────────────────────────────────
// API Calls
// ─────────────────────────────────────────────

/**
 * Check if the API server is running
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Predict disaster risk for a single location
 *
 * Example:
 *   const result = await predictDisaster({
 *     rainfall: 25, rain_3d: 80, rain_7d: 130,
 *     temperature: 27.5, humidity: 85,
 *     latitude: 6.9271, longitude: 79.8612,
 *     elevation: 8
 *   });
 *   console.log(result.event);     // "Flood"
 *   console.log(result.risk_level) // "HIGH"
 */
export async function predictDisaster(
  input: WeatherInput
): Promise<PredictionResult> {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(input),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Prediction failed");
  }

  return res.json();
}

/**
 * Predict disaster risk for multiple locations (map grid)
 * Max 100 locations per call
 *
 * Example:
 *   const result = await predictBatch([
 *     { rainfall: 25, rain_3d: 80, ...location1 },
 *     { rainfall: 10, rain_3d: 30, ...location2 },
 *   ]);
 */
export async function predictBatch(
  locations: BatchLocation[]
): Promise<BatchResult> {
  const res = await fetch(`${API_BASE_URL}/predict/batch`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ locations }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Batch prediction failed");
  }

  return res.json();
}

// ─────────────────────────────────────────────
// Helper: Risk level → color (for UI)
// ─────────────────────────────────────────────
export function getRiskColor(riskLevel: string): string {
  switch (riskLevel) {
    case "LOW":      return "#4CAF50"; // green
    case "MODERATE": return "#FF9800"; // orange
    case "MEDIUM":   return "#FF9800"; // orange
    case "HIGH":     return "#F44336"; // red
    default:         return "#9E9E9E"; // grey
  }
}

export function getRiskEmoji(event: string): string {
  switch (event) {
    case "Flood":     return "🌊";
    case "Landslide": return "⛰️";
    case "Fog":       return "🌫️";
    case "Warning":   return "⚠️";
    case "No Risk":   return "✅";
    default:          return "❓";
  }
}
