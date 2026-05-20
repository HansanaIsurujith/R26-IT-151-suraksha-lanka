/**
 * MapScreen.js
 * Location: src/screens/MapScreen.js
 *
 * Features:
 *  - Destination search bar (Google Places)
 *  - Route display with colored risk segments
 *  - Flood / Landslide / Fog risk zone circles
 *  - Auto disaster prediction every 5 mins
 *  - Real-time route re-check every 2 mins
 *  - Risk timeline modal
 */

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  View, StyleSheet, Text, ActivityIndicator,
  Platform, Pressable, Alert, TextInput,
  TouchableOpacity, Modal, FlatList,
} from "react-native";
import * as Location from "expo-location";
import { predictBatch, getEventEmoji } from "../services/disasterApi";
import { getWeatherForLocation, getWeatherGrid } from "../services/weatherService";

// ── Safely load maps ───────────────────────────────────────
let MapView         = null;
let Marker          = null;
let Circle          = null;
let Polyline        = null;
let PROVIDER_GOOGLE = null;
let mapModuleAvailable = false;

if (Platform.OS !== "web") {
  try {
    const M        = require("react-native-maps");
    MapView        = M.default || M;
    Marker         = M.Marker;
    Circle         = M.Circle;
    Polyline       = M.Polyline;
    PROVIDER_GOOGLE = M.PROVIDER_GOOGLE;
    mapModuleAvailable = true;
  } catch (e) {
    console.warn("MapView not available:", e.message);
  }
}

// ── Config ─────────────────────────────────────────────────
const GOOGLE_API_KEY       = "YOUR_GOOGLE_API_KEY_HERE"; // app.json eke same key
const DISASTER_REFRESH_MS  = 5 * 60 * 1000; // 5 mins — area disaster check
const ROUTE_RECHECK_MS     = 2 * 60 * 1000; // 2 mins — route risk re-check
const GRID_RADIUS          = 0.15;
const GRID_SIZE            = 3;
const ROUTE_SAMPLE_COUNT   = 8;

// ── Risk config ─────────────────────────────────────────────
const RISK_CONFIG = {
  Flood:     { color: "#E53935", circleColor: "rgba(229,57,53,0.2)",   radius: 2500, label: "Flood Risk" },
  Landslide: { color: "#FB8C00", circleColor: "rgba(251,140,0,0.2)",   radius: 2000, label: "Landslide Risk" },
  Fog:       { color: "#7E57C2", circleColor: "rgba(126,87,194,0.15)", radius: 3000, label: "Fog Alert" },
  Warning:   { color: "#FDD835", circleColor: "rgba(253,216,53,0.2)",  radius: 1500, label: "Warning" },
  "No Risk": { color: "#43A047", circleColor: "rgba(67,160,71,0.08)",  radius: 1200, label: "No Risk" },
};

function segmentColor(event) {
  switch (event) {
    case "Flood":     return "#E53935";
    case "Landslide": return "#FB8C00";
    case "Fog":       return "#7E57C2";
    case "Warning":   return "#FDD835";
    default:          return "#43A047";
  }
}

// ── Decode Google polyline ──────────────────────────────────
function decodePolyline(encoded) {
  const pts = [];
  let i = 0, lat = 0, lng = 0;
  while (i < encoded.length) {
    let b, shift = 0, result = 0;
    do { b = encoded.charCodeAt(i++) - 63; result |= (b & 0x1f) << shift; shift += 5; } while (b >= 0x20);
    lat += result & 1 ? ~(result >> 1) : result >> 1;
    shift = 0; result = 0;
    do { b = encoded.charCodeAt(i++) - 63; result |= (b & 0x1f) << shift; shift += 5; } while (b >= 0x20);
    lng += result & 1 ? ~(result >> 1) : result >> 1;
    pts.push({ latitude: lat / 1e5, longitude: lng / 1e5 });
  }
  return pts;
}

function samplePoints(pts, count) {
  if (pts.length <= count) return pts;
  const step = Math.floor(pts.length / count);
  return Array.from({ length: count }, (_, i) => pts[i * step]);
}

// ════════════════════════════════════════════════════════════
export default function MapScreen() {
  const [userLocation,    setUserLocation]    = useState(null);
  const [isLoading,       setIsLoading]       = useState(true);

  // Search
  const [searchQuery,     setSearchQuery]     = useState("");
  const [suggestions,     setSuggestions]     = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);

  // Route
  const [destination,     setDestination]     = useState(null);
  const [routePoints,     setRoutePoints]     = useState([]);
  const [routeSegments,   setRouteSegments]   = useState([]);
  const [routeLoading,    setRouteLoading]    = useState(false);
  const [routeAnalyzing,  setRouteAnalyzing]  = useState(false);
  const [overallRisk,     setOverallRisk]     = useState(null);
  const [lastRouteCheck,  setLastRouteCheck]  = useState(null);

  // Disaster zones (area around user)
  const [disasterZones,   setDisasterZones]   = useState([]);
  const [lastAreaCheck,   setLastAreaCheck]   = useState(null);

  // Timeline modal
  const [riskTimeline,    setRiskTimeline]    = useState([]);
  const [timelineVisible, setTimelineVisible] = useState(false);

  const mapRef         = useRef(null);
  const searchTimer    = useRef(null);
  const areaRefreshRef = useRef(null);
  const routeRecheckRef = useRef(null);

  // ── Get GPS + initial disaster check ─────────────────────
  useEffect(() => {
    (async () => {
      try {
        const { status } = await Location.requestForegroundPermissionsAsync();
        let loc = { latitude: 6.9271, longitude: 79.8612 };
        if (status === "granted") {
          const pos = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced });
          loc = { latitude: pos.coords.latitude, longitude: pos.coords.longitude };
        }
        setUserLocation(loc);
        await fetchAreaDisasters(loc);
      } finally {
        setIsLoading(false);
      }
    })();

    return () => {
      clearInterval(areaRefreshRef.current);
      clearInterval(routeRecheckRef.current);
      clearTimeout(searchTimer.current);
    };
  }, []);

  // ── Auto refresh area disasters every 5 mins ─────────────
  useEffect(() => {
    if (!userLocation) return;
    areaRefreshRef.current = setInterval(() => fetchAreaDisasters(userLocation), DISASTER_REFRESH_MS);
    return () => clearInterval(areaRefreshRef.current);
  }, [userLocation]);

  // ── Fetch disaster zones around user ─────────────────────
  const fetchAreaDisasters = async (loc) => {
    try {
      const grid   = await getWeatherGrid(loc.latitude, loc.longitude, GRID_RADIUS, GRID_SIZE);
      const result = await predictBatch(grid);
      setDisasterZones(result.predictions || []);
      setLastAreaCheck(new Date());
    } catch (err) {
      console.error("Area disaster fetch error:", err);
    }
  };

  // ── Search autocomplete ───────────────────────────────────
  const handleSearchChange = (text) => {
    setSearchQuery(text);
    clearTimeout(searchTimer.current);
    if (text.length < 3) { setSuggestions([]); setShowSuggestions(false); return; }
    searchTimer.current = setTimeout(async () => {
      try {
        const loc = userLocation ? `${userLocation.latitude},${userLocation.longitude}` : "6.9271,79.8612";
        const url = `https://maps.googleapis.com/maps/api/place/autocomplete/json?input=${encodeURIComponent(text)}&location=${loc}&radius=100000&components=country:lk&key=${GOOGLE_API_KEY}`;
        const res  = await fetch(url);
        const data = await res.json();
        setSuggestions(data.predictions || []);
        setShowSuggestions(true);
      } catch (err) {
        console.error("Autocomplete error:", err);
      }
    }, 400);
  };

  // ── Select destination ────────────────────────────────────
  const handleSelectDestination = async (place) => {
    try {
      setSearchQuery(place.description);
      setSuggestions([]);
      setShowSuggestions(false);
      setRouteLoading(true);

      const url  = `https://maps.googleapis.com/maps/api/place/details/json?place_id=${place.place_id}&fields=geometry&key=${GOOGLE_API_KEY}`;
      const res  = await fetch(url);
      const data = await res.json();
      const loc  = data.result?.geometry?.location;
      if (!loc) throw new Error("Location not found");

      const dest = { latitude: loc.lat, longitude: loc.lng, name: place.description };
      setDestination(dest);
      await fetchRoute(dest);
    } catch {
      Alert.alert("Error", "Could not find destination. Please try again.");
    } finally {
      setRouteLoading(false);
    }
  };

  // ── Fetch route ───────────────────────────────────────────
  const fetchRoute = useCallback(async (dest) => {
    if (!userLocation) return;
    try {
      setRouteAnalyzing(true);
      const url  = `https://maps.googleapis.com/maps/api/directions/json?origin=${userLocation.latitude},${userLocation.longitude}&destination=${dest.latitude},${dest.longitude}&mode=driving&key=${GOOGLE_API_KEY}`;
      const res  = await fetch(url);
      const data = await res.json();

      if (!data.routes?.length) {
        Alert.alert("No Route", "Could not find a route to this destination.");
        return;
      }

      const pts = decodePolyline(data.routes[0].overview_polyline.points);
      setRoutePoints(pts);

      mapRef.current?.fitToCoordinates(pts, {
        edgePadding: { top: 160, right: 40, bottom: 200, left: 40 },
        animated: true,
      });

      await analyzeRouteRisk(pts);
      setLastRouteCheck(new Date());

      // Start route re-check every 2 mins
      clearInterval(routeRecheckRef.current);
      routeRecheckRef.current = setInterval(async () => {
        await analyzeRouteRisk(pts);
        setLastRouteCheck(new Date());
      }, ROUTE_RECHECK_MS);

    } catch {
      Alert.alert("Route Error", "Could not calculate route. Check your connection.");
    } finally {
      setRouteAnalyzing(false);
    }
  }, [userLocation]);

  // ── Analyze risk along route ──────────────────────────────
  const analyzeRouteRisk = async (pts) => {
    try {
      const sampled = samplePoints(pts, ROUTE_SAMPLE_COUNT);
      const weather = await Promise.all(sampled.map(p => getWeatherForLocation(p.latitude, p.longitude)));
      const result  = await predictBatch(weather);
      const preds   = result.predictions || [];

      // Colored segments
      const segs = preds.map((pred, i) => ({
        points: pts.slice(
          Math.floor(i * pts.length / preds.length),
          Math.floor((i + 1) * pts.length / preds.length) + 1
        ),
        color: segmentColor(pred.event),
        event: pred.event,
      }));
      setRouteSegments(segs);

      // Timeline
      setRiskTimeline(preds.map((pred, idx) => ({
        index:      idx,
        position:   Math.round((idx / preds.length) * 100),
        event:      pred.event,
        riskLevel:  pred.risk_level,
        message:    pred.message,
        confidence: pred.confidence,
        latitude:   pred.latitude,
        longitude:  pred.longitude,
      })));

      // Overall risk
      const hasHigh = preds.some(p => p.event === "Flood" || p.event === "Landslide");
      const hasMid  = preds.some(p => p.event === "Warning" || p.event === "Fog");
      setOverallRisk(hasHigh ? "HIGH" : hasMid ? "MEDIUM" : "LOW");

      if (hasHigh) {
        Alert.alert(
          "⚠️ Route Risk Warning",
          "Your route passes through high-risk disaster zones!\n\nProceed with caution.",
          [
            { text: "View Timeline", onPress: () => setTimelineVisible(true) },
            { text: "OK" },
          ]
        );
      }
    } catch (err) {
      console.error("Route risk error:", err);
    }
  };

  // ── Clear route ───────────────────────────────────────────
  const clearRoute = () => {
    clearInterval(routeRecheckRef.current);
    setDestination(null);
    setRoutePoints([]);
    setRouteSegments([]);
    setOverallRisk(null);
    setRiskTimeline([]);
    setSearchQuery("");
    setLastRouteCheck(null);
  };

  // ── Flood/Landslide/Wildlife button handlers ──────────────
  const handleFloodAlert = () => {
    const floods = disasterZones.filter(z => z.event === "Flood");
    Alert.alert("🌊 Flood Status",
      floods.length > 0
        ? `${floods.length} flood risk zone(s) detected nearby!\n\nStay away from low-lying areas.`
        : "No flood risk detected in your area currently.",
      [{ text: "OK" }]
    );
  };

  const handleLandslideAlert = () => {
    const slides = disasterZones.filter(z => z.event === "Landslide");
    Alert.alert("⛰️ Landslide Status",
      slides.length > 0
        ? `${slides.length} landslide risk zone(s) detected!\n\nAvoid hilly terrain.`
        : "No landslide risk detected in your area currently.",
      [{ text: "OK" }]
    );
  };

  const handleWildlifeAlert = () => {
    Alert.alert("🐘 Wildlife Detection", "Wildlife detection module — coming soon!", [{ text: "OK" }]);
  };

  const bannerColor = { HIGH: "#E53935", MEDIUM: "#FB8C00", LOW: "#43A047" };

  // ── Web / no maps fallback ────────────────────────────────
  if (!mapModuleAvailable || !MapView || Platform.OS === "web") {
    return (
      <View style={styles.container}>
        <View style={styles.placeholderMap}>
          <Text style={styles.title}>Suraksha Lanka</Text>
          <Text style={styles.subtitle}>Location: Colombo, Sri Lanka</Text>
          <Text style={styles.coordinates}>6.9271°N, 79.8612°E</Text>
        </View>
        <QuickButtons onFlood={handleFloodAlert} onLandslide={handleLandslideAlert} onWildlife={handleWildlifeAlert} />
      </View>
    );
  }

  if (isLoading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#0ea5e9" />
        <Text style={styles.loadingText}>Loading map...</Text>
      </View>
    );
  }

  const region = userLocation
    ? { latitude: userLocation.latitude, longitude: userLocation.longitude, latitudeDelta: 0.2, longitudeDelta: 0.2 }
    : { latitude: 6.9271, longitude: 79.8612, latitudeDelta: 0.2, longitudeDelta: 0.2 };

  return (
    <View style={styles.container}>

      {/* ── Map ── */}
      <MapView
        ref={mapRef}
        provider={PROVIDER_GOOGLE}
        style={styles.map}
        initialRegion={region}
        showsUserLocation
        showsMyLocationButton={false}
        scrollEnabled
        zoomEnabled
      >
        {/* Area disaster zones (around user) */}
        {disasterZones.map((zone, idx) => {
          const cfg = RISK_CONFIG[zone.event] || RISK_CONFIG["No Risk"];
          return (
            <React.Fragment key={`area-${idx}`}>
              <Circle
                center={{ latitude: zone.latitude, longitude: zone.longitude }}
                radius={cfg.radius}
                fillColor={cfg.circleColor}
                strokeColor={cfg.color}
                strokeWidth={1.5}
              />
              {zone.event !== "No Risk" && (
                <Marker
                  coordinate={{ latitude: zone.latitude, longitude: zone.longitude }}
                  pinColor={cfg.color}
                  title={cfg.label}
                  description={zone.message}
                />
              )}
            </React.Fragment>
          );
        })}

        {/* Route colored segments */}
        {routeSegments.map((seg, idx) => (
          <Polyline key={`seg-${idx}`} coordinates={seg.points} strokeColor={seg.color} strokeWidth={5} />
        ))}

        {/* Plain route fallback */}
        {routeSegments.length === 0 && routePoints.length > 0 && (
          <Polyline coordinates={routePoints} strokeColor="#0EA5E9" strokeWidth={4} />
        )}

        {/* Destination marker */}
        {destination && (
          <Marker
            coordinate={{ latitude: destination.latitude, longitude: destination.longitude }}
            pinColor="#0EA5E9"
            title="Destination"
            description={destination.name}
          />
        )}
      </MapView>

      {/* ── Search bar ── */}
      <View style={styles.searchWrapper}>
        <View style={styles.searchBox}>
          <Text style={styles.searchIcon}>🔍</Text>
          <TextInput
            style={styles.searchInput}
            placeholder="Search destination..."
            placeholderTextColor="#94A3B8"
            value={searchQuery}
            onChangeText={handleSearchChange}
            onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
          />
          {searchQuery.length > 0 && (
            <TouchableOpacity onPress={() => { setSearchQuery(""); setSuggestions([]); setShowSuggestions(false); }}>
              <Text style={styles.clearX}>✕</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* Suggestions dropdown */}
        {showSuggestions && suggestions.length > 0 && (
          <View style={styles.suggestions}>
            {suggestions.slice(0, 5).map((place) => (
              <TouchableOpacity
                key={place.place_id}
                style={styles.suggestionItem}
                onPress={() => handleSelectDestination(place)}
              >
                <Text style={styles.suggestionIcon}>📍</Text>
                <View style={{ flex: 1 }}>
                  <Text style={styles.suggestionMain} numberOfLines={1}>
                    {place.structured_formatting?.main_text || place.description}
                  </Text>
                  <Text style={styles.suggestionSub} numberOfLines={1}>
                    {place.structured_formatting?.secondary_text || ""}
                  </Text>
                </View>
              </TouchableOpacity>
            ))}
          </View>
        )}
      </View>

      {/* ── Loading bars ── */}
      {(routeLoading || routeAnalyzing) && (
        <View style={styles.loadingBar}>
          <ActivityIndicator size="small" color="#fff" />
          <Text style={styles.loadingBarText}>
            {routeLoading ? "Finding destination..." : "Analyzing route risks..."}
          </Text>
        </View>
      )}

      {/* ── Route risk banner ── */}
      {overallRisk && destination && (
        <View style={[styles.riskBanner, { backgroundColor: bannerColor[overallRisk] }]}>
          <View style={{ flex: 1 }}>
            <Text style={styles.riskBannerTitle}>
              {overallRisk === "HIGH" ? "⚠️" : overallRisk === "MEDIUM" ? "⚡" : "✅"}
              {"  "}{overallRisk} RISK ROUTE
            </Text>
            <Text style={styles.riskBannerSub}>
              {lastRouteCheck ? `Checked: ${lastRouteCheck.toLocaleTimeString()}` : "Checking..."}
            </Text>
          </View>
          <View style={styles.riskBannerRight}>
            <TouchableOpacity style={styles.timelineBtn} onPress={() => setTimelineVisible(true)}>
              <Text style={styles.timelineBtnText}>📊 Timeline</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.clearBtn} onPress={clearRoute}>
              <Text style={styles.clearBtnText}>✕</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      {/* ── Area status bar (no route) ── */}
      {!destination && lastAreaCheck && (
        <View style={styles.statusBar}>
          <View style={styles.statusDot} />
          <Text style={styles.statusText}>
            Monitoring area • {lastAreaCheck.toLocaleTimeString()}
          </Text>
        </View>
      )}

      {/* ── Legend ── */}
      <View style={styles.legend}>
        {["Flood", "Landslide", "Warning", "No Risk"].map(key => (
          <View key={key} style={styles.legendRow}>
            <View style={[styles.legendDot, { backgroundColor: RISK_CONFIG[key].color }]} />
            <Text style={styles.legendText}>{RISK_CONFIG[key].label}</Text>
          </View>
        ))}
      </View>

      {/* ── Quick buttons (original style) ── */}
      <QuickButtons onFlood={handleFloodAlert} onLandslide={handleLandslideAlert} onWildlife={handleWildlifeAlert} />

      {/* ════ RISK TIMELINE MODAL ════ */}
      <Modal visible={timelineVisible} animationType="slide" transparent onRequestClose={() => setTimelineVisible(false)}>
        <View style={styles.modalOverlay}>
          <View style={styles.timelineModal}>

            <View style={styles.timelineHeader}>
              <Text style={styles.timelineTitle}>🗺️ Route Risk Timeline</Text>
              <TouchableOpacity onPress={() => setTimelineVisible(false)}>
                <Text style={styles.timelineClose}>✕</Text>
              </TouchableOpacity>
            </View>

            {destination && (
              <View style={styles.destInfo}>
                <Text style={styles.destInfoText} numberOfLines={2}>📍 {destination.name}</Text>
              </View>
            )}

            {/* Progress bar */}
            <View style={styles.progressBar}>
              {riskTimeline.map((item, idx) => (
                <View key={idx} style={[styles.progressSeg, { backgroundColor: segmentColor(item.event) }]} />
              ))}
            </View>
            <View style={styles.progressLabels}>
              <Text style={styles.progressLabel}>Start</Text>
              <Text style={styles.progressLabel}>Destination</Text>
            </View>

            <FlatList
              data={riskTimeline}
              keyExtractor={(_, i) => String(i)}
              style={styles.timelineList}
              renderItem={({ item }) => {
                const cfg = RISK_CONFIG[item.event] || RISK_CONFIG["No Risk"];
                return (
                  <View style={styles.tlItem}>
                    <View style={styles.tlDotCol}>
                      <View style={[styles.tlDot, { backgroundColor: cfg.color }]} />
                      {item.index < riskTimeline.length - 1 && <View style={styles.tlLine} />}
                    </View>
                    <View style={[styles.tlCard, { borderLeftColor: cfg.color }]}>
                      <View style={styles.tlCardTop}>
                        <Text style={styles.tlEvent}>{getEventEmoji(item.event)}  {cfg.label}</Text>
                        <View style={[styles.tlBadge, { backgroundColor: cfg.color }]}>
                          <Text style={styles.tlBadgeText}>{item.riskLevel}</Text>
                        </View>
                      </View>
                      <Text style={styles.tlPos}>~{item.position}% along route</Text>
                      <Text style={styles.tlMsg}>{item.message}</Text>
                      <Text style={styles.tlConf}>Confidence: {item.confidence}%</Text>
                    </View>
                  </View>
                );
              }}
            />

            <View style={styles.tlFooter}>
              <Text style={styles.tlFooterText}>🔄 Route re-checked every 2 minutes</Text>
            </View>
          </View>
        </View>
      </Modal>

    </View>
  );
}

// ── Quick buttons component ────────────────────────────────
function QuickButtons({ onFlood, onLandslide, onWildlife }) {
  return (
    <View style={styles.buttonContainer}>
      <Pressable style={({ pressed }) => [styles.button, styles.floodButton,     pressed && styles.buttonPressed]} onPress={onFlood}>
        <Text style={styles.buttonText}>🌊</Text>
      </Pressable>
      <Pressable style={({ pressed }) => [styles.button, styles.landslideButton, pressed && styles.buttonPressed]} onPress={onLandslide}>
        <Text style={styles.buttonText}>⛰️</Text>
      </Pressable>
      <Pressable style={({ pressed }) => [styles.button, styles.wildlifeButton,  pressed && styles.buttonPressed]} onPress={onWildlife}>
        <Text style={styles.buttonText}>🐘</Text>
      </Pressable>
    </View>
  );
}

// ── Styles ─────────────────────────────────────────────────
const styles = StyleSheet.create({
  container:      { flex: 1, backgroundColor: "#f5f5f5" },
  map:            { flex: 1 },
  centered:       { flex: 1, alignItems: "center", justifyContent: "center" },
  loadingText:    { marginTop: 10, color: "#555", fontSize: 14 },

  placeholderMap: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#e0f2fe" },
  title:          { fontSize: 24, fontWeight: "bold", textAlign: "center", color: "#0c4a6e" },
  subtitle:       { fontSize: 16, textAlign: "center", marginTop: 10, color: "#0369a1" },
  coordinates:    { fontSize: 14, textAlign: "center", marginTop: 10, color: "#0ea5e9" },

  // Search
  searchWrapper: { position: "absolute", top: 14, left: 16, right: 16, zIndex: 10 },
  searchBox: {
    flexDirection: "row", alignItems: "center", backgroundColor: "#fff",
    borderRadius: 14, paddingHorizontal: 14, paddingVertical: 10,
    elevation: 6, shadowColor: "#000", shadowOpacity: 0.15, shadowRadius: 8, shadowOffset: { width: 0, height: 3 },
  },
  searchIcon:  { fontSize: 16, marginRight: 8 },
  searchInput: { flex: 1, fontSize: 14, color: "#1E293B" },
  clearX:      { fontSize: 14, color: "#94A3B8", paddingLeft: 8 },

  suggestions: {
    backgroundColor: "#fff", borderRadius: 12, marginTop: 6,
    elevation: 6, shadowColor: "#000", shadowOpacity: 0.12, shadowRadius: 6, shadowOffset: { width: 0, height: 2 },
  },
  suggestionItem: { flexDirection: "row", alignItems: "center", paddingVertical: 12, paddingHorizontal: 14, borderBottomWidth: 1, borderBottomColor: "#F1F5F9" },
  suggestionIcon: { fontSize: 16, marginRight: 10 },
  suggestionMain: { fontSize: 13, fontWeight: "600", color: "#1E293B" },
  suggestionSub:  { fontSize: 11, color: "#94A3B8", marginTop: 2 },

  // Loading
  loadingBar:     { position: "absolute", top: 78, left: 16, right: 16, backgroundColor: "#0EA5E9", borderRadius: 10, flexDirection: "row", alignItems: "center", paddingVertical: 10, paddingHorizontal: 16, gap: 10, elevation: 4 },
  loadingBarText: { color: "#fff", fontSize: 13, fontWeight: "600" },

  // Route risk banner
  riskBanner:      { position: "absolute", bottom: 190, left: 16, right: 16, borderRadius: 14, paddingVertical: 12, paddingHorizontal: 16, flexDirection: "row", alignItems: "center", justifyContent: "space-between", elevation: 6, shadowColor: "#000", shadowOpacity: 0.2, shadowRadius: 8, shadowOffset: { width: 0, height: 3 } },
  riskBannerTitle: { fontSize: 14, fontWeight: "700", color: "#fff" },
  riskBannerSub:   { fontSize: 11, color: "rgba(255,255,255,0.8)", marginTop: 2 },
  riskBannerRight: { flexDirection: "row", gap: 8 },
  timelineBtn:     { backgroundColor: "rgba(255,255,255,0.25)", borderRadius: 8, paddingHorizontal: 10, paddingVertical: 5 },
  timelineBtnText: { color: "#fff", fontSize: 12, fontWeight: "600" },
  clearBtn:        { backgroundColor: "rgba(255,255,255,0.25)", borderRadius: 8, paddingHorizontal: 10, paddingVertical: 5 },
  clearBtnText:    { color: "#fff", fontSize: 12, fontWeight: "600" },

  // Status bar
  statusBar:  { position: "absolute", top: 78, left: 16, right: 16, backgroundColor: "rgba(255,255,255,0.92)", borderRadius: 10, flexDirection: "row", alignItems: "center", paddingVertical: 8, paddingHorizontal: 14, gap: 8, elevation: 3 },
  statusDot:  { width: 8, height: 8, borderRadius: 4, backgroundColor: "#43A047" },
  statusText: { fontSize: 12, color: "#475569" },

  // Legend
  legend:     { position: "absolute", bottom: 190, left: 16, backgroundColor: "rgba(255,255,255,0.95)", borderRadius: 12, padding: 10, elevation: 4, shadowColor: "#000", shadowOpacity: 0.1, shadowRadius: 4, shadowOffset: { width: 0, height: 2 }, gap: 5 },
  legendRow:  { flexDirection: "row", alignItems: "center", gap: 6 },
  legendDot:  { width: 10, height: 10, borderRadius: 5 },
  legendText: { fontSize: 11, color: "#333" },

  // Original buttons
  buttonContainer: { position: "absolute", bottom: 100, right: 20, gap: 15 },
  button:          { width: 60, height: 60, borderRadius: 30, alignItems: "center", justifyContent: "center", shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.25, shadowRadius: 3.84, elevation: 5 },
  floodButton:     { backgroundColor: "#3b82f6" },
  landslideButton: { backgroundColor: "#ef4444" },
  wildlifeButton:  { backgroundColor: "#10b981" },
  buttonPressed:   { opacity: 0.7 },
  buttonText:      { color: "#fff", fontSize: 24, fontWeight: "600" },

  // Modal
  modalOverlay:   { flex: 1, backgroundColor: "rgba(0,0,0,0.5)", justifyContent: "flex-end" },
  timelineModal:  { backgroundColor: "#fff", borderTopLeftRadius: 24, borderTopRightRadius: 24, maxHeight: "85%", overflow: "hidden" },
  timelineHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", padding: 20, borderBottomWidth: 1, borderBottomColor: "#F1F5F9" },
  timelineTitle:  { fontSize: 17, fontWeight: "700", color: "#0F172A" },
  timelineClose:  { fontSize: 18, color: "#94A3B8", padding: 4 },
  destInfo:       { paddingHorizontal: 20, paddingVertical: 10, backgroundColor: "#F8FAFC" },
  destInfoText:   { fontSize: 13, color: "#475569" },
  progressBar:    { flexDirection: "row", height: 10, marginHorizontal: 20, borderRadius: 5, overflow: "hidden", marginTop: 12 },
  progressSeg:    { flex: 1, height: "100%" },
  progressLabels: { flexDirection: "row", justifyContent: "space-between", marginHorizontal: 20, marginTop: 4, marginBottom: 8 },
  progressLabel:  { fontSize: 10, color: "#94A3B8" },
  timelineList:   { flex: 1, paddingHorizontal: 20 },
  tlItem:         { flexDirection: "row", marginBottom: 8, paddingTop: 8 },
  tlDotCol:       { alignItems: "center", width: 24, marginRight: 12 },
  tlDot:          { width: 14, height: 14, borderRadius: 7 },
  tlLine:         { width: 2, flex: 1, backgroundColor: "#E2E8F0", marginTop: 4 },
  tlCard:         { flex: 1, backgroundColor: "#F8FAFC", borderRadius: 12, padding: 12, borderLeftWidth: 4, marginBottom: 4 },
  tlCardTop:      { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 4 },
  tlEvent:        { fontSize: 13, fontWeight: "700", color: "#0F172A" },
  tlBadge:        { borderRadius: 6, paddingHorizontal: 8, paddingVertical: 2 },
  tlBadgeText:    { fontSize: 10, color: "#fff", fontWeight: "700" },
  tlPos:          { fontSize: 11, color: "#64748B", marginBottom: 4 },
  tlMsg:          { fontSize: 12, color: "#475569", lineHeight: 18 },
  tlConf:         { fontSize: 11, color: "#94A3B8", marginTop: 4 },
  tlFooter:       { padding: 16, borderTopWidth: 1, borderTopColor: "#F1F5F9", backgroundColor: "#F8FAFC" },
  tlFooterText:   { fontSize: 11, color: "#64748B", textAlign: "center" },
});
