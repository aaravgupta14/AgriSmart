-- =====================================================================
-- AgriSmart  02_views.sql
-- =====================================================================
USE agrismart;

-- Latest reading of every plot -----------------------------------------
CREATE OR REPLACE VIEW v_latest_soil_log AS
SELECT * FROM (
    SELECT l.*, ROW_NUMBER() OVER (PARTITION BY l.plot_id ORDER BY l.log_ts DESC, l.log_id DESC) AS rn
    FROM soil_health_logs l
    WHERE l.moisture_pct IS NOT NULL OR l.air_temp_c IS NOT NULL
) t WHERE rn = 1;

-- Latest valid pH per plot (manual soil test or optional probe), max 180 days old
CREATE OR REPLACE VIEW v_latest_ph AS
SELECT plot_id, ph_value, log_ts AS ph_measured_on, source AS ph_source
FROM (
    SELECT l.plot_id, l.ph_value, l.log_ts, l.source,
           ROW_NUMBER() OVER (PARTITION BY l.plot_id ORDER BY l.log_ts DESC) AS rn
    FROM soil_health_logs l
    WHERE l.ph_value IS NOT NULL AND l.log_ts >= NOW() - INTERVAL 180 DAY
) t WHERE rn = 1;

-- 7-day consensus of each plot (window ends at the plot's latest reading;
-- readings rejected by the quality check, quality_score = 0, are excluded)
CREATE OR REPLACE VIEW v_plot_recent_conditions AS
SELECT p.plot_id, p.soil_type,
       MIN(l.log_ts)                AS window_start,
       MAX(l.log_ts)                AS window_end,
       COUNT(*)                     AS n_readings,
       ROUND(AVG(l.moisture_pct),2) AS moisture_avg,
       ROUND(AVG(l.soil_temp_c),2)  AS soil_temp_avg,
       ROUND(AVG(l.air_temp_c),2)   AS air_temp_avg,
       ROUND(AVG(l.humidity_pct),2) AS humidity_avg
FROM plots p
JOIN soil_health_logs l ON l.plot_id = p.plot_id
JOIN (SELECT plot_id, MAX(log_ts) AS last_ts FROM soil_health_logs GROUP BY plot_id) m
     ON m.plot_id = p.plot_id
WHERE l.log_ts >= m.last_ts - INTERVAL 7 DAY
  AND (l.quality_score IS NULL OR l.quality_score > 0)
  AND (l.moisture_pct IS NOT NULL OR l.air_temp_c IS NOT NULL)
GROUP BY p.plot_id, p.soil_type;

-- Most recent cultivation cycle per plot --------------------------------
CREATE OR REPLACE VIEW v_last_crop AS
SELECT plot_id, cycle_id, crop_id, crop_name, family, is_perennial, sow_date, status
FROM (
    SELECT cc.plot_id, cc.cycle_id, cc.crop_id, c.crop_name, c.family, c.is_perennial,
           cc.sow_date, cc.status,
           ROW_NUMBER() OVER (PARTITION BY cc.plot_id ORDER BY cc.sow_date DESC, cc.cycle_id DESC) AS rn
    FROM cultivation_cycles cc JOIN crops c ON c.crop_id = cc.crop_id
) t WHERE rn = 1;

-- RULE-BASED recommendation inside the database (Review-1 deliverable).
-- Each crop is scored by how many of the plot's 7-day average conditions
-- fall inside the crop's learned envelope; the crop grown last time is
-- excluded for annual crops (rotation).  The ML recommender in the web app
-- is the primary engine; this view is the explainable SQL baseline.
CREATE OR REPLACE VIEW v_recommended_crops AS
SELECT * FROM (
    SELECT s.plot_id, c.crop_id, c.crop_name, c.season, c.family,
           ( (s.moisture_avg  BETWEEN c.moisture_min  AND c.moisture_max)
           + (s.soil_temp_avg BETWEEN c.soil_temp_min AND c.soil_temp_max)
           + (s.air_temp_avg  BETWEEN c.air_temp_min  AND c.air_temp_max)
           + (s.humidity_avg  BETWEEN c.humidity_min  AND c.humidity_max)
           + COALESCE(ph.ph_value BETWEEN c.ph_min AND c.ph_max, 0) )            AS conditions_met,
           4 + (ph.ph_value IS NOT NULL)                                         AS conditions_checked,
           ROUND(100 * ( (s.moisture_avg  BETWEEN c.moisture_min  AND c.moisture_max)
                       + (s.soil_temp_avg BETWEEN c.soil_temp_min AND c.soil_temp_max)
                       + (s.air_temp_avg  BETWEEN c.air_temp_min  AND c.air_temp_max)
                       + (s.humidity_avg  BETWEEN c.humidity_min  AND c.humidity_max)
                       + COALESCE(ph.ph_value BETWEEN c.ph_min AND c.ph_max, 0) )
                 / (4 + (ph.ph_value IS NOT NULL)), 1)                            AS match_pct,
           -- tie-break: normalised distance from the centre of each envelope
           ROUND( ABS(s.moisture_avg - (c.moisture_min + c.moisture_max)/2) / (c.moisture_max - c.moisture_min + 0.01)
                + ABS(s.air_temp_avg - (c.air_temp_min + c.air_temp_max)/2)  / (c.air_temp_max - c.air_temp_min + 0.01)
                + ABS(s.humidity_avg - (c.humidity_min + c.humidity_max)/2)  / (c.humidity_max - c.humidity_min + 0.01), 3)
                                                                                  AS centre_distance,
           s.window_end AS based_on_readings_until
    FROM v_plot_recent_conditions s
    CROSS JOIN crops c
    LEFT JOIN v_latest_ph ph ON ph.plot_id = s.plot_id
    LEFT JOIN v_last_crop lc ON lc.plot_id = s.plot_id
    WHERE NOT (lc.crop_id <=> c.crop_id AND c.is_perennial = FALSE)
) x
ORDER BY plot_id, match_pct DESC, centre_distance ASC;

-- Performance of every cultivation cycle (inputs -> output traceability)
CREATE OR REPLACE VIEW v_cycle_performance AS
SELECT cc.cycle_id, cc.plot_id, p.plot_name, f.full_name AS farmer, c.crop_name, c.season,
       cc.sow_date, cc.harvest_date, cc.status, p.area_acres,
       DATEDIFF(cc.harvest_date, cc.sow_date)                       AS actual_days,
       c.maturation_days                                            AS catalogue_days,
       cc.yield_kg,
       ROUND(cc.yield_kg / p.area_acres, 1)                         AS yield_kg_per_acre,
       c.expected_yield_kg_acre,
       ROUND(100 * cc.yield_kg / p.area_acres / c.expected_yield_kg_acre, 1) AS yield_efficiency_pct,
       COALESCE(t.n_kg, 0)                                          AS n_applied_kg,
       COALESCE(t.cost_inr, 0)                                      AS input_cost_inr,
       ROUND(COALESCE(t.cost_inr, 0) / NULLIF(cc.yield_kg, 0), 2)   AS input_cost_per_kg_yield
FROM cultivation_cycles cc
JOIN plots p   ON p.plot_id = cc.plot_id
JOIN farmers f ON f.farmer_id = p.farmer_id
JOIN crops c   ON c.crop_id = cc.crop_id
LEFT JOIN (SELECT cycle_id, SUM(n_kg) AS n_kg, SUM(cost_inr) AS cost_inr
           FROM treatments GROUP BY cycle_id) t ON t.cycle_id = cc.cycle_id;

-- One row per plot for the dashboard -------------------------------------
CREATE OR REPLACE VIEW v_plot_overview AS
SELECT p.plot_id, p.plot_name, p.area_acres, p.soil_type, p.latitude, p.longitude,
       f.farmer_id, f.full_name AS farmer, f.village,
       (SELECT GROUP_CONCAT(n.node_id) FROM sensor_nodes n WHERE n.plot_id = p.plot_id AND n.is_active) AS nodes,
       ll.log_ts AS last_reading_ts, ll.moisture_pct, ll.soil_temp_c, ll.air_temp_c, ll.humidity_pct,
       ph.ph_value AS latest_ph,
       lc.crop_name AS last_crop, lc.status AS last_crop_status
FROM plots p
JOIN farmers f ON f.farmer_id = p.farmer_id
LEFT JOIN v_latest_soil_log ll ON ll.plot_id = p.plot_id
LEFT JOIN v_latest_ph ph       ON ph.plot_id = p.plot_id
LEFT JOIN v_last_crop lc       ON lc.plot_id = p.plot_id;

-- Sensor-node health -------------------------------------------------------
CREATE OR REPLACE VIEW v_node_health AS
SELECT n.node_id, n.plot_id, p.plot_name, n.firmware_ver, n.last_seen_ts, n.is_active,
       TIMESTAMPDIFF(MINUTE, n.last_seen_ts, NOW())                 AS minutes_since_seen,
       (SELECT COUNT(*) FROM soil_health_logs l
         WHERE l.node_id = n.node_id AND l.log_ts >= n.last_seen_ts - INTERVAL 1 DAY) AS readings_last_24h,
       (SELECT ROUND(AVG(l.quality_score), 3) FROM soil_health_logs l
         WHERE l.node_id = n.node_id AND l.log_ts >= n.last_seen_ts - INTERVAL 7 DAY) AS avg_quality_7d
FROM sensor_nodes n JOIN plots p ON p.plot_id = n.plot_id;
