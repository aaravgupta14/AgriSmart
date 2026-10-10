-- =====================================================================
-- AgriSmart  03_routines.sql  -  triggers, stored procedures, function
-- Run with the mysql client (handles DELIMITER) or scripts/setup_database.py
-- =====================================================================
USE agrismart;

DROP TRIGGER  IF EXISTS trg_cycle_rotation_check;
DROP TRIGGER  IF EXISTS trg_soil_log_before_insert;
DROP TRIGGER  IF EXISTS trg_soil_log_after_insert;
DROP PROCEDURE IF EXISTS sp_yield_report;
DROP PROCEDURE IF EXISTS sp_plot_history;
DROP PROCEDURE IF EXISTS sp_start_cycle;
DROP PROCEDURE IF EXISTS sp_record_harvest;
DROP FUNCTION  IF EXISTS fn_virtual_n_index;

DELIMITER $$

-- ---------------------------------------------------------------------
-- TRIGGER 1: crop-rotation rule enforced at data-entry time
--   * an ANNUAL crop may not follow itself on the same plot
--   * a plot may not have two GROWING cycles at once
--   * a new cycle may not start before the previous one was harvested
-- ---------------------------------------------------------------------
CREATE TRIGGER trg_cycle_rotation_check
BEFORE INSERT ON cultivation_cycles
FOR EACH ROW
BEGIN
    DECLARE v_last_crop INT;
    DECLARE v_last_harvest DATE;
    DECLARE v_perennial BOOLEAN;
    DECLARE v_growing INT;
    DECLARE v_msg VARCHAR(255);

    SELECT COUNT(*) INTO v_growing
    FROM cultivation_cycles WHERE plot_id = NEW.plot_id AND status = 'GROWING';
    IF v_growing > 0 AND NEW.status = 'GROWING' THEN
        SIGNAL SQLSTATE '45000'
            SET MESSAGE_TEXT = 'Plot already has a GROWING cycle - record its harvest first.';
    END IF;

    SELECT crop_id, harvest_date INTO v_last_crop, v_last_harvest
    FROM cultivation_cycles
    WHERE plot_id = NEW.plot_id AND sow_date <= NEW.sow_date
    ORDER BY sow_date DESC, cycle_id DESC LIMIT 1;

    SELECT is_perennial INTO v_perennial FROM crops WHERE crop_id = NEW.crop_id;

    IF v_last_crop IS NOT NULL AND v_last_crop = NEW.crop_id AND v_perennial = FALSE THEN
        SET v_msg = CONCAT('Crop rotation violation: ',
                           (SELECT crop_name FROM crops WHERE crop_id = NEW.crop_id),
                           ' was the previous crop on plot ', NEW.plot_id,
                           '. Choose a different crop (ideally another family).');
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = v_msg;
    END IF;

    IF v_last_harvest IS NOT NULL AND NEW.sow_date < v_last_harvest THEN
        SIGNAL SQLSTATE '45000'
            SET MESSAGE_TEXT = 'Sowing date is before the previous cycle''s harvest date.';
    END IF;
END$$

-- ---------------------------------------------------------------------
-- TRIGGER 2: clean sensor rows before they are stored
--   * DS18B20 fault codes (-127 = disconnected, 85 = power-on reset) -> NULL
--   * plot_id is always taken from the node (prevents mis-attribution)
-- ---------------------------------------------------------------------
CREATE TRIGGER trg_soil_log_before_insert
BEFORE INSERT ON soil_health_logs
FOR EACH ROW
BEGIN
    IF NEW.soil_temp_c IN (-127, 85) THEN
        SET NEW.soil_temp_c = NULL;
        SET NEW.note = CONCAT_WS('; ', NEW.note, 'DS18B20 fault code removed');
    END IF;
    IF NEW.node_id IS NOT NULL THEN
        SET NEW.plot_id = (SELECT plot_id FROM sensor_nodes WHERE node_id = NEW.node_id);
        SET NEW.source = 'SENSOR';
    END IF;
    IF NEW.log_ts IS NULL THEN
        SET NEW.log_ts = NOW();
    END IF;
END$$

-- TRIGGER 3: keep sensor_nodes.last_seen_ts up to date
CREATE TRIGGER trg_soil_log_after_insert
AFTER INSERT ON soil_health_logs
FOR EACH ROW
BEGIN
    IF NEW.node_id IS NOT NULL THEN
        UPDATE sensor_nodes
           SET last_seen_ts = GREATEST(COALESCE(last_seen_ts, NEW.log_ts), NEW.log_ts)
         WHERE node_id = NEW.node_id;
    END IF;
END$$

-- ---------------------------------------------------------------------
-- FUNCTION: Virtual Nitrogen Balance Index (novelty N4, no NPK sensor)
--   last 3 cycles, weights 1, 0.6, 0.36
--   crop effect : FIXER +1, LOW 0, MEDIUM -0.5, HIGH -1
--   fertiliser  : + LEAST(1, N kg per acre / 40)
--   (identical to recommender.virtual_nitrogen_index in Python)
-- ---------------------------------------------------------------------
CREATE FUNCTION fn_virtual_n_index(p_plot INT)
RETURNS DECIMAL(5,3)
READS SQL DATA
BEGIN
    DECLARE v_idx DECIMAL(6,3) DEFAULT 0;
    SELECT COALESCE(SUM(w * (crop_effect + LEAST(1, n_per_acre / 40))), 0) INTO v_idx
    FROM (
        SELECT POW(0.6, ROW_NUMBER() OVER (ORDER BY cc.sow_date DESC, cc.cycle_id DESC) - 1) AS w,
               CASE c.n_demand WHEN 'FIXER' THEN 1 WHEN 'LOW' THEN 0
                               WHEN 'MEDIUM' THEN -0.5 ELSE -1 END                 AS crop_effect,
               COALESCE((SELECT SUM(t.n_kg) FROM treatments t WHERE t.cycle_id = cc.cycle_id), 0)
                   / p.area_acres                                                  AS n_per_acre,
               ROW_NUMBER() OVER (ORDER BY cc.sow_date DESC, cc.cycle_id DESC)     AS rn
        FROM cultivation_cycles cc
        JOIN crops c ON c.crop_id = cc.crop_id
        JOIN plots p ON p.plot_id = cc.plot_id
        WHERE cc.plot_id = p_plot
    ) h
    WHERE rn <= 3;
    RETURN v_idx;
END$$

-- ---------------------------------------------------------------------
-- PROCEDURE: yield report for a crop and year (Review-1 deliverable)
--   total yield, actual vs catalogue maturation time, yield per acre,
--   efficiency vs expected yield, input cost per kg of produce
-- ---------------------------------------------------------------------
CREATE PROCEDURE sp_yield_report(IN p_crop VARCHAR(40), IN p_year INT)
BEGIN
    IF NOT EXISTS (SELECT 1 FROM crops WHERE crop_name = p_crop) THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Unknown crop name.';
    END IF;

    -- result set 1: one row per harvested cycle
    SELECT cycle_id, plot_name, farmer, sow_date, harvest_date, area_acres,
           actual_days, catalogue_days, actual_days - catalogue_days AS days_vs_catalogue,
           yield_kg, yield_kg_per_acre, yield_efficiency_pct, n_applied_kg, input_cost_inr,
           input_cost_per_kg_yield
    FROM v_cycle_performance
    WHERE crop_name = p_crop AND status = 'HARVESTED' AND YEAR(harvest_date) = p_year
    ORDER BY yield_kg_per_acre DESC;

    -- result set 2: totals
    SELECT p_crop AS crop, p_year AS year,
           COUNT(*)                                  AS cycles,
           ROUND(SUM(yield_kg), 1)                   AS total_yield_kg,
           ROUND(SUM(area_acres), 2)                 AS total_area_acres,
           ROUND(SUM(yield_kg) / SUM(area_acres), 1) AS yield_kg_per_acre,
           ROUND(AVG(actual_days), 1)                AS avg_actual_days,
           MAX(catalogue_days)                       AS catalogue_days,
           ROUND(AVG(yield_efficiency_pct), 1)       AS avg_efficiency_pct,
           ROUND(SUM(input_cost_inr) / NULLIF(SUM(yield_kg), 0), 2) AS input_cost_per_kg
    FROM v_cycle_performance
    WHERE crop_name = p_crop AND status = 'HARVESTED' AND YEAR(harvest_date) = p_year;
END$$

-- PROCEDURE: complete history of one plot
CREATE PROCEDURE sp_plot_history(IN p_plot INT)
BEGIN
    SELECT * FROM v_cycle_performance WHERE plot_id = p_plot ORDER BY sow_date;
    SELECT fn_virtual_n_index(p_plot) AS virtual_n_index;
END$$

-- PROCEDURE: start a new cycle (rotation trigger fires inside)
CREATE PROCEDURE sp_start_cycle(IN p_plot INT, IN p_crop VARCHAR(40), IN p_sow DATE, IN p_run BIGINT)
BEGIN
    DECLARE v_crop INT;
    SELECT crop_id INTO v_crop FROM crops WHERE crop_name = p_crop;
    IF v_crop IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Unknown crop name.';
    END IF;
    INSERT INTO cultivation_cycles (plot_id, crop_id, sow_date, status, rec_run_id)
    VALUES (p_plot, v_crop, p_sow, 'GROWING', p_run);
    SELECT LAST_INSERT_ID() AS cycle_id;
END$$

-- PROCEDURE: record a harvest atomically
CREATE PROCEDURE sp_record_harvest(IN p_cycle INT, IN p_date DATE, IN p_yield DECIMAL(10,2))
BEGIN
    DECLARE v_status VARCHAR(10);
    DECLARE v_sow DATE;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION BEGIN ROLLBACK; RESIGNAL; END;

    START TRANSACTION;
    SELECT status, sow_date INTO v_status, v_sow
    FROM cultivation_cycles WHERE cycle_id = p_cycle FOR UPDATE;
    IF v_status IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Cycle not found.';
    ELSEIF v_status <> 'GROWING' THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Cycle is already closed.';
    ELSEIF p_date < v_sow THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Harvest date is before sowing date.';
    END IF;
    UPDATE cultivation_cycles
       SET harvest_date = p_date, yield_kg = p_yield,
           status = IF(p_yield > 0, 'HARVESTED', 'FAILED')
     WHERE cycle_id = p_cycle;
    COMMIT;
END$$

DELIMITER ;
