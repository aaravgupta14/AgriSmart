-- =====================================================================
-- AgriSmart : Smart Farm & Crop Recommendation System
-- 01_schema.sql  -  tables (3NF)                       MySQL 8.0.16+
-- ---------------------------------------------------------------------
-- Changes from the Review-1 ER diagram
--   * N-P-K columns removed (no NPK sensor on the node). Nitrogen status
--     is instead ESTIMATED from cultivation + treatment history
--     (fn_virtual_n_index, see 03_routines.sql).
--   * Soil_Health_Logs gains air_temp_c / humidity_pct (DHT11), a
--     `source` column (SENSOR / MANUAL) and a data-quality score.
--   * Crops stores learned suitability envelopes (P5-P95 of the training
--     data) instead of n_req/p_req/k_req.
--   * Treatments records fertiliser N/P/K actually applied.
--   * Recommendation_Runs / Recommendation_Items store every ML
--     recommendation for audit (which reading -> which advice).
-- =====================================================================

CREATE DATABASE IF NOT EXISTS agrismart
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
USE agrismart;

SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS recommendation_items, recommendation_runs, treatments,
                     cultivation_cycles, soil_health_logs, sensor_nodes,
                     plots, crops, farmers;
SET FOREIGN_KEY_CHECKS = 1;

-- ---------------------------------------------------------------- FARMERS
CREATE TABLE farmers (
    farmer_id   INT AUTO_INCREMENT PRIMARY KEY,
    full_name   VARCHAR(100) NOT NULL,
    phone_no    CHAR(10)     NOT NULL UNIQUE,
    village     VARCHAR(80)  NOT NULL,
    coop_id     VARCHAR(20)  NOT NULL,
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_phone CHECK (phone_no REGEXP '^[6-9][0-9]{9}$')
) ENGINE = InnoDB;

-- ------------------------------------------------------------------ PLOTS
CREATE TABLE plots (
    plot_id     INT AUTO_INCREMENT PRIMARY KEY,
    farmer_id   INT           NOT NULL,
    plot_name   VARCHAR(60)   NOT NULL,
    area_acres  DECIMAL(6,2)  NOT NULL,
    soil_type   ENUM('Alluvial','Black','Red','Laterite','Sandy') NOT NULL,
    latitude    DECIMAL(9,6),
    longitude   DECIMAL(9,6),
    CONSTRAINT fk_plot_farmer FOREIGN KEY (farmer_id) REFERENCES farmers(farmer_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT chk_area CHECK (area_acres > 0 AND area_acres <= 500),
    CONSTRAINT chk_lat  CHECK (latitude  BETWEEN -90  AND 90),
    CONSTRAINT chk_lon  CHECK (longitude BETWEEN -180 AND 180),
    UNIQUE KEY uq_plot_name (farmer_id, plot_name)
) ENGINE = InnoDB;

-- ------------------------------------------------------------------ CROPS
CREATE TABLE crops (
    crop_id                 INT AUTO_INCREMENT PRIMARY KEY,
    crop_name               VARCHAR(40)  NOT NULL UNIQUE,
    season                  ENUM('Kharif','Rabi','Zaid','Perennial') NOT NULL,
    family                  VARCHAR(20)  NOT NULL,
    n_demand                ENUM('HIGH','MEDIUM','LOW','FIXER') NOT NULL,
    is_perennial            BOOLEAN      NOT NULL DEFAULT FALSE,
    maturation_days         SMALLINT     NOT NULL,
    expected_yield_kg_acre  DECIMAL(8,1) NOT NULL,
    -- learned suitability envelopes (P5 - P95 of training data)
    ph_min        DECIMAL(4,2), ph_max        DECIMAL(4,2),
    moisture_min  DECIMAL(5,2), moisture_max  DECIMAL(5,2),
    soil_temp_min DECIMAL(5,2), soil_temp_max DECIMAL(5,2),
    air_temp_min  DECIMAL(5,2), air_temp_max  DECIMAL(5,2),
    humidity_min  DECIMAL(5,2), humidity_max  DECIMAL(5,2),
    CONSTRAINT chk_ph_rng  CHECK (ph_min <= ph_max),
    CONSTRAINT chk_days    CHECK (maturation_days > 0)
) ENGINE = InnoDB;

-- ----------------------------------------------------------- SENSOR_NODES
CREATE TABLE sensor_nodes (
    node_id       VARCHAR(20) PRIMARY KEY,            -- e.g. 'NODE-01' (printed by the Arduino)
    plot_id       INT          NOT NULL,
    mac_address   VARCHAR(17)  UNIQUE,                -- Uno has no MAC: optional / USB serial no.
    firmware_ver  VARCHAR(10)  NOT NULL DEFAULT '1.0.0',
    installed_on  DATE         NOT NULL,
    last_seen_ts  DATETIME,
    is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
    -- which sensors are wired to this node (chosen at registration)
    sensors       SET('MOISTURE','AIR_TEMP_HUMIDITY','SOIL_TEMP','PH') NOT NULL
                  DEFAULT 'MOISTURE,AIR_TEMP_HUMIDITY,SOIL_TEMP',
    CONSTRAINT fk_node_plot FOREIGN KEY (plot_id) REFERENCES plots(plot_id)
        ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE = InnoDB;

-- ------------------------------------------------------- SOIL_HEALTH_LOGS
CREATE TABLE soil_health_logs (
    log_id         BIGINT AUTO_INCREMENT PRIMARY KEY,
    plot_id        INT          NOT NULL,
    node_id        VARCHAR(20),                       -- NULL for manual entries
    log_ts         DATETIME     NOT NULL,
    source         ENUM('SENSOR','MANUAL') NOT NULL DEFAULT 'SENSOR',
    moisture_pct   DECIMAL(5,2),                      -- capacitive sensor, volumetric %
    soil_temp_c    DECIMAL(6,3),                      -- DS18B20
    air_temp_c     DECIMAL(5,2),                      -- DHT11
    humidity_pct   DECIMAL(5,2),                      -- DHT11
    ph_value       DECIMAL(4,2),                      -- manual / optional probe
    quality_score  DECIMAL(4,3),                      -- filled by the app (QW-TC)
    note           VARCHAR(200),
    CONSTRAINT fk_log_plot FOREIGN KEY (plot_id) REFERENCES plots(plot_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_log_node FOREIGN KEY (node_id) REFERENCES sensor_nodes(node_id)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT chk_moist CHECK (moisture_pct IS NULL OR moisture_pct BETWEEN 0 AND 100),
    CONSTRAINT chk_stemp CHECK (soil_temp_c  IS NULL OR soil_temp_c  BETWEEN -10 AND 70),
    CONSTRAINT chk_atemp CHECK (air_temp_c   IS NULL OR air_temp_c   BETWEEN -10 AND 60),
    CONSTRAINT chk_hum   CHECK (humidity_pct IS NULL OR humidity_pct BETWEEN 0 AND 100),
    CONSTRAINT chk_ph    CHECK (ph_value     IS NULL OR ph_value     BETWEEN 3 AND 10),
    CONSTRAINT chk_any   CHECK (moisture_pct IS NOT NULL OR soil_temp_c IS NOT NULL OR
                                air_temp_c IS NOT NULL OR humidity_pct IS NOT NULL OR ph_value IS NOT NULL),
    INDEX idx_plot_ts (plot_id, log_ts),
    INDEX idx_node_ts (node_id, log_ts)
) ENGINE = InnoDB;

-- ----------------------------------------------------- CULTIVATION_CYCLES
CREATE TABLE cultivation_cycles (
    cycle_id      INT AUTO_INCREMENT PRIMARY KEY,
    plot_id       INT          NOT NULL,
    crop_id       INT          NOT NULL,
    sow_date      DATE         NOT NULL,
    harvest_date  DATE,
    yield_kg      DECIMAL(10,2),
    status        ENUM('GROWING','HARVESTED','FAILED') NOT NULL DEFAULT 'GROWING',
    rec_run_id    BIGINT,                             -- recommendation that led to this sowing
    CONSTRAINT fk_cyc_plot FOREIGN KEY (plot_id) REFERENCES plots(plot_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_cyc_crop FOREIGN KEY (crop_id) REFERENCES crops(crop_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT chk_dates  CHECK (harvest_date IS NULL OR harvest_date >= sow_date),
    CONSTRAINT chk_yield  CHECK (yield_kg IS NULL OR yield_kg >= 0),
    CONSTRAINT chk_status CHECK ((status = 'GROWING' AND harvest_date IS NULL)
                              OR (status <> 'GROWING' AND harvest_date IS NOT NULL)),
    INDEX idx_plot_sow (plot_id, sow_date)
) ENGINE = InnoDB;

-- ------------------------------------------------------------- TREATMENTS
CREATE TABLE treatments (
    treatment_id  INT AUTO_INCREMENT PRIMARY KEY,
    cycle_id      INT          NOT NULL,
    applied_on    DATE         NOT NULL,
    category      ENUM('FERTILISER','MANURE','PESTICIDE','FUNGICIDE','HERBICIDE') NOT NULL,
    product_name  VARCHAR(60)  NOT NULL,
    qty_kg        DECIMAL(8,2) NOT NULL,
    n_kg          DECIMAL(7,2) NOT NULL DEFAULT 0,    -- nutrient actually supplied
    p_kg          DECIMAL(7,2) NOT NULL DEFAULT 0,
    k_kg          DECIMAL(7,2) NOT NULL DEFAULT 0,
    cost_inr      DECIMAL(10,2) NOT NULL DEFAULT 0,
    CONSTRAINT fk_trt_cycle FOREIGN KEY (cycle_id) REFERENCES cultivation_cycles(cycle_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT chk_qty CHECK (qty_kg > 0 AND n_kg >= 0 AND p_kg >= 0 AND k_kg >= 0 AND cost_inr >= 0)
) ENGINE = InnoDB;

-- ---------------------------------------------- RECOMMENDATION_RUNS / ITEMS
CREATE TABLE recommendation_runs (
    run_id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    plot_id         INT          NOT NULL,
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_used      ENUM('with_pH','without_pH') NOT NULL,
    status          ENUM('CONFIDENT','LOW_CONFIDENCE','OUT_OF_DISTRIBUTION','NO_VALID_DATA') NOT NULL,
    window_start    DATETIME,
    window_end      DATETIME,
    n_readings      INT          NOT NULL,
    n_readings_used INT          NOT NULL,
    mean_quality    DECIMAL(4,3),
    vnbi            DECIMAL(5,3),                     -- virtual nitrogen balance index
    season_at_run   ENUM('Kharif','Rabi','Zaid')  NOT NULL,
    CONSTRAINT fk_run_plot FOREIGN KEY (plot_id) REFERENCES plots(plot_id)
        ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE recommendation_items (
    run_id       BIGINT      NOT NULL,
    rank_no      TINYINT     NOT NULL,
    crop_id      INT         NOT NULL,
    p_ml         DECIMAL(6,4) NOT NULL,               -- calibrated model probability
    season_f     DECIMAL(4,2) NOT NULL,
    rotation_f   DECIMAL(4,2) NOT NULL,
    nutrient_f   DECIMAL(4,2) NOT NULL,
    final_score  DECIMAL(6,4) NOT NULL,
    PRIMARY KEY (run_id, rank_no),
    CONSTRAINT fk_item_run  FOREIGN KEY (run_id)  REFERENCES recommendation_runs(run_id) ON DELETE CASCADE,
    CONSTRAINT fk_item_crop FOREIGN KEY (crop_id) REFERENCES crops(crop_id)
) ENGINE = InnoDB;

ALTER TABLE cultivation_cycles
    ADD CONSTRAINT fk_cyc_run FOREIGN KEY (rec_run_id) REFERENCES recommendation_runs(run_id)
    ON DELETE SET NULL;
