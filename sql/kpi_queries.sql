-- Example KPI queries on the star schema in data/warehouse/dancake.db
-- (SQLite syntax; identical or near-identical in PostgreSQL / MySQL)

-- 1. Annual net sales, units and average order value
SELECT d.year,
       COUNT(*)                         AS orders,
       SUM(f.quantity)                  AS units,
       ROUND(SUM(f.net_sales_bdt), 0)   AS net_sales,
       ROUND(AVG(f.net_sales_bdt), 1)   AS avg_order_value
FROM fact_sales f JOIN dim_date d ON d.date_id = f.date_id
GROUP BY d.year ORDER BY d.year;

-- 2. Category share of net sales
SELECT p.category,
       ROUND(SUM(f.net_sales_bdt) * 100.0 / (SELECT SUM(net_sales_bdt) FROM fact_sales), 2) AS share_pct
FROM fact_sales f JOIN dim_product p ON p.product_id = f.product_id
GROUP BY p.category ORDER BY share_pct DESC;

-- 3. Region x channel matrix with average units per order
SELECT s.region, s.channel,
       ROUND(SUM(f.net_sales_bdt) / 1e6, 2) AS net_sales_m,
       ROUND(AVG(f.quantity), 2)            AS units_per_order
FROM fact_sales f JOIN dim_segment s ON s.segment_id = f.segment_id
GROUP BY s.region, s.channel ORDER BY net_sales_m DESC;

-- 4. Month-of-year seasonality (average monthly sales over complete years)
SELECT d.month, ROUND(AVG(m.net) / 1e6, 2) AS avg_net_sales_m
FROM (SELECT d.year, d.month, SUM(f.net_sales_bdt) AS net
      FROM fact_sales f JOIN dim_date d ON d.date_id = f.date_id
      WHERE d.year BETWEEN 2020 AND 2025
      GROUP BY d.year, d.month) m
JOIN (SELECT DISTINCT month FROM dim_date) d ON d.month = m.month
GROUP BY d.month ORDER BY d.month;

-- 5. Top 10 products with year-over-year growth 2024 -> 2025
WITH y AS (
  SELECT f.product_id, d.year, SUM(f.net_sales_bdt) AS net
  FROM fact_sales f JOIN dim_date d ON d.date_id = f.date_id
  GROUP BY f.product_id, d.year)
SELECT p.product_name, ROUND(a.net / 1e6, 2) AS net_2025_m,
       ROUND((a.net / b.net - 1) * 100, 1)    AS growth_pct
FROM y a JOIN y b ON a.product_id = b.product_id AND a.year = 2025 AND b.year = 2024
JOIN dim_product p ON p.product_id = a.product_id
ORDER BY a.net DESC LIMIT 10;

-- 6. Discount cost by channel
SELECT s.channel, ROUND(SUM(f.discount_bdt) / 1e6, 2) AS discount_m,
       ROUND(SUM(f.discount_bdt) * 100.0 / SUM(f.gross_sales_bdt), 2) AS discount_rate_pct
FROM fact_sales f JOIN dim_segment s ON s.segment_id = f.segment_id
GROUP BY s.channel;
