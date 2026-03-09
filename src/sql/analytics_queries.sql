-- ═══════════════════════════════════════════════════════════════════════════
-- E-Commerce Revenue Intelligence — SQL Analytics Queries
-- Database: PostgreSQL 14+
-- ═══════════════════════════════════════════════════════════════════════════

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE SETUP (PostgreSQL DDL)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS olist_customers (
    customer_id         VARCHAR(50) PRIMARY KEY,
    customer_unique_id  VARCHAR(50),
    customer_zip_code   VARCHAR(10),
    customer_city       VARCHAR(100),
    customer_state      CHAR(2),
    customer_segment    VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS olist_orders (
    order_id                        VARCHAR(50) PRIMARY KEY,
    customer_id                     VARCHAR(50) REFERENCES olist_customers(customer_id),
    order_status                    VARCHAR(20),
    order_purchase_timestamp        TIMESTAMPTZ,
    order_approved_at               TIMESTAMPTZ,
    order_delivered_carrier_date    TIMESTAMPTZ,
    order_delivered_customer_date   TIMESTAMPTZ,
    order_estimated_delivery_date   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS olist_order_items (
    order_id            VARCHAR(50) REFERENCES olist_orders(order_id),
    order_item_id       INT,
    product_id          VARCHAR(50),
    seller_id           VARCHAR(50),
    shipping_limit_date TIMESTAMPTZ,
    price               NUMERIC(10,2),
    freight_value       NUMERIC(10,2),
    PRIMARY KEY (order_id, order_item_id)
);

CREATE TABLE IF NOT EXISTS olist_order_payments (
    order_id                VARCHAR(50) REFERENCES olist_orders(order_id),
    payment_sequential      INT,
    payment_type            VARCHAR(20),
    payment_installments    INT,
    payment_value           NUMERIC(10,2),
    PRIMARY KEY (order_id, payment_sequential)
);

CREATE TABLE IF NOT EXISTS olist_products (
    product_id              VARCHAR(50) PRIMARY KEY,
    product_category_name   VARCHAR(100),
    product_weight_g        INT,
    unit_price              NUMERIC(10,2)
);

CREATE TABLE IF NOT EXISTS olist_sellers (
    seller_id       VARCHAR(50) PRIMARY KEY,
    seller_city     VARCHAR(100),
    seller_state    CHAR(2),
    seller_rating   NUMERIC(3,2)
);

CREATE TABLE IF NOT EXISTS olist_order_reviews (
    review_id               VARCHAR(50),
    order_id                VARCHAR(50) REFERENCES olist_orders(order_id),
    review_score            INT CHECK (review_score BETWEEN 1 AND 5),
    review_comment_message  TEXT,
    review_creation_date    TIMESTAMPTZ,
    PRIMARY KEY (review_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_orders_customer   ON olist_orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_status     ON olist_orders(order_status);
CREATE INDEX IF NOT EXISTS idx_orders_timestamp  ON olist_orders(order_purchase_timestamp);
CREATE INDEX IF NOT EXISTS idx_items_product     ON olist_order_items(product_id);
CREATE INDEX IF NOT EXISTS idx_items_seller      ON olist_order_items(seller_id);


-- ═══════════════════════════════════════════════════════════════════════════
-- QUERY 1: Daily Revenue with 7-Day Moving Average (Window Function)
-- ═══════════════════════════════════════════════════════════════════════════
WITH daily_revenue AS (
    SELECT
        DATE_TRUNC('day', o.order_purchase_timestamp)::DATE  AS order_date,
        COUNT(DISTINCT o.order_id)                           AS total_orders,
        SUM(p.payment_value)                                 AS total_revenue,
        AVG(p.payment_value)                                 AS avg_order_value,
        COUNT(DISTINCT o.customer_id)                        AS unique_customers
    FROM olist_orders o
    JOIN olist_order_payments p USING (order_id)
    WHERE o.order_status = 'delivered'
    GROUP BY 1
),
revenue_with_ma AS (
    SELECT
        order_date,
        total_orders,
        total_revenue,
        avg_order_value,
        unique_customers,
        -- 7-day moving average
        AVG(total_revenue) OVER (
            ORDER BY order_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS ma_7d_revenue,
        -- 30-day moving average
        AVG(total_revenue) OVER (
            ORDER BY order_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS ma_30d_revenue,
        -- Day-over-day change
        LAG(total_revenue) OVER (ORDER BY order_date) AS prev_day_revenue,
        total_revenue - LAG(total_revenue) OVER (ORDER BY order_date) AS revenue_delta,
        -- Cumulative revenue
        SUM(total_revenue) OVER (ORDER BY order_date) AS cumulative_revenue,
        -- Week-over-week
        LAG(total_revenue, 7) OVER (ORDER BY order_date) AS revenue_wow_ago
    FROM daily_revenue
)
SELECT
    order_date,
    total_orders,
    ROUND(total_revenue, 2)      AS total_revenue,
    ROUND(avg_order_value, 2)    AS avg_order_value,
    unique_customers,
    ROUND(ma_7d_revenue, 2)      AS ma_7d,
    ROUND(ma_30d_revenue, 2)     AS ma_30d,
    ROUND(revenue_delta, 2)      AS daily_delta,
    ROUND(
        CASE WHEN prev_day_revenue > 0
             THEN (revenue_delta / prev_day_revenue) * 100
             ELSE NULL END, 2
    )                            AS pct_change,
    ROUND(cumulative_revenue, 2) AS cumulative_revenue,
    ROUND(
        CASE WHEN revenue_wow_ago > 0
             THEN ((total_revenue - revenue_wow_ago) / revenue_wow_ago) * 100
             ELSE NULL END, 2
    )                            AS wow_growth_pct
FROM revenue_with_ma
ORDER BY order_date;


-- ═══════════════════════════════════════════════════════════════════════════
-- QUERY 2: RFM Segmentation (pure SQL)
-- ═══════════════════════════════════════════════════════════════════════════
WITH customer_orders AS (
    SELECT
        c.customer_unique_id                                           AS customer_id,
        MAX(o.order_purchase_timestamp)                                AS last_purchase,
        COUNT(DISTINCT o.order_id)                                     AS frequency,
        SUM(p.payment_value)                                           AS monetary
    FROM olist_orders o
    JOIN olist_customers c USING (customer_id)
    JOIN olist_order_payments p USING (order_id)
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
),
rfm_raw AS (
    SELECT
        customer_id,
        EXTRACT(DAY FROM NOW() - last_purchase)::INT  AS recency,
        frequency,
        ROUND(monetary::NUMERIC, 2)                   AS monetary
    FROM customer_orders
),
rfm_scored AS (
    SELECT
        customer_id, recency, frequency, monetary,
        NTILE(5) OVER (ORDER BY recency DESC)   AS r_score,
        NTILE(5) OVER (ORDER BY frequency)      AS f_score,
        NTILE(5) OVER (ORDER BY monetary)       AS m_score
    FROM rfm_raw
),
rfm_segmented AS (
    SELECT *,
        r_score * 100 + f_score * 10 + m_score AS rfm_score,
        CASE
            WHEN r_score >= 4 AND f_score >= 4              THEN 'Champions'
            WHEN r_score >= 3 AND f_score >= 3              THEN 'Loyal Customers'
            WHEN r_score >= 4 AND f_score < 3               THEN 'Potential Loyalist'
            WHEN r_score <= 2 AND f_score >= 4              THEN 'At Risk'
            WHEN r_score = 1  AND f_score <= 2              THEN 'Lost'
            WHEN r_score >= 4 AND f_score = 1               THEN 'New Customers'
            ELSE 'Need Attention'
        END AS segment
    FROM rfm_scored
)
SELECT
    segment,
    COUNT(*)                    AS customer_count,
    ROUND(AVG(recency), 0)     AS avg_recency_days,
    ROUND(AVG(frequency), 1)   AS avg_frequency,
    ROUND(AVG(monetary), 2)    AS avg_monetary,
    ROUND(SUM(monetary), 2)    AS total_revenue,
    ROUND(
        SUM(monetary) / SUM(SUM(monetary)) OVER () * 100, 1
    )                           AS revenue_share_pct
FROM rfm_segmented
GROUP BY segment
ORDER BY total_revenue DESC;


-- ═══════════════════════════════════════════════════════════════════════════
-- QUERY 3: Cohort Retention Analysis (CTE + Pivot)
-- ═══════════════════════════════════════════════════════════════════════════
WITH first_orders AS (
    SELECT
        c.customer_unique_id,
        DATE_TRUNC('month', MIN(o.order_purchase_timestamp)) AS cohort_month
    FROM olist_orders o
    JOIN olist_customers c USING (customer_id)
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
),
customer_activity AS (
    SELECT
        c.customer_unique_id,
        DATE_TRUNC('month', o.order_purchase_timestamp) AS activity_month
    FROM olist_orders o
    JOIN olist_customers c USING (customer_id)
    WHERE o.order_status = 'delivered'
),
cohort_data AS (
    SELECT
        f.cohort_month,
        EXTRACT(YEAR FROM AGE(a.activity_month, f.cohort_month)) * 12 +
        EXTRACT(MONTH FROM AGE(a.activity_month, f.cohort_month)) AS period,
        COUNT(DISTINCT f.customer_unique_id) AS customers
    FROM first_orders f
    JOIN customer_activity a USING (customer_unique_id)
    WHERE a.activity_month >= f.cohort_month
    GROUP BY 1, 2
),
cohort_sizes AS (
    SELECT cohort_month, customers AS cohort_size
    FROM cohort_data
    WHERE period = 0
)
SELECT
    cd.cohort_month,
    cs.cohort_size,
    cd.period,
    cd.customers,
    ROUND(cd.customers::NUMERIC / cs.cohort_size * 100, 1) AS retention_rate_pct
FROM cohort_data cd
JOIN cohort_sizes cs USING (cohort_month)
WHERE cd.period <= 12
ORDER BY cd.cohort_month, cd.period;


-- ═══════════════════════════════════════════════════════════════════════════
-- QUERY 4: Revenue Anomaly Detection (Z-Score in SQL)
-- ═══════════════════════════════════════════════════════════════════════════
WITH daily_revenue AS (
    SELECT
        DATE_TRUNC('day', order_purchase_timestamp)::DATE AS dt,
        SUM(p.payment_value)  AS revenue,
        COUNT(DISTINCT o.order_id) AS orders
    FROM olist_orders o
    JOIN olist_order_payments p USING (order_id)
    WHERE o.order_status IN ('delivered','shipped')
    GROUP BY 1
),
stats AS (
    SELECT
        AVG(revenue) AS global_mean,
        STDDEV(revenue) AS global_std
    FROM daily_revenue
),
scored AS (
    SELECT
        dr.dt,
        dr.revenue,
        dr.orders,
        s.global_mean,
        s.global_std,
        (dr.revenue - s.global_mean) / NULLIF(s.global_std, 0) AS z_score,
        -- Rolling 7-day mean/std
        AVG(dr.revenue) OVER (ORDER BY dr.dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS roll_mean_7d,
        STDDEV(dr.revenue) OVER (ORDER BY dr.dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS roll_std_7d,
        LAG(dr.revenue, 1) OVER (ORDER BY dr.dt) AS prev_day
    FROM daily_revenue dr, stats s
)
SELECT
    dt,
    ROUND(revenue, 2)        AS revenue,
    orders,
    ROUND(z_score::NUMERIC, 3)     AS z_score,
    ROUND(roll_mean_7d::NUMERIC, 2)  AS roll_mean_7d,
    CASE
        WHEN ABS(z_score) > 3.0   THEN 'CRITICAL'
        WHEN ABS(z_score) > 2.5   THEN 'HIGH'
        WHEN ABS(z_score) > 2.0   THEN 'MEDIUM'
        ELSE 'NORMAL'
    END                      AS severity,
    CASE
        WHEN z_score >  2.0  THEN 'SPIKE'
        WHEN z_score < -2.0  THEN 'DROP'
        ELSE 'NORMAL'
    END                      AS anomaly_type,
    ABS(z_score) > 2.0       AS is_anomaly
FROM scored
ORDER BY dt;


-- ═══════════════════════════════════════════════════════════════════════════
-- QUERY 5: Root Cause Attribution — Revenue Drop by Category
-- ═══════════════════════════════════════════════════════════════════════════
WITH period_revenue AS (
    SELECT
        pr.product_category_name,
        DATE_TRUNC('week', o.order_purchase_timestamp) AS week,
        SUM(oi.price)                                   AS weekly_revenue,
        COUNT(DISTINCT o.order_id)                      AS orders
    FROM olist_orders o
    JOIN olist_order_items oi USING (order_id)
    JOIN olist_products pr USING (product_id)
    WHERE o.order_status IN ('delivered','shipped')
    GROUP BY 1, 2
),
week_comparison AS (
    SELECT
        product_category_name,
        week,
        weekly_revenue,
        orders,
        LAG(weekly_revenue) OVER (
            PARTITION BY product_category_name ORDER BY week
        ) AS prev_week_revenue,
        LAG(orders) OVER (
            PARTITION BY product_category_name ORDER BY week
        ) AS prev_week_orders
    FROM period_revenue
)
SELECT
    product_category_name,
    week,
    ROUND(weekly_revenue, 2)       AS this_week_revenue,
    ROUND(prev_week_revenue, 2)    AS last_week_revenue,
    ROUND(weekly_revenue - prev_week_revenue, 2) AS revenue_delta,
    ROUND(
        CASE WHEN prev_week_revenue > 0
             THEN (weekly_revenue - prev_week_revenue) / prev_week_revenue * 100
             ELSE NULL END, 1
    )                              AS pct_change,
    orders                         AS this_week_orders,
    prev_week_orders               AS last_week_orders
FROM week_comparison
WHERE prev_week_revenue IS NOT NULL
ORDER BY revenue_delta ASC
LIMIT 20;


-- ═══════════════════════════════════════════════════════════════════════════
-- QUERY 6: Customer Lifetime Value by Cohort
-- ═══════════════════════════════════════════════════════════════════════════
WITH cohort_revenue AS (
    SELECT
        c.customer_unique_id,
        DATE_TRUNC('month', MIN(o.order_purchase_timestamp)) OVER (
            PARTITION BY c.customer_unique_id
        ) AS cohort_month,
        DATE_TRUNC('month', o.order_purchase_timestamp)  AS order_month,
        p.payment_value
    FROM olist_orders o
    JOIN olist_customers c USING (customer_id)
    JOIN olist_order_payments p USING (order_id)
    WHERE o.order_status = 'delivered'
),
monthly_ltv AS (
    SELECT
        cohort_month,
        EXTRACT(YEAR FROM AGE(order_month, cohort_month)) * 12 +
        EXTRACT(MONTH FROM AGE(order_month, cohort_month)) AS months_since_first,
        COUNT(DISTINCT customer_unique_id) AS active_customers,
        SUM(payment_value)                 AS period_revenue,
        AVG(payment_value)                 AS avg_payment
    FROM cohort_revenue
    GROUP BY cohort_month, months_since_first
),
cohort_start AS (
    SELECT cohort_month, active_customers AS cohort_size
    FROM monthly_ltv WHERE months_since_first = 0
)
SELECT
    ml.cohort_month,
    ml.months_since_first,
    cs.cohort_size,
    ml.active_customers,
    ROUND(ml.period_revenue, 2)                 AS period_revenue,
    ROUND(
        SUM(ml.period_revenue) OVER (
            PARTITION BY ml.cohort_month
            ORDER BY ml.months_since_first
        ) / cs.cohort_size, 2
    )                                            AS cumulative_ltv
FROM monthly_ltv ml
JOIN cohort_start cs USING (cohort_month)
ORDER BY ml.cohort_month, ml.months_since_first;


-- ═══════════════════════════════════════════════════════════════════════════
-- QUERY 7: Top Seller Performance Dashboard
-- ═══════════════════════════════════════════════════════════════════════════
WITH seller_metrics AS (
    SELECT
        oi.seller_id,
        s.seller_city,
        s.seller_state,
        s.seller_rating,
        COUNT(DISTINCT oi.order_id)             AS total_orders,
        COUNT(DISTINCT oi.product_id)           AS unique_products,
        SUM(oi.price)                           AS total_revenue,
        AVG(oi.price)                           AS avg_item_price,
        AVG(r.review_score)                     AS avg_review_score,
        COUNT(DISTINCT DATE_TRUNC('month', o.order_purchase_timestamp)) AS active_months
    FROM olist_order_items oi
    JOIN olist_orders o USING (order_id)
    JOIN olist_sellers s ON oi.seller_id = s.seller_id
    LEFT JOIN olist_order_reviews r USING (order_id)
    WHERE o.order_status = 'delivered'
    GROUP BY oi.seller_id, s.seller_city, s.seller_state, s.seller_rating
),
ranked_sellers AS (
    SELECT *,
        RANK() OVER (ORDER BY total_revenue DESC)      AS revenue_rank,
        RANK() OVER (ORDER BY avg_review_score DESC)   AS review_rank,
        PERCENT_RANK() OVER (ORDER BY total_revenue)   AS revenue_percentile
    FROM seller_metrics
)
SELECT
    seller_id,
    seller_city,
    seller_state,
    seller_rating,
    total_orders,
    unique_products,
    ROUND(total_revenue, 2)       AS total_revenue,
    ROUND(avg_item_price, 2)      AS avg_item_price,
    ROUND(avg_review_score, 2)    AS avg_review_score,
    active_months,
    revenue_rank,
    review_rank,
    ROUND(revenue_percentile * 100, 1) AS revenue_percentile
FROM ranked_sellers
ORDER BY revenue_rank
LIMIT 50;


-- ═══════════════════════════════════════════════════════════════════════════
-- QUERY 8: Payment Method Analysis with Running Totals
-- ═══════════════════════════════════════════════════════════════════════════
WITH payment_stats AS (
    SELECT
        p.payment_type,
        DATE_TRUNC('month', o.order_purchase_timestamp) AS month,
        COUNT(*) AS transaction_count,
        SUM(p.payment_value) AS total_value,
        AVG(p.payment_value) AS avg_value,
        AVG(p.payment_installments) AS avg_installments
    FROM olist_order_payments p
    JOIN olist_orders o USING (order_id)
    WHERE o.order_status = 'delivered'
    GROUP BY p.payment_type, DATE_TRUNC('month', o.order_purchase_timestamp)
)
SELECT
    payment_type,
    month,
    transaction_count,
    ROUND(total_value, 2) AS monthly_revenue,
    ROUND(avg_value, 2) AS avg_order_value,
    ROUND(avg_installments, 1) AS avg_installments,
    -- Running total per payment type
    ROUND(SUM(total_value) OVER (
        PARTITION BY payment_type ORDER BY month
    ), 2) AS cumulative_revenue,
    -- Market share
    ROUND(
        total_value / SUM(total_value) OVER (PARTITION BY month) * 100, 1
    ) AS monthly_share_pct
FROM payment_stats
ORDER BY payment_type, month;


-- ═══════════════════════════════════════════════════════════════════════════
-- QUERY 9: Geographic Revenue Heatmap
-- ═══════════════════════════════════════════════════════════════════════════
SELECT
    c.customer_state,
    COUNT(DISTINCT o.order_id)           AS total_orders,
    COUNT(DISTINCT c.customer_unique_id) AS unique_customers,
    ROUND(SUM(p.payment_value), 2)       AS total_revenue,
    ROUND(AVG(p.payment_value), 2)       AS avg_order_value,
    ROUND(AVG(r.review_score), 2)        AS avg_review_score,
    ROUND(
        SUM(p.payment_value) / SUM(SUM(p.payment_value)) OVER () * 100, 2
    )                                    AS revenue_share_pct,
    -- Orders per customer
    ROUND(
        COUNT(DISTINCT o.order_id)::NUMERIC /
        NULLIF(COUNT(DISTINCT c.customer_unique_id), 0), 2
    )                                    AS orders_per_customer
FROM olist_orders o
JOIN olist_customers c USING (customer_id)
JOIN olist_order_payments p USING (order_id)
LEFT JOIN olist_order_reviews r USING (order_id)
WHERE o.order_status = 'delivered'
GROUP BY c.customer_state
ORDER BY total_revenue DESC;


-- ═══════════════════════════════════════════════════════════════════════════
-- QUERY 10: Real-Time Streaming Aggregation View
-- ═══════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE VIEW v_realtime_kpis AS
WITH last_hour AS (
    SELECT
        COUNT(DISTINCT order_id)     AS orders_last_hour,
        SUM(p.payment_value)         AS revenue_last_hour,
        AVG(p.payment_value)         AS aov_last_hour
    FROM olist_orders o
    JOIN olist_order_payments p USING (order_id)
    WHERE order_purchase_timestamp >= NOW() - INTERVAL '1 hour'
),
today AS (
    SELECT
        COUNT(DISTINCT order_id)     AS orders_today,
        SUM(p.payment_value)         AS revenue_today
    FROM olist_orders o
    JOIN olist_order_payments p USING (order_id)
    WHERE DATE_TRUNC('day', order_purchase_timestamp) = CURRENT_DATE
),
this_month AS (
    SELECT
        COUNT(DISTINCT order_id)     AS orders_mtd,
        SUM(p.payment_value)         AS revenue_mtd,
        COUNT(DISTINCT o.customer_id) AS customers_mtd
    FROM olist_orders o
    JOIN olist_order_payments p USING (order_id)
    WHERE DATE_TRUNC('month', order_purchase_timestamp) = DATE_TRUNC('month', CURRENT_DATE)
)
SELECT
    h.orders_last_hour,
    ROUND(h.revenue_last_hour::NUMERIC, 2)  AS revenue_last_hour,
    ROUND(h.aov_last_hour::NUMERIC, 2)      AS aov_last_hour,
    t.orders_today,
    ROUND(t.revenue_today::NUMERIC, 2)      AS revenue_today,
    m.orders_mtd,
    ROUND(m.revenue_mtd::NUMERIC, 2)        AS revenue_mtd,
    m.customers_mtd,
    NOW()                                    AS snapshot_time
FROM last_hour h, today t, this_month m;

-- Usage: SELECT * FROM v_realtime_kpis;
