--
-- PostgreSQL database dump
--

\restrict erHTrpB7YebdTaHMPrDotwKAVMaVge0Ra8gblr8u15mLLCaPuYOrxs2aHOgugNn

-- Dumped from database version 17.7 (Debian 17.7-3.pgdg12+1)
-- Dumped by pg_dump version 17.6 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: gold_predictor_user
--

-- *not* creating schema, since initdb creates it


ALTER SCHEMA public OWNER TO gold_predictor_user;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: historical_predictions; Type: TABLE; Schema: public; Owner: gold_predictor_user
--

CREATE TABLE public.historical_predictions (
    id integer NOT NULL,
    date date NOT NULL,
    predicted_price real NOT NULL,
    actual_price real,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.historical_predictions OWNER TO gold_predictor_user;

--
-- Name: historical_predictions_id_seq; Type: SEQUENCE; Schema: public; Owner: gold_predictor_user
--

CREATE SEQUENCE public.historical_predictions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.historical_predictions_id_seq OWNER TO gold_predictor_user;

--
-- Name: historical_predictions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: gold_predictor_user
--

ALTER SEQUENCE public.historical_predictions_id_seq OWNED BY public.historical_predictions.id;


--
-- Name: predictions; Type: TABLE; Schema: public; Owner: gold_predictor_user
--

CREATE TABLE public.predictions (
    id integer NOT NULL,
    prediction_date date NOT NULL,
    predicted_price real NOT NULL,
    actual_price real,
    accuracy_percentage real,
    prediction_method text DEFAULT 'Lasso Regression'::text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.predictions OWNER TO gold_predictor_user;

--
-- Name: predictions_id_seq; Type: SEQUENCE; Schema: public; Owner: gold_predictor_user
--

CREATE SEQUENCE public.predictions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.predictions_id_seq OWNER TO gold_predictor_user;

--
-- Name: predictions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: gold_predictor_user
--

ALTER SEQUENCE public.predictions_id_seq OWNED BY public.predictions.id;


--
-- Name: historical_predictions id; Type: DEFAULT; Schema: public; Owner: gold_predictor_user
--

ALTER TABLE ONLY public.historical_predictions ALTER COLUMN id SET DEFAULT nextval('public.historical_predictions_id_seq'::regclass);


--
-- Name: predictions id; Type: DEFAULT; Schema: public; Owner: gold_predictor_user
--

ALTER TABLE ONLY public.predictions ALTER COLUMN id SET DEFAULT nextval('public.predictions_id_seq'::regclass);


--
-- Data for Name: historical_predictions; Type: TABLE DATA; Schema: public; Owner: gold_predictor_user
--

COPY public.historical_predictions (id, date, predicted_price, actual_price, created_at) FROM stdin;
\.


--
-- Data for Name: predictions; Type: TABLE DATA; Schema: public; Owner: gold_predictor_user
--

COPY public.predictions (id, prediction_date, predicted_price, actual_price, accuracy_percentage, prediction_method, created_at, updated_at) FROM stdin;
2	2025-11-10	3995.4087	4061.3	98.37758	Lasso Regression	2025-11-09 17:33:43.819624	2025-11-10 05:14:13.062108
4	2025-11-11	4046.199	4152.4	97.44242	Lasso Regression	2025-11-10 17:30:16.535723	2025-11-11 04:10:03.489823
6	2025-11-12	4089.694	4116.2	99.35605	Lasso Regression	2025-11-11 17:33:55.688089	2025-11-12 05:10:06.76394
7	2025-11-13	4086.96	4214.3	96.978386	Lasso Regression (Fallback)	2025-11-12 05:09:58.751212	2025-11-13 07:22:04.116633
9	2025-11-14	4173.9834	4207.8	99.19634	Lasso Regression	2025-11-13 17:33:07.706556	2025-11-14 04:10:09.06834
12	2025-11-16	4104.99	4094.2	99.73646	Lasso Regression (Fallback)	2025-11-15 00:00:10.011646	2025-11-16 23:18:11.926159
11	2025-11-17	4113.6064	4077.4	99.11202	Lasso Regression	2025-11-14 17:32:59.423288	2025-11-17 05:14:46.990544
14	2025-11-18	4073.1453	4009.8	98.42024	Lasso Regression	2025-11-17 17:30:42.942365	2025-11-18 05:20:51.420828
16	2025-11-19	4055.9119	4074.5	99.54379	Lasso Regression	2025-11-18 17:31:00.207191	2025-11-19 03:10:24.51757
18	2025-11-20	4059.19	4069.2	99.754005	Lasso Regression	2025-11-19 17:30:52.953152	2025-11-20 03:10:10.148885
23	2025-11-23	4052.98	4070.8	99.56225	Lasso Regression (Fallback)	2025-11-22 00:10:00.210156	2025-11-23 23:21:30.790419
30	2025-11-30	4152.691	\N	\N	Lasso Regression (Fallback)	2025-11-28 20:06:46.47088	2025-11-28 20:06:46.47088
28	2025-11-28	4138.9116	4254.9	97.27401	Lasso Regression (Fallback)	2025-11-27 07:37:10.609407	2025-11-29 11:58:01.542964
10	2025-11-15	4166.22	4087.6	98.07662	Lasso Regression (Fallback)	2025-11-14 00:28:17.84288	2025-11-29 12:11:47.147784
21	2025-11-22	4071.06	4076.7	99.86166	Lasso Regression (Fallback)	2025-11-21 00:12:39.78191	2025-11-29 12:11:50.112388
29	2025-11-29	4135.616	4254.9	97.19655	Lasso Regression (Fallback)	2025-11-28 08:01:40.941382	2025-11-29 12:11:56.157574
20	2025-11-21	4059.556	4065	99.86607	Lasso Regression	2025-11-20 17:33:17.27492	2025-11-30 11:19:28.371547
25	2025-11-25	4083.255	4130	98.86816	Lasso Regression	2025-11-24 17:33:51.372007	2025-11-30 11:19:34.569418
26	2025-11-26	4112.7	4165	98.7443	Lasso Regression (Fallback)	2025-11-25 00:02:23.731514	2025-11-30 11:19:37.600966
27	2025-11-27	4135.1587	4156	99.49853	Lasso Regression (Fallback)	2025-11-26 07:09:02.718432	2025-11-30 11:19:40.816243
22	2025-11-24	4065.0532	4135	98.30842	Lasso Regression	2025-11-21 17:34:47.212287	2025-11-30 11:45:45.080502
31	2025-12-01	4152.691	4239.3	97.957	Lasso Regression (Fallback)	2025-11-30 06:25:42.282246	2025-12-05 05:41:12.669311
32	2025-12-02	4173.7827	4186.6	99.69385	Lasso Regression (Fallback)	2025-12-01 05:51:31.823331	2025-12-05 05:41:17.445207
33	2025-12-03	4182.149	4199.3	99.591576	Lasso Regression (Fallback)	2025-12-01 18:31:57.49505	2025-12-05 05:41:23.545099
34	2025-12-04	4162.219	4211.8	98.822815	Lasso Regression (Fallback)	2025-12-03 08:19:26.146527	2025-12-05 05:41:28.127422
35	2025-12-08	4205.6396	4236.9	99.26219	Lasso Regression (Fallback)	2025-12-05 05:34:14.999089	2025-12-08 00:22:24.16166
36	2025-12-09	4201.2734	\N	\N	Lasso Regression (Fallback)	2025-12-08 05:38:57.472631	2025-12-08 05:38:57.472631
37	2025-12-15	4269.845	\N	\N	Lasso Regression (Fallback)	2025-12-13 08:03:00.363215	2025-12-13 08:03:00.363215
\.


--
-- Name: historical_predictions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: gold_predictor_user
--

SELECT pg_catalog.setval('public.historical_predictions_id_seq', 1, false);


--
-- Name: predictions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: gold_predictor_user
--

SELECT pg_catalog.setval('public.predictions_id_seq', 37, true);


--
-- Name: historical_predictions historical_predictions_pkey; Type: CONSTRAINT; Schema: public; Owner: gold_predictor_user
--

ALTER TABLE ONLY public.historical_predictions
    ADD CONSTRAINT historical_predictions_pkey PRIMARY KEY (id);


--
-- Name: predictions predictions_pkey; Type: CONSTRAINT; Schema: public; Owner: gold_predictor_user
--

ALTER TABLE ONLY public.predictions
    ADD CONSTRAINT predictions_pkey PRIMARY KEY (id);


--
-- Name: predictions predictions_prediction_date_key; Type: CONSTRAINT; Schema: public; Owner: gold_predictor_user
--

ALTER TABLE ONLY public.predictions
    ADD CONSTRAINT predictions_prediction_date_key UNIQUE (prediction_date);


--
-- Name: idx_actual_price; Type: INDEX; Schema: public; Owner: gold_predictor_user
--

CREATE INDEX idx_actual_price ON public.predictions USING btree (actual_price);


--
-- Name: idx_created_at; Type: INDEX; Schema: public; Owner: gold_predictor_user
--

CREATE INDEX idx_created_at ON public.predictions USING btree (created_at);


--
-- Name: idx_prediction_date; Type: INDEX; Schema: public; Owner: gold_predictor_user
--

CREATE INDEX idx_prediction_date ON public.predictions USING btree (prediction_date);


--
-- Name: idx_predictions_created_at; Type: INDEX; Schema: public; Owner: gold_predictor_user
--

CREATE INDEX idx_predictions_created_at ON public.predictions USING btree (created_at);


--
-- Name: idx_predictions_date; Type: INDEX; Schema: public; Owner: gold_predictor_user
--

CREATE INDEX idx_predictions_date ON public.predictions USING btree (prediction_date);


--
-- Name: idx_predictions_date_created; Type: INDEX; Schema: public; Owner: gold_predictor_user
--

CREATE INDEX idx_predictions_date_created ON public.predictions USING btree (prediction_date, created_at);


--
-- Name: DEFAULT PRIVILEGES FOR SEQUENCES; Type: DEFAULT ACL; Schema: -; Owner: postgres
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres GRANT ALL ON SEQUENCES TO gold_predictor_user;


--
-- Name: DEFAULT PRIVILEGES FOR TYPES; Type: DEFAULT ACL; Schema: -; Owner: postgres
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres GRANT ALL ON TYPES TO gold_predictor_user;


--
-- Name: DEFAULT PRIVILEGES FOR FUNCTIONS; Type: DEFAULT ACL; Schema: -; Owner: postgres
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres GRANT ALL ON FUNCTIONS TO gold_predictor_user;


--
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: -; Owner: postgres
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres GRANT ALL ON TABLES TO gold_predictor_user;


--
-- PostgreSQL database dump complete
--

\unrestrict erHTrpB7YebdTaHMPrDotwKAVMaVge0Ra8gblr8u15mLLCaPuYOrxs2aHOgugNn

