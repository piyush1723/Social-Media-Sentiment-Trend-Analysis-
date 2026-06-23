-- ==========================================
-- Social Media Sentiment Trend Analysis
-- Database Schema
-- ==========================================

CREATE DATABASE IF NOT EXISTS sentiment_analysis;

USE sentiment_analysis;

-- Main tweets table

CREATE TABLE tweets (
    tweet_id INT PRIMARY KEY,
    entity VARCHAR(100) NOT NULL,
    sentiment VARCHAR(20) NOT NULL,
    tweet TEXT NOT NULL
);

-- Indexes for faster querying

CREATE INDEX idx_entity
ON tweets(entity);

CREATE INDEX idx_sentiment
ON tweets(sentiment);

TRUNCATE TABLE tweets;
DROP TABLE tweets;

CREATE TABLE tweets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    tweet_id INT,
    entity VARCHAR(100) NOT NULL,
    sentiment VARCHAR(20) NOT NULL,
    tweet TEXT NOT NULL
);

DESCRIBE tweets;