-- Supabase SQL Schema for Paper Trading

-- Create paper_trades table
CREATE TABLE IF NOT EXISTS paper_trades (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    token VARCHAR(20) NOT NULL,
    exchange VARCHAR(10) NOT NULL,
    signal_type VARCHAR(10) NOT NULL,
    entry_price DECIMAL(10, 2) NOT NULL,
    exit_price DECIMAL(10, 2),
    stop_loss DECIMAL(10, 2) NOT NULL,
    take_profit DECIMAL(10, 2) NOT NULL,
    quantity INTEGER NOT NULL,
    signal_strength INTEGER,
    signal_quality INTEGER,
    rsi DECIMAL(5, 2),
    trend VARCHAR(20),
    status VARCHAR(10) NOT NULL DEFAULT 'OPEN',
    profit_loss DECIMAL(12, 2),
    profit_loss_percent DECIMAL(8, 2),
    entry_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    exit_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT check_signal_type CHECK (signal_type IN ('BUY', 'SELL')),
    CONSTRAINT check_status CHECK (status IN ('OPEN', 'CLOSED')),
    CONSTRAINT check_signal_strength CHECK (signal_strength BETWEEN 0 AND 4),
    CONSTRAINT check_signal_quality CHECK (signal_quality BETWEEN 0 AND 4)
);

-- Create indexes for better query performance
CREATE INDEX idx_paper_trades_symbol ON paper_trades(symbol);
CREATE INDEX idx_paper_trades_status ON paper_trades(status);
CREATE INDEX idx_paper_trades_entry_time ON paper_trades(entry_time DESC);
CREATE INDEX idx_paper_trades_symbol_status ON paper_trades(symbol, status);

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_paper_trades_updated_at 
    BEFORE UPDATE ON paper_trades 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create a view for performance statistics
CREATE OR REPLACE VIEW trade_statistics AS
SELECT 
    COUNT(*) as total_trades,
    COUNT(*) FILTER (WHERE status = 'CLOSED') as closed_trades,
    COUNT(*) FILTER (WHERE status = 'OPEN') as open_trades,
    COUNT(*) FILTER (WHERE profit_loss > 0) as winning_trades,
    COUNT(*) FILTER (WHERE profit_loss < 0) as losing_trades,
    ROUND(AVG(CASE WHEN profit_loss > 0 THEN profit_loss END), 2) as avg_win,
    ROUND(AVG(CASE WHEN profit_loss < 0 THEN profit_loss END), 2) as avg_loss,
    ROUND(SUM(profit_loss), 2) as total_pnl,
    ROUND(
        (COUNT(*) FILTER (WHERE profit_loss > 0)::DECIMAL / 
         NULLIF(COUNT(*) FILTER (WHERE status = 'CLOSED'), 0) * 100), 
        2
    ) as win_rate_percent
FROM paper_trades;

-- Grant permissions (adjust as needed for your Supabase setup)
-- ALTER TABLE paper_trades ENABLE ROW LEVEL SECURITY;

-- Example: Allow authenticated users to read/write their own trades
-- CREATE POLICY "Users can view their own trades" 
--     ON paper_trades FOR SELECT 
--     USING (auth.uid() = user_id);

COMMENT ON TABLE paper_trades IS 'Paper trading records for backtesting and simulation';
COMMENT ON COLUMN paper_trades.signal_type IS 'BUY or SELL signal';
COMMENT ON COLUMN paper_trades.status IS 'OPEN or CLOSED position status';
COMMENT ON COLUMN paper_trades.signal_strength IS 'Signal strength from 0-4';
COMMENT ON COLUMN paper_trades.signal_quality IS 'Signal quality score from 0-4';
