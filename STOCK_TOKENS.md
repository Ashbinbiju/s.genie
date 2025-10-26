# Popular Stock Tokens for Angel One

## How to Add New Stocks

1. Use the **🔍 Search Stock** feature in the sidebar
2. Enter symbol name (e.g., "INFY" for Infosys)
3. Select exchange (NSE/BSE)
4. Copy the token from search results
5. Add to `config_secure.py` in this format:

```python
'SYMBOLNAME': {'token': 'TOKEN_NUMBER', 'exchange': 'NSE', 'lot_size': 1},
```

## Popular NSE Stock Tokens

### Banking & Finance
- **HDFCBANK**: Token: `1333`, Exchange: NSE
- **ICICIBANK**: Token: `1330`, Exchange: NSE
- **SBIN**: Token: `3045`, Exchange: NSE
- **AXISBANK**: Token: `5900`, Exchange: NSE
- **KOTAKBANK**: Token: `1922`, Exchange: NSE

### IT Sector
- **TCS**: Token: `11536`, Exchange: NSE
- **INFY**: Token: `1594`, Exchange: NSE
- **WIPRO**: Token: `3787`, Exchange: NSE
- **HCLTECH**: Token: `7229`, Exchange: NSE
- **TECHM**: Token: `13538`, Exchange: NSE

### Auto Sector
- **TATAMOTORS**: Token: `3456`, Exchange: NSE
- **MARUTI**: Token: `10999`, Exchange: NSE
- **M&M**: Token: `2031`, Exchange: NSE
- **BAJAJ-AUTO**: Token: `16669`, Exchange: NSE
- **EICHERMOT**: Token: `910`, Exchange: NSE

### Pharma
- **SUNPHARMA**: Token: `3351`, Exchange: NSE
- **DRREDDY**: Token: `881`, Exchange: NSE
- **CIPLA**: Token: `694`, Exchange: NSE
- **DIVISLAB**: Token: `10940`, Exchange: NSE
- **LUPIN**: Token: `10440`, Exchange: NSE

### FMCG
- **ITC**: Token: `1660`, Exchange: NSE
- **HINDUNILVR**: Token: `1394`, Exchange: NSE
- **NESTLEIND**: Token: `17963`, Exchange: NSE
- **BRITANNIA**: Token: `547`, Exchange: NSE

### Metals & Mining
- **TATASTEEL**: Token: `3499`, Exchange: NSE
- **HINDALCO**: Token: `1363`, Exchange: NSE
- **JSWSTEEL**: Token: `11723`, Exchange: NSE
- **COALINDIA**: Token: `5215`, Exchange: NSE

### Energy
- **RELIANCE**: Token: `2885`, Exchange: NSE
- **ONGC**: Token: `2475`, Exchange: NSE
- **POWERGRID**: Token: `14977`, Exchange: NSE
- **NTPC**: Token: `11630`, Exchange: NSE

### Telecom
- **BHARTIARTL**: Token: `10604`, Exchange: NSE
- **IDEA**: Token: `7929`, Exchange: NSE

## Nifty 50 Index Tokens

- **NIFTY 50**: Token: `99926000`, Exchange: NSE
- **NIFTY BANK**: Token: `99926009`, Exchange: NSE

## How to Verify Token

Use Angel One's official scrip master:
https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json

Or use the search feature in the dashboard!

## Notes

- All tokens are for **equity (EQ)** segment
- Tokens may change rarely - verify if data not loading
- Use search feature to find latest tokens
- Only NSE equity trading supported in current strategy
