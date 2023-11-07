
start_date = datetime('01/01/2022', 'InputFormat', 'dd/MM/yyyy');
end_date = datetime('01/01/2023', 'InputFormat', 'dd/MM/yyyy');
tickers = {'AAPL', 'MSFT'};
data = generate_weights(start_date, end_date, tickers);
disp(data.AAPL)

function weights = generate_weights(start_date, end_date, tickers)
% start_date: Date from which the stock market data is to be fetched.
% end_date: The end date of the historical data.
closing_data = struct();
for ticker = tickers
    symbol = ticker{1}; % Convert the cell to a string
    data = getMarketDataViaYahoo(symbol, start_date, end_date);
    closing_data.(symbol) = data.Close; % Use 'symbol' as the field name

end
weights = closing_data;

end
