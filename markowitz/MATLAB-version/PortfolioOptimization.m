classdef PortfolioOptimization
    properties
        Stocks
        StartDate
        EndDate
        NumTradingDays
        NumPortfolios
        RiskFreeRate
        PortfolioReturns
        PortfolioData
        Weights
    end

    methods
        function obj = PortfolioOptimization(stocks, start, end_date)
            obj.Stocks = stocks;
            obj.StartDate = start;
            obj.EndDate = end_date;
            obj.NumTradingDays = 252;
            obj.NumPortfolios = 50000;
            obj.RiskFreeRate = 0.045 / 252; % Based on government bond data
        end

        function data = GetDataFromYahoo(obj)
            stockData = containers.Map();
            for i = 1:length(obj.Stocks)
                stock = obj.Stocks{i};
                ticker = yahoo.Financial(stock);
                data = fetch(ticker, obj.StartDate, obj.EndDate, 'd');
                stockData(stock) = data.AdjClose;
            end
            data = struct(stockData);
        end

        function ShowData(obj)
            data = obj.GetDataFromYahoo();
            dates = obj.StartDate:obj.EndDate;
            figure;
            plot(dates, data);
            title('Stock Prices Over Time');
            xlabel('Date');
            ylabel('Price');
        end

        function returns = CalculateReturns(obj)
            data = obj.GetDataFromYahoo();
            returns = price2ret(data);
            obj.PortfolioReturns = returns;
        end

        function portfolioData = GeneratePortfolios(obj)
            weights = zeros(obj.NumPortfolios, length(obj.Stocks));
            portfolioData = struct('mean', zeros(obj.NumPortfolios, 1), 'risk', zeros(obj.NumPortfolios, 1));
            returns = obj.CalculateReturns();
            fprintf('Generating portfolios\n');
            progressBar = waitbar(0, 'Generating portfolios');
            for i = 1:obj.NumPortfolios
                weight = rand(1, length(obj.Stocks));
                weight = weight / sum(weight);
                weights(i, :) = weight;

                portfolioReturn = sum(mean(returns) .* weight) * obj.NumTradingDays;
                excessReturn = returns - obj.RiskFreeRate;
                portfolioVolatility = sqrt(weight * (excessReturn' * excessReturn) * weight');

                portfolioData(i).mean = portfolioReturn;
                portfolioData(i).risk = portfolioVolatility;

                waitbar(i / obj.NumPortfolios, progressBar);
            end
            close(progressBar);
            obj.Weights = weights;
            obj.PortfolioData = portfolioData;
        end

        function optimal = OptimizePortfolio(obj)
            obj.GeneratePortfolios();
            returns = obj.PortfolioReturns;
            weights0 = obj.Weights(1, :);
            fun = @(x) -Statistics(x, returns).(3); % Minimize -Sharpe Ratio
            Aeq = ones(1, length(obj.Stocks));
            beq = 1;
            lb = zeros(1, length(obj.Stocks));
            ub = ones(1, length(obj.Stocks));
            options = optimoptions('fmincon', 'Algorithm', 'sqp');
            optimalWeights = fmincon(fun, weights0, [], [], Aeq, beq, lb, ub, [], options);
            optimal = optimalWeights;
        end

        function DisplayStats(obj, weights)
            stats = Statistics(weights, obj.PortfolioReturns);
            fprintf('Expected return: %f\n', stats(1));
            fprintf('Expected volatility: %f\n', stats(2));
            fprintf('Sharpe ratio: %f\n', stats(3));
        end

        function DisplayAndPrintPortfolio(obj)
            optimal = obj.OptimizePortfolio();
            obj.ShowData();
            obj.DisplayStats(optimal);
            fprintf('The optimum portfolio is:\n');
            disp(array2table(optimal, 'RowNames', obj.Stocks, 'VariableNames', {'Weight'}));
            obj.PlotPortfolio();
        end

        function PlotPortfolio(obj)
            portfolioData = obj.PortfolioData;
            figure;
            scatter(portfolioData.risk, portfolioData.mean, [], (portfolioData.mean - obj.RiskFreeRate * 252) ./ portfolioData.risk, 'filled');
            colorbar;
            xlabel('Expected Volatility');
            ylabel('Expected Return');
            title('Efficient Frontier with Optimal Portfolio');
            hold on;
            plot(Statistics(obj.OptimizePortfolio(), obj.PortfolioReturns)(2), Statistics(obj.OptimizePortfolio(), obj.PortfolioReturns)(1), 'g*', 'MarkerSize', 15);
            hold off;
        end
    end
end

function stats = Statistics(weights, returns)
    portfolioReturn = sum(mean(returns) .* weights) * 252;
    excessReturn = returns - obj.RiskFreeRate;
    portfolioVolatility = sqrt(weights * (excessReturn' * excessReturn) * weights');
    sharpeRatio = (portfolioReturn - obj.RiskFreeRate * 252) / portfolioVolatility;
    stats = [portfolioReturn, portfolioVolatility, sharpeRatio];
end
