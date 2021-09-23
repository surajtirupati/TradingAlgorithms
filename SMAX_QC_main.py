class SMAX(QCAlgorithm):
    '''In this code we setup a simple moving average crossover strategy. We define the long term moving average (ltma) and short term
    moving avergae (stma) in the __init__() function. '''

    def __init__(self):
        self.symbol = "BTCUSD"
        self.ltma = 84
        self.stma = 7
        self.cash = 10000
        self.previous = None
        self.fast = None
        self.slow = None

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        # Set Strategy Cash - this is ignored when trading live
        self.SetCash(self.cash)

        # Set Backtest start date - this is ignored when trading live
        self.SetStartDate(2015, 2, 1)

        # Find more symbols here: http://quantconnect.com/data
        self.AddSecurity(SecurityType.Crypto, self.symbol, Resolution.Daily)

        # create a 15 day exponential moving average
        self.fast = self.SMA(self.symbol, self.stma, Resolution.Daily)

        # create a 30 day exponential moving average
        self.slow = self.SMA(self.symbol, self.ltma, Resolution.Daily)

        # warmup period equal to the ltma
        self.SetWarmUp(timedelta(days=self.ltma))

    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.

        Arguments:
            data: TradeBars IDictionary object with your stock data
        '''
        # a couple things to notice in this method:
        #  1. We never need to 'update' our indicators with the data, the engine takes care of this for us
        #  2. We can use indicators directly in math expressions
        #  3. We can easily plot many indicators at the same time

        # only once per day
        if self.previous is not None and self.previous == self.Time:
            return

        # define a small tolerance on our checks to avoid bouncing
        tolerance = 0.00015;

        holdings = self.Portfolio[self.symbol].Quantity

        # we only want to go long if we're currently short or flat
        if holdings <= 0:
            # if the fast is greater than the slow, we'll go long
            if self.fast.Current.Value > self.slow.Current.Value * (1 + tolerance):
                self.Log("BUY  >> {0}".format(self.Securities[self.symbol].Price))
                self.SetHoldings(self.symbol, 1.0)

        # we only want to liquidate if we're currently long
        # if the fast is less than the slow we'll liquidate our long
        if holdings > 0 and self.fast.Current.Value < self.slow.Current.Value:
            self.Log("SELL >> {0}".format(self.Securities[self.symbol].Price))
            self.Liquidate(self.symbol)

        self.previous = self.Time

