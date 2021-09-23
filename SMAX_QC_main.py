class SMAX(QCAlgorithm):
    '''In this example we look at the canonical 15/30 day moving average cross. This algorithm
    will go long when the 15 crosses above the 30 and will liquidate when the 15 crosses
    back below the 30.'''

    def __init__(self):
        self.symbol = "BTCUSD"
        self.previous = None
        self.fast = None
        self.slow = None

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        # Set Strategy Cash - this is ignored when trading live
        self.SetCash(10000)

        # Set Backtest start date - this is ignored when trading live
        self.SetStartDate(2015, 2, 1)

        # Find more symbols here: http://quantconnect.com/data
        self.AddSecurity(SecurityType.Crypto, self.symbol, Resolution.Daily)

        # create a 15 day exponential moving average
        self.fast = self.SMA(self.symbol, 7, Resolution.Daily)

        # create a 30 day exponential moving average
        self.slow = self.SMA(self.symbol, 84, Resolution.Daily)

        self.SetWarmUp(timedelta(days=84))

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

