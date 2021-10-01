from bs4 import BeautifulSoup
import requests
import pandas as pd

def create_url(stock, statement, period):
    """
    Function to create the relevant URL based on two inputs:
    - stock: name of the ticker to scrape
    - statement: abbreviation of the statement desired
    - period: the period of interest - must be either 'quarter' or 'annual'

    Returns the relevant URL.
    """

    if statement == "inc":
        url = "https://www.marketwatch.com/investing/stock/{}/financials/income/{}".format(stock, period)
        return url

    elif statement == "bs":
        url = "https://www.marketwatch.com/investing/stock/{}/financials/balance-sheet/{}".format(stock, period)
        return url

    elif statement == "cf":
        url = "https://www.marketwatch.com/investing/stock/{}/financials/cash-flow/{}".format(stock, period)
        return url

    else:
        print(
            "Invalid statement type. Please specify 'inc', 'bs', or 'cf' for income statement, balance sheet, or cash flow statement.")
        return None


def obtain_table_as_df(url):
    """
    This function takes inputs:
    - url: a MarketWatch URL containing one of the three financial statements of a particular company.

    Irrespective of which statement and which company,
    """

    # Make a GET request to fetch the raw HTML content
    html_content = requests.get(url).text

    # get the HTML content
    soup = BeautifulSoup(html_content, "lxml")

    # obtaining tables
    table = soup.find_all('table')

    # income statement: 3
    # balance sheet: 3, 4
    # cash flow: 3, 4, 5

    # indexing the relevant tables to obtain data of interest
    dfs = [pd.read_html(str(table))[i] for i in range(3, len(table) - 1)]

    # setting the index of each dataframe and
    dfs = [dfs[i].iloc[:, :-1].set_index(dfs[i].columns[0]) for i in range(len(dfs))]

    # initialise dataframe to hold all dataframes
    df = pd.DataFrame()

    # append each dataframe in list of dataframes to df
    for i in range(0, len(dfs)):
        # appending
        df = df.append(dfs[i])

        # checking if dataframe is empty
    if df.empty:
        # print error
        print("Error: please check ticker or period input.")

        return

    return df


def clean_data(df):
    """
    This function cleans the marketwatch scraped dataframes specifically by:

    - removing duplicate words in the row indices
    - converting the values from strings to ints

    It returns the cleaned dataframe.
    """

    def repeated_word(word):

        n = len(word) // 2

        return word[:n]

    def clean_duplicate_rows(df):

        new_idx = []

        for string in df.index:
            string = string.replace(" ", "")
            string = repeated_word(string)
            new_idx.append(string)

        df.index = new_idx

        return df

    def convert_values_to_int(df):

        try:
            for i in range(len(df.values)):
                for j in range(len(df.values[i])):

                    if df.values[i][j][-1] == "-":
                        pass

                    elif df.values[i][j][-1] == "B":
                        df.values[i][j] = int(float(df.values[i][j][:-1]) * 1 * 10 ** 9)

                    elif df.values[i][j][-1] == "M":
                        df.values[i][j] = int(float(df.values[i][j][:-1]) * 1 * 10 ** 6)

                    elif df.values[i][j][-1] == "K":
                        df.values[i][j] = int(float(df.values[i][j][:-1]) * 1 * 10 ** 3)

                    elif df.values[i][j][-2] == "B":
                        df.values[i][j] = int(-float(df.values[i][j][1:-2]) * 1 * 10 ** 9)

                    elif df.values[i][j][-2] == "M":
                        df.values[i][j] = int(-float(df.values[i][j][1:-2]) * 1 * 10 ** 6)

                    elif df.values[i][j][-2] == "K":
                        df.values[i][j] = int(-float(df.values[i][j][1:-2]) * 1 * 10 ** 3)

                    else:
                        pass

            return df

        except:

            print("DataFrame already cleaned.")

            return df

    df = clean_duplicate_rows(df)
    df = convert_values_to_int(df)

    return df

def obtain_fundamentals(stock, statement, period):
    """
    Simple function that compiles all the previous code and returns the dataframe of interest.
    """
    url = create_url(stock, statement, period)
    df = obtain_table_as_df(url)
    df = clean_data(df)

    return df