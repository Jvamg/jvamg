import requests
from agno.tools import tool


class CurrencyConverterTools:
    @tool
    def convert_usd_to_other(self, to_currency: str, amount: float = 1.0) -> str:
        """
        Converts a given amount from USD to another currency using the UniRateAPI.
        This function should only be used when a currency other than USD is requested.

        :param to_currency: The target currency code (e.g., 'BRL', 'EUR').
        :param amount: The amount in USD to convert.
        :return: A string with the converted amount and currency, or an error message.
        """
        try:
            url = f"https://api.unirateapi.com/api/rates"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if data and data.get("success"):
                rates = data.get("rates", {})
                if to_currency.upper() in rates:
                    converted_amount = amount * rates[to_currency.upper()]
                    return f"{amount} USD is equal to {converted_amount:.2f} {to_currency.upper()}"
                else:
                    return f"Currency '{to_currency}' not found."
            else:
                return "Could not retrieve exchange rates."

        except requests.exceptions.RequestException as e:
            return f"Error connecting to the API: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"
