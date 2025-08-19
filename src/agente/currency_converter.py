import requests
import os
from typing import Any, Dict, List, Optional
from agno.tools import Toolkit
from dotenv import load_dotenv
load_dotenv()


class CurrencyConverterTools(Toolkit):
    """
    CurrencyConverterTools is a Python library for converting currencies.
    It uses the UniRateAPI to convert currencies.

    Args:
        timeout (Optional[int]): Timeout for the request, default is 10 seconds.
    """

    def __init__(self, timeout: Optional[int] = 10):
        super().__init__(name="currency_converter_tools")
        self.timeout: Optional[int] = timeout
        self.register(self.convert_usd_to_other)

    def convert_usd_to_other(self, to_currency: str, amount: float = 1.0) -> str:
        """
        Converts a given amount from USD to another currency using the UniRateAPI.
        This function should only be used when a currency other than USD is requested.

        :param to_currency: The target currency code (e.g., 'BRL', 'EUR').
        :param amount: The amount in USD to convert.
        :return: A string with the converted amount and currency, or an error message.
        """
        try:
            api_key = os.getenv("UNIRATE_API_KEY")
            if not api_key:
                return "Missing UNIRATE_API_KEY environment variable."

            url = "https://api.unirateapi.com/api/rates"
            params = {"api_key": api_key, "from": "USD"}
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # Accept both documented and legacy schemas
            rates = None
            if isinstance(data, dict):
                if "rates" in data:
                    rates = data["rates"]
                elif data.get("success") and "rates" in data:
                    rates = data["rates"]

            if isinstance(rates, dict):
                code = to_currency.upper()
                if code in rates:
                    try:
                        rate_value = float(rates[code])
                    except (TypeError, ValueError):
                        return "Received invalid rate value from API."
                    converted_amount = amount * rate_value
                    return f"{amount} USD is equal to {converted_amount:.2f} {code}"
                return f"Currency '{to_currency}' not found."
            return "Could not retrieve exchange rates."

        except requests.exceptions.RequestException as e:
            return f"Error connecting to the API: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"
