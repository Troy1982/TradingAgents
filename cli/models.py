from enum import StrEnum


class AnalystType(StrEnum):
    MARKET = "market"
    # Wire value stays "social" for saved-config and string-keyed-caller
    # back-compat; the user-facing label is "Sentiment Analyst".
    SOCIAL = "social"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    # Taiwan-listed stocks only: offered when the ticker is one.
    CHIPS = "chips"


class AssetType(StrEnum):
    STOCK = "stock"
    CRYPTO = "crypto"
