# OpenRouter is the single model gateway, including Jev

Every model, including Jev, is called through OpenRouter, and each Model role names its model in config. One key gives access to every model family, so the benchmark can be re-run by readers and swept across families by changing config alone. The trade-off: a gateway hop adds latency to every call, including Jev's. Latency figures are therefore measured end-to-end through the same gateway for all Levels, so the comparison stays fair.
