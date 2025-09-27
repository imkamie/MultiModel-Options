from pricing_models.black_scholes import BlackScholes
from pricing_models.binomial_tree import BinomialTree
from pricing_models.bachelier import Bachelier
from pricing_models.monte_carlo_gbm import MonteCarloGBM


def make_models_dict():
    return {
        "Black-Scholes-Merton": {
            "constructor": BlackScholes,
            "params": BlackScholes.PARAMS,
            "run": lambda inst: (inst.run(), *BlackScholes.show_prices(inst)),
            "show_prices": BlackScholes.show_prices,
        },
        "Binomial (CRR)": {
            "constructor": BinomialTree,
            "params": BinomialTree.PARAMS,
            "run": lambda inst: (inst.run(), *BinomialTree.show_prices(inst)),
            "show_prices": BinomialTree.show_prices,
        },
        "Monte Carlo (GBM)": {
            "constructor": MonteCarloGBM,
            "params": MonteCarloGBM.PARAMS,
            "run": lambda inst: (inst.run(), *MonteCarloGBM.show_prices(inst)),
            "show_prices": MonteCarloGBM.show_prices,
        },
        "Bachelier (Normal)": {
            "constructor": Bachelier,
            "params": Bachelier.PARAMS,
            "run": lambda inst: (inst.run(), *Bachelier.show_prices(inst)),
            "show_prices": Bachelier.show_prices,
        },
    }
