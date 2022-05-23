from objective import square
from algo import Vanila, RankMu, RankOne, RankOne_Cum, RankOne_Cum_RankMu, CMAES
import matplotlib.pyplot as plt

algo = RankOne(N = 5,
              step_size=0.5,
              cost_func = square)

algo.run()