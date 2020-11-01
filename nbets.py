#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import random
from tqdm import tqdm

### CONFIG
sns.set()

class Round:
    """ define one round of the game """
    def __init__(self, p = None, b = None, a = 1):

        if p is None:
            p = random.uniform(0.05, 0.95)

        if b is None:
            b = min(random.expovariate(lambd = p/0.5) + 1.1, 50)

        self.p = p
        self.q = 1-p
        self.b = b
        self.a = a

    def play(self):
        return random.choices(population = [self.b, -self.a],
                              k = 1,
                              weights= [self.p, self.q])[0]


class Game:
    """  define the game. Rounds of bets where you have a p chance of multiplying wager by b, and
         q chance of reducing capital by a (default 1, meaning lose the wager money)
     """
    def __init__(self, starting_money = 10000, **kwargs):
        self.starting_money = starting_money
        self.cash = starting_money

    def play(self, strategy, rounds = 50, reset = False, newbet = True, describe = False, *args, **kwargs):

        if reset:
            cash = self.starting_money
        else:
            cash = self.cash

        if describe:
            print(f'Starting cash is ${cash}. Playing {rounds} rounds.')

        bet = Round(*args, **kwargs)

        for i in range(rounds):
            strategy.setup(bet = bet)
            wager = max(strategy.wager(bank = cash), 0)
            aorb = bet.play()
            cash += wager * aorb

            if describe:
                print(f'Round {i+1}: Multiply wager by {bet.b:.2f} with probability {bet.p:.2%}, lose ', end='')
                if bet.a == 1:
                    print('wager ', end='')
                else:
                    print(f'{bet.a:.2%} ', end='')
                print('otherwise. ', end = '')

                print(str(strategy) + ' ', end = '')

                if wager > 0:
                    print(f'Wager is $ {wager:.2f}. ', end = '')

                    if aorb < 0:
                        print('Lost bet. ', end = '')
                    else:
                        print('Won bet. ', end='')

                else:
                    print('No bet. ', end = '')

                if cash <= 0.05:
                    print('Broke.')
                    break

                else:
                    print(f'Remaining cash is $ {cash:.2f}.')

            if newbet:
                bet = Round(*args, **kwargs)

        self.cash = cash
        return cash


class Strategy:

    def setup(self, bet):
        self.p = bet.p
        self.q = bet.q
        self.a = bet.a
        self.b = bet.b

    def wager(self, *args, **kwargs):
        return kwargs['bank']

    def __str__(self):
        return 'All-in strategy.'


class StrategyKelly(Strategy):
    """ define the strategy to play the game as one based on the kelly criterion """

    def setup(self, bet):
        super().setup(bet)
        self.f = (self.b * self.p - self.a * self.q) / (self.a * self.b)

    def wager(self, bank):
        return bank * self.f

    def __str__(self):
        return f'Kelly Strategy, f = {self.f:.3f}.'


class StrategyFixedF(Strategy):
    """ strategy fixed f. Every wager is a fixed predetermined fraction of bank """
    def __init__(self, f, *args, **kwargs):
        self.f = f

    def setup(self, bet):
        pass

    def wager(self, bank):
        return bank * self.f

    def __str__(self):
        return f'Fixed fraction strategy, f = {self.f:.3f}.'


class Simulation:
    def __init__(self, sims = 10**2, rounds = 10**2, *args, **kwargs):
        self.rounds = rounds
        self.args = args
        _ = kwargs.pop('p', None)
        self.kwargs = kwargs

        self.fs = np.linspace(0.05, 0.95, 21)
        self.ps = np.linspace(0, 1, 11)
        idx = pd.MultiIndex.from_product([self.fs, self.ps, np.arange(1, sims) ], names = ['f', 'p', 'sim'])
        df = pd.DataFrame([], columns=['growth'], index = idx, dtype=float)
        df = df.reset_index()
        self.results = df

    def run(self):
        money_begin = self.kwargs.get('starting_money', 100)
        game = Game(starting_money = money_begin)

        for i in tqdm(self.results.index, desc='Progress'):
            f, p = self.results[['f', 'p']].loc[i]
            money_end = game.play(strategy = StrategyFixedF(f = f), rounds = self.rounds, reset = True, p = p,
                                  *self.args, **self.kwargs)
            r = (money_end / money_begin)**(1/self.rounds) - 1



            self.results.loc[i, 'growth'] = r


if __name__ == '__main__':
    game = Game(starting_money = 1000)
    finalcash = game.play(strategy = StrategyKelly(), rounds = 50)

    res = Simulation(b = 1, sims=10**2, rounds=50)
    res.run()

    sns.lineplot(data = res.results, x='f', y='growth', hue = 'p')