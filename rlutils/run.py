from rlutils.algos.tf.mb import pets
from rlutils.algos.tf.mf import a2c, a2c_q, ddpg, ppo, sac, td3, trpo, dqn
from rlutils.algos.tf.offline import cql, bracp, plas
from rlutils.runner import get_argparser_from_func

__all__ = ['ppo', 'td3', 'trpo', 'sac', 'a2c', 'a2c_q', 'ddpg', 'cql', 'bracp', 'plas', 'dqn', 'pets']


def main():
    import argparse

    parser = argparse.ArgumentParser('Running rl algorithms', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    algorithm_parsers = parser.add_subparsers(title='algorithm', help='algorithm specific parser', dest='algo')
    for algo in __all__:
        algo_parser = algorithm_parsers.add_parser(algo, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        get_argparser_from_func(eval(f'{algo}.Runner.main'), algo_parser)

    kwargs = vars(parser.parse_args())
    algo = kwargs.pop('algo')
    eval(f'{algo}.Runner.main')(**kwargs)


if __name__ == '__main__':
    main()
