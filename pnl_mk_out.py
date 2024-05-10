import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from tqdm import tqdm
#===========================================

def plot_ts(ts, flds=['bid', 'ask'], trds=None, mkt=None):
    

    plt.figure(figsize=(12, 6))

    colors = sns.color_palette("Paired")

    if isinstance(flds, list):
        for i, row in enumerate(flds):
            sns.lineplot(data=ts, x='ts_ms', y= row, label=row, color=colors[i])
    else:
        sns.lineplot(data=ts, x='ts_ms', y=flds, label=flds, color=colors[0])

    if trds is not None:

        longs = trds[trds['side']=='B']
        shorts = trds[trds['side']=='S']

        plt.scatter(longs['ts_ms'], longs['px'], marker='o', color='g', label='Longs')
        plt.scatter(shorts['ts_ms'], shorts['px'], marker='o', color='r', label='Shorts')

    plt.xlabel('TimeStamp (milliseconds)')
    plt.ylabel('Price')
    plt.title(f'{mkt} Bid & Ask Prices Over Period')
    plt.legend()

    plt.show()


def plot_mark_out(df, normalized=True, mkt=None):
    plt.figure(figsize=(12,6))

    sns.lineplot(data=df['pnl'], label='pnl', color='b')

    plt.xlabel('Horizon Interval (ms)')
    if normalized:
        plt.ylabel('Normalized PNL')
        plt.title(f'{mkt} Normalized PNL')
    else:
        plt.title(f'{mkt} PNL')
        plt.ylabel('PNL')

    plt.show()

#===========================================


def closest_px(trds, data, fld='mid'):
    i = data['ts_ms'].searchsorted(trds) - 1
    px = data.iloc[i][fld]
    return px

def update_trd_list(data, trds):
    i = trds['ts_ms'].searchsorted(data)
    return i


def calc_pnl_by_interval(row, trds):
    if row['trd_idx'] == 0:
        return 0.00

    _trds = trds.iloc[:int(row['trd_idx'])]

    long_trds = _trds[_trds['side']=='B']
    short_trds = _trds[_trds['side']=='S']

    long_pnl = np.sum((row['mid'] - long_trds['px'].values)*long_trds['size'].values)
    short_pnl = np.sum((row['mid'] - short_trds['px'].values)*short_trds['size'].values*-1)
    
    _pnl = long_pnl - short_pnl
    
    return _pnl



def check_trd(row):
    diff_bid = abs(row['px'] - row['bid'])
    diff_ask = abs(row['px'] - row['ask'])
    if diff_ask < diff_bid:
        px = 'ask'
    else:
        px = 'bid'

    if row['side'] == 'B' and px == 'ask' or row['side'] == 'S' and px == 'bid':
        return 'taker'
    else:
        return 'maker'
    
    
def calc_agg_pnl(data, trds, horizon, normalized=True):
    trds['m2m'] = trds['ts_ms'] + horizon
    data['mid'] = (data['bid'] + data['ask'])/2

    sorted_trds = trds.sort_values(by='m2m')
    sorted_data = data.sort_values(by='ts_ms')

    sorted_trds['last_px'] = sorted_trds['m2m'].apply(closest_px, data=sorted_data) 

    if normalized:
        sorted_trds['pnl']  = sorted_trds.apply(lambda row: row['last_px'] - row['px'] if row['side'] == 'B' else (row['last_px'] - row['px'])*-1, axis=1)
        #sorted_trds['pnl']  = sorted_trds.apply(lambda row: (row['last_px'] - row['px'])*-1 if row['side'] == 'B' else row['last_px'] - row['px'], axis=1)
    else:
        sorted_trds['pnl']  = sorted_trds.apply(lambda row: (row['last_px'] - row['px'])*row['size'] if row['side'] == 'B' else (row['last_px'] - row['px'])*row['size']*-1, axis=1)
        #sorted_trds['pnl']  = sorted_trds.apply(lambda row: (row['last_px'] - row['px'])* -1 *row['size'] if row['side'] == 'B' else (row['last_px'] - row['px'])*row['size'], axis=1) 

    total_pnl = sorted_trds['pnl'].sum()

    return total_pnl


def maker_or_taker(data, trds):
    sorted_trds = trds.sort_values(by='ts_ms')
    sorted_data = data.sort_values(by='ts_ms')

    sorted_trds['bid'] = sorted_trds['ts_ms'].apply(closest_px, data=sorted_data, fld='bid')
    sorted_trds['ask'] = sorted_trds['ts_ms'].apply(closest_px, data=sorted_data, fld='ask')

    sorted_trds['maker_or_taker'] = sorted_trds.apply(check_trd, axis=1)

    print(sorted_trds['maker_or_taker'].value_counts())
    return sorted_trds

def calc_pnl(data, trds):
    data['mid'] = (data['bid'] + data['ask'])/2
    sorted_trds = trds.sort_values(by='ts_ms')
    sorted_data = data.sort_values(by='ts_ms')

    sorted_data['trd_idx']    = sorted_data['ts_ms'].apply(update_trd_list, trds=sorted_trds)
    sorted_data['pnl']        = sorted_data.apply(calc_pnl_by_interval, axis=1, trds=sorted_trds)
    sorted_data['peak']       = sorted_data['pnl'].cummax()
    sorted_data['drawdown']   = abs(sorted_data['peak'] - sorted_data['pnl'])
    sorted_data['%_drawdown'] = (sorted_data['drawdown'] / sorted_data['peak'])*100

    max_drawdown_idx = sorted_data['%_drawdown'].idxmax()
    max_drawdown = sorted_data.iloc[max_drawdown_idx]

    print(f"max_drawdown: {max_drawdown['%_drawdown']} timestamp: {max_drawdown['ts_ms']}")

    return sorted_data
    
#===========================================

def main(data, trds, horizons, normalized=True):

    tmp = {'horizon': [], 'pnl': []}

    for row in tqdm(horizons):
        tmp['horizon'].append(row)
        if normalized:
            tmp['pnl'].append(calc_agg_pnl(data, trds, row))
        else:
            tmp['pnl'].append(calc_agg_pnl(data, trds, row, normalized=False))

    df = pd.DataFrame(tmp)
    df.set_index('horizon', inplace=True)

    return df



#===========================================

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mkt', type=str, default=None)
    parser.add_argument('--plot-data', action='store_true')
    parser.add_argument('--check-strat', action='store_true')
    parser.add_argument('--max-drawdown', action='store_true')
    parser.add_argument('--exec', action='store_true')
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--plot-pnl', action='store_true')
    parser.add_argument('--num-horizons', type=int, required=False)

    args = parser.parse_args()

#===========================================

    DATA_FILE_NAME = '_md.csv'
    TRDS_FILE_NAME = '_trades.csv'

    with open('horizon_ticks', 'r') as file:
        content = file.read()

    horizons = json.loads(content)
    
    if args.num_horizons:
        horizons = horizons[:args.num_horizons]

    if args.mkt:
        data_csv = args.mkt + DATA_FILE_NAME
        trds_csv = args.mkt + TRDS_FILE_NAME

        data = pd.read_csv(data_csv)
        trds = pd.read_csv(trds_csv)


        if args.exec:
            if args.normalized:
                df = main(data, trds, horizons)
                if args.plot_pnl:
                    plot_mark_out(df, mkt=args.mkt)
            else:
                df = main(data, trds, horizons, normalized=False)
                if args.plot_pnl:
                    plot_mark_out(df, normalized=False, mkt=args.mkt)
        else:
            if args.check_strat:
                df = maker_or_taker(data, trds)

            if args.max_drawdown:
                calc_pnl(data[:100000], trds)

            if args.plot_data:
                plot_ts(data, trds, horizons)

    else:
        print('Need to specify mkt')




          

