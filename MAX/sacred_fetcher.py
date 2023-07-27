#!/usr/bin/env python
import pymongo
import gridfs

from bson.objectid import ObjectId

import json
import csv

import os

import argparse

import pandas as pd
import altair as alt


def fetch_experiment(run_id, db_name, mongo_url='mongodb://localhost:27017', get_output=False, get_metrics=False, get_artifacts=False):
    client = pymongo.MongoClient(mongo_url)
    runs_db = client[db_name].runs
    metrics_db = client[db_name].metrics
    files_db = gridfs.GridFS(client[db_name])

    result = runs_db.find_one({'_id': run_id})

    os.makedirs(f'logs/{run_id}/', exist_ok=True)

    if get_output:
        with open(f'logs/{run_id}/config.json', 'w') as f:
            j = json.dumps(result['config'], indent=4)
            print(j, file=f)

        with open(f'logs/{run_id}/stdout.txt', 'w') as f:
            f.write(result['captured_out'])

    if get_metrics:
        metrics_path = f'logs/{run_id}/metrics'
        os.makedirs(metrics_path, exist_ok=True)

        for metric in result['info']['metrics']:
            metric_name = metric['name']
            metric_entry = list(metrics_db.find({'_id': ObjectId(metric['id'])}))[0]

            curve = list(zip(metric_entry['steps'], metric_entry['values']))

            metric_filename_root = f'{metrics_path}/{metric_name}'
            metric_filename_csv = f'{metric_filename_root}.csv'
            with open(metric_filename_csv, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(curve)

            # save plot
            df = pd.read_csv(metric_filename_csv)
            df.columns = ['step', metric_name]

            alt.Chart(df, width=1000, height=500).mark_line(point=True).encode(
                x='step',
                y=metric_name
            ).interactive().save(f'{metric_filename_root}.html', format='html')

    if get_artifacts:
        os.makedirs(f'logs/{run_id}/artifacts/', exist_ok=True)
        for artifact in result['artifacts']:
            artifact_file_name = artifact['name']
            artifact_file_id = artifact['file_id']
            with open(f'logs/{run_id}/artifacts/{artifact_file_name}', 'wb') as f:
                f.write(files_db.get(artifact_file_id).read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fetch results from sacred and dump in a directory',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--id', type=int, help='run id of the experiment')
    parser.add_argument('--db', type=str, help='name of sacred database')
    parser.add_argument('--url', type=str, default='mongodb://localhost:27017', help='url of mongo db')
    parser.add_argument('-o', action='store_true', help='get captured output')
    parser.add_argument('-m', action='store_true', help='get metrics')
    parser.add_argument('-a', action='store_true', help='get artifacts')
    args = parser.parse_args()

    fetch_experiment(run_id=args.id, db_name=args.db, mongo_url=args.url, get_output=args.o, get_metrics=args.m, get_artifacts=args.a)
