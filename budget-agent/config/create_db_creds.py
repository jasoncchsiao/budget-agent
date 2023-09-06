import yaml

db_creds = {
    'REDSHIFT':{'db': 'bids',
                'host': 'ignite-bid-cluster.ce0ui3hfe3ue.us-west-2.redshift.amazonaws.com',
                'port': '5439',
                'user': 'jasonhsiao',
                'password': 'r8MiC08x07@Z'},
    'FUELASSET':{'db': 'fuelAsset',
                 'host': 'fuelasset.db.fuel451.com',
                 'port': 3306,
                 'user': 'jasonhsiao',
                 'password': 'y1U$hcSPVsQW'},
    'FUELBIDDING':{'db': 'fuelData',
                   'host': '162.222.180.40',
                   'port': 3306,
                   'user': 'jasonhsiao',
                   'password': 'a0X@vHLsMzVaz'},
    'FUELASSET_DEV':{'db': 'fuelAsset',
                     'host': '35.226.210.107',
                     'port': 3306,
                     'user': 'jasonhsiao',
                     'password': '123'},
    'GMAIL_API':{'email': 'jhsiao@fuelx.com',
                 'password': 'osnjggckmfjafhlq'}
}

with open('db_creds.yaml', 'w') as fh:
    yaml.dump(db_creds, fh)
