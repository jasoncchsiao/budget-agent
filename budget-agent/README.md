Hey folks! Welcome to budget-agent project repo.

The completed project can be seen at
https://github.com/fuelx/budget-agent

The scope of this project aims to calibrate the very next day pacing and performance based on historical data

The various files in this project are to set up your environment, database connection, and etc to help you run the code.

Currently running in one vm instance w/ below script
fuelx_com/.local/share/virtualenvs/p-p-balancer-project-PzsUuV3y/bin/python $pp_home/adjustIO_write_to_db_v3.py
sudo /home/jhsiao_fuelx_com/.local/share/virtualenvs/p-p-balancer-project-PzsUuV3y/bin/python $pp_home/send_dailyio_adjusted_report.py
sudo gsutil cp $pp_home/data/*_$FILEDATE.* $gcs_home/
sudo gsutil cp $pp_home/logs/*$FILEDATE.log $gcs_home/

#!/usr/bin/env bash
# app.sh
export pp_home=/home/jhsiao_fuelx_com/Projects/p-p-balancer-project
export FILEDATE=$(date +%F)
export LOGFILE=$pp_home/logs/model_output.$FILEDATE.log
export gcs_home=gs://shop-like-model/BudgetCalculator
touch $LOGFILE
#$pp_home/pipenv run python app.py
#$pp_home/pipenv run python output_write_to_db2.py
sudo /home/jhsiao_fuelx_com/.local/share/virtualenvs/p-p-balancer-project-PzsUuV3y/bin/python $pp_home/app2.py
sudo /home/jhsiao_fuelx_com/.local/share/virtualenvs/p-p-balancer-project-PzsUuV3y/bin/python $pp_home/output_write_to_db2.py
sudo /home/jhsiao_fuelx_com/.local/share/virtualenvs/p-p-balancer-project-PzsUuV3y/bin/python $pp_home/adjustIO_write_to_db_v3.py
sudo /home/jhsiao_fuelx_com/.local/share/virtualenvs/p-p-balancer-project-PzsUuV3y/bin/python $pp_home/send_dailyio_adjusted_report.py
