# spark-ml
Trying out Spark for ML

Steps:
  1. `docker-compose up -d --scale worker=2`
  1. `docker-compose exec master /bin/bash bin/spark-shell`
  1. paste the contents of `random_forest.scala` into spark shell
  1. congrats, you just trained a RandomForest model
