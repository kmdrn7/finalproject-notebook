#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
import pyspark
import numpy as np
import pyspark.sql.functions as F
from joblib import load
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler 
from pywebhdfs.webhdfs import PyWebHdfsClient
from io import BytesIO

hdfs = PyWebHdfsClient(host='192.168.100.38',port='50070', user_name='vagrant')
bytes = BytesIO(hdfs.read_file('models/without-scaler/RandomForest.joblib'))
clf = load(bytes)

@F.udf(returnType=IntegerType())
def predict_udf(data):
    print(data)
    return 1
    # pred = clf.predict([data])
    # return int(pred[0])

spark = (SparkSession
      .builder
      .master('spark://192.168.100.38:7077')
      .appName('MalwareDetection')
      .config("spark.driver.memory", "512m")
      .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1,org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1")
      .config("spark.mongodb.input.uri","mongodb://ta:ta@192.168.100.29:27018/MalwareDetection.data?authSource=admin")
      .config("spark.mongodb.output.uri","mongodb://ta:ta@192.168.100.29:27018/MalwareDetection.data?authSource=admin")
      .getOrCreate())

spark.conf.set("spark.sql.caseSensitive", "true")

sc = spark.sparkContext

schema = spark.read.load("hdfs://lab-1:9000/schema/schema.json", format="json")

raw = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "192.168.100.29:19092").option("subscribe", "netflow_formatted").option("startingOffsets", "latest").load()

cols = ['Protocol', 'Flow Duration', 'Fwd Packet Length Max',
      'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Fwd IAT Total',
      'Bwd IAT Total', 'Packet Length Max', 'Packet Length Mean',
      'Packet Length Std', 'FIN Flag Count', 'SYN Flag Count',
      'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg',
      'Bwd Segment Size Avg', 'FWD Init Win Bytes', 'Fwd Seg Size Min',
      'Idle Mean', 'Idle Max', 'Idle Min']

va = VectorAssembler(inputCols=cols, outputCol="SS_features")

parsed = raw.selectExpr("cast (value as string) as json").select(F.from_json("json",schema.schema).alias("data")).select("data.*")
extracted = parsed.select(cols).drop("Label")
extracted = va.transform(extracted)
predicted = extracted.withColumn('Prediction', predict_udf(extracted["SS_features"]))

def foreach_batch_function(df, idx):
    print(df.show())
    pass

stream = predicted.writeStream.foreachBatch(foreach_batch_function).start()
stream.awaitTermination()

