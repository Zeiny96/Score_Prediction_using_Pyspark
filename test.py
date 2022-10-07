################################################ Imports
import sqlite3
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import Tokenizer,StopWordsRemover,CountVectorizer,IDF
from pyspark.ml.feature import StringIndexer
import os
import shutil
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import PipelineModel
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt


################################################ Inputs
sql_database_file = "./database.sqlite"
csv_file = "./database.csv"
model_path = "./model"
output_csv_file = "./output.csv"

################################################ Reading SQL database file and dumping it into CSV database file
if not os.path.exists(csv_file):
	database = sqlite3.connect(sql_database_file)
	db_df = pd.read_sql_query("SELECT * FROM Reviews", database)
	db_df.to_csv(csv_file, index=False)


################################################ Reading CSV database file
spark = SparkSession.builder.appName("App").config("spark.driver.memory", "30g").getOrCreate()
df = spark.read.csv("database.csv",header=True,inferSchema=True)


################################################ Defining desired columns and their types
float_columns = set(["Id"])
str_columns = set(["Summary", "Text"])
columns = list(float_columns.union(str_columns))
df = df.select(columns)

################################################ Casting columns to their types after removing wrong types or Nulls
for colum in df.columns:
	if colum in float_columns:
		df = df.filter(col(colum).cast("float").isNotNull())
		df = df.withColumn(colum, df[colum].cast('float'))
	else:
		df = df.filter(col(colum).cast("string").isNotNull())
		df = df.withColumn(colum, df[colum].cast('string'))


################################################ Preparing the data
df_concatenated = df.withColumn('Summary_Text', concat(col('Summary'),lit('and'), col('Text')))


################################################ Defining, training and saving the model
Cl_model = PipelineModel.load(model_path)


################################################ Make predictions based on ids
predictions = Cl_model.transform(df_concatenated)
df = df.withColumn("Id", monotonically_increasing_id())
predictions = predictions.withColumn("prediction", monotonically_increasing_id())
df_pred = df.join(predictions, "Id", "left").select("Id", predictions.prediction.alias("prediction"))


################################################ Dumping the results
df_pred.repartition(1).write.format('com.databricks.spark.csv').save(output_csv_file+"_tmp",header = 'true')
os.system(f"mv {output_csv_file}_tmp/part* {output_csv_file}")
shutil.rmtree(output_csv_file+"_tmp")
