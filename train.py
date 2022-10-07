################################################ Imports
import sqlite3
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import Tokenizer,StopWordsRemover,CountVectorizer,IDF
from pyspark.ml.feature import StringIndexer
import os
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import PipelineModel
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt


################################################ Inputs
sql_database_file = "database.sqlite"
csv_file = "database.csv"
model_path = "./model"


################################################ Reading SQL database file and dumping it into CSV database file
if not os.path.exists(csv_file):
	database = sqlite3.connect(sql_database_file)
	db_df = pd.read_sql_query("SELECT * FROM Reviews", database)
	db_df.to_csv(csv_file, index=False)


################################################ Reading CSV database file
spark = SparkSession.builder.appName("App").config("spark.driver.memory", "30g").getOrCreate()
df = spark.read.csv("database.csv",header=True,inferSchema=True)


################################################ Checking the dataframe
df.show()


################################################ Defining desired columns and their types
float_columns = set(["HelpfulnessNumerator", "Score"])
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
df = df[df.HelpfulnessNumerator > 0]
df = df.withColumn('Summary_Text', concat(col('Summary'),lit(' and '), col('Text')))
tokenizer = Tokenizer(inputCol="Summary_Text",outputCol="Summary_Text_tokenized")
stopwords_remover = StopWordsRemover(inputCol="Summary_Text_tokenized",outputCol="Summary_Text_filtered_tokenized")
vectorizer = CountVectorizer(inputCol="Summary_Text_filtered_tokenized",outputCol="Summary_Text_raw")
idf = IDF(inputCol="Summary_Text_raw",outputCol="Summary_Text_vectorized")


################################################ Splitting the data
(trainDF,testDF) = df.randomSplit((0.75,0.25),seed=42)


################################################ Defining, training and saving the model
Cl = LogisticRegression(featuresCol="Summary_Text_vectorized", labelCol="Score")
pipeline = Pipeline(stages=[tokenizer, stopwords_remover, vectorizer, idf, Cl])
Cl_model = pipeline.fit(trainDF)
Cl_model.write().overwrite().save(model_path)


################################################ Loading the best model
Cl_model = PipelineModel.load(model_path)


################################################ Evaluating the model
predictions = Cl_model.transform(testDF)
y_true = predictions.select(['Score']).collect()
y_pred = predictions.select(['prediction']).collect()
cm = confusion_matrix(y_true, y_pred)
sn.heatmap(cm, annot=True, fmt='d')
plt.show()
print(classification_report(y_true, y_pred))


