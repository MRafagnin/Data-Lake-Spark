import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    '''
        - Instantiate Spark session
    '''
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    '''
        - Read Song Data files
        - Create Tables
        - Write data back into S3
    '''
    # get filepath to song data file
    song_data = f"{input_data}song_data/*/*/*/*.json"
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select("song_id", "title", "artist_id", "year", "duration") \
                    .dropDuplicates(['song_id'])
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id') \
                     .parquet(os.path.join(output_data, 'songs'), 'overwrite')

    # extract columns to create artists table
    df.createOrReplaceTempView("artists_table_DF")
    
    artists_table = spark.sql("""
                                SELECT  artist_id        AS artist_id,
                                        artist_name      AS name,
                                        artist_location  AS location,
                                        artist_latitude  AS latitude,
                                        artist_longitude AS longitude
                                            FROM artists_table_DF
                            """)
    
    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data, 'artists'), 'overwrite')


def process_log_data(spark, input_data, output_data):
    '''
        - Read Log Data files
        - Create Tables
        - Write data back into S3
    '''
    # get filepath to log data file
    log_data = f"{input_data}log-data/*/*/*.json"

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table
    df.createOrReplaceTempView("users_table_DF")
    
    users_table = spark.sql("""
                            SELECT DISTINCT 
                                userId    AS user_id, 
                                firstName AS first_name,
                                lastName  AS last_name,
                                gender    AS gender,
                                level     AS level
                                    FROM users_table_DF
                                        WHERE userId IS NOT NULL
                        """)
    
    # write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, 'users'), 'overwrite')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x:  datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
    df = df.withColumn('start_time', get_timestamp('ts'))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d'))
    df = df.withColumn("datetime", get_datetime(df.ts))
    
    # extract columns to create time table
    time_table = df.withColumn("hour", hour("start_time")) \
                    .withColumn("day", dayofmonth("start_time")) \
                    .withColumn("week", weekofyear("start_time")) \
                    .withColumn("month", month("start_time")) \
                    .withColumn("year", year("start_time")) \
                    .select("ts","start_time","hour", "day", "week", "month", "year").drop_duplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.parquet(os.path.join(output_data, "time_table/"), \
                                 mode='overwrite', \
                                 partitionBy=["year","month"])

    # read in song data to use for songplays table
    song_df = spark.read\
                   .format("parquet")\
                   .option("basePath", os.path.join(output_data, "songs/"))\
                   .load(os.path.join(output_data, "songs/*/*/"))

    # extract columns from joined song and log datasets to create songplays table  
    songplays_table = df.join(song_df, (df.song == song_df.title) & \
                                       (df.artist == song_df.artist_name) & \
                                       (df.length == song_df.duration), 'left_outer'). \
            select(
                df.timestamp,
                col("userId").alias('user_id'),
                df.level,
                song_df.song_id,
                song_df.artist_id,
                col("sessionId").alias("session_id"),
                df.location,
                col("useragent").alias("user_agent"),
                year('datetime').alias('year'),
                month('datetime').alias('month'))
    
    songplays_table = songplays_table.join(time_table, songplays_table.start_time == time_table.start_time, how="inner")
    
    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.parquet(os.path.join(output_data, 'songplays_table/'), \
                                  mode='overwrite', \
                                  partitionBy=["year", "month"])

def main():
    '''
        - Calls on above functions
        - Defines S3 bucket links
    '''
    
    spark = create_spark_session()
    
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://test-output-spark"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()