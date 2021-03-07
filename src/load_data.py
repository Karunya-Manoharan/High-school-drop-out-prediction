from typing import List

import sys
import psycopg2 as pg2  # Preferred cursor connection
from sqlalchemy import create_engine  # preferred for pushing back to DB
import yaml
import pandas as pd
import numpy as np

_Loader_Registry = {}


def add_loader(name):
    def _add_loader(fn):
        _Loader_Registry[name] = fn
        return fn

    """
    Any function annotated with add_loader must implement support for database cursor at the key 'cur'
    """

    return _add_loader


def get_loader(name):
    return _Loader_Registry[name]


def connect_cursor(USERPATH):
    """Connect a cursor to database.

    Input: Path to authorized user "secrets.yaml" file that is
           expected to include connection parameters
    Output: Database cursor object

    Future consideration:
    enable user to input latest year of available data instead
    of hardcoding (e.g., 2011 current cutoff based on 2015 latest data)
    """

    # Prepare secrets file to connect to DB
    with open(USERPATH, 'r') as f:
        # loads contents of secrets.yaml into a python dictionary
        secret_config = yaml.safe_load(f.read())

    # Set database connection to `conn`
    db_params = secret_config['db']
    conn = pg2.connect(host=db_params['host'],
                       port=db_params['port'],
                       dbname=db_params['dbname'],
                       user=db_params['user'],
                       password=db_params['password'])

    # Connect cursor with psycopg2 database connection
    cur = conn.cursor()

    return cur


def get_graduation_info(cur) -> pd.DataFrame:
    """Fetch relevant student_lookups as rows with columns indicating known and
    unknown withdraw reasons (i.e., year graduated, year dropped out, year
    transferred, year withdrew)

    Input: Database cursor
    Output: Pandas DataFrame of student_lookups (IDs) and their
            known withdraw outcomes based on raw database data

    Future consideration:
    enable user to input latest year of available data instead
    of hardcoding (e.g., 2013 current cutoff based on 2015 latest data
    and expected cohort of incoming 10th grade students + overall 4-yr on-time graduation)
    """

    # SQL SELECT statement to gather all students that entered 9th grade
    # prior to 2011, the latest year for which we could feasibly see
    # graduation outcomes.

    # Along with unique entrants, get their noted withdraw status
    # which covers graduation, dropout, transfers and withdraws.
    cur.execute('''
                select *
                from (
		               SELECT *, ROW_NUMBER() OVER
		                    (PARTITION BY student_lookup
                            ORDER BY student_lookup) AS rnum
		               FROM sketch.hs_withdraw_info_2 hwi) t
                where t.rnum = 1
                and t.entry_year >= 2007 and t.entry_year <= 2013
                ''')

    # Use cursor to fetch all rows into a list
    rows = cur.fetchall()

    # Build dataframe from rows
    df = pd.DataFrame(rows, columns=[name[0] for name in cur.description])

    # Make sure student_id is an int
    df['student_lookup'] = df['student_lookup'].astype('int')

    # Additional step to drop the students with no 9th grade history
    s = no_ninth_grade(cur)

    drop_set = list(set(df['student_lookup']).intersection(set(s)))

    df = df[~(df['student_lookup'].isin(drop_set))]

    return df

def no_ninth_grade(cur):
    """Fetch student_lookups for students that entered 10th grade but
    for which we have no 9th grade data. 

    Input: Database cursor
    Output: Pandas DataFrame of student_lookups (IDs)
    """
    cur.execute('''
    select distinct student_lookup from sketch.hs_withdraw_info_2 hwi 
     where student_lookup not in 
           (select distinct student_lookup from clean.all_snapshots where grade = 9) 
           and hwi.entry_year between 2007 and 2013;
           ''')

    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=[name[0] for name in cur.description])
    s = df['student_lookup'].astype('int')
    return s

def to_sql(USERPATH, df):
    """Function will push processed DataFrame to the database.

    Input: Path to secrets file to connect to db and dataframe

    Output: None

    Future work: Generalize
    """
    # Prepare secrets file to connect to DB
    with open(USERPATH, 'r') as f:
        # loads contents of secrets.yaml into a python dictionary
        secret_config = yaml.safe_load(f.read())

    # Set connection to `engine` for uploading table
    db_params = secret_config['db']
    engine = create_engine(
        'postgres://{user}:{password}@{host}:{port}/{dbname}'.format(
            host=db_params['host'],
            port=db_params['port'],
            dbname=db_params['dbname'],
            user=db_params['user'],
            password=db_params['password']))

    # Drop table if exists since it will be replaced
    engine.execute('''drop table if exists sketch.ninth_grade_entries''')
    engine.execute("COMMIT")

    # Create the table in sketch as 'ninth_grade_entries'
    create_sql = '''CREATE TABLE sketch.ninth_grade_entries ('''

    # This loop concatenates column names with SQL VARCHAR datatype
    # and appends it to the `create_sql' string. It calls all columns
    # into VARCHAR as default so it could handle columns with text
    # and/or number data.
    for col in df.columns:
        create_sql += col.strip("'") + ' varchar,'

    # Execute create query and commit table to database.
    engine.execute(create_sql.rstrip(',') + ');')
    engine.execute("COMMIT")

    df.pg_copy_to('ninth_grade_entries',
                  engine,
                  schema='sketch',
                  index=False,
                  if_exists='replace')


@add_loader("snapshot")
def get_snapshot_features(col_names: List[str], cur):
    """Function to get features from all_snapshots.

    Input: Databse cursor object and selected columns from
           clean.all_snapshots table in schools database

    Output: DataFrame
    """
    # Make col_names into a string for the OUTER query
    tcol_str = ''
    for col in col_names[:-1]:
        tcol_str += ('t.' + col + ', ')

    # Append last element without comma delimiter
    tcol_str = tcol_str + 't.' + col_names[-1]

    # Make col_names into a string for the INNER query
    col_str = ''
    for col in col_names[:-1]:
        col_str += (col + ', ')

    # Append last element without comma delimiter
    col_str = col_str + col_names[-1]

    qry = '''select distinct t.student_lookup, ''' + tcol_str + ''' from (select distinct student_lookup, school_year, ROW_NUMBER() OVER (PARTITION BY student_lookup) as rnum, ''' + col_str + ''' from clean.all_snapshots where grade = 9 order by student_lookup, school_year) t where t.rnum=1;'''

    cur.execute(qry)

    rows = cur.fetchall()

    df = pd.DataFrame([[int(row[0])] + list(row)[1:] for row in rows],
                      columns=[name[0] for name in cur.description])

    # Make sure student_id is an int
    df['student_lookup'] = df['student_lookup'].astype('int')

    return df


@add_loader("disability_and_intervention")
def get_disadvantage_features(cur):

    qry = ''' select * from sketch.disadv_and_intervention;'''

    cur.execute(qry)

    rows = cur.fetchall()

    df = pd.DataFrame([[int(row[0])] + list(row)[1:] for row in rows],
                      columns=[name[0] for name in cur.description])

    # Make sure student_id is an int
    df = df.astype('int')
    return df


@add_loader("marks")
def get_course_mark_features(cur):

    qry = ''' select * from sketch.marks_by_subject; '''

    cur.execute(qry)

    rows = cur.fetchall()

    df = pd.DataFrame([[int(row[0])] + list(row)[1:] for row in rows],
                      columns=[name[0] for name in cur.description])

    # Letter and numeric are duplicative in a way, so drop 1
    df = df[df.columns[~df.columns.str.contains('letter')]]

    # Make sure student_id is an int
    df['student_lookup'] = df['student_lookup'].astype('int')

    return df


@add_loader("absence_and_suspension")
def get_absence_features(cur):

    qry = ''' select * from sketch.absences; '''

    cur.execute(qry)

    rows = cur.fetchall()

    df = pd.DataFrame([[int(row[0])] + list(row)[1:] for row in rows],
                      columns=[name[0] for name in cur.description])

    df = df.replace(np.nan, 0)

    # Make sure student_id is an int
    df['student_lookup'] = df['student_lookup'].astype('int')

    return df

@add_loader("absence_discipline")
def get_absence_features(cur):

    qry = ''' select * from sketch.absence_discipline; '''

    cur.execute(qry)

    rows = cur.fetchall()

    df = pd.DataFrame([[int(row[0])] + list(row)[1:] for row in rows],
                      columns=[name[0] for name in cur.description])

    df = df.replace(np.nan, 0)

    # Make sure student_id is an int
    df['student_lookup'] = df['student_lookup'].astype('int')

    return df


@add_loader("std_test")
def get_test_score_features(cur):

    qry = ''' select * from sketch.std_test; '''

    cur.execute(qry)

    rows = cur.fetchall()

    df = pd.DataFrame([[int(row[0])] + list(row)[1:] for row in rows],
                      columns=[name[0] for name in cur.description])

    # We don't have anything going back so far, so keep it 7-9 grades
    df = df[df.columns[~df.columns.str.contains('ogt')]] 

    # Make sure student_id is an int
    df['student_lookup'] = df['student_lookup'].astype('int')
    return df


@add_loader("snapshot_non_mark")
def get_snapshot_non_mark_features(cur):

    qry = '''select student_lookup, ethnicity, gifted, limited_english
            from clean.all_snapshots a where a.grade = 9'''
    cur.execute(qry)

    rows = cur.fetchall()

    data = pd.DataFrame([[int(row[0])] + list(row)[1:] for row in rows],
                        columns=[name[0] for name in cur.description])
    df = data.drop_duplicates(subset=['student_lookup'], keep='first')

    df['limited_english_cate'] = np.where(
        df['limited_english'].str.contains('N'), 1, 0)

    df['gifted_c'] = np.where(df['gifted'].str.contains('N'), 1, 0)

    df = pd.get_dummies(df, columns=['ethnicity'], prefix=['ethnicity'])
    df = df.drop(['gifted', 'limited_english'], axis=1)
    #cur.execute(qry)

    #rows = cur.fetchall()

    #df = pd.DataFrame([[int(row[0])] + list(row)[1:] for row in rows],
    #                 columns=[name[0] for name in cur.description])

    # Make sure student_id is an int
    df['student_lookup'] = df['student_lookup'].astype('int')

    return df


@add_loader("repeat_grade_count")
def repeat_grade_count(cur):
    qry = '''select distinct student_lookup, grade, school_year
            from clean.all_grades a'''

    cur.execute(qry)

    rows = cur.fetchall()

    data = pd.DataFrame([[int(row[0])] + list(row)[1:] for row in rows],
                        columns=[name[0] for name in cur.description])

    data = data[data.duplicated(subset=['student_lookup', 'grade'],
                                keep=False)]

    df = data.groupby(['student_lookup', 'grade']).school_year.count()
    df = pd.DataFrame(df)
    df.reset_index(inplace=True)
    df2 = pd.DataFrame(df.groupby(['student_lookup']).grade.count())
    df2.reset_index(inplace=True)
    df2.columns = ['student_lookup', 'repeat_grade_count']
    df2['student_lookup'] = df2['student_lookup'].astype('int')

    return df2


@add_loader("grade_9_gpa")
def grade_9_gpa(cur):

    qry = ''' select student_lookup, gpa_9, school_gpa_9_rank, school_gpa_9_decile
from sketch.grade_9_gpa; '''

    cur.execute(qry)

    rows = cur.fetchall()

    df = pd.DataFrame([[int(row[0])] + list(row)[1:] for row in rows],
                      columns=[name[0] for name in cur.description])

    # Make sure student_id is an int
    df['student_lookup'] = df['student_lookup'].astype('int')
    return df


@add_loader("demographics")
def get_demographic_features(cur):

    qry = ''' select * from sketch.demographics; '''

    cur.execute(qry)

    rows = cur.fetchall()

    df = pd.DataFrame([[int(row[0])] + list(row)[1:] for row in rows],
                      columns=[name[0] for name in cur.description])

    # Make sure student_id is an int
    df['student_lookup'] = df['student_lookup'].astype('int')
    return df


@add_loader("demographics_grade_10")
def get_demographic_features(cur):

    qry = ''' select * from sketch.demographics_grade_10; '''

    cur.execute(qry)

    rows = cur.fetchall()

    df = pd.DataFrame([[int(row[0])] + list(row)[1:] for row in rows],
                      columns=[name[0] for name in cur.description])

    # Make sure student_id is an int
    df['student_lookup'] = df['student_lookup'].astype('int')
    return df


@add_loader("school_district")
def get_school_features(cur):

    qry = ''' select * from sketch.school_district; '''

    cur.execute(qry)

    rows = cur.fetchall()

    df = pd.DataFrame([[int(row[0])] + list(row)[1:] for row in rows],
                      columns=[name[0] for name in cur.description])

    # We don't have anything going back so far, so keep it 7-9 grades
    df = df[df.columns[~df.columns.str.contains('grade_6')]] 

    # Make sure student_id is an int
    df['student_lookup'] = df['student_lookup'].astype('int')
    return df


## Deprecated
def deduplicate_columns(df):
    """Function will deduplicate instances of student_lookup. Most arise from
    students with multiple transfer years, and so this will focus on 1) purging
    students who transferred out of the area and 2) maintaining only one row
    for within-area transfers and capturing a single graduated/not graduated
    outcome.

    Output a cleaned dataframe with unique student_lookups now as index
    """
    pass


def impute_labels(df):
    """Function will complete label creation by imputing withdraw reason for
    all students that lack such outcomes."""
    pass


def aggregate_features(cur, df):
    """Function to add aggregate features to student_lookups.

    Input: Databse cursor object and relevant students dataframe

    Output: Merged dataframe linking aggregate features to student dataframe

    Future work: generalize feature aggregation and linking
    """
    # Get aggregate absence feature
    absence_query = '''
    select student_lookup, avg(days_absent) as avg_abs, avg(days_absent_unexcused) as avg_abs_unex
        from clean.all_snapshots
        where grade <9
        group by student_lookup;
    '''
    cur.execute(absence_query)

    rows = cur.fetchall()

    absence_df = pd.DataFrame(rows,
                              columns=[name[0] for name in cur.description])

    absence_df = absence_df.astype({'student_lookup': 'Int32'})

    # Get average grade feature
    grade_query = '''
    select distinct student_lookup, ogt_socstudies_ss, ogt_science_ss, ogt_write_ss, ogt_math_ss, ogt_read_ss, eighth_socstudies_ss, eighth_science_ss, eighth_math_ss, eighth_read_ss
        from clean.oaaogt;
    '''
    cur.execute(grade_query)

    rows = cur.fetchall()

    grade_df = pd.DataFrame(rows,
                            columns=[name[0] for name in cur.description])

    # Function to purge DNS [did not sit] and other non-numeric outcomes for now
    f = lambda x: None if x in ['DNA', 'INV', 'DNS'] else x

    for col in grade_df.columns:
        grade_df[col] = grade_df[col].apply(f)

    # Change dtypes to float and get averages for OGT tests and Eighth grade marks
    grade_df = grade_df.astype('float')

    grade_df['ogt_avg'] = grade_df[[
        'ogt_socstudies_ss', 'ogt_science_ss', 'ogt_write_ss', 'ogt_math_ss',
        'ogt_read_ss'
    ]].mean(axis=1)
    grade_df['grade_8_avg'] = grade_df[[
        'eighth_socstudies_ss', 'eighth_science_ss', 'eighth_math_ss',
        'eighth_read_ss'
    ]].mean(axis=1)

    grade_df = grade_df[['student_lookup', 'ogt_avg',
                         'grade_8_avg']].astype({'student_lookup': 'Int32'})

    # Merge features to relevant students
    feature_df = pd.merge(absence_df, grade_df, on='student_lookup')

    df = pd.merge(df, feature_df, how='left', on='student_lookup')

    # Close cursor and connection to database
    cur.close()

    return (df)
