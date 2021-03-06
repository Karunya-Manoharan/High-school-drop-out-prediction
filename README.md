# Setup

Runs on Python 3.7

1. Activate a virtual environment `venv activate <name of environment>`
2. `pip install -r requirements.txt`

# Getting Started

Running the pipeline revolves around `run.py`. Running the entire pipeline with the following command.
`python src/run.py run --config_path <path to spec file> --secret_path <path to file with PGSQL db connection info>`

# Structure
- `configs` consists of pipeline configurations in YAML format for various experiments.

Inside `src`, we have the following modules/files to our code:
- `load_data.py`: This file is in charge of providing data from a variety of sources, but ultimately converting everything into a dataframe indexed by a student's unique lookup id. This includes aggregating records and fetching columns from SQL tables.
- `preprocessors/`: This directory consists of all the preprocessing functions that can be applied to the dataframes that are loaded. These functions are in charge of labeling data, processing features, and performing splits into training and test data. All preprocessors implement a `transform` method that applies a specific transformation of the data.
- `model.py`: This file specifies the evaluation and training of the model in the pipeline. It takes the train and test sets generated by preprocessors and fits and evaluates a model on them. It then saves the model and evaluation results.
- `configloader.py`: This contains `load_config` which builds the entire pipeline from the loaded configuration. Examples may be seen in the `configs` directory. There main components in a configuration are as follows:
    - `model`: This is the specification for the ML model. Available models are in the `models/` folder.
    - `preprocessor`: This is a list of preprocessor specifications. The loaded data is processed sequentially by this list of preprocessors. All possible specifiable preprocessors are listed in the `preprocessor/` directory.
    - `loaders`: This is dictionary with the key value as the name of a data loader, and the value as a dictionary of arguments that are passed to the data loader function. Loader functions are in `load_data.py` and annotated by their data loader name with `add_loader`.
    - `labeler`: Type of labeler for deciding what label to assign students. We aim to have all students that *graduate within 3 years of entering 10th grade* to be assigned are 0, and otherwise assigned 1. Types of labelers can be found in `preprocessor/label.py`
    - `splitter`: How we split up our data into train test splits. Types of splitters can be found in `preprocessor/data_splitter.py`

# Help
## Supported Models
To specify a model, use a config file in the config folder. 
Each model name links to the corresponding scikit learn documentation.
Please look at that documentation to see the parameters that can be changed and the assumptions that are made for each model.
The following models are currently supported:
- [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
- Naive Bayes
  - [Gaussian](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB)

## Available Features
Features are included in the model via "Loaders" in the `config` file. These are maintained as "feature store" tables in the database, where each table row contains a unique `student_lookup` with all other information about the student expressed as column attributes. The following [spreadsheet](https://drive.google.com/file/d/1mcrw67T3OkFktmZxgUOmRwRp0aRohp1E/view?usp=sharing) contains running lists of features/attributes available in each table. 

## Constructed SQL Tables and Views
The "sketch" schema is writable and so holds all of the database objects we create from the data provided in the "clean" schema. 

### View: sketch.grade_9_gpa

#### What the query returns:

student_lookup | school_year | district | school | gpa_9 | gpa_9_missing | school_gpa_9_rank | school_gpa_9_decile

which gives us a row giving student ID, school year, school district, school code/school name (as available), ninth grade gpa, binary where 1 means student is missing GPA data, within-school and class gpa rank, within-school and class gpa decile.

#### NOTE: Rank and Decile only calculated for those with GPA data available. If you have a NULL gpa, you are listed as GPA = 0, GPA_MISSING = 1 and GPA DECILE = NULL.
