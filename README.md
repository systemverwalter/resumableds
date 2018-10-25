# resumableds
A Python class that supports Data Science projects.

resumableds supports you in writing data science scripts including save/resume functionality.

Data can be saved and resumed avoiding unnessary retrievals of raw data from data storages.

The data directory structure is inspired by cookiecutter-data-science (https://drivendata.github.io/cookiecutter-data-science/).

The class also supports the statement 'Analysis is a DAG' (https://drivendata.github.io/cookiecutter-data-science/#analysis-is-a-dag).


resumableds is written in pure Python and it is intended to be used within Jupyter notebooks.


### Example
<code>
  
proj1 = RdsProject('project1') # create object from class (creates the dir if it doesn't exist yet)

proj1.raw.df1 = pd.DataFrame() # create dataframe as attribute of proj1.raw (RdsFs 'raw')

proj1.defs.variable1 = 'foo' # create simple objects as attribute of proj1.defs (RdsFs 'defs')

proj1.save() # saved attributes of all RfdFs in proj1 to disk

</code>

This will result in the following directory structure (plus some overhead of internals):

- <output_dir>/defs/var_variable1.pkl
- <output_dir>/raw/df1.pkl
- <output_dir>/raw/df1.csv

Note, pandas dataframes are always dumped as pickle for further processing and as csv for easy exploration. The csv files are never read back anymore.

Later on or in another python session, you can do this:

<code>
  
proj2 = RdsProject('project1') # vars and data are read back to their original names

proj2.defs.variable1 == 'foo' # ==> True

isinstance(proj2.raw.df1, pd.DataFrame) # ==> True

</code>



