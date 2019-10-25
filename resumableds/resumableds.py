import logging
import os
from time import time
import shutil
import glob
import pandas as pd
import datetime
import pickle
import yaml
import subprocess
import platform
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError
from hashlib import md5
from pandas.util import hash_pandas_object
import multiprocessing as mp
#import networkx

class PandasObjectHasher:
    '''
    Class to compare two dataframes (or the same dataframe at different times).
    This class will be used to determine if a loaded data object has changed
    since the last load from disk.
    '''
    
    def __init__(self, df):
        self.data_hash_exception_occured = False
        self.index_hash = self.__create_index_hash(df)
        self.columns_hash = self.__create_columns_hash(df)
        self.data_hash = self.__create_data_hash(df)
    
    def __create_index_hash(self, df):
        return df.index.values.tolist()
    
    def __create_columns_hash(self, df):
        if isinstance(df, pd.DataFrame):
            return df.columns.values.tolist()
        return None
    
    def __create_data_hash(self, df):
        data_hash = None
        try:
            data_hash = md5(hash_pandas_object(df).values).hexdigest()
        except Exception as e:
            # hashing dataframes with mutable objects like lists inside will throw an exception
            logging.debug(e) # debug because lib is also working without hashes
            self.data_hash_exception_occured = True
        return data_hash
        
    def index_changed(self, df):
        return self.__create_index_hash(df) != self.index_hash
    
    def columns_changed(self, df):
        return self.__create_columns_hash(df) != self.columns_hash
    
    def data_changed(self, df):
        return self.__create_data_hash(df) != self.data_hash

    def obj_changed(self, df):
        
        if self.data_hash_exception_occured:
            #no data hash available, play safe -> presume data is changed
            return True
    
        if self.index_changed(df):
            return True
        if self.columns_changed(df):
            return True
        if self.data_changed(df):
            return True
        return False


class RdsFs:
    '''
    Data Science "file system"
    The class  RdsFs handles syncing between memory and disk of python objects and pandas dataframes.
    It supports saving and resuming of abritrary python objects by means of pickling.
    Pandas dataframes are pickled for further processing and saved as csv files for easy exploration.
    The csv files are only saved but never read back.
    The directory can be copied or moved on file system level to another location and later resumed in python.
    The file names on disk correspond with the object name in python.
    Python objects (as well as dataframes) must be created as attribute of the object of this class.
    All attributes of this object will be synced between ram and disk when using ram2disk() or disk2ram()
    During loading from disk, the data objects are hashed for later comparison.
    During dumping to disk, a check is done to only dump if there is a change compared to the disk version.
    The class may not very useful on its own. It is used by class RdsProject.
    Users acutally should use RdsProject.

    Parameters
    ----------
    output_dir: string
        Path to the data directory; location of the data files on disk.

    Example
    -------
    proj1 = RdsFs('/mnt/data/project1') # create object from class
    proj1.df1 = pd.DataFrame() # create dataframe as attribute of proj1
    proj1.variable1 = 'foo' # create simple objects as attribute of proj1
    proj1.sync2disk() # save attributes of proj1 to disk

    This will result in two files in /mnt/data/project1 (plus some overhead of internals):
    - var_variable1.pkl
    - df1.pkl

    Later on or in another python session, you can do this:
    proj2 = RdsFs('/mnt/data/project1') # create object from class
    proj2.disk2ram() # reads files back to python objects
    proj2.variable1 == 'foo' ==> True
    isinstance(proj2.df1, pd.DataFrame) ==> True
    '''

    def __init__(self, output_dir, nof_processes, backend):

        self.internal_obj_prefix = 'var_'
        self.backend_file_extensions = {
                                         'pickle': '.pkl',
                                         'feather': '.feather',
                                         'parquet': '.parquet',
                                        }
        self.pandas_dump_functions = {
                                         'pickle': 'to_pickle',
                                         'feather': 'to_feather',
                                         'parquet': 'to_parquet',
                                        }
        
        self.pandas_read_functions = {
                                         'pickle': 'read_pickle',
                                         'feather': 'read_feather',
                                         'parquet': 'read_parquet',
                                        }
        
        self.output_dir = output_dir
        self.nof_processes = nof_processes
        self.backend = backend
        
        self.hash_objects = {}
        
        # max memory usage in GB allowed to do a csv dump
        self.max_memory_for_csv_dump = 2
        
        # names of internal objects to be excluded from pickle dump
        self.internals = (
                            'internals',
                            'internal_obj_prefix',
                            'pickle_file_ext',
                            'output_dir',
                            'hash_objects',
                            'max_memory_for_csv_dump',
                            'nof_processes',
                            'backend',
                            'backend_file_extensions',
                            'pandas_dump_functions',
                            'pandas_read_functions',
                          )

        logging.debug('output directory set to "%s"' % self.output_dir)
        self.make_output_dir()

    def make_output_dir(self):
        '''
        Creates the output directory to read/write files.
        '''
        #logging.debug('create "%s" if not exists' % self.output_dir)
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except FileExistsError as e:
            logging.debug(e)

    def clean(self):
        '''
        Deletes the output directory including all its content and recreates an empty directory.
        '''

        logging.debug('clean output directory "%s"' % self.output_dir)
        try:
            shutil.rmtree(self.output_dir)
        except Exception as e:
            logging.error(e)
            return False

        # recreate empty dir structure
        self.make_output_dir()

        return True
        
    def __load_pd_df(self, filename):
        '''
        Loads a pickle file into a dataframe.
        Returns a tuple of dataframe name (file name w/o extension), dataframe and dataframe hash

        Parameters
        ----------
        filename: string
            The absolute file name
        '''
        
        # find backend of a used file extension
        reverse_backend_lookup = {v:k for k,v in self.backend_file_extensions.items()}
        dataframe_name = os.path.basename(filename).split('.')[0]
        ext = os.path.basename(filename).split('.')[-1]
        ext = '.' + ext #dict contains leading dot in name :/
        use_backend = reverse_backend_lookup[ext]
        read_func = self.pandas_read_functions[use_backend]
        logging.debug('execute {} = pd.{}("{}")'.format(dataframe_name, read_func, filename))
        dataframe = getattr(pd, read_func)(filename)
        
        # create data hash object and add it to dict of hash objects
        logging.debug('create hash object to track changes')
        dataframe_hash = PandasObjectHasher(dataframe)
        
        return (dataframe_name, dataframe, dataframe_hash)

    def __dump_pd_df(self, dataframe, filename):
        '''
        Dumps a pandas dataframe / series to file.
        File format depends on backend setting.

        Parameters
        ----------
        dataframe: pd.DataFrame object
            The dataframe that should be pickled
        filename: string
            The base name of the file w/o extension
        '''
        
        # check if dump is required
        dump_required = True
        if filename in self.hash_objects.keys():
            if self.hash_objects[filename].obj_changed(dataframe):
                dump_required = True
            else:
                dump_required = False
        else:
            dump_required = True

        if not dump_required:
            logging.debug('no new dump required. Skip!')
            return False
        
        # actual dump    
        abs_fn = os.path.join(self.output_dir, filename)
        # Series will always be pickled; dataframes only if backend is pickle
        if isinstance(dataframe, pd.Series) or (self.backend == 'pickle'):
            abs_fn_pickle = abs_fn + self.backend_file_extensions['pickle']
            logging.debug('execute {}.to_pickle("{}")'.format(filename, abs_fn_pickle))
            dataframe.to_pickle(abs_fn_pickle)
        else:
            abs_fn_w_ext = abs_fn + self.backend_file_extensions[self.backend]
            dump_func_name = self.pandas_dump_functions[self.backend]
            logging.debug('execute {}.{}("{}")'.format(filename, dump_func_name, abs_fn_w_ext))
            getattr(dataframe, dump_func_name)(abs_fn_w_ext)
        
            
        # create new data hash object and add it to dict of hash objects.
        logging.debug('create hash object to track changes')
        self.hash_objects[filename] = PandasObjectHasher(dataframe) # new hash or updated hash    
        
        return True
                
    def __dump_df_pd_csv(self, dataframe, filename, sep=';', decimal=','):
        '''
        Dumps a dataframe to csv:
        - csv file for easy exploration (this file will not be read anymore)

        Parameters
        ----------
        dataframe: pd.DataFrame object
            The dataframe that should be pickled
        filename: string
            The base name of the file w/o extension
        sep: string, optional
            The csv field separator, defaults to ';'
        decimal: string, optional
            The csv decimal separator, defaults to ','
        '''
        
        #df mem usage in GB
        mem_usage = dataframe.memory_usage(index=True, deep=True)
        # dataframes return series; series return int
        if isinstance(mem_usage, pd.Series):
            mem_usage = mem_usage.sum()
        mem_usage = mem_usage / 1024 / 1024 / 1024

        if mem_usage < self.max_memory_for_csv_dump:
            abs_fn_csv = os.path.join(self.output_dir, filename) + '.csv'
            logging.debug('dump "%s" with sep="%s" and decimal="%s"' % (abs_fn_csv, sep, decimal))
            dataframe.to_csv(abs_fn_csv, sep=sep, decimal=decimal, header=True)
            return True
        else:
            logging.debug('no dump to csv since dataframe memory usage is too large. Skip!')
            return False
        
    def _ls(self):
        '''
        Returns output directory content including mtime.

        Returns
        -------
        Dict with file names as keys and mtime as values.
        '''

        #logging.debug('ls "%s"' % self.output_dir)
        ls_content = glob.glob(os.path.join(self.output_dir, '*'))
        ls_content = {f:str(datetime.datetime.fromtimestamp(os.path.getmtime(f))) for f in ls_content}
        #for k, v in ls_content.items():
        #    logging.debug('\t%s modified on %s' % (k, v))
        return ls_content

    def ls(self):
        '''
        Prints dataframe files from the output directory including mtime as returned by _ls().
        Internal python objects are skipped and not shown.
        '''
        return {os.path.basename(k): v for k, v in self._ls().items() if not os.path.basename(k).startswith(self.internal_obj_prefix)}

    def ram2disk(self, csv):
        '''
        Saves all attributes of this object as files to the output directory.
        '''
        t0 = time()
        pool = mp.Pool(processes=self.nof_processes)
        
        # for all attributes in object (except internals)...
        to_save = {k:v for k,v in self.__dict__.items() if k not in self.internals}
        saved = []
        for name, obj in to_save.items():
            logging.debug('save %s...' % name)
            pool.apply_async(
                              self._ram2disk1obj,
                              args=(obj, name, csv),
                              callback=lambda x: saved.append(x),
                            )
        pool.close()
        pool.join()
        
        if len(saved) == len(to_save.keys()):
            logging.debug('sync to disk done for "%s": %d objects in %.2fs' % (
                                                                                self.output_dir,
                                                                                len(saved),
                                                                                time() - t0
                                                                               )
                         )
        else:
            not_saved = [k for k in to_save.keys() if k not in saved]
            logging.error('sync to disk failed for "%s": objects "%s" not saved' % (self.output_dir, not_saved))
        

    def _ram2disk1obj(self, obj, name, csv):
        '''
        Saves obj in file (name).
        '''
        
        if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
                #if object is dataframe, dump it
                self.__dump_pd_df(obj, name)
                if csv:
                    self.__dump_df_pd_csv(obj, name)
        #TODO: if isinstance(obj, dask)
        #        #if object is dask dataframe, dump it
        #        self.__dump_dask_df(obj, name, csv)
        else:
            # if not a dataframe, pickle it
            base_name = self.internal_obj_prefix + name + self.backend_file_extensions['pickle']
            abs_fn = os.path.join(self.output_dir, base_name)
            logging.debug('dump "%s"' % abs_fn)
            with open(abs_fn, 'wb') as f:
                pickle.dump(obj, f)
        return name # to collect saved items back in a list

    def disk2ram(self):
        '''
        Reads all pickle files from the output directory
        and loads them as attributes of this object.
        '''
                
        t0 = time()
        pool = mp.Pool(processes=self.nof_processes)
        
        # get all data objects from dir
        to_load = {k:v for k, v in self._ls().items() if (k.endswith(self.backend_file_extensions['pickle'])) or (k.endswith(self.backend_file_extensions[self.backend]))}
        loaded = []
        for fn, mtime in to_load.items():
            logging.debug('load %s from %s...' % (fn, mtime))
            pool.apply_async(
                              self._disk2ram1obj,
                              args=(fn,),
                              callback=lambda x: loaded.append(x)
                            )
        pool.close()
        pool.join()
        
        # from list to internal dict
        for obj in loaded:
            self.__load_in_class(obj)
        
        loaded_class_objects = [k for k in self.__dict__.keys() if k not in self.internals]
        
        if len(loaded_class_objects) == len(to_load.keys()):
            logging.debug('sync to ram done for "%s": %d objects in %.2fs' % (
                                                                                self.output_dir,
                                                                             len(loaded_class_objects),
                                                                                time() - t0
                                                                               )
                         )
            return True
        else:
            not_loaded = [k for k in to_load.keys() if k not in loaded_class_objects]
            logging.error('sync to ram failed for "%s": files "%s" not loaded' % (self.output_dir, not_loaded))
            return False

    def _disk2ram1obj(self, fn):
        '''
        Reads a pickle file from the output directory
        and loads it as attribute of this object.
        '''
        
        var_name = None
        var = None
        var_hash = None
        
        base_name = os.path.basename(fn)
        
        if base_name.startswith(self.internal_obj_prefix):
            # internal objects (no dataframes)
            var_name = base_name[len(self.internal_obj_prefix):-1*len(self.backend_file_extensions['pickle'])]
            try:
                with open(fn, 'rb') as f:
                    var = pickle.load(f)
            except Exception as e:
                logging.error(e)
                logging.error('skip "%s" from loading into memory due to an exception. Functionality might be broken!' % var_name)
        else:
            #if object is dataframe, load dump
            if (fn.endswith(self.backend_file_extensions[self.backend])) or (fn.endswith(self.backend_file_extensions['pickle'])):
                var_name, var, var_hash = self.__load_pd_df(fn)
        
        # hash is None for non-dataframes (so for regular Python objects)
        return (var_name, var, var_hash)

    def __load_in_class(self, var_info):
        '''
        load results of multiproccesing functions into class objects
        '''
        var_name, var, var_hash = var_info
        if var_name:
            self.__dict__[var_name] = var
        if var_hash:
            self.hash_objects[var_name] = var_hash
    
    def __str__(self):
        '''
        Returns
        -------
        The output directory (w/o full path) as string.
        '''

        return 'DsProject directory "%s"' % self.output_dir.split(os.path.sep)[-1]

    def __repr__(self):
        '''
        Returns
        -------
        Returns a string that contains:
        - an overview of all files in the output directory
        - an overview of all loaded python objects (name and content) (for dataframes the shape is shown rather than the full content)
        '''

        files = '\n'.join(['\t%s: %s' % (str(k), str(v)) for k, v in self.ls().items()])
        objects = '\n'.join(['\t%s: %s' % (str(k), str(v)) if (not isinstance(v, pd.DataFrame)) and (not isinstance(v, pd.Series)) else '\t%s: %s' % (str(k), str(v.shape)) for k, v in {k:v for k,v in self.__dict__.items()if k not in self.internals}.items()])

        return '''
{caption}
{underline}
existing files:
{files}
loaded objects:
{objects}
'''.format(caption=str(self),
           underline='=' * len(str(self)),
           files=files,
           objects=objects)


class RdsProject:
    '''
    RdsProject incl. save/resume functionality.
    This class supports you in writing data science scripts.
    Data can be saved and resumed avoiding unnessary retrievals of raw data from data storages.

    Parameters
    ----------
    project_name: string
        The project name
    output_dir: string, optional
        Path to the data directory; location of the data files on disk.
        Defaults to the current working directory.
    dirs: list, optional
        List of sub-directory names that should be used in the project.
        Defaults to ['defs', 'external', 'raw', 'interim', 'processed']
    output_dir: string, optional
        Location of data files, defaults to ./<project_name>
    analysis_start_date: datetime (can also be string, will be converted automatically), optional
        Start date of the analysis.
        Defaults to today - analysis_timespan
    analysis_end_date: datetime (can also be string, will be converted automatically), optional
        End date of the analysis.
        Defaults to today.
    analysis_timespan: timedelta (can also be string, will be converted automatically), optional
        Defaults to 180 days.
    cell_execution_timeout: int, optional
        The execution timeout of a single cell in a process chain
        Defaults to 3600.
    make_configs: dict, optional
        'Make' configurations.
        Example: {'raw': ['get_sql_data.ipynb', 'get_nosql_data.ipynb']}
        Defaults to {}.
    start_clean: boolean, optional
        Skip resume if true.
        Defaults to False.
    nof_processes: int, optional
        Configure the max number of parallel processes used to read/write data
        Defaults to mp.cpu_count().
        
    Example
    -------
    proj1 = RdsProject('project1') # create object from class (creates the dir if it doesn't exist yet)
    proj1.raw.df1 = pd.DataFrame() # create dataframe as attribute of proj1.raw (RdsFs 'raw')
    proj1.defs.variable1 = 'foo' # create simple objects as attribute of proj1.defs (RdsFs 'defs')
    proj1.save() # saved attributes of all RfdFs in proj1 to disk

    This will result in the following directory structure (plus some overhead of internals):
    - <output_dir>/defs/var_variable1.pkl
    - <output_dir>/raw/df1.pkl
    - <output_dir>/raw/df1.csv

    Note, pandas dataframes are always dumped as pickle for further processing and as csv for easy exploration. The csv files are never read back anymore.


    Later on or in another python session, you can do this:
    proj2 = RdsProject('project1') # create object from class (doesn't touch the dir as it already exists) All vars and data is read back to their original names.
    proj2.defs.variable1 == 'foo' ==> True
    isinstance(proj2.raw.df1, pd.DataFrame) ==> True
    '''

    def __init__(self, 
                 project_name,
                 dirs=None,
                 output_dir=None,
                 analysis_start_date=None,
                 analysis_end_date=None,
                 analysis_timespan='180 days',
                 cell_execution_timeout=3600,
                 make_configs={},
                 start_clean=False,
                 nof_processes=100,
                 backend='pickle',
                 ):
        
        # project name
        self.project_name = project_name

        # define project's status file name
        self.status_file = '%s.yml' % self.project_name       
        
        # set number of processes (multiprocessing)
        # this is done here and will be used in start / resume towards RdsFs
        self.nof_processes = nof_processes if nof_processes <= mp.cpu_count() else mp.cpu_count() 
        
        
        # set names of output directories
        # external: files from outside this project,
        # external files can be copied here for further use
        self.EXTERNAL = 'external'
        # raw: raw data retrieved from a data storage (like SQL server)
        self.RAW = 'raw'
        # half ready results / in-between steps
        self.INTERIM = 'interim'
        # analysis results
        self.PROCESSED = 'processed'

        # defs: save definitions like column names, etc
        self.DEFS = 'defs'
        
        # get a list of data dirs that should be used
        self.output_dirs = []
        self.output_dirs = self.__update_dir_specs(dirs)
        
        
        # start clean if desired
        if start_clean:
            self.start(
                        self.output_dirs,
                        output_dir,
                        analysis_start_date,
                        analysis_end_date,
                        analysis_timespan,
                        cell_execution_timeout,
                        make_configs,
                        backend,
                       )
            self.clean()
            self.save()
        # resume if possible
        elif self.resume(dirs):
            logging.info('Project "%s" resumed' % self.project_name)
        else:
            self.start(
                        self.output_dirs,
                        output_dir,
                        analysis_start_date,
                        analysis_end_date,
                        analysis_timespan,
                        cell_execution_timeout,
                        make_configs,
                        backend
                       )

        logging.debug('output_dir set to "%s"' % self.output_dir)
        logging.debug('backend set to "%s"' % self.backend)
        logging.debug('analysis_start_date set to "%s"' % self.analysis_start_date)
        logging.debug('analysis_end_date set to "%s"' % self.analysis_end_date)
        logging.debug('analysis_timespan set to "%s"' % self.analysis_timespan)
        logging.debug('ready to rumble')

            
    def start(
                self,
                dirs,
                output_dir,
                analysis_start_date,
                analysis_end_date,
                analysis_timespan,
                cell_execution_timeout,
                make_configs,
                backend,
              ):
        '''
        Initiate new project.
        No files will be touched!

        Parameters
        ----------
        dirs: list, optional
            List of sub-directory names that should be used in the project.
            By default all subdirectories defined in the contructor are taken into account.
        '''

        # set ouput_dir
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = os.path.join('.', self.project_name)       
        
        # set backend binary format to read/write dataframes
        self.backend = backend
        
        # analsysis timespan
        self.analysis_timespan = analysis_timespan
        if not isinstance(self.analysis_timespan, pd.Timedelta):
            try:
                self.analysis_timespan = pd.Timedelta(self.analysis_timespan)
            except Exception as e:
                logging.error(e)
        
        # analysis start date
        self.analysis_start_date = analysis_start_date
        if self.analysis_start_date is None:
            self.analysis_start_date = pd.datetime.today() - self.analysis_timespan
        
        # analysis end date
        # defaults to today
        self.analysis_end_date = analysis_end_date
        if self.analysis_end_date is None:
            self.analysis_end_date = pd.datetime.today()
        
        # re-calculate timespan as it might be wrong due to overwritten start or end date
        self.analysis_timespan = self.analysis_end_date - self.analysis_start_date

        # set the exec timeout of a single cell for notebooks execution
        self.cell_execution_timeout = cell_execution_timeout

        # set make_configs
        self.make_configs = make_configs
        
        # dict ot store successful execution dates
        self.execution_dates_make_configs = {}
        
        # init working directories
        for sub_dir in dirs:
            self.__dict__[sub_dir] = RdsFs(
                                            os.path.join(self.output_dir, sub_dir), 
                                            nof_processes=self.nof_processes,
                                            backend=self.backend,
                                          )
        
        # save project properties in defs
        self.__kwargs2defs()
        
        logging.info('Project "%s" created' % self.project_name)
        self._status('started')
        self.save()

    def save(self, dirs=None, csv=False):
        '''
        Saves the state of ds project to disk.

        Parameters
        ----------
        dirs: list, optional
            List of sub-directoies that should be saved to disk.
            By default all subdirectories defined in the contructor are taken into account.
        csv: boolean, optional
            Save data files also as csv
            Defaults to false
        '''

        dirs = self.__update_dir_specs(dirs)

        for sub_dir in dirs:
            self.__dict__[sub_dir].ram2disk(csv)
        
        # write status file
        y_out = {
                    'output_dir': self.output_dir,
                    'backend': self.backend,
                }
        with open(self.status_file, 'w') as ymlfile:
            ymlfile.write(yaml.dump(y_out))

        self._status('saved')
        logging.info('Project "%s" saved' % self.project_name)

    def resume(self, dirs=None, force=False):
        '''
        Resumes an existing project.
        Check if this project has been saved, if so, resume
        check for save can be skipped by forcing resume

        Parameters
        ----------
        dirs: list, optional
            List of sub-directoies that should be resumed.
            By default all subdirectories defined in the contructor are taken into account.
        force: boolean, optional
            switch to forcefully resume, even though the project state is not 'saved'.
            Defaults to False.
        '''

        if os.path.isfile(self.status_file):
            logging.info('saved project state found; resuming from last saved state')
            
            # read output_dir and backend from status file
            # this eliminates the need to always provide an output_dir in the constructor
            # backend is required to instantiate the RdsFs class correctly.
            with open(self.status_file, 'r') as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)
                self.output_dir = cfg['output_dir']
                self.backend = cfg['backend'
                                  ]
            logging.debug('resuming from "%s"' % self.output_dir)
            result = self.__disk2ram(dirs)
            if result:
                # read defs to project properties
                self.__defs2kwargs()
                self._status('resumed')
                return True
            else:
                return False
        elif force:
            logging.info('forcefully resuming from last saved state')
            result = self.__disk2ram(dirs)
            if result:
                # read defs to project properties
                self.__defs2kwargs()
                self._status('forcefully resumed')
                return True
            else:
                return False
        return False

    def __disk2ram(self, dirs=None):

        dirs = self.__update_dir_specs(dirs)

        for sub_dir in dirs:
            self.__dict__[sub_dir] = RdsFs(
                                            os.path.join(self.output_dir, sub_dir),
                                            nof_processes=self.nof_processes,
                                            backend=self.backend,
                                           )
            result = self.__dict__[sub_dir].disk2ram()
            if result:
                continue
            else:
                return False
        return True

    def __kwargs2defs(self):
        '''
        sync project properties to status file and defs container
        '''
        # save in defs
        self.__dict__[self.DEFS].project_output_dir = self.output_dir
        self.__dict__[self.DEFS].analysis_start_date = self.analysis_start_date
        self.__dict__[self.DEFS].analysis_end_date = self.analysis_end_date
        self.__dict__[self.DEFS].analysis_timespan = self.analysis_timespan
        self.__dict__[self.DEFS].cell_execution_timeout = self.cell_execution_timeout
        self.__dict__[self.DEFS].make_configs = self.make_configs
        self.__dict__[self.DEFS].execution_dates_make_configs = self.execution_dates_make_configs
        self.__dict__[self.DEFS].nof_processes = self.nof_processes
        self.__dict__[self.DEFS].backend = self.backend

    def __defs2kwargs(self):
        '''
        sync project properties to status file and defs container
        '''
        # load from defs
        self.output_dir = self.__dict__[self.DEFS].project_output_dir
        self.analysis_start_date = self.__dict__[self.DEFS].analysis_start_date
        self.analysis_end_date = self.__dict__[self.DEFS].analysis_end_date
        self.analysis_timespan = self.__dict__[self.DEFS].analysis_timespan
        self.cell_execution_timeout = self.__dict__[self.DEFS].cell_execution_timeout
        self.make_configs = self.__dict__[self.DEFS].make_configs
        try:
            self.execution_dates_make_configs = self.__dict__[self.DEFS].execution_dates_make_configs
        except AttributeError:
            logging.debug('create empty dict "execution_dates_make_configs"')
            self.__dict__[self.DEFS].execution_dates_make_configs = {}
            self.execution_dates_make_configs = self.__dict__[self.DEFS].execution_dates_make_configs
        self.nof_processes = self.__dict__[self.DEFS].nof_processes
        self.backend = self.__dict__[self.DEFS].backend
        
    def reset(self, dirs=None):
        '''
        Reset the project state.
        This includes deleting all files from the output_dir.

        Parameters
        ----------
        dirs: list, optional
            List of sub-directoies that should be reset.
            By default all subdirectories defined in the contructor are taken into account.
        '''
        self.clean(dirs)
        self.save(dirs)

    def clean(self, dirs=None):
        '''
        Delete all files in data dirs.
        
        Parameters
        ----------
        dirs: list, optional
            List of sub-directoies that should be reset.
            By default all subdirectories defined in the contructor are taken into account.
        '''

        dirs = self.__update_dir_specs(dirs)

        for sub_dir in dirs:
            self.__dict__[sub_dir].clean()
        logging.info('directories "%s" cleaned' % str(dirs))
        # push back the project props to defs
        self.__kwargs2defs()
        self._status('cleaned')

    def _status(self, status):
        '''
        Change the internal status of project.
        The internal attributes will be synced to defs and to the status file as well.
        
        Parameters
        ----------
        status: string
            New status as text.
        '''
        logging.debug('"%s" status changed to "%s"' % (self.project_name, status))
        self.status = status
        
        return self.status

    def __update_dir_specs(self, dirs):
        '''
        Do a precheck for output dirs and return a list with currently managed output dirs.
        '''
        
        if dirs is None:
            # bootstrap
            if not self.output_dirs:
                dirs = sorted(
                              [
                                self.EXTERNAL,
                                self.RAW,
                                self.INTERIM,
                                self.PROCESSED,
                                self.DEFS,
                              ]
                             )
            # if no dirs are added, return currenly managed list
            else:
                return self.output_dirs

        # if single directory is given, make it a list for generic processing
        if not isinstance(dirs, list):
            dirs = [dirs]

        # always add defs
        dirs.append(self.DEFS)

        # update data_dirs based on maybe newly added items
        self.output_dirs.extend(dirs)
        self.output_dirs = list(set(self.output_dirs))
        self.output_dirs = sorted(self.output_dirs)
        return sorted(set(dirs)) # only return new items for save / resume actions

    def make_config(self, make_name, notebooks=None):
        '''
        Get/Set 'make' process by name.
        
        Parameters
        ----------
        make_name: string
            Name of the 'make' process.
            Defaults to None which means the 
        notebooks:  list
            A list of notebooks that will be executed in given order when executing this process.
            Defaults to None. Then the make config is returned but not updated.
            If the process name doesn't exist, None is returned.
        Returns the process chain (a list of notebooks) of the given make name.
        '''
        if notebooks:
            self.make_configs[make_name] = notebooks
            # sync to defs
            self.__kwargs2defs()
            logging.debug('Make config "%s" registered as "%s"' % (str(notebooks), make_name))
            return notebooks
            
        return self.make_configs.get(make_name, None)
    
    def make(self, make_name, subprocess=False):
        '''
        Run a make config that is previously defined by make_config().
        
        Parameters
        ----------
        make_name:  string
            Name of the make config as defined by the method make_config().
        subprocess: boolean
            Defines if the notebook execution is done using subprocesses or not.
            Defaults to False.
        '''
        
        logging.info('make "%s"' % make_name)
        notebooks = self.make_configs[make_name]

        if subprocess:
            logging.debug('run notebooks as subprocesses')
            result = self._run_notebooks_as_subprocess(notebooks)
        else:
            result = self._run_notebooks_in_python(notebooks)
        
        if result:
            # save execution date of successful run of a make_config
            self.execution_dates_make_configs[make_name] = datetime.datetime.now()
        
        return result

    def _run_notebooks_in_python(self, notebooks):
        '''
        Run list of notebooks (as python implementation)
        '''

        total_t0 = time()
        # save current working directory
        pwd = os.getcwd()
        for k, abs_notebook_path in enumerate(notebooks):
            notebook = os.path.basename(abs_notebook_path)
            
            w_dir = os.path.dirname(abs_notebook_path)

            #executed_notebook = os.path.join(w_dir, '_'.join(('executed', notebook)))
            executed_notebook = os.path.join(pwd, '_'.join(('executed', notebook)))

            logging.info('Execute item %d / %d' % (k+1, len(notebooks)))
            #logging.debug('change directory to "%s"' % w_dir)
            #os.chdir(w_dir)
            logging.info('running "%s"', abs_notebook_path)

            # start timer
            t0 = time()

            with open(abs_notebook_path) as f:
                nb = nbformat.read(f, as_version=4)

            # configure preprocessor with cell execution timeout
            ep = ExecutePreprocessor(timeout=self.cell_execution_timeout)

            try:
                # execute notebook in working directory
                out = ep.preprocess(nb, {'metadata': {'path': w_dir}})
            except CellExecutionError:
                out = None
                msg = 'Error executing the notebook "%s".\n\n' % notebook
                msg += 'See notebook "%s" for the traceback.' % executed_notebook
                logging.error(msg)
                raise
            finally:
                logging.info('process execution took %d seconds' % (time()-t0))
                with open(executed_notebook, mode='wt') as f:
                    try:
                        nbformat.write(nb, f)
                    except Exception as e:
                        logging.warning("Couldn't save notebook %s to disk. Continuing anyway." % executed_notebook)

        logging.info('all %d notebooks sucessfully executed in %d seconds' % (len(notebooks), (time()-total_t0)))
        return True

    def _run_notebooks_as_subprocess(self, notebooks):
        '''
        Run list of notebooks (as subprocesses)
        '''
        
        total_t0 = time()
        # save current working directory
        pwd = os.getcwd()
        for k, abs_notebook_path in enumerate(notebooks):
            notebook = os.path.basename(abs_notebook_path)
            w_dir = os.path.dirname(abs_notebook_path)
            logging.info('Execute item %d / %d' % (k+1, len(notebooks)))
            logging.debug('change directory to "%s"' % w_dir)
            os.chdir(w_dir)
            logging.info('running "%s"', abs_notebook_path)

            # start timer
            t0 = time()
            # run from command line
            process = subprocess.run(['jupyter',
                                      'nbconvert',
                                      '--ExecutePreprocessor.timeout=%d' % self.__dict__[self.DEFS].cell_execution_timeout, # this is required for long running cells like fetches
                                      '--execute',
                                      notebook],
                                     shell=False,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)

            logging.debug(process.stdout)
            logging.debug(process.stderr)
            logging.debug('process exited with returncode  %d' % process.returncode)
            logging.info('process execution took %d seconds' % (time()-t0))

            if process.returncode != 0:
                logging.error('stopped process chain due to errors in subprocess at item %d / %d' % (k+1, len(notebooks)))
                os.chdir(pwd)
                return False

        # change back to original working directory
        os.chdir(pwd)
        logging.info('all %d notebooks sucessfully executed in %d seconds' % (len(notebooks), (time()-total_t0)))
        return True

    def __str__(self):
        return 'DsProject "%s"' % self.project_name

    def __repr__(self):
        return '''
{caption}
{underline}
Analysis time:\t{a_start} - {a_end} ({a_delta})
State:\t\t{state}
output dir:\t{output_dir}
loaded dirs:\t{dirs}
'''.format(caption=str(self),
           underline='=' * len(str(self)),
           state=self.status,
           a_start=str(self.analysis_start_date),
           a_end=str(self.analysis_end_date),
           a_delta=str(self.analysis_timespan),
           output_dir=self.output_dir,
           dirs=str(self.output_dirs),)


    def run_subprocess(self, cmd_args, check=False):
        '''
        Helper function to make external command execution somewhat easier.
        '''
        # start timer
        t0 = time()

        command = ' '.join(cmd_args)

        # run from command line
        logging.debug('executing command: "%s"' % command)
        process = subprocess.run(
                                 cmd_args,
                                 shell=False,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 #capture_output=True,
                                 check=check,
                                )

        #logging.debug(process.output)
        logging.debug(process.stdout)
        logging.debug(process.stderr)


        if process.returncode != 0:
            logging.error('command "%s" failed after %d seconds' % (command, time()-t0))
            return False

        logging.debug('process exited with returncode  %d' % process.returncode)
        logging.info('command "%s" executed in %d seconds' % (command, time()-t0))
        return True
        
    def create_notebook_templates(self):
        '''
        Create notebook templates in current working directory.
        The notebooks contain a skeleton to support the resumableds workflow.
        '''
        
        nb_defs = {
                    '''\
# Definitions

Define project variables, etc.''': nbformat.v4.new_markdown_cell,
                    '''\
import resumableds''': nbformat.v4.new_code_cell,
                    '''\
# DS project name
project = '%s'

# create project
rds = resumableds.RdsProject(project, 'defs')''' % self.project_name: nbformat.v4.new_code_cell,
                    '''\
# your variables / definitions go here...

#rds.defs.a = 'a variable'
''': nbformat.v4.new_code_cell,
                    '''\
# save defs to disk
rds.save('defs')''': nbformat.v4.new_code_cell,
            '''\
*(Notebook is based on resumableds template)*''': nbformat.v4.new_markdown_cell,
                }


        nb_collection = {
                    '''\
# Data collection

Get raw data from data storages.''': nbformat.v4.new_markdown_cell,
                    '''\
import resumableds''': nbformat.v4.new_code_cell,
                    '''\
# DS project name
project = '%s'

# create project
rds = resumableds.RdsProject(project, 'raw')''' % self.project_name: nbformat.v4.new_code_cell,
                    '''\
# your data retrieval here

#rds.raw.customer_details = pd.read_sql_table('customer_details', example_con)
''': nbformat.v4.new_code_cell,
                    '''\
# save project
rds.save('raw')''': nbformat.v4.new_code_cell,
                    '''\
*(Notebook is based on resumableds template)*''': nbformat.v4.new_markdown_cell,
                        }

        nb_processing = {
                    '''\
# Processing

Manipulate your data.''': nbformat.v4.new_markdown_cell,
                    '''\
import resumableds''': nbformat.v4.new_code_cell,
                    '''\
# DS project name
project = '%s'

# create project
rds = resumableds.RdsProject(project, ['raw', 'interim', 'processed'])''' % self.project_name: nbformat.v4.new_code_cell,
                    '''\
# your data processing here

#rds.interim.german_customers = rds.raw.customer_details.loc[rds.raw.customer_details['country'] == 'Germany']
#rds.processed.customers_by_city = rds.interim.german_customers.groupby('city').customer_name.count()
''': nbformat.v4.new_code_cell,
                    '''\
# save project
rds.save(['interim', 'processed'])''': nbformat.v4.new_code_cell,
                    '''\
*(Notebook is based on resumableds template)*''': nbformat.v4.new_markdown_cell,
                        }

        nb_graphs = {
                    '''\
# Graphical output

Visualize your data.''': nbformat.v4.new_markdown_cell,
                    '''\
import resumableds''': nbformat.v4.new_code_cell,
                    '''\
# DS project name
project = '%s'

# create project
rds = resumableds.RdsProject(project, ['processed'])''' % self.project_name: nbformat.v4.new_code_cell,
                    '''\
# your data visualization here

#rds.processed.customers_by_city.plot()
''': nbformat.v4.new_code_cell,
                    '''\
# save project
rds.save('defs')''': nbformat.v4.new_code_cell,
                    '''\
*(Notebook is based on resumableds template)*''': nbformat.v4.new_markdown_cell,
                  }


        nb_templates = {
                            '01_definitions.ipynb': nb_defs,
                            '10_collection.ipynb': nb_collection,
                            '20_processing.ipynb': nb_processing,
                            '30_graphs.ipynb': nb_graphs,
                            #'40_publication.ipynb': nb_publication,
                       }

        for nb_name, nb_cells in nb_templates.items():
            logging.debug('create notebook "%s" from template' % nb_name)
            nb = nbformat.v4.new_notebook()
            nb['cells'] = [f(arg) for arg, f in nb_cells.items()]
            nbformat.write(nb, nb_name)
