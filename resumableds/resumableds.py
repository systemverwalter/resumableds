import logging
import os
from time import time
import shutil
import glob
import pandas as pd
import datetime
import pickle
import subprocess
import platform
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError
#import networkx


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

    def __init__(self, output_dir):

        self.internal_obj_prefix = 'var_'
        self.pickle_file_ext = '.pkl'

        self.output_dir = output_dir

        logging.debug('ouptut directory set to "%s"' % self.output_dir)
        self.make_output_dir()

    def make_output_dir(self):
        '''
        Creates the ouput directory to read/write files.
        '''
        logging.debug('create "%s" if not exists' % self.output_dir)
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

    def load_dump(self, filename):
        '''
        Loads a pickle file into a dataframe.
        The file name (w/o extension) will be used as python object name.

        Parameters
        ----------
        filename: string
            The absolute file name
        '''

        dataframe_name = os.path.basename(filename).split('.')[0]
        logging.debug('load dump "%s" into "%s"' % (filename, dataframe_name))
        self.__dict__[dataframe_name] = pd.read_pickle(filename)

    def dump(self, dataframe, filename, sep=';', decimal=','):
        '''
        Dumps a dataframe to files:
        - pickle file for further processing
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

        abs_fn = os.path.join(self.output_dir, filename)
        abs_fn_pickle = abs_fn + self.pickle_file_ext
        abs_fn_csv = abs_fn + '.csv'
        logging.info('dump "%s"' % filename)
        logging.debug('dump "%s"' % abs_fn_pickle)
        dataframe.to_pickle(abs_fn_pickle)
        logging.debug('dump "%s" with sep="%s" and decimal="%s"' % (abs_fn_csv, sep, decimal))
        dataframe.to_csv(abs_fn_csv, sep=sep, decimal=decimal)

    def _ls(self):
        '''
        Returns output directory content including mtime.

        Returns
        -------
        Dict with file names as keys and mtime as values.
        '''

        logging.debug('ls "%s"' % self.output_dir)
        ls_content = glob.glob(os.path.join(self.output_dir, '*'))
        ls_content = {f:str(datetime.datetime.fromtimestamp(os.path.getmtime(f))) for f in ls_content}
        for k, v in ls_content.items():
            logging.debug('\t%s modified on %s' % (k, v))
        return ls_content

    def ls(self):
        '''
        Prints dataframe files from the output directory including mtime as returned by _ls().
        Internal python objects are skipped and not shown.
        '''
        return {os.path.basename(k): v for k, v in self._ls().items() if not os.path.basename(k).startswith(self.internal_obj_prefix)}

    def ram2disk(self):
        '''
        Saves all attributes of this object as files (pickles) to the output directory.
        '''

        # for all attributes in object...
        for name, obj in self.__dict__.items():
            if isinstance(obj, pd.DataFrame):
                #if object is dataframe, dump it
                self.dump(obj, name)
            else:
                # if not a dataframe, pickle it
                base_name = self.internal_obj_prefix + name + self.pickle_file_ext
                abs_fn = os.path.join(self.output_dir, base_name)
                logging.debug('dump "%s"' % abs_fn)
                with open(abs_fn, 'wb') as f:
                    pickle.dump(obj, f)
        logging.debug('sync to disk done for "%s"' % self.output_dir)

    def disk2ram(self):
        '''
        Reads all pickle files from the output directory
        and loads them as attributes of this object.
        '''

        files = self._ls()
        for fn, mtime in files.items():
            logging.debug('load %s from %s' % (fn, mtime))
            base_name = os.path.basename(fn)
            if  base_name.startswith(self.internal_obj_prefix):
                # internal objects (no dataframes)
                var_name = base_name[len(self.internal_obj_prefix):-1*len(self.pickle_file_ext)]
                with open(fn, 'rb') as f:
                    self.__dict__[var_name] = pickle.load(f)
            else:
                #if object is dataframe, load dump
                if fn.endswith(self.pickle_file_ext):
                    self.load_dump(fn)
        logging.debug('sync to ram done for "%s"' % self.output_dir)

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
        objects = '\n'.join(['\t%s: %s' % (str(k), str(v)) if not isinstance(v, pd.DataFrame) else '\t%s: %s' % (str(k), str(v.shape)) for k, v in self.__dict__.items()])

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
        Example: {'raw': ['get_sql_data.ipynb', 'get_no_sql_data.ipynb']}
        Defaults to {}.

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

    def __init__(self, project_name, dirs=None, **kwargs):

        #
        self.__always_load_defs = True

        # set names of output directories
        # external: files from outside this project, external files will be copied here for further use
        self.EXTERNAL = 'external'
        # raw: raw info as got from e.g. a database or the odas fetch command
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

        self.project_name = project_name
        self.kwargs = kwargs

        self.output_dir = self.kwargs.get('output_dir', os.path.join('.', self.project_name))

        # define project's status file name
        self.status_file = '%s.status' % self.project_name
        self.status_file = os.path.join(self.output_dir, self.status_file)

        # resume from file if possible
        if self.resume(dirs):
            logging.info('Project "%s" resumed' % self.project_name)
        else:
            self.start(dirs)
            logging.info('Project "%s" created' % self.project_name)

    def start(self, dirs=None):
        '''
        Initiate new project.
        No files will be touched!

        Parameters
        ----------
        dirs: list, optional
            List of sub-directory names that should be used in the project.
            By default all subdirectories defined in the contructor are taken into account.
        '''

        dirs = self.__update_dir_specs(dirs)

        for sub_dir in dirs:
            self.__dict__[sub_dir] = RdsFs(os.path.join(self.output_dir, sub_dir))
        # add some default definitions
        self.__kwargs2defs()
        self._status('started')

    def save(self, dirs=None):
        '''
        Saves the state of ds project to disk.

        Parameters
        ----------
        dirs: list, optional
            List of sub-directoies that should be saved to disk.
            By default all subdirectories defined in the contructor are taken into account.
        '''

        dirs = self.__update_dir_specs(dirs)

        for sub_dir in dirs:
            self.__dict__[sub_dir].ram2disk()
        
        self._status('saved')
        
        with open(self.status_file, 'w') as f:
            f.write('%s %s %s' % (datetime.datetime.now(), self.status, str(dirs)))
        logging.info('Project "%s" saved' % self.project_name)

    def fast_save(self):
        '''
        Only change status of project without acutally dumping any data (handy for debugging).
        Handle with care. Data (in memory but on disk yet) can be lost.
        '''

        self._status('fast saved')
        with open(self.status_file, 'w') as f:
            f.write('%s %s' % (datetime.datetime.now(), self._status))
        logging.info('Project "%s" saved' % self.project_name)

    def resume(self, dirs=None, force=False):
        '''
        Resumes
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
            self.__disk2ram(dirs)
            os.unlink(self.status_file)
            self._status('resumed')
            return True
        elif force:
            logging.info('forcefully resuming from last saved state')
            self.__disk2ram(dirs)
            self._status('forcefully resumed')
            return True

        return False

    def __disk2ram(self, dirs=None):

        dirs = self.__update_dir_specs(dirs)

        for sub_dir in dirs:
            self.__dict__[sub_dir] = RdsFs(os.path.join(self.output_dir, sub_dir))
            self.__dict__[sub_dir].disk2ram()

    def __kwargs2defs(self):
        '''
        Get some definitions from kwargs
        Defaults to:
            analysis_start_date: 180 days from now
            analysis_end_date:   today
        '''

        if hasattr(self, self.DEFS):
            # analsysis timespan
            self.__dict__[self.DEFS].analysis_timespan = self.kwargs.get('analysis_timespan', '180 days')
            if not isinstance(self.__dict__[self.DEFS].analysis_timespan, pd.Timedelta):
                try:
                    self.__dict__[self.DEFS].analysis_timespan = pd.Timedelta(self.__dict__[self.DEFS].analysis_timespan)
                except Exception as e:
                    logging.error(e)

            # analysis start date
            # defaults to today - analysis timespan
            self.__dict__[self.DEFS].analysis_start_date = self.kwargs.get('analysis_start_date',
                                                                   pd.datetime.today() - self.__dict__[self.DEFS].analysis_timespan)

            logging.info('analysis start date set to %s' % self.__dict__[self.DEFS].analysis_start_date)

            # analysis end date
            # defaults to today
            self.__dict__[self.DEFS].analysis_end_date = self.kwargs.get('analysis_end_date', pd.datetime.today())
            logging.info('analysis end date set to %s' % self.__dict__[self.DEFS].analysis_end_date)

            # re-calculate timespan as it might be wrong due to overwritten start or end date
            self.__dict__[self.DEFS].analysis_timespan = self.__dict__[self.DEFS].analysis_end_date - self.__dict__[self.DEFS].analysis_start_date
            logging.info('analysis timespan set to %s' % self.__dict__[self.DEFS].analysis_timespan)

            # set the exec timeout of a single cell for notebooks execution
            self.__dict__[self.DEFS].cell_execution_timeout = self.kwargs.get('cell_execution_timeout', 3600)

            # set make_configs
            self.__dict__[self.DEFS].make_configs = self.kwargs.get('make_configs', {})


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
        self.kill(dirs)
        self.start(dirs)

    def kill(self, dirs=None):
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
        self._status('killed')

    def _status(self, status):
        '''
        Change the internal status of project.
        
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

        if self.__always_load_defs:
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
            self.__dict__[self.DEFS].make_configs[make_name] = notebooks
            logging.debug('Make config "%s" registered as "%s"' % (str(notebooks), make_name))
            return notebooks
            
        return self.__dict__[self.DEFS].make_configs.get(make_name, None)
  
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
        
        logging.info('run process chain "%s"' % make_name)
        notebooks = self.__dict__[self.DEFS].make_configs[make_name]

        if subprocess:
            logging.debug('run notebooks as subprocesses')
            return self._run_notebooks_as_subprocess(notebooks)
        return self._run_notebooks_in_python(notebooks)          

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
            executed_notebook = os.path.join(w_dir, '_'.join(('executed', notebook)))

            logging.info('Execute item %d / %d' % (k+1, len(specs)))
            #logging.debug('change directory to "%s"' % w_dir)
            #os.chdir(w_dir)
            logging.info('running "%s"', abs_notebook_path)

            # start timer
            t0 = time()

            with open(abs_notebook_path) as f:
                nb = nbformat.read(f, as_version=4)

            # configure preprocessor with cell execution timeout
            ep = ExecutePreprocessor(timeout=self.__dict__[self.DEFS].cell_execution_timeout)

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

        logging.info('all %d notebooks sucessfully executed in %d seconds' % (len(specs), (time()-total_t0)))
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
            logging.info('Execute item %d / %d' % (k+1, len(specs)))
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
                logging.error('stopped process chain due to errors in subprocess at item %d / %d' % (k+1, len(specs)))
                os.chdir(pwd)
                return False

        # change back to original working directory
        os.chdir(pwd)
        logging.info('all %d notebooks sucessfully executed in %d seconds' % (len(specs), (time()-total_t0)))
        return True

    def __str__(self):
        return 'DsProject "%s"' % self.project_name

    def __repr__(self):
        return '''
{caption}
{underline}
Analysis time:\t{a_start} - {a_end} ({a_delta})
State:\t\t{state}
output dir:\t{output_dir}'
loaded dirs:\t{dirs}
'''.format(caption=str(self),
           underline='=' * len(str(self)),
           state=self.status,
           a_start=str(self.__dict__[self.DEFS].analysis_start_date),
           a_end=str(self.__dict__[self.DEFS].analysis_end_date),
           a_delta=str(self.__dict__[self.DEFS].analysis_timespan),
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
