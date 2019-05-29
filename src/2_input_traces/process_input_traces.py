import os
import re

import pandas as pd
import DataHandler


class ProcessTraces:
    def __init__(self, core_data_dir):
        # Core data directory
        self.core_data_dir = core_data_dir

        # Map between wind bubbles and associated files
        self.wind_bubble_file_map = self._get_wind_bubble_file_map()

    def _get_wind_bubble_file_map(self):
        """Map between wind bubbles and associated files"""

        return pd.read_csv(os.path.join(self.core_data_dir, 'maps', 'wind_bubble_file_map.csv'), index_col='BUBBLE_ID')

    @staticmethod
    def _process_solar_trace(data_dir, filename):
        """
        Process solar traces for a single file

        Parameters
        ----------
        data_dir : str
            Directory containing solar trace data

        filename : str
            Name of solar trace file to process

        Returns
        -------
        df : pandas DataFrame
            Solar trace information for given file
        """

        # Load solar traces as CSV
        df = pd.read_csv(os.path.join(data_dir, filename))

        # Set index and unstack (want year, month, day, and interval ID as index)
        df = df.set_index(['Year', 'Month', 'Day']).stack().to_frame(name='capacity_factor')

        # Construct timestamps from index
        def _construct_timestamp(row):
            """Construct timestamp object given year, month, day, and interval ID"""

            # Convert index to integer and assign to year, month, day, interval ID
            year, month, day, interval_id = [int(i) for i in row.name]

            # Construct timestamp
            timestamp = pd.Timestamp(year, month, day, 0) + pd.Timedelta(minutes=30 * interval_id)

            return timestamp

        # Construct timestamps
        df['timestamp'] = df.apply(_construct_timestamp, axis=1)

        # Set timestamp as index
        df = df.set_index('timestamp')

        # Re-sample to hourly resolution (if label 04:00:00, this denotes the end
        # of the trading interval i.e. represents the period from 03:00:00 - 04:00:00)
        df = df.resample('1h', label='right', closed='right').mean()

        # Technology extracted from filename
        technology = re.findall(r'_(.+)\.csv', filename)[0]

        # Zone extracted from filename
        zone = filename.split(' ')[0]

        # Add technology type and zone to DataFrame
        df['technology'] = technology
        df['zone'] = zone

        return df

    def process_solar_traces(self, data_dir, output_dir, save=False):
        """
        Process solar traces for each technology type, planning zone, and year in model horizon

        Parameters
        ----------
        data_dir : str
            Directory containing solar trace data

        output_dir : str
            Directory where output files will be stored

        save : bool (default=False)
            Specify if traces should be saved. Default is to not save.

        Returns
        -------
        df_o : pandas DataFrame
            All solar trace information in a single DataFrame
        """

        # Container for DataFrames describing solar capacity factors for each zone
        # and technology type
        dfs = []

        # All files in directory
        files = os.listdir(data_dir)

        for i, file in enumerate(files):
            print(f'Processing solar traces, file: {i + 1}/{len(files)}')

            # Process file
            df = self._process_solar_trace(data_dir, file)

            # Append to container
            dfs.append(df)

        # All solar traces
        df_o = pd.concat(dfs)

        # Save file if specified
        if save:
            # Save DataFrame
            df_o.to_hdf(os.path.join(output_dir, 'solar_traces.h5'), key='df')

        return df_o

    @staticmethod
    def _process_wind_trace(data_dir, filename, bubble_id):
        """
        Process wind traces for a single file

        Parameters
        ----------
        data_dir : str
            Directory containing wind trace data

        filename : str
            Name of wind trace file to process

        Returns
        -------
        df : pandas DataFrame
            Wind trace information for given file
        """

        # Load solar traces as CSV
        df = pd.read_csv(os.path.join(data_dir, filename))

        # Set index and unstack (want year, month, day, and interval ID as index)
        df = df.set_index(['Year', 'Month', 'Day'])

        # Total number of intervals per day
        intervals_per_day = int(df.columns[-1])

        # Interval duration in hours
        interval_duration = 24 / intervals_per_day

        # Stack columns
        df = df.stack().to_frame(name='capacity_factor')

        # Construct timestamps from index
        def _construct_timestamp(row):
            """Construct timestamp object given year, month, day, and interval ID"""

            # Convert index to integer and assign to year, month, day, interval ID
            year, month, day, interval_id = [int(i) for i in row.name]

            # Construct timestamp
            timestamp = pd.Timestamp(year, month, day, 0) + pd.Timedelta(hours=interval_id * interval_duration)

            return timestamp

        # Construct timestamps
        df['timestamp'] = df.apply(_construct_timestamp, axis=1)

        # Set timestamp as index
        df = df.set_index('timestamp')

        # Re-sample to hourly resolution (if label 04:00:00, this denotes the end
        # of the trading interval i.e. represents the period from 03:00:00 - 04:00:00)
        df = df.resample('1h', label='right', closed='right').mean()

        # Add wind bubble name to DataFrame
        df['bubble'] = bubble_id

        return df

    def process_wind_traces(self, data_dir, output_dir, save=False):
        """
        Process wind traces for each wind bubble

        Parameters
        ----------
        data_dir : str
            Directory containing wind trace data

        output_dir : str
            Directory where output files will be stored

        save : bool (default=False)
            Specify if traces should be saved. Default is to not save.

        Returns
        -------
        df_o : pandas DataFrame
            All wind trace information in a single DataFrame
        """

        # Container for DataFrames describing solar capacity factors for each zone
        # and technology type
        dfs = []

        # Counter
        i = 0

        for index, row in self.wind_bubble_file_map.iterrows():
            print(f'Processing wind traces, file: {i + 1}/{self.wind_bubble_file_map.shape[0]}')

            # Process file
            df = self._process_wind_trace(data_dir, row['FILE'], row.name)

            # Append to container
            dfs.append(df)

            # Increment counter
            i += 1

        # All wind traces. Drop duplicates.
        df_o = pd.concat(dfs)

        # Check if duplicated values found
        assert not df_o.reset_index().duplicated().any(), 'Duplicated wind trace records identified'

        # Save file if specified
        if save:
            # Save DataFrame
            df_o.to_hdf(os.path.join(output_dir, 'wind_traces.h5'), key='df')

        return df_o

    @staticmethod
    def _process_demand_trace(data_dir, filename):
        """
        Process demand traces for a single file

        Parameters
        ----------
        data_dir : str
            Directory containing wind trace data

        filename : str
            Name of wind trace file to process

        Returns
        -------
        df : pandas DataFrame
            Demand trace information for given file
        """

        # Load demand traces as CSV
        df = pd.read_csv(os.path.join(data_dir, filename))

        # Set index and unstack (want year, month, day, and interval ID as index)
        df = df.set_index(['Year', 'Month', 'Day']).stack().to_frame(name='demand')

        # Construct timestamps from index
        def _construct_timestamp(row):
            """Construct timestamp object given year, month, day, and interval ID"""

            # Convert index to integer and assign to year, month, day, interval ID
            year, month, day, interval_id = [int(i) for i in row.name]

            # Construct timestamp
            timestamp = pd.Timestamp(year, month, day, 0) + pd.Timedelta(minutes=30 * interval_id)

            return timestamp

        # Construct timestamps
        df['timestamp'] = df.apply(_construct_timestamp, axis=1)

        # Set timestamp as index
        df = df.set_index('timestamp')

        # Re-sample to hourly resolution (if label 04:00:00, this denotes the end
        # of the trading interval i.e. represents the period from 03:00:00 - 04:00:00)
        df = df.resample('1h', label='right', closed='right').mean()

        # NEM region extracted from filename
        region = filename.split(' ')[1]

        # Demand scenario extracted from filename
        scenario = filename.split(' ')[2]

        # Add technology type and zone to DataFrame
        df['region'] = region
        df['scenario'] = scenario

        return df

    def process_demand_traces(self, data_dir, output_dir, save=False):
        """
        Process demand traces for each region

        Parameters
        ----------
        data_dir : str
            Directory containing wind trace data

        output_dir : str
            Directory where output files will be stored

        save : bool (default=False)
            Specify if traces should be saved. Default is to not save.

        Returns
        -------
        df_o : pandas DataFrame
            All demand trace information in a single DataFrame
        """

        # Container for DataFrames describing solar capacity factors for each zone
        # and technology type
        dfs = []

        # All files in directory
        files = os.listdir(data_dir)

        for i, file in enumerate(files):
            print(f'Processing demand traces, file: {i + 1}/{len(files)}')

            # Process file
            df = self._process_demand_trace(data_dir, file)

            # Append to container
            dfs.append(df)

        # All solar traces
        df_o = pd.concat(dfs)

        # Check if duplicated demand values found
        assert not df_o.reset_index().duplicated().any(), 'Duplicated demand trace entries identified'

        # Save file if specified
        if save:
            # Save DataFrame
            df_o.to_hdf(os.path.join(output_dir, 'demand_traces.h5'), key='df')

        return df_o

    @staticmethod
    def _process_hydro_trace(archive_name, hydro_duids):
        """
        Process traces for hydro units for a given month

        Parameters
        ----------
        archive_name : str
            Directory containing wind trace data

        hydro_duids : list
            List of existing hydro generators

        Returns
        -------
        df : pandas DataFrame
            Hydro trace information for each generator
        """

        # Extract SCADA dispatch data for given archive
        df = MMSDM.parse_dispatch_unit_scada(archive_name)

        # Only retain hydro generators
        df = df.reindex(columns=hydro_duids)

        # Reindex to hourly resolution
        df = df.resample('1H', closed='right', label='right').mean()

        return df

    def process_hydro_traces(self, generator_data_dir, output_dir, save=False):
        """Process hydro traces"""
        """
        Process traces for hydro generators (multiple months)

        Parameters
        ----------
        generator_data_dir : str
            Directory containing generator information

        output_dir : str
            Directory where output files will be stored

        save : bool (default=False)
            Specify if traces should be saved. Default is to not save.

        Returns
        -------
        df_o : pandas DataFrame
            All hydro trace information in a single DataFrame
        """

        # Existing generator information
        df_g = pd.read_csv(os.path.join(generator_data_dir, 'generators.csv'))

        # DUIDs for existing hydro generators
        existing_hydro_duids = df_g.loc[df_g['FUEL_CAT'] == 'Hydro', 'DUID'].to_list()

        # Container for hydro output traces
        dfs = []

        # Archive names from which to extract data (only consider 2016)
        archives = [f'MMSDM_2016_{i:02}.zip' for i in range(1, 13)]

        for i, archive in enumerate(archives):
            print(f'Processing hydro traces, file: {i + 1}/{len(archives)}')

            # Process data in archive
            df = self._process_hydro_trace(archive, existing_hydro_duids)

            # Append processed data to main container
            dfs.append(df)

        # Combine all hydro traces into a single DataFrame and drop duplicated rows
        df_o = pd.concat(dfs)

        # Drop duplicated entries
        df_o = df_o.reset_index().drop_duplicates(subset=['SETTLEMENTDATE'], keep='last').set_index('SETTLEMENTDATE')

        # Check for duplicated values
        assert not df_o.reset_index().duplicated().any(), 'Duplicated hydro trace entries identified'

        # Save DataFrame
        if save:
            df_o.to_hdf(os.path.join(output_dir, 'hydro_traces.h5'), key='df')

        return df_o


if __name__ == '__main__':
    # Paths
    # -----
    # Directory in which output files are stored
    output_directory = os.path.join(os.path.curdir, 'output')

    # Core data directory
    data_directory = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data')

    # Root directory for NTNDP information
    ntndp_directory = os.path.join(data_directory, 'files', '2016 NTNDP Database Input Data Traces')

    # Directory containing solar traces
    solar_data_directory = os.path.join(ntndp_directory, 'Solar traces', 'Solar traces', '2016 Future Solar Traces')

    # Directory containing wind traces
    wind_data_directory = os.path.join(ntndp_directory, 'Wind traces', 'Wind traces', '2016 Future Wind Traces')

    # Directory containing demand traces
    demand_data_directory = os.path.join(ntndp_directory, '2016 Regional Demand Traces', '2016 Regional Demand Traces')

    # Directory containing zipped MMSDM archive files
    mmsdm_archive_directory = r'C:\Users\eee\Desktop\nemweb\Reports\Data_Archive\MMSDM\zipped'

    # Directory containing parameters for existing generators
    generator_data_directory = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data', 'files',
                                            'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603',
                                            'generators')

    # Map directory
    map_directory = os.path.join(data_directory, 'maps')

    # Data processing objects
    # -----------------------
    # Object used to process different NTNDP traces
    traces = ProcessTraces(data_directory)

    # Initialise object used to extract information from MMSDM data archive
    MMSDM = DataHandler.ParseMMSDMTables(mmsdm_archive_directory)

    # Process signals
    # ---------------
    # Process solar traces
    # df_solar = traces.process_solar_traces(solar_data_directory, output_directory, save=True)
    #
    # # Process wind traces
    # df_wind = traces.process_wind_traces(wind_data_directory, output_directory, save=True)
    #
    # # Process demand traces
    # df_demand = traces.process_demand_traces(demand_data_directory, output_directory, save=True)
    #
    # # Process hydro generator traces
    # df_hydro = traces.process_hydro_traces(generator_data_directory, output_directory, save=True)

    df = pd.read_hdf(os.path.join(output_directory, 'wind_traces.h5'))
