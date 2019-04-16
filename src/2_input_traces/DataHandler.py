import os
import re
import zipfile
from io import BytesIO

import pandas as pd


class ParseMMSDMTables:
    """Used to extract and format data from AEMO MMSDM tables"""

    def __init__(self, archive_dir):
        """Initialise object used to extract MMSDM data

        Parameters
        ----------
        archive_dir : str
            Path to directory containing MMSDM data (zipped)
        """

        # Path to folder containing MMSDM data
        self.archive_dir = archive_dir

    def mmsdm_table_to_dataframe(self, archive_name, table_name):
        """Read MMSDM table into pandas DataFrame

        Parameters
        ----------
        archive_name : str
            Name of zip folder containing MMSDM information for a given year

        table_name : str
            Name of MMSDM table to be read into pandas DataFrame

        Returns
        -------
        df : pandas DataFrame
            DataFrame containing contents of given MMSDM table
        """

        # Path to MMSDM archive for a given year and month
        archive_path = os.path.join(self.archive_dir, archive_name)

        # Open zipped archive
        with zipfile.ZipFile(archive_path) as outer_zip:
            # Filter files in archive by table name and file type
            compressed_files = [f for f in outer_zip.filelist
                                if (table_name in f.filename)
                                and ('.zip' in f.filename)
                                and re.search(r'_{}_\d'.format(table_name), f.filename)]

            # Check only 1 file in list
            if len(compressed_files) != 1:
                raise Exception('Encountered {} files, should only encounter 1'.format(len(compressed_files)))
            else:
                compressed_file = compressed_files[0]

                # Construct name of compressed csv file
                csv_name = compressed_file.filename.replace('.zip', '.CSV').split('/')[-1]

                # Convert opened zip into bytes IO object to read inner zip file
                zip_data = BytesIO(outer_zip.read(compressed_file))

                # Open inner zip file
                with zipfile.ZipFile(zip_data) as inner_zip:
                    # Read csv from inner zip file
                    with inner_zip.open(csv_name) as f:
                        # Read into Pandas DataFrame
                        df = pd.read_csv(f, skiprows=1)

                # Remove last row of DataFrame (End File row)
                df = df[:-1]

        return df

    def parse_biddayoffer_d(self, archive_name, column_index=None):
        """Read BIDDAYOFFER_D table into DataFrame and apply formatting

        Parameters
        ----------
        archive_name : str
            Name of zip folder containing MMSDM information for a given year

        column_index : list
            List containing new column labels for outputted DataFrame

        Returns
        -------
        df_o : pandas DataFrame
            Formatted data from BIDDAYOFFER_D table
        """

        # Read MMSDM table into DataFrame
        df = self.mmsdm_table_to_dataframe(archive_name=archive_name, table_name='BIDDAYOFFER_D')

        # Columns to keep
        cols = ['SETTLEMENTDATE', 'DUID', 'PRICEBAND1', 'PRICEBAND2', 'PRICEBAND3', 'PRICEBAND4',
                'PRICEBAND5', 'PRICEBAND6', 'PRICEBAND7', 'PRICEBAND8', 'PRICEBAND9', 'PRICEBAND10']

        # Filter energy bids, remove duplicate (keeping last bid)
        df_o = df.loc[df['BIDTYPE'] == 'ENERGY', cols].drop_duplicates(keep='last')

        # Convert settlement date to datetime object
        df_o['SETTLEMENTDATE'] = pd.to_datetime(df_o['SETTLEMENTDATE'])

        # Shift settlement date forward by 4hrs and 5mins. Note that each trading
        # day starts at 4.05am, but the settlement date starts at 12am. Price bands
        # for a given settlementdate are applicable to trading intervals when the
        # trading day actually starts, hence the need to shift the time forward
        # by 4hrs and 5 mins.
        df_o['SETTLEMENTDATE'] = df_o['SETTLEMENTDATE'] + pd.Timedelta(hours=4, minutes=5)

        # List of columns corresponding to priceband values
        value_columns = df_o.columns.drop(['SETTLEMENTDATE', 'DUID'])

        # Pivot DataFrame and rename columns
        df_o = df_o.pivot_table(index='SETTLEMENTDATE', columns='DUID', values=value_columns)
        df_o.columns = df_o.columns.map('|'.join)

        # If new column index exists, reindex columns accordingly
        if column_index:
            # Reindex columns
            df_o = df_o.reindex(columns=column_index)

        # Fill nan values with -999
        df_o = df_o.fillna(-999)

        return df_o

    def parse_bidperoffer_d(self, archive_name, column_index=None):
        """Read BIDPEROFFER_D table into DataFrame and apply formatting

        Parameters
        ----------
        archive_name : str
            Name of zip folder containing MMSDM information for a given year

        column_index : list
            List containing new column labels for outputted DataFrame

        Returns
        -------
        df_o : pandas DataFrame
            Formatted data from BIDPEROFFER_D table
        """

        # Read MMSDM table into DataFrame
        df = self.mmsdm_table_to_dataframe(archive_name=archive_name, table_name='BIDPEROFFER_D')

        # Columns to keep
        cols = ['INTERVAL_DATETIME', 'DUID', 'BANDAVAIL1', 'BANDAVAIL2', 'BANDAVAIL3', 'BANDAVAIL4',
                'BANDAVAIL5', 'BANDAVAIL6', 'BANDAVAIL7', 'BANDAVAIL8', 'BANDAVAIL9', 'BANDAVAIL10',
                'MAXAVAIL', 'ROCUP', 'ROCDOWN']

        df_o = df.loc[df['BIDTYPE'] == 'ENERGY', cols].drop_duplicates(keep='last')

        # Convert interval datetime to datetime object
        df_o['INTERVAL_DATETIME'] = pd.to_datetime(df_o['INTERVAL_DATETIME'])

        # List of columns corresponding to priceband values
        value_columns = df_o.columns.drop(['INTERVAL_DATETIME', 'DUID'])

        # Pivot DataFrame and rename columns
        df_o = df_o.pivot_table(index='INTERVAL_DATETIME', columns='DUID', values=value_columns)
        df_o.columns = df_o.columns.map('|'.join)

        # If new column index exists, reindex columns accordingly
        if column_index:
            # Reindex columns
            df_o = df_o.reindex(columns=column_index)

        # Fill nan values with -999
        df_o = df_o.fillna(-999)

        return df_o

    def parse_dispatchregionsum(self, archive_name, column_index=None):
        """Read DISPATCHREGIONSUM table into DataFrame and apply formatting

        Parameters
        ----------
        archive_name : str
            Name of zip folder containing MMSDM information for a given year

        column_index : list
            List containing new column labels for outputted DataFrame

        Returns
        -------
        df_o : pandas DataFrame
            Formatted data from DISPATCHREGIONSUM table
        """

        # Read MMSDM table into DataFrame
        df = self.mmsdm_table_to_dataframe(archive_name=archive_name, table_name='DISPATCHREGIONSUM')

        # Note sure why, but duplicates occur in table. Suspect it has to do with INTERVENTION field.
        df_o = df.copy().drop_duplicates(['SETTLEMENTDATE', 'REGIONID'], keep='last')

        # Convert settlement date to datetime object
        df_o['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])

        # Apply pivot
        df_o = df_o.pivot(index='SETTLEMENTDATE', columns='REGIONID', values='TOTALDEMAND').add_prefix('TOTALDEMAND|')

        # If new column index exists, reindex columns accordingly
        if column_index:
            # Reindex columns
            df_o = df_o.reindex(columns=column_index)

        # Fill nan values with -999
        df_o = df_o.fillna(-999)

        return df_o

    def parse_dispatch_unit_scada(self, archive_name, column_index=None):
        """Read DISPATCH_UNIT_SCADA table into DataFrame and apply formatting

        Parameters
        ----------
        archive_name : str
            Name of zip folder containing MMSDM information for a given year

        column_index : list
            List containing new column labels for outputted DataFrame

        Returns
        -------
        df_o : pandas DataFrame
            Formatted data from DISPATCH_UNIT_SCADA table
        """

        # Read MMSDM table into DataFrame
        df = self.mmsdm_table_to_dataframe(archive_name=archive_name, table_name='DISPATCH_UNIT_SCADA')

        # Convert settlement date to datetime object
        df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])

        # Apply pivot
        df_o = df.pivot(index='SETTLEMENTDATE', columns='DUID', values='SCADAVALUE')

        # If new column index exists, reindex columns accordingly
        if column_index:
            # Reindex columns
            df_o = df_o.reindex(columns=column_index)

        # Fill nan values with -999
        df_o = df_o.fillna(-999)

        return df_o

    def parse_dispatchprice(self, archive_name, column_index=None):
        """Read DISPATCHPRICE table into DataFrame and apply formatting

        Parameters
        ----------
        archive_name : str
            Name of zip folder containing MMSDM information for a given year

        column_index : list
            List containing new column labels for outputted DataFrame

        Returns
        -------
        df_o : pandas DataFrame
            Formatted data from DISPATCH_UNIT_SCADA table
        """

        # Read MMSDM table into DataFrame
        df = self.mmsdm_table_to_dataframe(archive_name=archive_name, table_name='DISPATCHPRICE')

        # Drop duplicates - think this has to do with the INTERVENTION flag. Keeping last
        # value as this is most recent.
        df_o = df.drop_duplicates(subset=['SETTLEMENTDATE', 'REGIONID'], keep='last').copy()

        # Convert settlement date to datetime object
        df_o['SETTLEMENTDATE'] = pd.to_datetime(df_o['SETTLEMENTDATE'])

        # Apply pivot
        df_o = df_o.pivot(index='SETTLEMENTDATE', columns='REGIONID', values='RRP')

        # Add prefix to column labels
        df_o = df_o.add_prefix('RRP|')

        # If new column index exists, reindex columns accordingly
        if column_index:
            # Reindex columns
            df_o = df_o.reindex(columns=column_index)

        # Fill nan values with -999
        df_o = df_o.fillna(-999)

        return df_o
