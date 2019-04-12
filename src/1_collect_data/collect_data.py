import os

import numpy as np
import pandas as pd


class ConstructDataset:
    def __init__(self, data_dir, scenario='neutral'):
        # Directory containing core data files
        self.data_dir = data_dir

        # Demand scenario (either 'neutral' or 'low')
        self.scenario = scenario

        # Files
        # -----
        # NTNDP database spreadsheets
        self.ntndp_filename = '2016 Planning Studies - Additional Modelling Data and Assumptions summary.xlsm'

        # ACIL Allen spreadsheets
        self.acil_filename = 'Fuel_and_Technology_Cost_Review_Data_ACIL_Allen.xlsx'

        # Mappings
        # --------
        # Map between zones and regions
        self.df_zones_map = pd.read_csv(os.path.join(data_dir, 'maps', 'zones.csv'), index_col=0)

        # Map between wind bubbles, zones and regions
        self.df_bubble_map = pd.read_csv(os.path.join(data_dir, 'maps', 'bubbles.csv'), index_col=0)

        # Map between thermal unit types, fuel categories, and unit categories
        self.df_thermal_map = pd.read_csv(os.path.join(data_dir, 'maps', 'thermal_unit_types.csv'), index_col=0)

        # Map between existing gas and coal generators and fuel cost IDs
        self.df_fuel_cost_map = pd.read_csv(os.path.join(data_dir, 'maps', 'existing_fuel_cost_map.csv'), index_col=0)

        # Map between candidate coal generator IDs and fuel cost IDs
        self.df_ntndp_coal_cost_map = pd.read_csv(os.path.join(data_dir, 'maps', 'candidate_coal_cost_map.csv'),
                                                  index_col=0)

        # Map between existing generators and IDs for FOM costs in NTNDP database
        self.df_fom_cost_map = pd.read_csv(os.path.join(data_dir, 'maps', 'existing_fom_cost_map.csv'), index_col=0)

        # NTNDP Data
        # ----------
        # Load fixed operating and maintenance cost data
        self.df_ntndp_fom = self._load_ntndp_fom()

        # Variable operating and maintenance cost data
        self.df_ntndp_vom = self._load_ntndp_vom()

        # Heat rates
        self.df_ntndp_heat_rates = self._load_ntndp_heat_rates()

        # Emissions
        self.df_ntndp_emissions = self._load_ntndp_emissions_rates()

        # Build costs
        self.df_ntndp_build_cost = self._load_ntndp_build_costs_neutral()

        # Coal cost
        self.df_ntndp_coal_cost = self._load_ntndp_coal_cost_neutral()

        # Gas cost
        self.df_ntndp_gas_cost = self._load_ntndp_gas_cost_neutral()

        # ACIL Allen Data
        # ---------------
        # Fuel costs for existing units
        self.df_acil_fuel_cost = self._load_acil_existing_fuel_cost()

        # Technical parameters for candidate units
        self.df_acil_technical_parameters = self._load_acil_candidate_technical_parameters()

        # NEM Dataset
        # -----------
        # All existing generators from NEM dataset
        self.df_g = pd.read_csv(
            os.path.join(data_dir, 'files', 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603',
                         'generators', 'generators.csv'), index_col='DUID')

    def _load_ntndp_fom(self):
        """Load fixed operating and maintenance cost spreadsheet from NTNDP database"""

        # Fixed operating and maintenance costs
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.ntndp_filename),
                            sheet_name='FOM', skiprows=2, header=None)
              .rename(columns={0: 'NTNDP_ID', 1: 'FOM'}).set_index('NTNDP_ID'))

        return df

    def _load_ntndp_vom(self):
        """Load variable operating and maintenance (VOM) cost spreadsheet from NTNDP database"""

        # Variable operating and maintenance costs
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.ntndp_filename),
                            sheet_name='VOM', skiprows=2, header=None)
              .rename(columns={0: 'NTNDP_ID', 1: 'VOM'}).set_index('NTNDP_ID'))

        return df

    def _load_ntndp_heat_rates(self):
        """Load heat rate spreadsheet from NTNDP database"""

        # Heat rates
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.ntndp_filename),
                            sheet_name='Heat Rates', skiprows=1))

        # Rename columns and set index
        df = (df.rename(columns={'Generators': 'NTNDP_ID', 'Heat Rate (GJ/MWh)': 'HEAT_RATE',
                                 'Date From': 'DATE_FROM', 'Date To': 'DATE_TO',
                                 'Category': 'CATEGORY'})
              .set_index('NTNDP_ID'))

        return df

    def _load_ntndp_emissions_rates(self):
        """Load emissions rate data from NTNDP database"""

        # Emissions rates
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.ntndp_filename),
                            sheet_name='Emissions Rate', skiprows=1))

        # Rename columns and set index
        df = (df.rename(columns={'Generator': 'NTNDP_ID', 'Comb Co2 (kg/MWh)': 'EMISSIONS',
                                 'Date From': 'DATE_FROM',
                                 'Fugi Co2 (kg/MWh)': 'FUGITIVE_EMISSIONS'})
              .set_index('NTNDP_ID'))

        return df

    def _load_ntndp_build_costs_neutral(self):
        """Load candidate unit build costs (assuming neutral demand scenario)"""

        # Build costs - neutral demand scenario
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.ntndp_filename),
                            sheet_name='Build Cost', skiprows=3, nrows=137))

        # Rename columns and set index
        df = df.rename(columns={'$/kW': 'NTNDP_ID'}).set_index('NTNDP_ID')

        return df

    def _load_ntndp_build_costs_low(self):
        """Load candidate unit build costs (assuming low demand scenario)"""

        # Build costs - low demand scenario
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.ntndp_filename),
                            sheet_name='Build Cost', skiprows=144, nrows=137))

        # Rename columns and set index
        df = df.rename(columns={'$/kW': 'NTNDP_ID'}).set_index('NTNDP_ID')

        return df

    def _load_ntndp_coal_cost_neutral(self):
        """Coal cost information (neutral demand scenario)"""

        # Coal costs - neutral demand scenario
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.ntndp_filename),
                            sheet_name='Coal Cost', skiprows=2, nrows=22))

        # Rename columns and set index
        df = df.rename(columns={'Fuel Cost ($/GJ)': 'FUEL_COST_ID'}).set_index('FUEL_COST_ID')

        return df

    def _load_ntndp_coal_cost_low(self):
        """Coal cost information (low demand scenario)"""

        # Coal costs - low demand scenario
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.ntndp_filename),
                            sheet_name='Coal Cost', skiprows=28, nrows=22))

        # Rename columns and set index
        df = df.rename(columns={'Fuel Cost ($/GJ)': 'FUEL_COST_ID'}).set_index('FUEL_COST_ID')

        return df

    def _load_ntndp_gas_cost_neutral(self):
        """Gas cost information (neutral demand scenario)"""

        # Gas costs - neutral demand scenario
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.ntndp_filename),
                            sheet_name='Gas Cost', skiprows=2, nrows=63))

        # Rename columns and set index
        df = df.rename(columns={'Fuel Cost ($/GJ)': 'FUEL_COST_ID'}).set_index('FUEL_COST_ID')

        return df

    def _load_ntndp_gas_cost_low(self):
        """Gas cost information (low demand scenario)"""

        # Gas costs - low demand scenario
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.ntndp_filename),
                            sheet_name='Gas Cost', skiprows=69, nrows=63))

        # Rename columns and set index
        df = df.rename(columns={'Fuel Cost ($/GJ)': 'FUEL_COST_ID'}).set_index('FUEL_COST_ID')

        return df

    def _load_ntndp_battery_build_cost_neutral(self):
        """Battery build cost information (neutral demand scenario)"""

        # Battery build costs
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.ntndp_filename),
                            sheet_name='Battery Build Cost', skiprows=1, nrows=16))

        # Rename columns and set index
        df = df.rename(columns={'$/kW': 'UNIT_ID'}).set_index('UNIT_ID')

        return df

    def _load_ntndp_battery_build_cost_low(self):
        """Battery build cost information (low demand scenario)"""

        # Battery build costs
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.ntndp_filename),
                            sheet_name='Battery Build Cost', skiprows=20, nrows=16))

        # Rename columns and set index
        df = df.rename(columns={'$/kW': 'UNIT_ID'}).set_index('UNIT_ID')

        return df

    def _load_acil_existing_fuel_cost(self):
        """Existing generator fuel costs"""

        # Load existing generator fuel costs
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.acil_filename),
                            sheet_name='Existing Fuel Costs', skiprows=2, nrows=154))

        # Rename columns and set index
        df = df.rename(columns={'Profile': 'FUEL_COST_ID'}).set_index('FUEL_COST_ID')

        return df

    def _load_acil_candidate_technical_parameters(self):
        """Load sheet containing new candidate generator technical parameters"""

        # Load technical parameters for candidate units
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.acil_filename),
                            sheet_name='New Technologies', skiprows=1, nrows=19, usecols=[i for i in range(1, 28)]))

        # Rename columns and set index
        df = df.rename(columns={'Technology': 'ACIL_TECHNOLOGY_ID'}).set_index('ACIL_TECHNOLOGY_ID')

        return df

    def _load_acil_build_limits(self):
        """Load sheet containing technology build limits per zone"""

        # Load build limits by technology
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.acil_filename),
                            sheet_name='Build Limits', skiprows=1, nrows=62))

        # Rename columns and set index
        df = df.rename(columns={'Technology': 'ACIL_TECHNOLOGY_ID'})

        return df

    def get_existing_duids(self, fuel_type):
        """
        Get existing generator DUIDs by fuel type

        Parameters
        ----------
        fuel_type : str
            Type of fuel. Options: 'ALL', 'COAL', 'GAS', 'LIQUID', 'HYDRO', 'WIND'

        Returns
        -------
        duids : list
            List of existing generators with corresponding fuel type
        """

        # Filter for existing coal units
        mask_coal = self.df_g['FUEL_TYPE'].isin(['Brown coal', 'Black coal'])

        # Filter for exsiting gas units
        mask_gas = self.df_g['FUEL_TYPE'].isin(['Natural Gas (Pipeline)', 'Coal seam methane'])

        # Filter for exising liquid (e.g. diesel) units
        mask_liquid = self.df_g['FUEL_TYPE'].isin(['Kerosene - non aviation', 'Diesel oil'])

        # Filter for exsiting wind units
        mask_wind = self.df_g['FUEL_TYPE'].isin(['Wind'])

        # Filter for existing hydro units
        mask_hydro = self.df_g['FUEL_TYPE'].isin(['Hydro'])

        if fuel_type == 'ALL':
            duids = self.df_g.index

        elif fuel_type == 'COAL':
            duids = self.df_g.loc[mask_coal, :].index

        elif fuel_type == 'GAS':
            duids = self.df_g.loc[mask_gas, :].index

        elif fuel_type == 'LIQUID':
            duids = self.df_g.loc[mask_liquid, :].index

        elif fuel_type == 'HYDRO':
            duids = self.df_g.loc[mask_hydro, :].index

        elif fuel_type == 'WIND':
            duids = self.df_g.loc[mask_wind, :].index

        else:
            raise (Exception(f"Unexpected fuel type: '{fuel_type}' encountered"))

        return duids

    def get_all_candidate_units(self):
        """Extract information for all candidate units"""

        # All candidate unit IDs from FOM spreadsheet (use this as the basis for candidate unit IDs)
        all_candidate_units = self.df_ntndp_fom[329:].index

        # Candidate units and their respective locations (could be a wind bubble for wind generators, or zone ID)
        df_candidate = pd.DataFrame([(c, c.split(' ')[0]) for c in all_candidate_units],
                                    columns=['NTNDP_UNIT_ID', 'LOCATION'])

        # Merge zone IDs - mapping wind bubbles to NEM zones
        df_candidate = pd.merge(df_candidate, self.df_bubble_map[['ZONE']], how='left', left_on=['LOCATION'],
                                right_index=True)

        # Fill missing zones (LOCATION and ZONE will be the same for these generators)
        df_candidate['ZONE'] = df_candidate.apply(lambda x: x['LOCATION'] if pd.isnull(x['ZONE']) else x['ZONE'],
                                                  axis=1)

        def _get_technology_and_fuel_type(row):
            """Get technology and fuel type for each candidate unit"""

            # Wind farms
            if 'WIND' in row['NTNDP_UNIT_ID']:
                fuel_cat_1, fuel_cat_2, technology_cat_1, technology_cat_2 = 'WIND', 'WIND', 'WIND', 'WIND'

            else:
                # Get technology subcategory from NTNDP ID
                technology_cat_2 = '-'.join([i.upper() for i in row['NTNDP_UNIT_ID'].split(' ')[1:]])

                # Brown coal assumed in LV
                if (row['ZONE'] == 'LV') and ('COAL' in technology_cat_2):

                    # Technology primary category
                    technology_cat_1 = 'COAL'

                    # Primary and secondary fuel categories
                    fuel_cat_1, fuel_cat_2 = 'COAL', 'BROWN-COAL'

                # Black coal assumed everywhere else
                elif 'COAL' in technology_cat_2:

                    # Technology primary category
                    technology_cat_1 = 'COAL'

                    fuel_cat_1, fuel_cat_2 = 'COAL', 'BLACK-COAL'

                # Gas generators
                elif ('OCGT' in technology_cat_2) or ('CCGT' in technology_cat_2):

                    # Primary technology category
                    technology_cat_1 = 'GAS'

                    # Primary and secondary fuel category            
                    fuel_cat_1, fuel_cat_2 = 'GAS', 'GAS'

                # Solar farms
                elif 'SOLAR' in technology_cat_2:

                    # Primary technology category
                    technology_cat_1 = 'SOLAR'

                    # Primary and secondary fuel category            
                    fuel_cat_1, fuel_cat_2 = 'SOLAR', 'SOLAR'

                # Biomass plant
                elif 'BIOMASS' in technology_cat_2:

                    # Primary technology category
                    technology_cat_1 = 'BIOMASS'

                    # Primary and secondary fuel category            
                    fuel_cat_1, fuel_cat_2 = 'BIOMASS', 'BIOMASS'

                # Wave power
                elif 'WAVE' in technology_cat_2:

                    # Primary technology category
                    technology_cat_1 = 'WAVE'

                    # Primary and secondary fuel category            
                    fuel_cat_1, fuel_cat_2 = 'WAVE', 'WAVE'

                # Set value to NaN
                else:

                    # Primary technology category
                    technology_cat_1 = np.nan

                    # Primary and secondary fuel category            
                    fuel_cat_1, fuel_cat_2 = np.nan, np.nan

            return technology_cat_1, technology_cat_2, fuel_cat_1, fuel_cat_2

        # Assign technology and fuel type as columns
        df_candidate[
            ['TECHNOLOGY_PRIMARY', 'TECHNOLOGY_SUBCAT', 'FUEL_TYPE_PRIMARY', 'FUEL_TYPE_SUBCAT']] = df_candidate.apply(
            _get_technology_and_fuel_type, axis=1, result_type='expand')

        # Check no missing values
        assert not df_candidate[['TECHNOLOGY_PRIMARY', 'TECHNOLOGY_SUBCAT', 'FUEL_TYPE_PRIMARY',
                                 'FUEL_TYPE_SUBCAT']].isna().any().any(), 'Missing technology or fuel type label'

        return df_candidate

    def get_candidate_wind_units(self):
        """Get candidate wind generators"""

        # All candidate wind generators
        df = self.get_all_candidate_units().copy()
        mask_wind = df['TECHNOLOGY_PRIMARY'] == 'WIND'
        mask_40pc = df['NTNDP_UNIT_ID'].str.contains('40pc')

        # Only retain one wind bubble per zone
        df_wind = df.loc[mask_wind & ~mask_40pc, :].drop_duplicates(subset=['ZONE'], keep='first')

        # Candidate wind units
        df_wind['UNIT_ID'] = df_wind.apply(lambda x: x['ZONE'] + '-' + x['TECHNOLOGY_SUBCAT'], axis=1)

        # Set index and rename columns
        df_wind = df_wind.set_index('UNIT_ID').rename(columns={'LOCATION': 'WIND_BUBBLE'})

        def _get_acil_id(row):
            """Get ACIL Allen technology ID"""

            # Solar - dual axis tracking
            if row.name.endswith('WIND'):
                acil_id = 'Wind - (100 MW)'

            else:
                raise (Exception('Missing ACIL Allen technology ID assignment'))

            return acil_id

        # ACIL Allen technology ID
        df_wind['ACIL_TECHNOLOGY_ID'] = df_wind.apply(_get_acil_id, axis=1)

        return df_wind

    def get_candidate_solar_units(self):
        """Get candidate solar generators"""

        # All solar units
        df = self.get_all_candidate_units().copy()
        mask_solar = df['TECHNOLOGY_PRIMARY'] == 'SOLAR'

        # Remove generators with 40pc in name (duplicates)
        mask_40pc = df['NTNDP_UNIT_ID'].str.contains('40pc')

        # All solar units
        df_solar = df.loc[mask_solar & ~mask_40pc, :].copy()

        # Construct unit ID from zone name and solar unit type
        df_solar['UNIT_ID'] = df_solar['NTNDP_UNIT_ID'].apply(lambda x: x.upper().replace(' ', '-'))

        # Set index and drop redundant columns
        df_solar = df_solar.set_index('UNIT_ID').drop('LOCATION', axis=1)

        def _get_acil_id(row):
            """Get ACIL Allen technology ID"""

            # Solar - dual axis tracking
            if row.name.endswith('PV-DAT'):
                acil_id = 'Solar PV DAT'

            # Solar - single axis tracking
            elif row.name.endswith('PV-SAT'):
                acil_id = 'Solar PV SAT'

            # Solar - fixed
            elif row.name.endswith('PV-FFP'):
                acil_id = 'Solar PV FFP'

            else:
                raise (Exception('Missing candidate solar unit ACIL technology ID assignment'))

            return acil_id

        # ACIL Allen technology ID
        df_solar['ACIL_TECHNOLOGY_ID'] = df_solar.apply(_get_acil_id, axis=1)

        return df_solar

    def get_candidate_coal_units(self):
        """Get candidate coal units"""

        # Old candidate units
        df = self.get_all_candidate_units().copy()

        # Candidate coal generators
        mask_coal = df['FUEL_TYPE_PRIMARY'] == 'COAL'
        df_coal = df.loc[mask_coal, :].copy()

        # Construct unit ID
        df_coal['UNIT_ID'] = df_coal['NTNDP_UNIT_ID'].apply(lambda x: x.upper().replace(' ', '-'))

        # Set index and drop columns
        df_coal = df_coal.set_index('UNIT_ID').drop('LOCATION', axis=1)

        def _get_acil_id(row):
            """Get ACIL Allen technology ID

            Note: Assuming black coal geneators are candidate units 
            except for LV (brown coal assumed)
            """

            # Supercritical black coal (no CCS)
            if row.name.endswith('SC') and row['ZONE'] != 'LV':
                acil_id = 'Supercritical PC - Black coal without CCS'

            # Supercritical black coal (with CCS)
            elif row.name.endswith('SC-CCS') and row['ZONE'] != 'LV':
                acil_id = 'Supercritical PC - Black coal with CCS'

            # Supercritical brown coal (no CCS)
            elif row.name.endswith('SC') and row['ZONE'] == 'LV':
                acil_id = 'Supercritical PC - Brown coal without CCS'

            # Supercritical brown coal (with CCS)
            elif row.name.endswith('SC-CCS') and row['ZONE'] == 'LV':
                acil_id = 'Supercritical PC - Brown coal with CCS'

            else:
                raise (Exception('Missing assignment'))

            return acil_id

        # ACIL Allen technology ID
        df_coal['ACIL_TECHNOLOGY_ID'] = df_coal.apply(_get_acil_id, axis=1)

        return df_coal

    def get_candidate_gas_units(self):
        """Get candidate gas units"""

        # Old candidate units
        df = self.get_all_candidate_units().copy()

        # Candidate coal generators
        mask_gas = df['FUEL_TYPE_PRIMARY'] == 'GAS'
        df_gas = df.loc[mask_gas, :].copy()

        # Construct unit ID
        df_gas['UNIT_ID'] = df_gas['NTNDP_UNIT_ID'].apply(lambda x: x.upper().replace(' ', '-'))

        # Set index and drop columns
        df_gas = df_gas.set_index('UNIT_ID').drop('LOCATION', axis=1)

        def _get_acil_id(row):
            """Get ACIL Allen technology ID"""

            # OCGT (no CCS)
            if row.name.endswith('OCGT'):
                acil_id = 'OCGT - Without CCS'

            # CCGT (no CCS)
            elif row.name.endswith('CCGT'):
                acil_id = 'CCGT - Without CCS'

            # CCGT with CCS
            elif row.name.endswith('CCGT-CCS'):
                acil_id = 'CCGT - With CCS'

            else:
                raise (Exception('Missing ACIL Allen technology ID assignment'))

            return acil_id

        # ACIL Allen technology ID
        df_gas['ACIL_TECHNOLOGY_ID'] = df_gas.apply(_get_acil_id, axis=1)

        return df_gas

    def get_all_units(self):
        """Get all existing and candidate unit IDs"""

        # All unit IDs
        all_units = (self.df_g.index
                     .union(self.get_candidate_coal_units().index)
                     .union(self.get_candidate_gas_units().index)
                     .union(self.get_candidate_solar_units().index)
                     .union(self.get_candidate_wind_units().index))

        return all_units

    def get_candidate_coal_fuel_cost_profiles(self):
        """Get candidate coal generator fuel cost profiles"""

        # All candidate coal generators
        df = self.get_candidate_coal_units().copy()

        # Add fuel cost ID for each candidate generator
        df['FUEL_COST_ID'] = df.apply(lambda x: self.df_ntndp_coal_cost_map.loc[x.name, 'NTNDP_FUEL_COST_ID'], axis=1)

        # Coal cost profiles
        df_coal_cost = self.df_ntndp_coal_cost.copy()

        # Update columns
        new_cols = {i: int(i.split('-')[0]) for i in df_coal_cost}
        df_coal_cost = df_coal_cost.rename(columns=new_cols)

        # Candidate coal generator fuel costs
        df_candidate_coal_cost = (pd.merge(df[['FUEL_COST_ID']], df_coal_cost,
                                           how='left', left_on='FUEL_COST_ID', right_index=True)
                                  .drop('FUEL_COST_ID', axis=1))

        # Check for missing values
        assert not df_candidate_coal_cost.isna().any().any(), 'Missing coal fuel cost profile values'

        return df_candidate_coal_cost

    def get_candidate_gas_fuel_cost_profiles(self):
        """Get candidate generator fuel cost profiles"""

        # All candidate gas generators
        df = self.get_candidate_gas_units().copy()

        def _get_gas_fuel_cost_id(row):
            """Construct gas fuel cost ID from unit ID"""

            # NTNDP fuel cost ID
            cost_id = row.name.replace('-', ' ').replace('CCS', '(CCS)')

            return cost_id

        df['FUEL_COST_ID'] = df.apply(_get_gas_fuel_cost_id, axis=1)

        # Must manually correct these fuel cost ID assignments
        manual_map = [
            ('NCEN-CCGT', 'NCEN CCGT (CCS)'),
            ('NNS-CCGT', 'NNS CCGT (CCS)'),
            ('CQ-CCGT', 'CQ CCGT (CCS)'),
            ('NQ-CCGT', 'NQ CCGT (CCS)'),
            ('SEQ-CCGT', 'SEQ CCGT (CCS)'),
            ('SWQ-CCGT', 'SWQ CCGT (CCS)'),
            ('SESA-CCGT', 'SESA CCGT (CCS)'),
            ('LV-CCGT', 'LV CCGT (CCS)'),
            ('TAS-CCGT', 'TAS CCGT (CCS)')
        ]

        # Update fuel cost IDs for selected gas generators
        for unit_id, fuel_cost_id in manual_map:
            # Add fuel cost ID
            df.loc[unit_id, 'FUEL_COST_ID'] = fuel_cost_id

        # Gas cost
        df_gas_cost = self.df_ntndp_gas_cost.copy()

        # Rename columns
        new_cols = {i: int(i.split('-')[0]) for i in df_gas_cost.columns}
        df_gas_cost = df_gas_cost.rename(columns=new_cols)

        # Candidate gas cost
        df_candidate_gas_cost = (pd.merge(df[['FUEL_COST_ID']], df_gas_cost, how='left',
                                          left_on=['FUEL_COST_ID'], right_index=True)
                                 .drop('FUEL_COST_ID', axis=1))

        # Check for missing values
        assert not df_candidate_gas_cost.isna().any().any(), 'Missing gas fuel cost profile values'

        return df_candidate_gas_cost

    def get_candidate_gas_heat_rates(self):
        """Get heat rates for candidate gas generators"""

        # All candidate gas generators
        df = self.get_candidate_gas_units().copy()

        # Heat rates for all generators
        df_heat_rates = self.df_ntndp_heat_rates.copy()

        # Remove duplicates
        df_heat_rates = df_heat_rates[~df_heat_rates.index.duplicated(keep='first')]

        df_candidate_gas_heat_rates = (pd.merge(df, df_heat_rates[['HEAT_RATE']],
                                                how='left', left_on=['NTNDP_UNIT_ID'],
                                                right_index=True)[['HEAT_RATE']])
        # Check for missing values
        assert not df_candidate_gas_heat_rates['HEAT_RATE'].isna().any(), 'Missing candidate gas generator heat rates'

        return df_candidate_gas_heat_rates

    def get_candidate_coal_heat_rates(self):
        """Get heat rates for candidate gas generators"""

        # All candidate coal generators
        df = self.get_candidate_coal_units().copy()

        # Heat rates for all generators
        df_heat_rates = self.df_ntndp_heat_rates.copy()

        # Remove duplicates
        df_heat_rates = df_heat_rates[~df_heat_rates.index.duplicated(keep='first')]

        df_candidate_coal_heat_rates = (pd.merge(df, df_heat_rates[['HEAT_RATE']],
                                                 how='left', left_on=['NTNDP_UNIT_ID'],
                                                 right_index=True)[['HEAT_RATE']])

        # Check for missing values
        assert not df_candidate_coal_heat_rates['HEAT_RATE'].isna().any(), 'Missing candidate coal generator heat rates'

        return df_candidate_coal_heat_rates

    def get_candidate_solar_heat_rates(self):
        """Get heat rates for candidate solar generators (assume=0)"""

        # Get all candidate solar units
        df = self.get_candidate_solar_units().copy()

        # Set emission=0 for all solar units
        df['HEAT_RATE'] = 0

        # Only retain 'emissions' column
        df_o = df[['HEAT_RATE']]

        return df_o

    def get_candidate_wind_heat_rates(self):
        """Get heat rates for candidate wind generators (assume=0)"""

        # Get all candidate solar units
        df = self.get_candidate_wind_units().copy()

        # Set emission=0 for all wind units
        df['HEAT_RATE'] = 0

        # Only retain 'emissions' column
        df_o = df[['HEAT_RATE']]

        return df_o

    def get_candidate_wind_vom_cost(self):
        """Get candidate wind generator variable operating and maintenance cost"""

        # All candidate wind generators
        df = self.get_candidate_wind_units().copy()

        # Variable operating and maintenance cost
        df_vom = pd.merge(df, self.df_ntndp_vom, how='left', left_on='NTNDP_UNIT_ID', right_index=True)[['VOM']]

        # Check for missing values
        assert not df_vom.isna().any().any(), 'Missing wind VOM values'

        return df_vom

    def get_candidate_solar_vom_cost(self):
        """Get candidate solar generator variable operating and maintenance cost"""

        # All candidate solar generators
        df = self.get_candidate_solar_units().copy()

        # Variable operating and maintenance cost
        df_vom = pd.merge(df, self.df_ntndp_vom, how='left', left_on='NTNDP_UNIT_ID', right_index=True)[['VOM']]

        # Check for missing values
        assert not df_vom.isna().any().any(), 'Missing solar VOM values'

        return df_vom

    def get_candidate_coal_vom_cost(self):
        """Get candidate coal generator variable operating and maintenance cost"""

        # All candidate coal generators
        df = self.get_candidate_coal_units().copy()

        # Variable operating and maintenance cost
        df_vom = pd.merge(df, self.df_ntndp_vom, how='left', left_on='NTNDP_UNIT_ID', right_index=True)[['VOM']]

        # Check for missing values
        assert not df_vom.isna().any().any(), 'Missing coal VOM values'

        return df_vom

    def get_candidate_gas_vom_cost(self):
        """Get candidate gas generator variable operating and maintenance cost"""

        # All candidate gas generators
        df = self.get_candidate_gas_units().copy()

        # Variable operating and maintenance cost
        df_vom = pd.merge(df, self.df_ntndp_vom, how='left', left_on='NTNDP_UNIT_ID', right_index=True)[['VOM']]

        # Check for missing values
        assert not df_vom.isna().any().any(), 'Missing gas VOM values'

        return df_vom

    def get_candidate_wind_fom_cost(self):
        """Get candidate wind generator fixed operating and maintenance cost"""

        # All candidate wind generators
        df = self.get_candidate_wind_units().copy()

        # Variable operating and maintenance cost
        df_fom = pd.merge(df, self.df_ntndp_fom, how='left', left_on='NTNDP_UNIT_ID', right_index=True)[['FOM']]

        # Check for missing values
        assert not df_fom.isna().any().any(), 'Missing wind FOM values'

        return df_fom

    def get_candidate_solar_fom_cost(self):
        """Get candidate solar generator fixed operating and maintenance cost"""

        # All candidate solar generators
        df = self.get_candidate_solar_units().copy()

        # Variable operating and maintenance cost
        df_fom = pd.merge(df, self.df_ntndp_fom, how='left', left_on='NTNDP_UNIT_ID', right_index=True)[['FOM']]

        # Check for missing values
        assert not df_fom.isna().any().any(), 'Missing solar FOM values'

        return df_fom

    def get_candidate_coal_fom_cost(self):
        """Get candidate coal generator fixed operating and maintenance cost"""

        # All candidate coal generators
        df = self.get_candidate_coal_units().copy()

        # Variable operating and maintenance cost
        df_fom = pd.merge(df, self.df_ntndp_fom, how='left', left_on='NTNDP_UNIT_ID', right_index=True)[['FOM']]

        # Check for missing values
        assert not df_fom.isna().any().any(), 'Missing coal FOM values'

        return df_fom

    def get_candidate_gas_fom_cost(self):
        """Get candidate gas generator fixed operating and maintenance cost"""

        # All candidate gas generators
        df = self.get_candidate_gas_units().copy()

        # Variable operating and maintenance cost
        df_fom = pd.merge(df, self.df_ntndp_fom, how='left', left_on='NTNDP_UNIT_ID', right_index=True)[['FOM']]

        # Check for missing values
        assert not df_fom.isna().any().any(), 'Missing gas FOM values'

        return df_fom

    def get_candidate_coal_emissions_rates(self):
        """Get emissions rates for candidate coal generators"""

        # NTNDP emissions
        df_emissions = self.df_ntndp_emissions.copy()

        # Remove duplicates
        df_emissions = df_emissions.loc[~df_emissions.index.duplicated(keep='first'), :]

        # Divide kgCO2/MWh by 1000 to get tCO2/MWh
        df_emissions[['EMISSIONS', 'FUGITIVE_EMISSIONS']] = df_emissions[['EMISSIONS', 'FUGITIVE_EMISSIONS']].div(1000)

        # Candidate coal units
        df = self.get_candidate_coal_units()

        # Join candidate coal emissions rates
        df_coal_emissions = (pd.merge(df[['NTNDP_UNIT_ID']], df_emissions,
                                      how='left', left_on=['NTNDP_UNIT_ID'],
                                      right_index=True)[['EMISSIONS']])

        assert not df_coal_emissions.isna().any().any(), 'Missing candidate coal emissions rates'

        return df_coal_emissions

    def get_candidate_gas_emissions_rates(self):
        """Get emissions rates for candidate gas generators"""

        # NTNDP emissions
        df_emissions = self.df_ntndp_emissions.copy()

        # Remove duplicates
        df_emissions = df_emissions.loc[~df_emissions.index.duplicated(keep='first'), :]

        # Divide kgCO2/MWh by 1000 to get tCO2/MWh
        df_emissions[['EMISSIONS', 'FUGITIVE_EMISSIONS']] = df_emissions[['EMISSIONS', 'FUGITIVE_EMISSIONS']].div(1000)

        # Candidate coal units
        df = self.get_candidate_gas_units()

        # Join candidate coal emissions rates
        df_gas_emissions = (pd.merge(df[['NTNDP_UNIT_ID']], df_emissions,
                                     how='left', left_on=['NTNDP_UNIT_ID'],
                                     right_index=True)[['EMISSIONS']])

        assert not df_gas_emissions.isna().any().any(), 'Missing candidate gas emissions rates'

        return df_gas_emissions

    def get_candidate_solar_emissions_rates(self):
        """Get emissions rates for candidate solar generators (assume=0)"""

        # Get all candidate solar units
        df = self.get_candidate_solar_units().copy()

        # Set emission=0 for all solar units
        df['EMISSIONS'] = 0

        # Only retain 'emissions' column
        df_o = df[['EMISSIONS']]

        return df_o

    def get_candidate_wind_emissions_rates(self):
        """Get emissions rates for candidate wind generators (assume=0)"""

        # Get all candidate wind units
        df = self.get_candidate_wind_units().copy()

        # Set emission=0 for all solar units
        df['EMISSIONS'] = 0

        # Only retain 'emissions' column
        df_o = df[['EMISSIONS']]

        return df_o

    def get_existing_coal_fuel_cost_profiles(self):
        """Get fuel cost profiles for existing coal generators"""

        # Get DUIDs for existing coal generators
        existing_coal_duids = self.get_existing_duids('COAL')

        # All existing coal generators
        df = self.df_g.reindex(existing_coal_duids).copy()

        # Join fuel cost profile IDs
        df = df.join(self.df_fuel_cost_map[['FUEL_COST_ID']], how='left')

        # Check every existing coal generator has a corresponding fuel cost profile ID
        assert not df['FUEL_COST_ID'].isna().any(), 'Existing coal generator missing fuel cost ID'

        # Existing coal generators
        df_existing_coal = pd.merge(df[['FUEL_COST_ID']], self.df_ntndp_coal_cost, how='left',
                                    left_on=['FUEL_COST_ID'], right_index=True).drop('FUEL_COST_ID', axis=1)

        # Rename columns
        new_columns = {i: int(i.split('-')[0]) for i in df_existing_coal.columns}
        df_existing_coal = df_existing_coal.rename(columns=new_columns)

        # Check for missing values
        assert not df_existing_coal.isna().any().any(), 'Missing fuel cost values for existing coal generators'

        return df_existing_coal

    def get_existing_gas_and_liquid_fuel_cost_profiles(self):
        """Get fuel cost profiles for existing gas and liquid fuel generators"""

        # Get DUIDs for existing gas and liquid fuel generators
        existing_gas_duids = self.get_existing_duids('GAS')
        existing_liquid_fuel_duids = self.get_existing_duids('LIQUID')

        # All existing gas and liquid fuel generators
        df = self.df_g.reindex(existing_gas_duids + existing_liquid_fuel_duids).copy()

        # Join fuel cost profile IDs
        df = df.join(self.df_fuel_cost_map[['FUEL_COST_ID']], how='left')

        # Check every existing gas and liquid fuel generator has a corresponding fuel cost profile ID
        assert not df['FUEL_COST_ID'].isna().any(), 'Existing gas generator missing fuel cost ID'

        # Existing gas generators
        df_existing = pd.merge(df[['FUEL_COST_ID']], self.df_ntndp_gas_cost, how='left',
                               left_on=['FUEL_COST_ID'], right_index=True)

        # Rename columns
        new_columns = {i: int(i.split('-')[0]) for i in df_existing.columns if '-' in i}
        df_existing = df_existing.rename(columns=new_columns)

        # Generators missing cost information for all years
        missing_duids = df_existing[df_existing.drop('FUEL_COST_ID', axis=1).isna().all(axis=1)].index

        # Join fuel cost IDs for missing generators
        df_missing = df_existing.reindex(missing_duids)[['FUEL_COST_ID']]

        # Copy ACIL Allen fuel cost profiles with reset index
        df_acil_cost = self.df_acil_fuel_cost.reset_index().copy()

        # Only consider fuel cost information for liquid fuel
        mask_fuel_type = df_acil_cost['FUEL_COST_ID'] == 'Liquid Fuel'

        # Screen scenarios
        if self.scenario == 'neutral':
            mask_scenario = df_acil_cost['Scenario'] == 'Medium'

        elif self.scenario == 'low':
            mask_scenario = df_acil_cost['Scenario'] == 'Low'

        else:
            raise (Exception(f"Unexpected scenario encountered {self.scenario}."))

        # Only retain row corresponding to liquid fuel type and given scenario
        df_acil_liquid_cost_profile = df_acil_cost.loc[mask_fuel_type & mask_scenario, :].set_index(
            'FUEL_COST_ID').copy()

        # Merge cost information for liquid fuel units
        df_liquid_cost_profile = pd.merge(df_missing, df_acil_liquid_cost_profile, how='left', left_on='FUEL_COST_ID',
                                          right_index=True).drop(['FUEL_COST_ID', 'Scenario'], axis=1)

        # Update cost profile information for liquid fuel generators that have
        # values missing in the NTNDP database
        df_existing.update(df_liquid_cost_profile)

        # Drop fuel cost ID column
        df_existing = df_existing.drop('FUEL_COST_ID', axis=1)

        # Check no rows missing all values for all years
        assert not df_existing.isna().all(axis=1).any(), 'At least one generator with no fuel costs for entire horizon'

        # Locations where zeros are identified in matrix
        # Note: some gas generators have their fuel cost = 0 (problem with NTNDP spreadsheet)
        # must identify these entries accordingly.
        zeros = np.where(df_existing == 0)

        # Denote these as missing values
        df_existing.values[zeros[0], zeros[1]] = np.nan

        # Forward fill missing values, then backfill
        df_existing = df_existing.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)

        # Check no missing values
        assert not df_existing.isna().any().any(), 'Missing values for gas and liquid fuel cost profile'

        return df_existing

    def get_existing_unit_fom_costs(self):
        """Get FOM costs for existing generators"""

        # Join FOM cost IDs to dataframe summarising unit information
        df_tmp = self.df_g.join(self.df_fom_cost_map[['FOM_COST_ID']], how='left').copy()

        # Use foreign key to join NTNDP FOM costs
        df_o = pd.merge(df_tmp, self.df_ntndp_fom, how='left', left_on='FOM_COST_ID', right_index=True)[['FOM']]

        # Check that no values are missing
        assert not df_o['FOM'].isna().any(), 'Missing FOM cost values'

        return df_o

    def get_all_fom_costs(self):
        """Get FOM costs for all unit and place in a single DataFrame"""

        # FOM costs for different types of units
        fom = [self.get_candidate_coal_fom_cost(),
               self.get_candidate_gas_fom_cost(),
               self.get_candidate_solar_fom_cost(),
               self.get_candidate_wind_fom_cost(),
               self.get_existing_unit_fom_costs()]

        # Concatenate FOM costs
        df_fom = pd.concat(fom)

        # Check no missing values
        assert not df_fom.isna().any().any(), 'Missing FOM cost values'

        return df_fom

    def get_all_vom_costs(self):
        """Get VOM costs for all unit and place in a single DataFrame"""

        # VOM costs for different types of units
        vom = [self.get_candidate_coal_vom_cost(),
               self.get_candidate_gas_vom_cost(),
               self.get_candidate_solar_vom_cost(),
               self.get_candidate_wind_vom_cost(),
               self.df_g[['VOM']]]

        # Concatenate VOM costs
        df_vom = pd.concat(vom)

        # Check no missing values
        assert not df_vom.isna().any().any(), 'Missing VOM cost values'

        return df_vom

    def get_all_heat_rates(self):
        """Get heat rates for all units and place in a single DataFrame"""

        # Heat rates for different types of units
        heat_rates = [self.get_candidate_coal_heat_rates(),
                      self.get_candidate_gas_heat_rates(),
                      self.get_candidate_solar_heat_rates(),
                      self.get_candidate_wind_heat_rates(),
                      self.df_g[['HEAT_RATE']]]

        # Concatenate heat rates
        df_heat_rates = pd.concat(heat_rates)

        # Check no missing values
        assert not df_heat_rates.isna().any().any(), 'Missing heat rate values'

        return df_heat_rates

    def get_all_emissions_rates(self):
        """Get emissions rates for all units and place in a single DataFrame"""

        # Emissions rates for different types of units
        emissions_rates = [self.get_candidate_coal_emissions_rates(),
                           self.get_candidate_gas_emissions_rates(),
                           self.get_candidate_solar_emissions_rates(),
                           self.get_candidate_wind_emissions_rates(),
                           self.df_g[['EMISSIONS']]]

        # Concatenate heat rates
        df_emissions_rates = pd.concat(emissions_rates)

        # Check no missing values
        assert not df_emissions_rates.isna().any().any(), 'Missing emissions rate values'

        return df_emissions_rates

    def get_static_data(self):
        """Compile parameters for all units that do not vary over time (static parameters)"""

        # FOM costs
        df_fom = self.get_all_fom_costs()

        # VOM costs
        df_vom = self.get_all_vom_costs()

        # Heat rates
        df_heat_rates = self.get_all_heat_rates()

        # Emissions rates
        df_emissions_rates = self.get_all_emissions_rates()

        # Concatenate in single DataFrame
        df_static = pd.concat([df_fom, df_vom, df_heat_rates, df_emissions_rates], axis=1)

        # Check no missing values
        assert not df_static.isna().any().any(), 'Missing static values'

        return df_static

    def get_time_varying_fuel_costs(self):
        """Compile fuel cost data for all units (costs vary over time)"""

        # All fuel cost information in a single DataFrame
        df_fuel_costs = pd.concat([self.get_candidate_coal_fuel_cost_profiles(),
                                   self.get_candidate_gas_fuel_cost_profiles(),
                                   self.get_existing_coal_fuel_cost_profiles(),
                                   self.get_existing_gas_and_liquid_fuel_cost_profiles()],
                                  sort=False)

        # Reindex so all units included. Fill forward along rows (2041 fuel cost value missing for some units).
        # Then fill missing entries with 0. Assume fuel cost for wind and solar generators = 0 for entire horizon.
        # Will check later that no thermal plant have fuel costs=0.
        df_fuel_costs = df_fuel_costs.reindex(self.get_all_units()).fillna(method='ffill', axis=1).fillna(0)

        return df_fuel_costs

    def get_time_varying_build_costs(self):
        """Compile build cost data into a single DataFrame"""

        # All candidate units which can be invested in
        df_candidate_units = pd.concat([self.get_candidate_coal_units()[['NTNDP_UNIT_ID']],
                                        self.get_candidate_gas_units()[['NTNDP_UNIT_ID']],
                                        self.get_candidate_wind_units()[['NTNDP_UNIT_ID']],
                                        self.get_candidate_solar_units()[['NTNDP_UNIT_ID']]])

        # candidate_units
        df_build_cost = pd.merge(df_candidate_units, self.df_ntndp_build_cost, how='left',
                                 left_on='NTNDP_UNIT_ID', right_index=True).drop('NTNDP_UNIT_ID', axis=1)

        # Update column names. Assume 2016-17 applies from Jan 1 2016 - Dec 31 2016
        new_cols = {i: int(i.split('-')[0]) for i in df_build_cost.columns}
        df_build_cost = df_build_cost.rename(columns=new_cols)

        # Check for missing values
        assert not df_build_cost.isna().any().any(), 'Missing build cost values'

        return df_build_cost

    def get_combined_unit_information(self):
        """Combine unit information into a single DataFrame"""

        # Static generator information (heat rates, FOM, VOM costs)
        df_static = self.get_static_data().copy()
        df_static.columns = pd.MultiIndex.from_product([['STATIC'], df_static.columns])

        # Time varying fuel costs
        df_fuel_costs = self.get_time_varying_fuel_costs().copy()
        df_fuel_costs.columns = pd.MultiIndex.from_product([['FUEL_COST'], df_fuel_costs.columns])

        # Candidate unit information
        df_candidate_coal = self.get_candidate_coal_units().copy()
        df_candidate_gas = self.get_candidate_gas_units().copy()
        df_candidate_solar = self.get_candidate_solar_units().copy()
        df_candidate_wind = self.get_candidate_wind_units().copy()

        # Combine into single DataFrame
        df_candidate = pd.concat([df_candidate_coal, df_candidate_gas, df_candidate_solar, df_candidate_wind],
                                 sort=False).drop(['NTNDP_UNIT_ID', 'WIND_BUBBLE', 'GEN_TYPE'], axis=1)
        df_candidate['UNIT_TYPE'] = 'CANDIDATE'

        # Map between fuel types and fuel categories for existing generators
        fuel_type_map = {'Wind': 'WIND',
                         'Brown coal': 'COAL',
                         'Black coal': 'COAL',
                         'Natural Gas (Pipeline)': 'GAS',
                         'Hydro': 'HYDRO',
                         'Coal seam methane': 'GAS',
                         'Solar': 'SOLAR',
                         'Kerosene - non aviation': 'LIQUID',
                         'Diesel oil': 'LIQUID'}

        # Existing generators
        df_existing = self.df_g[['FUEL_TYPE', 'NEM_ZONE']].copy()

        # Assign fuel categories
        df_existing['FUEL_CAT'] = df_existing.apply(lambda x: fuel_type_map[x['FUEL_TYPE']], axis=1)
        df_existing = df_existing.drop('FUEL_TYPE', axis=1).rename(columns={'NEM_ZONE': 'ZONE'})
        df_existing['UNIT_TYPE'] = 'EXISTING'

        # Combine candidate and existing unit information into single DataFrame
        df_unit_info = pd.concat([df_candidate, df_existing])
        df_unit_info.columns = pd.MultiIndex.from_product([['INFO'], df_unit_info.columns])
        df_units = df_unit_info.join(df_static, how='left').join(df_fuel_costs, how='left')

        # Check for missing values
        assert not df_units.isna().any().any(), 'Missing unit information'

        # Filters
        # -------
        # Renewable generators
        mask_renewables = df_units[('INFO', 'FUEL_CAT')].isin(['SOLAR', 'WIND', 'HYDRO'])

        # Non-renewables
        mask_non_renewables = df_units[('INFO', 'FUEL_CAT')].isin(['LIQUID', 'COAL', 'GAS'])

        # Perform checks
        # --------------
        # Check that fuel costs for renewables are zero
        assert not df_units.loc[mask_renewables, ('FUEL_COST', slice(None))].ne(
            0).any().any(), 'Fuel cost for renewables not 0'

        # Check that emisisons rates for renewables are zero
        assert not df_units.loc[mask_renewables, ('STATIC', 'EMISSIONS')].ne(
            0).any().any(), 'Emissions for renewables not 0'

        # Check that fuel costs for non-renewables are greater than zero
        assert df_units.loc[mask_non_renewables, ('FUEL_COST', slice(None))].gt(
            0).any().any(), 'Fuel cost for non-renewables less than or equal to 0'

        # Check that emisisons rates for renewables are zero
        assert df_units.loc[mask_non_renewables, ('STATIC', 'EMISSIONS')].gt(
            0).any().any(), 'Emissions for non-renewables less than or equal to 0'

        return df_units

    def get_battery_build_cost(self):
        """Load and format batter build cost information"""

        # Load data for neutral demand scenario
        if self.scenario == 'neutral':
            df = self._load_ntndp_battery_build_cost_neutral().copy()

        # Load data for low demand scenario
        elif self.scenario == 'low':
            df = self._load_ntndp_battery_build_cost_low().copy()

        else:
            raise (Exception(f"Unexpected scenario: {self.scenario} encountered"))

        # Format index
        df.index = [i.replace(' ', '-').upper() for i in df.index]

        # Rename columns
        new_cols = {i: int(i.split('-')[0]) for i in df.columns}
        df = df.rename(columns=new_cols)

        return df

    def get_candidate_unit_technical_parameters(self):
        """Candidate unit technical parameters"""

        # Candidate units
        dfs = [self.get_candidate_wind_units().copy(),
               self.get_candidate_solar_units().copy(),
               self.get_candidate_coal_units().copy(),
               self.get_candidate_gas_units().copy(),
               ]

        # Place all candidate unit information into a single DataFrame
        df = pd.concat(dfs, sort=False)

        # ACIL Allen technical information for candidate units
        df_acil = self._load_acil_candidate_technical_parameters()

        # Columns to retain along with their new names
        cols = {'Ramp Up Rate (MW/h)': 'RR_UP',
                'Ramp Down Rate (MW/h)': 'RR_DOWN',
                'Warm Start-up Costs ($/MW sent-out)': 'SU_COST_WARM_MW',
                'No Load Fuel Consumption (% of Full Load Fuel Consumption)': 'NL_FUEL_CONS',
                'Min Gen (%)': 'MIN_GEN_PERCENT'}

        # ACIL Allen technical parameter information for candidate units
        df_acil_merge = df_acil.rename(columns=cols)[cols.values()]

        # Columns to retain
        keep_cols = list(cols.values()) + ['ZONE', 'GEN_TYPE', 'FUEL_CAT']
        df_candidate_parameters = (pd.merge(df, df_acil_merge,
                                            how='left', left_on='ACIL_TECHNOLOGY_ID',
                                            right_index=True)[keep_cols])

        # Assume that startup ramp-rate is same as normal operation ramp rate (no other data)
        df_candidate_parameters['RR_STARTUP'] = df_candidate_parameters['RR_UP']

        # Assume that shutdown ramp-rate is same as normal operation ramp rate (no other data)
        df_candidate_parameters['RR_SHUTDOWN'] = df_candidate_parameters['RR_DOWN']

        # Divide by 100 to get min generation percentage as fraction
        df_candidate_parameters['MIN_GEN_PERCENT'] = df_candidate_parameters['MIN_GEN_PERCENT'].div(100)

        def _get_min_on_time(row):
            """Get minimum on time for candidate units"""

            # Based on existing OCGT unit in ACIL Allen spreadsheet
            if row.name.endswith('OCGT'):
                return 1

            # Based on existing CCGT unit in ACIL Allen spreadsheet
            elif row.name.endswith('CCGT') or row.name.endswith('CCGT-CCS'):
                return 4

            # Note: is = 1 in ACIL, but think this is misleading. Zero likely a better value.
            elif row.name.endswith('PV-DAT') or row.name.endswith('PV-SAT') or row.name.endswith('PV-FFP'):
                return 0

            # Note: is = 1 in ACIL, but think this is misleading. Zero likely a better value.
            elif row.name.endswith('WIND'):
                return 0

            # Assumption for black coal generators
            elif (row.name.endswith('COAL-SC') or row.name.endswith('COAL-SC-CCS')) and row['ZONE'] != 'LV':
                return 8

            # Assuming brown coal generators in LV have min on time of 16 hours
            elif (row.name.endswith('COAL-SC') or row.name.endswith('COAL-SC-CCS')) and row['ZONE'] == 'LV':
                return 16

            else:
                raise (Exception('Missing minimum on time assignment for candidate units'))

        # Minimum on time
        df_candidate_parameters['MIN_ON_TIME'] = df_candidate_parameters.apply(_get_min_on_time, axis=1)

        # Assume minimum off time is same as minimum on time (consistent with ACIL Allen existing technologies
        # spreadsheet)
        df_candidate_parameters['MIN_OFF_TIME'] = df_candidate_parameters.apply(_get_min_on_time, axis=1)

        return df_candidate_parameters

    def get_build_limits(self):
        """Get build limits for each candidate technology, for each zone"""

        # Load build limits from ACIL Allen spreadsheet
        df = self._load_acil_build_limits().copy()

        # Container for technology build limits for each zone
        build_limits = []

        # All NEM zones
        all_zones = self.df_g['NEM_ZONE'].unique()

        # All regions
        all_regions = self.df_g['NEM_REGION'].unique()

        for index, row in df.iterrows():
            if row['Region'] == 'NEM Wide':

                # Loop through all regions
                for r in all_regions:

                    # Loop through all zones
                    for z in self.df_zones_map.loc[self.df_zones_map['REGION'].eq(r), :].index:

                        # Build limit for zone
                        build_limit = row['Max Build Limit per Zone (MW)']

                        # If value missing, assume build limit is unconstrained
                        if pd.isnull(build_limit):
                            # Set build limit to arbitrarily large number
                            build_limit = 99999

                        # Append build limit information to container
                        build_limits.append((row['ACIL_TECHNOLOGY_ID'], r, z, build_limit))

            else:
                if row['Zone'] == 'All':
                    # All zones beloning to a given region
                    zones = self.df_zones_map.loc[self.df_zones_map['REGION'].eq(row['Region']), :].index

                elif row['ACIL_TECHNOLOGY_ID'] == 'Solar PV DAT':
                    # Note: Unclear build limits from ACIL spreadsheet, but this technology 
                    # type should have unconstrained build limits given that other solar options do.
                    # Setting build limit to a comparable level as other solar units.

                    # Loop through all regions
                    for r in all_regions:

                        # Loop through all zones
                        for z in self.df_zones_map.loc[self.df_zones_map['REGION'].eq(r), :].index:

                            # Build limit for zone
                            build_limit = row['Max Build Limit per Zone (MW)']

                            # If value missing, assume build limit is unconstrained
                            if pd.isnull(build_limit):
                                # Set build limit to arbitrarily large number
                                build_limit = 40000

                            # Append build limit information to container
                            build_limits.append((row['ACIL_TECHNOLOGY_ID'], r, z, build_limit))

                else:
                    zones = row['Zone'].replace(' ', '').split(',')

                    # For each zone
                    for z in zones:

                        # Build limit for zone
                        build_limit = row['Max Build Limit per Zone (MW)']

                        # If value missing, assume build limit is unconstrained
                        if pd.isnull(build_limit):
                            # Set build limit to arbitrarily large number
                            build_limit = 99999

                        # Append build limit information to container
                        build_limits.append((row['ACIL_TECHNOLOGY_ID'], row['Region'], z, build_limit))

                        # Build limits for each technology per zone
        df_build_limits = pd.DataFrame(build_limits, columns=['ACIL_TECHNOLOGY_ID', 'REGION', 'ZONE', 'BUILD_LIMIT'])

        # Pivot so different technologies comprise the index, and zones the columns. Cell values denote build limit
        # for technology for each zone.
        df_build_limits_pivot = df_build_limits.pivot(index='ACIL_TECHNOLOGY_ID', columns='ZONE',
                                                      values='BUILD_LIMIT').reindex(columns=all_zones).fillna(0)

        return df_build_limits_pivot


# Paths
# -----
# Directory containing core data files
data_directory = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data')

Dataset = ConstructDataset(data_directory, scenario='neutral')
self = Dataset

a = self.get_candidate_gas_fuel_cost_profiles()


