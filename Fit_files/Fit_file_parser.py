import pandas as pd
from fitparse import FitFile
import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog, messagebox
import inspect


def define_casette_sizes(casette: str) -> list:
    casette_options = {
        'ultegra_12_speed_11_30_cassette': [11, 12, 13, 14, 15, 16, 17, 19, 21, 24, 27, 30],
        'ultegra_12_speed_11_32_cassette': [11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30, 32]
    }
    return casette_options.get(casette, [])


def fitfile_to_dataframe(file_path: str) -> pd.DataFrame:
    """Converts a FIT file into a pandas DataFrame."""
    try:
        fitfile = FitFile(file_path)
        records = []
        for record in fitfile.get_messages('record'):
            record_data = {data.name: data.value for data in record}
            records.append(record_data)
        return pd.DataFrame(records)
    except Exception as e:
        raise ValueError(f"Error processing FIT file: {e}")


def estimate_gear_ratio(cadence: int, speed: float, tyre_width=0.028) -> float:
    tyre_diameter = 0.622 + 2 * tyre_width
    tyre_circumference = tyre_diameter * math.pi
    tyre_revolutions = speed / tyre_circumference
    crank_revolutions = cadence / 60
    return tyre_revolutions / crank_revolutions if crank_revolutions > 0 else float('nan')


def calculate_approximate_gear(gear_ratio: float, chain_ring: int, cassette_sizes: list) -> int:
    if not cassette_sizes:
        raise ValueError("Cassette sizes are required to calculate approximate gear.")
    gear_ratios = [chain_ring / cassette for cassette in cassette_sizes]
    closest_gear_ratio = min(gear_ratios, key=lambda x: abs(x - gear_ratio))
    return cassette_sizes[gear_ratios.index(closest_gear_ratio)]


def estimate_cda(speed, power, mass, elevation=None, wind_speed=1.79, wind_direction=45,
                 rolling_resistance=0.00309, air_density=1.225):
    """
    Estimate CdA (aerodynamic drag coefficient) from speed and power data.

    Parameters:
        speed (array-like): Speeds in meters per second.
        power (array-like): Power in watts.
        mass (float): Cyclist + bike mass in kilograms.
        elevation (array-like, optional): Elevation data in meters.
        wind_speed (float, optional): Wind speed in m/s (default: 0).
        wind_direction (float, optional): Wind direction in degrees relative to cyclist (default: 0).
        rolling_resistance (float): Rolling resistance coefficient (default: 0.005).
        air_density (float): Air density in kg/m^3 (default: 1.225).

    Returns:
        tuple: Valid CdA values, corresponding speeds, and powers, along with a plot.
    """
    speed = np.array(speed)
    power = np.array(power)

    g = 9.81  # m/s^2

    relative_speed = speed + wind_speed * np.cos(np.radians(wind_direction))

    power_rolling_resistance = mass * g * rolling_resistance * speed

    power_gradient = 0
    if elevation is not None:
        gradient = np.gradient(elevation) / speed
        power_gradient = mass * g * np.sin(np.arctan(gradient)) * speed

    power_aero = power - power_rolling_resistance - power_gradient
    power_aero = np.maximum(power_aero, 0)

    speed_cubed = relative_speed ** 3
    cda_values = (2 * power_aero) / (air_density * speed_cubed)

    valid_indices = (speed > 3) & (power > 50) & (cda_values > 0) & (cda_values < 1)
    valid_speeds = speed[valid_indices]
    valid_power = power[valid_indices]
    valid_cda = cda_values[valid_indices]

    if len(valid_cda) == 0:
        raise ValueError("Insufficient valid data to estimate CdA.")

    weights = power_aero[valid_indices] / power_aero[valid_indices].sum()
    weighted_cda = np.average(valid_cda, weights=weights)

    def aerodynamic_model(speed, cda):
        return 0.5 * cda * air_density * speed ** 3

    params = curve_fit(aerodynamic_model, valid_speeds, power_aero[valid_indices])
    cda_estimated = params[0]

    plt.scatter(valid_speeds, valid_cda, label='Data Points')
    slope, intercept, r_value, p_value, std_err = linregress(valid_speeds, valid_cda)
    trend_line = slope * valid_speeds + intercept
    plt.plot(valid_speeds, trend_line, color='red', label=f'Trend Line (R²={r_value ** 2:.2f})')

    plt.plot(valid_speeds, aerodynamic_model(valid_speeds, cda_estimated), color='green', linestyle='--',
             label='Fitted Model')

    plt.xlabel('Speed (m/s)')
    plt.ylabel('CdA')
    plt.title('CdA vs Speed')
    plt.legend()
    plt.show()

    return valid_cda, valid_speeds, valid_power, weighted_cda, cda_estimated


class FitFileParser:

    def __init__(self, fitfile_path, casette='ultegra_12_speed_11_30_cassette'):
        self.fitfile_path = fitfile_path
        self.df = fitfile_to_dataframe(fitfile_path)
        self.casette_chosen = define_casette_sizes(casette)
        self.casette_as_string = [str(label) for label in self.casette_chosen]
        self.teeth_no_to_investigate = list(range(40, 67, 2))
        self.teeth_no_to_investigate_str = [str(label) for label in self.teeth_no_to_investigate]
        self.time_in_middle_2_gears = []
        self.time_in_middle_4_gears = []

        # Ensure best gear is calculated after all attributes are initialized
        self.calculate_best_gear()

    def calculate_best_gear(self):
        if self.df.empty or not self.casette_chosen:
            raise ValueError("DataFrame is empty or cassette is not defined.")

        middle_2_gears = [self.casette_chosen[5], self.casette_chosen[6]]
        middle_4_gears = [self.casette_chosen[4], self.casette_chosen[5], self.casette_chosen[6],
                          self.casette_chosen[7]]

        self.df['altitude_change'] = self.df['altitude'].diff()
        self.df = self.df.loc[(self.df['power'] > 100) & (self.df['cadence'] > 60) & (self.df['altitude_change'] > 0)]

        self.df['gear_ratio'] = self.df.apply(
            lambda row: estimate_gear_ratio(row['cadence'], row['speed']), axis=1
        )
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)

        for teeth in self.teeth_no_to_investigate:
            column_name = f'gear_with_{teeth}'
            self.df[column_name] = self.df['gear_ratio'].apply(
                lambda x: calculate_approximate_gear(x, teeth, self.casette_chosen)
            )

            counts = self.df[column_name].value_counts().reindex(self.casette_chosen, fill_value=0)
            time_in_middle_2_gear = round(counts[middle_2_gears].sum() / 60, 1)
            time_in_middle_4_gear = round(counts[middle_4_gears].sum() / 60, 1)
            self.time_in_middle_2_gears.append(time_in_middle_2_gear)
            self.time_in_middle_4_gears.append(time_in_middle_4_gear)

    def plot_teeth(self):
        plt.bar(self.teeth_no_to_investigate_str, self.time_in_middle_4_gears, label='Middle 4 gears', color='hotpink')
        plt.bar(self.teeth_no_to_investigate_str, self.time_in_middle_2_gears, label='Middle 2 gears', color='yellow')
        plt.xlabel("Chain ring teeth")
        plt.ylabel("Time in minutes")
        plt.title("Time in middle 2 and 4 gears")
        plt.legend()
        plt.show()

    def plot_power_distribution(self, max_threshold=300):
        if 'power' not in self.df.columns:
            raise ValueError("Power data is missing in the DataFrame.")

        df_trimmed = self.df.loc[self.df['power'] < max_threshold]
        plt.hist(df_trimmed['power'], bins=50, color='blue', alpha=0.7)
        plt.xlabel('Power [W]')
        plt.ylabel('Frequency')
        plt.title('Power Distribution')
        plt.show()

    def plot_cadence_distribution(self, min_threshold=50, max_threshold=120):
        if 'cadence' not in self.df.columns:
            raise ValueError("Cadence data is missing in the DataFrame.")

        df_trimmed = self.df.loc[(self.df['cadence'] > min_threshold) & (self.df['cadence'] < max_threshold)]
        cadence_data = df_trimmed['cadence']

        plt.hist(cadence_data, bins=35, color='blue', alpha=0.7, density=True, label='Cadence Histogram')
        mu, std = norm.fit(cadence_data)
        x = np.linspace(min_threshold, max_threshold, 100)
        plt.plot(x, norm.pdf(x, mu, std), 'r-', label=f'Normal Fit: $\mu={mu:.2f}, \sigma={std:.2f}$')
        plt.xlabel('Cadence [rpm]')
        plt.ylabel('Density')
        plt.title('Cadence Distribution with Normal Fit')
        plt.legend()
        plt.show()

    def plot_time_in_each_casette_tooth(self, teeth=str(56)):
        column_name = f'gear_with_{teeth}'
        if column_name not in self.df.columns:
            raise ValueError(f"Column {column_name} not found in DataFrame.")

        gear_counts = self.df[column_name].value_counts().reindex(self.casette_chosen, fill_value=0)

        # Plot the general gear counts
        plt.bar(self.casette_as_string, gear_counts, width=0.4, color='lightgray', label='Other Gears')

        # Highlight the middle 2 gears
        middle_2_gears = [self.casette_chosen[5], self.casette_chosen[6]]
        for gear in middle_2_gears:
            plt.bar(str(gear), gear_counts.get(gear, 0), width=0.4, color='yellow', label=f'Middle 2 Gear: {gear}')

        # Highlight the middle 4 gears
        middle_4_gears = [self.casette_chosen[4], self.casette_chosen[5], self.casette_chosen[6],
                          self.casette_chosen[7]]
        for gear in middle_4_gears:
            plt.bar(str(gear), gear_counts.get(gear, 0), width=0.4, color='hotpink', label=f'Middle 4 Gear: {gear}')

        plt.xlabel('Cassette Tooth')
        plt.ylabel('Time [secs]')
        plt.title('Time in Each Cassette Tooth')
        plt.legend()
        plt.show()


    def plot_speed_distribution(self):
        if 'speed' not in self.df.columns:
            raise ValueError("Speed data is missing in the DataFrame.")

        speed_in_kmph = self.df['speed'] * 3.6
        plt.hist(speed_in_kmph, bins=int(speed_in_kmph.max() - speed_in_kmph.min()), color='blue', alpha=0.7)
        plt.xlabel('Speed [km/h]')
        plt.ylabel('Frequency')
        plt.title('Speed Distribution')
        plt.show()


fitfile_path = "outlaw.FIT"
parser = FitFileParser(fitfile_path)
parser.calculate_best_gear()
parser.plot_time_in_each_casette_tooth(teeth=str(56))

# zwift_file = 'Zwift1.fit'
# zwift = FitFileParser(zwift_file)
# zwift_df = zwift.df
# zwift_df = zwift_df.rename(columns={"cadence": "cadence_zwift", "power": "power_zwift", "speed": "speed_zwift"})
# zwift_df.index = pd.to_datetime(zwift_df['timestamp'])
# zwift_df = zwift_df[['cadence_zwift', 'power_zwift', 'speed_zwift']]
# z
#
#
# garmin_file = 'Afternoon_Ride.fit'
# garmin = FitFileParser(garmin_file)
# garmin_df = garmin.df
# garmin_df = garmin_df.rename(columns={"cadence": "cadence_garmin", "power": "power_garmin", "speed": "speed_garmin"})
# garmin_df.index = pd.to_datetime(garmin_df['timestamp'])
# garmin_df = garmin_df[['cadence_garmin', 'power_garmin']]
#
# time_delta = zwift_df.index[0]-garmin_df.index[0]
# zwift_df.index = zwift_df.index - time_delta
#
# df = pd.merge(zwift_df, garmin_df, how='inner', left_index=True, right_index=True)
# plt.plot(zwift_df['power_zwift'], label='Turbo power')
# plt.plot(garmin_df['power_garmin'], label='Pedal Power')
# plt.legend()


# class GUI:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("FIT File Analyzer")
#
#         self.file_path = None
#         self.parser = None
#
#         self.setup_ui()
#
#     def setup_ui(self):
#         """Sets up the user interface."""
#         self.label = tk.Label(self.root, text="Drag and drop a FIT file or click 'Browse'")
#         self.label.pack(pady=10)
#
#         self.browse_button = tk.Button(self.root, text="Browse", command=self.browse_file)
#         self.browse_button.pack(pady=5)
#
#         self.method_frame = tk.Frame(self.root)
#         self.method_frame.pack(pady=10)
#
#         self.run_buttons = []
#
#     def browse_file(self):
#         """Opens a file dialog to select a FIT file."""
#         self.file_path = filedialog.askopenfilename(filetypes=[("FIT files", "*.FIT"), ("All files", "*.*")])
#         if self.file_path:
#             self.initialize_parser()
#
#     def initialize_parser(self):
#         """Initializes the FitFileParser and sets up method buttons."""
#         try:
#             self.parser = FitFileParser(self.file_path)
#             self.label.config(text=f"Loaded: {self.file_path}")
#
#             # Clear previous buttons
#             for button in self.run_buttons:
#                 button.destroy()
#             self.run_buttons.clear()
#
#             # Add buttons for plot methods
#             methods = inspect.getmembers(FitFileParser, predicate=inspect.isfunction)
#             for name, method in methods:
#                 if name.startswith("plot"):
#                     btn = tk.Button(self.method_frame, text=name, command=lambda n=name: self.run_method(n))
#                     btn.pack(pady=2)
#                     self.run_buttons.append(btn)
#
#         except Exception as e:
#             messagebox.showerror("Error", str(e))
#
#     def run_method(self, method_name):
#         """Runs the selected method from the parser."""
#         try:
#             # Explicitly call the plot method on the parser
#             method = getattr(self.parser, method_name)
#             method()
#             plt.show()  # Ensure the plot is displayed
#         except Exception as e:
#             messagebox.showerror("Error", str(e))
#
#
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = GUI(root)
#     root.mainloop()


# Example usage
# file_path = 'outlaw.FIT'
# parser = FitFileParser(file_path)
# parser.plot_cadence_distribution()


# class FitFileParser:
#     def __init__(self, fitfile_path, casette='ultegra_12_speed_11_30_cassette'):
#         self.fitfile_path = fitfile_path
#         self.df = fitfile_to_dataframe(fitfile_path)
#         self.casette_chosen = define_casette_sizes(casette)
#         self.casette_as_string = [str(label) for label in self.casette_chosen]
#         self.teeth_no_to_investigate = list(range(40, 67, 2))
#         self.teeth_no_to_investigate_str = [str(label) for label in self.teeth_no_to_investigate]
#         self.time_in_middle_2_gears = []
#         self.time_in_middle_4_gears = []
#         self.gear_distributions = None  # To store gear distributions for each timestep
#
#     def calculate_gear_distributions(self, chainring=56):
#         """
#         Calculate a distribution of possible cassette gears for each timestep given a chainring.
#         """
#         if self.df.empty or not self.casette_chosen:
#             raise ValueError("DataFrame is empty or cassette is not defined.")
#
#         def calculate_gear_ratio(cadence, speed, tyre_width=0.028):
#             tyre_diameter = 0.622 + 2 * tyre_width  # Diameter in meters
#             tyre_circumference = tyre_diameter * math.pi  # Circumference in meters
#
#             # Revolutions per second
#             tyre_revolutions_per_sec = speed / tyre_circumference
#             crank_revolutions_per_sec = cadence / 60  # Convert RPM to revolutions per second
#
#             # Calculate gear ratio
#             return crank_revolutions_per_sec / tyre_revolutions_per_sec if tyre_revolutions_per_sec > 0 else float('nan')
#
#         def map_gear_ratios_to_distribution(gear_ratio, chainring, cassette):
#             # Calculate theoretical gear ratios for the cassette
#             cassette_ratios = [chainring / c for c in cassette]
#
#             # Compute distances (errors) from the actual gear ratio
#             distances = np.array([abs(gear_ratio - r) for r in cassette_ratios])
#
#             # Avoid division by zero
#             distances[distances == 0] = 1e-6
#
#             # Assign weights inversely proportional to distances
#             weights = 1 / distances
#
#             # Normalize to sum to 1
#             probabilities = weights / np.sum(weights)
#
#             return probabilities
#
#         # Calculate gear ratio and distributions
#         self.df['gear_ratio'] = self.df.apply(lambda row: calculate_gear_ratio(row['cadence'], row['speed']), axis=1)
#
#         distributions = []
#         for _, row in self.df.iterrows():
#             if np.isnan(row['gear_ratio']):
#                 distributions.append([0] * len(self.casette_chosen))  # If invalid ratio, no probabilities
#             else:
#                 distribution = map_gear_ratios_to_distribution(row['gear_ratio'], chainring, self.casette_chosen)
#                 distributions.append(distribution)
#
#         self.df['gear_distribution'] = distributions
#         self.gear_distributions = distributions
#
#     def plot_gear_distribution(self, timestep):
#         """
#         Plot the gear distribution for a specific timestep.
#         """
#         if 'gear_distribution' not in self.df.columns:
#             raise ValueError("Gear distributions have not been calculated. Run 'calculate_gear_distributions' first.")
#
#         if timestep >= len(self.df) or timestep < 0:
#             raise ValueError("Invalid timestep index.")
#
#         gear_probs = self.df.iloc[timestep]['gear_distribution']
#         plt.bar(self.casette_as_string, gear_probs, color='blue', alpha=0.7)
#         plt.xlabel('Cassette Tooth')
#         plt.ylabel('Probability')
#         plt.title(f'Gear Distribution at Timestep {timestep}')
#         plt.show()
#
#     def plot_overall_gear_distribution(self):
#         """
#         Plot the overall gear distribution across the entire ride.
#         """
#         if 'gear_distribution' not in self.df.columns:
#             raise ValueError("Gear distributions have not been calculated. Run 'calculate_gear_distributions' first.")
#
#         # Aggregate probabilities across all timesteps
#         total_distribution = np.sum(np.array(self.gear_distributions), axis=0)
#
#         # Normalize to get probabilities
#         overall_distribution = total_distribution / np.sum(total_distribution)
#
#         # Plot the overall distribution
#         plt.bar(self.casette_as_string, overall_distribution, color='blue', alpha=0.7)
#         plt.xlabel('Cassette Tooth')
#         plt.ylabel('Probability')
#         plt.title('Overall Gear Distribution Across Ride')
#         plt.show()
#
#     def calculate_best_gear(self):
#         if self.df.empty or not self.casette_chosen:
#             raise ValueError("DataFrame is empty or cassette is not defined.")
#
#         middle_2_gears = [self.casette_chosen[5], self.casette_chosen[6]]
#         middle_4_gears = [self.casette_chosen[4], self.casette_chosen[5], self.casette_chosen[6], self.casette_chosen[7]]
#
#         self.df['altitude_change'] = self.df['altitude'].diff()
#         self.df = self.df.loc[(self.df['power'] > 100) & (self.df['cadence'] > 60) & (self.df['altitude_change'] > 0)]
#
#         self.df['gear_ratio'] = self.df.apply(
#             lambda row: calculate_gear_ratio(row['cadence'], row['speed']), axis=1
#         )
#         self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
#         self.df.dropna(inplace=True)
#
#         for teeth in self.teeth_no_to_investigate:
#             column_name = f'gear_with_{teeth}'
#             self.df[column_name] = self.df['gear_ratio'].apply(
#                 lambda x: calculate_approximate_gear(x, teeth, self.casette_chosen)
#             )
#
#             counts = self.df[column_name].value_counts().reindex(self.casette_chosen, fill_value=0)
#             time_in_middle_2_gear = round(counts[middle_2_gears].sum() / 60, 1)
#             time_in_middle_4_gear = round(counts[middle_4_gears].sum() / 60, 1)
#             self.time_in_middle_2_gears.append(time_in_middle_2_gear)
#             self.time_in_middle_4_gears.append(time_in_middle_4_gear)
#
#
#
fitfile_path = "outlaw.FIT"
parser = FitFileParser(fitfile_path)
# parser.calculate_gear_distributions(chainring=56)
# parser.plot_overall_gear_distribution()
