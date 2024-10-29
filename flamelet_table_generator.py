import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
import h5py
import json
import numpy as np
import pandas as pd
import cantera as ct
from scipy import interpolate
from scipy.special import erfcinv

pyplot_params = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.xmargin": 0,
    "axes.ymargin": 0,
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.titlesize": 18
}

@dataclass
class InletCondition:
    """Represents inlet conditions for fuel or oxidizer stream"""
    composition: Dict[str, float]  # Species mole fractions
    temperature: float             # Temperature in Kelvin
    mass_flux: float               # Mass flux in kg/m²/s

class FlameletTableGenerator:
    """Generator for flamelet calculations including unstable branch traversal.
    
    This class handles the generation of flamelet solutions for counterflow diffusion flames,
    including the computation of complete S-curves with stable and unstable branches.
    
    Attributes:
        mechanism_file (str): Path to the chemical mechanism file
        fuel_inlet (InletCondition): Fuel stream inlet conditions
        oxidizer_inlet (InletCondition): Oxidizer stream inlet conditions
        pressure (float): Operating pressure in Pa
        width (float): Domain width in meters
        initial_chi_st (float): Initial scalar dissipation rate at stoichiometric mixture fraction
        gas (ct.Solution): Cantera Solution object for the mechanism
        flame (ct.CounterflowDiffusionFlame): Cantera flame object
        solutions (List[Dict]): List of computed solutions and their metadata
        Z_st (float): Stoichiometric mixture fraction
    """

    def __init__(
        self,
        mechanism_file: str,
        fuel_inlet: InletCondition,
        oxidizer_inlet: InletCondition,
        pressure: float,
        width: Optional[float] = None,
        initial_chi_st: float = 0.1,
    ):
        """Initialize the flamelet generator with mechanism and conditions.
        
        Args:
            mechanism_file: Path to the chemical mechanism file
            fuel_inlet: Fuel stream inlet conditions
            oxidizer_inlet: Oxidizer stream inlet conditions
            pressure: Operating pressure in Pa
            width: Domain width in meters. If None, estimated from initial conditions
            initial_chi_st: Initial scalar dissipation rate at stoichiometric mixture fraction
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Store input parameters
        self.mechanism_file = mechanism_file
        self.fuel_inlet = fuel_inlet
        self.oxidizer_inlet = oxidizer_inlet
        self.pressure = pressure
        self.initial_chi_st = initial_chi_st
        
        self.gas = ct.Solution(self.mechanism_file)
        self.Z_st = self._compute_stoichiometric_mixture_fraction()
        
        # Estimate appropriate domain width if not provided
        if width is None:
            # Estimate flame thickness based on thermal diffusivity and initial strain rate
            strain = self._estimate_strain_from_chi_st(initial_chi_st)
            self.gas.TPX = (
                0.5 * (fuel_inlet.temperature + oxidizer_inlet.temperature),
                pressure,
                fuel_inlet.composition
            )
            thermal_diffusivity = self.gas.thermal_conductivity / (self.gas.density * self.gas.cp_mass)
            flame_thickness = np.sqrt(thermal_diffusivity / strain)
            # Make domain width a multiple of flame thickness
            self.width = 20 * flame_thickness
            self.logger.info("Setting domain width automatically to {0:.3e} m".format(self.width))
        else:
            self.width = width
        
        self.flame = None
        self.solutions = []
        self._initialize_flame()
    
    def _compute_stoichiometric_mixture_fraction(self) -> float:
        """Compute the stoichiometric mixture fraction using Bilger's definition.
        
        Returns:
            float: Stoichiometric mixture fraction
        """
        self.gas.set_equivalence_ratio(1.0, self.fuel_inlet.composition, self.oxidizer_inlet.composition)
        return self.gas.mixture_fraction(fuel=self.fuel_inlet.composition,
                                         oxidizer=self.oxidizer_inlet.composition,
                                         basis='mole',
                                         element='Bilger')

    def _compute_mixture_fraction(self) -> np.ndarray:
        """Compute mixture fraction field for the current flame state using Bilger's definition.
        
        Returns:
            np.ndarray: Mixture fraction values at each grid point
        """
        Z = np.zeros_like(self.flame.grid)
        for i in range(len(self.flame.grid)):
            self.gas.TPY = self.flame.T[i], self.pressure, self.flame.Y[:, i]
            Z[i] = self.gas.mixture_fraction(fuel=self.fuel_inlet.composition,
                                             oxidizer=self.oxidizer_inlet.composition,
                                             basis='mole',
                                             element='Bilger')
        return Z
    
    def _compute_scalar_dissipation(
        self,
        mixture_fraction: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """Compute scalar dissipation rate field and its stoichiometric value.
        
        Args:
            mixture_fraction: Optional pre-computed mixture fraction field.
                If None, will be computed.
        
        Returns:
            Tuple containing:
                - np.ndarray: Scalar dissipation rate at each grid point
                - float: Scalar dissipation rate at stoichiometric mixture fraction
        """
        if mixture_fraction is None:
            mixture_fraction = self._compute_mixture_fraction()
        
        # Compute mixture fraction gradient
        dZ_dx = np.gradient(mixture_fraction, self.flame.grid)
        
        # Compute diffusivity (assuming Lewis number = 1)
        rho = self.flame.density
        D = self.flame.thermal_conductivity / (rho * self.flame.cp_mass)
        
        # Compute scalar dissipation rate
        chi = 2 * D * dZ_dx**2
        
        # Find scalar dissipation at stoichiometric mixture fraction
        interp = interpolate.interp1d(mixture_fraction, chi)
        chi_st = float(interp(self.Z_st))

        return chi, chi_st
    
    def _estimate_chi_st_from_strain(self, strain_rate: float) -> float:
        """Estimate the stoichiometric scalar dissipation rate from a target strain rate.

        Args:
            strain_rate: Target strain rate
        
        Returns:
            float: Estimated stoichiometric scalar dissipation rate

        Note:
            Uses Peters' relationship (Peters, PECS 1984)
        """
        chi_st = (strain_rate / np.pi) * np.exp(-2 * erfcinv(2 * self.Z_st)**2)
        return chi_st

    def _estimate_strain_from_chi_st(self, chi_st: float) -> float:
        """Estimate the strain rate from a given stoichiometric scalar dissipation rate.

        Args:
            chi_st: Scalar dissipation rate at stoichiometric mixture fraction [1/s]
        
        Returns:
            float: Estimated strain rate [1/s]
        """
        strain_rate = chi_st * np.pi / np.exp(-2 * erfcinv(2 * self.Z_st)**2)
        return strain_rate

    def _mdots_from_chi_st(self, chi_st: float) -> Tuple[float, float]:
        """Compute mass fluxes for fuel and oxidizer based on target chi_st.
        
        Args:
            chi_st: Target scalar dissipation rate at stoichiometric mixture fraction [1/s]
        
        Returns:
            Tuple containing:
                - mdot_fuel: Mass flux of fuel [kg/m²/s]
                - mdot_oxidizer: Mass flux of oxidizer [kg/m²/s]
        """
        # Set gas state for fuel
        self.gas.TPX = (self.fuel_inlet.temperature,
                        self.pressure,
                        self.fuel_inlet.composition)
        rho_fuel = self.gas.density
        
        # Set gas state for oxidizer
        self.gas.TPX = (self.oxidizer_inlet.temperature,
                        self.pressure,
                        self.oxidizer_inlet.composition)
        rho_ox = self.gas.density
        
        # Estimate strain rate needed for target chi_st
        target_strain = self._estimate_strain_from_chi_st(chi_st)
        
        # Set velocities to achieve target strain while maintaining momentum balance
        v_fuel = np.sqrt(target_strain * self.width / 4)
        v_ox = v_fuel * np.sqrt(rho_fuel / rho_ox)
        
        # Convert to mass fluxes
        mdot_fuel = rho_fuel * v_fuel
        mdot_oxidizer = rho_ox * v_ox
        
        return mdot_fuel, mdot_oxidizer

    def _initialize_flame(self):
        """Set up the initial counterflow diffusion flame configuration.
        
        Initializes the Cantera flame object with appropriate grid, inlet conditions,
        and refinement criteria. Estimates appropriate strain rate based on target
        scalar dissipation rate.
        """
        self.flame = ct.CounterflowDiffusionFlame(self.gas, width=self.width)
        
        # Set operating conditions
        self.flame.P = self.pressure
        
        # Use the new method to compute mass fluxes
        mdot_fuel, mdot_ox = self._mdots_from_chi_st(self.initial_chi_st)
        
        # Set inlet conditions
        self.flame.fuel_inlet.mdot = mdot_fuel
        self.flame.fuel_inlet.X = self.fuel_inlet.composition
        self.flame.fuel_inlet.T = self.fuel_inlet.temperature
        
        self.flame.oxidizer_inlet.mdot = mdot_ox
        self.flame.oxidizer_inlet.X = self.oxidizer_inlet.composition
        self.flame.oxidizer_inlet.T = self.oxidizer_inlet.temperature
        
        # Set refinement parameters
        self.flame.set_refine_criteria(ratio=4.0, slope=0.1, curve=0.2, prune=0.05)
    
    def _update_flame_width(self):
        """Update the flame width and reinitialize the flame object."""
        # Compute the current flame thickness
        strain = self._estimate_strain_from_chi_st(self.initial_chi_st)
        thermal_diffusivity = self.gas.thermal_conductivity / (self.gas.density * self.gas.cp_mass)
        flame_thickness = np.sqrt(thermal_diffusivity / strain)
        
        # Set new width to be approximately 10 times the flame thickness
        new_width = 10 * flame_thickness
        self.logger.info(f"Updating domain width from {self.width:.3e} m to {new_width:.3e} m")
        
        # Reinitialize the flame object with the new width
        old_width = self.width
        self.width = new_width
        self.flame = ct.CounterflowDiffusionFlame(self.gas, width=self.width)
        
        # Interpolate old solution onto the new domain
        if self.solutions:
            old_solution = self.solutions[-1]['state']
            old_grid = self.flame.grid
            
            # Calculate the new grid based on the new width
            new_grid = np.linspace(-self.width / 2, self.width / 2, len(old_grid))
            
            # Find the center of the flame in the old grid
            flame_center_index = np.argmax(old_solution.T)  # Assuming temperature profile indicates flame center
            flame_center_position = old_grid[flame_center_index]
            
            # Shift the old grid to align the flame center with the new grid
            shift_amount = flame_center_position - (self.width / 2)
            shifted_old_grid = old_grid - shift_amount
            
            # Interpolate the old state onto the new grid
            new_state = interpolate.interp1d(shifted_old_grid, old_solution, axis=0, fill_value="extrapolate")(new_grid)
            self.flame.from_array(new_state)
    
    def compute_s_curve(
        self,
        output_path: Optional[Path] = None,
        n_extinction_points: int = 10,
        **kwargs
    ) -> List[Dict]:
        """Compute complete S-curve including ignited branches and extinction branch.
        
        Computes the full S-curve by first traversing the upper (stable) and middle
        (unstable) branches, then computing the lower (extinction) branch.
        
        Args:
            output_path: Directory to save solution files
            n_extinction_points: Number of points to compute along extinction branch
            **kwargs: Additional arguments passed to compute_ignited_branches
        
        Returns:
            List[Dict]: List of solution metadata dictionaries containing properties
                of each computed solution point
        """
        # First compute the ignited and unstable branches
        self.logger.info("Computing ignited and unstable branches")
        ignited_data = self.compute_ignited_branches(output_path=output_path, **kwargs)
        
        # Get chi_st bounds for extinction branch
        chi_st_values = [sol['chi_st'] for sol in ignited_data]
        chi_st_max = max(chi_st_values)
        chi_st_min = ignited_data[-1]['chi_st']  # Last point on unstable branch
        
        # Compute extinction branch
        self.logger.info("Computing extinction branch")
        extinct_data = self.compute_extinct_branch(
            chi_st_min=chi_st_min,
            chi_st_max=chi_st_max,
            n_points=n_extinction_points,
            output_path=output_path,
            loglevel=kwargs.get('loglevel')
        )
        
        return ignited_data + extinct_data

    def compute_ignited_branches(
        self,
        output_path: Optional[Path] = None,
        restart_from: Optional[int] = None,
        n_max: int = 5000,
        initial_spacing: float = 0.6,
        unstable_spacing: float = 0.95,
        temperature_increment: float = 20.0,
        max_increment: float = 100.0,
        target_delta_T_max: float = 20.0,
        max_error_count: int = 3,
        strain_rate_tol: float = 0.10,
        loglevel: int = 0,
    ) -> List[Dict]:
        """Compute upper (stable) and middle (unstable) branches of the S-curve.
        
        This method traverses both the upper (stable) and middle (unstable) branches of the
        S-curve using temperature as a control parameter. It employs a two-point continuation
        method to track solutions through the turning point and down the unstable branch.
        
        Args:
            output_path: Directory to save solution files
            restart_from: Optional solution index to restart from if continuing a previous calculation
            n_max: Maximum number of iterations before stopping
            initial_spacing: Initial control point spacing for stable branch (0-1)
            unstable_spacing: Control point spacing for unstable branch (0-1)
            temperature_increment: Initial temperature increment between solutions [K]
            max_increment: Maximum allowed temperature increment [K]
            target_delta_T_max: Target maximum temperature change per step [K]
            max_error_count: Maximum number of successive solver errors before stopping
            strain_rate_tol: Tolerance for minimum strain rate relative to maximum
            loglevel: Cantera solver log level (0-3)
        
        Returns:
            List[Dict]: List of solution metadata dictionaries containing properties
                of each computed solution point. Each dictionary includes:
                - T_max: Maximum temperature [K]
                - T_st: Temperature at stoichiometric mixture fraction [K]
                - strain_rate: Maximum strain rate [1/s]
                - chi_st: Scalar dissipation rate at stoichiometric mixture fraction [1/s]
                - total_heat_release_rate: Integrated heat release rate [W/m³]
                - n_points: Number of grid points
                - flame_width: Width of the flame [m]
                - Tc_increment: Temperature increment used for this solution [K]
                - time_steps: Number of time steps taken by solver
                - eval_count: Number of right-hand side evaluations
                - cpu_time: Total CPU time for solution [s]
                - errors: Number of solver errors encountered
        
        Note:
            The method uses a two-point continuation strategy where control points are placed
            at specified fractions between the minimum and maximum temperatures. The temperature
            increment is adaptively adjusted based on solution behavior and convergence.
            Solutions are saved to HDF5 files if output_path is provided.
        """
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Handle restart case
        if restart_from is not None:
            breakpoint()
            if restart_from >= len(self.solutions):
                raise ValueError(f"Restart index {restart_from} exceeds number of available solutions")
            
            # Restore flame state and get previous temperature increment
            restart_solution = self.solutions[restart_from]
            self.flame.from_array(restart_solution['state'])
            temperature_increment = restart_solution['metadata']['Tc_increment']
            
            # Get maximum strain rate from previous solutions
            a_max = max(sol['metadata']['strain_rate'] for sol in self.solutions[:restart_from+1])
            strain_rate = restart_solution['metadata']['strain_rate']
            
            # Initialize data with previous solutions
            data = [sol['metadata'] for sol in self.solutions[:restart_from+1]]
            
            self.logger.info(f'Restarting from solution {restart_from} with T_increment = {temperature_increment:.2f}')
            start_iteration = restart_from + 1
            
        else:
            # Original initialization code
            self.logger.info('Computing the initial solution')
            self.flame.solve(loglevel=loglevel, auto=True)
            
            strain_rate = self.flame.strain_rate('max')
            a_max = strain_rate
            data = []
            start_iteration = 0

        # Enable and configure two-point control
        self.flame.two_point_control_enabled = True
        self.flame.flame.set_bounds(spread_rate=(-1e-5, 1e20))
        self.flame.max_time_step_count = 100
        
        # Initialize error tracking
        error_count = 0
        T_max = max(self.flame.T)

        # Rest of the method remains the same, but change the iteration range
        for i in range(start_iteration, n_max):
            # Update flame width before each solution attempt
            # self._update_flame_width()
            
            spacing = unstable_spacing if strain_rate <= 0.98 * a_max else initial_spacing
            control_temperature = (np.min(self.flame.T) + 
                                spacing * (np.max(self.flame.T) - np.min(self.flame.T)))
            
            # Store current state
            backup_state = self.flame.to_array()
            
            self.logger.debug(f'Iteration {i}: Control temperature = {control_temperature:.2f} K')
            self.flame.set_left_control_point(control_temperature)
            self.flame.set_right_control_point(control_temperature)
            
            # Update control points
            self.flame.left_control_point_temperature -= temperature_increment
            self.flame.right_control_point_temperature -= temperature_increment
            self.flame.clear_stats()
            
            # Check if we've reached inlet temperatures
            if (self.flame.left_control_point_temperature < self.flame.fuel_inlet.T + 100 or
                self.flame.right_control_point_temperature < self.flame.oxidizer_inlet.T + 100):
                self.logger.info("SUCCESS! Control point temperature near inlet temperature.")
                break
            
            try:
                self.flame.solve(loglevel=loglevel)
                
                # Adjust temperature increment based on convergence
                if abs(max(self.flame.T) - T_max) < 0.8 * target_delta_T_max:
                    temperature_increment = min(temperature_increment + 3, max_increment)
                elif abs(max(self.flame.T) - T_max) > target_delta_T_max:
                    temperature_increment *= (0.9 * target_delta_T_max / 
                                           abs(max(self.flame.T) - T_max))
                error_count = 0
                
            except ct.CanteraError as err:
                self.logger.debug(err)
                if strain_rate / a_max < strain_rate_tol:
                    self.logger.info(
                        'SUCCESS! Traversed unstable branch down to '
                        f'{100 * strain_rate / a_max:.2f}% of the maximum strain rate.'
                    )
                    break
                
                # Restore previous solution and reduce increment
                self.flame.from_array(backup_state)
                temperature_increment = 0.7 * temperature_increment
                error_count += 1
                if error_count > max_error_count:
                    self.logger.warning(
                        f'FAILURE! Stopping after {error_count} successive solver errors.'
                    )
                    break
                self.logger.warning(
                    f"Solver did not converge on iteration {i}. "
                    f"Trying again with dT = {temperature_increment:.2f}"
                )
            
            # Compute mixture fraction and scalar dissipation
            Z = self._compute_mixture_fraction()
            chi, chi_st = self._compute_scalar_dissipation(Z)

            # Get temperature at stoichiometric mixture fraction
            interp_T = interpolate.interp1d(Z, self.flame.T)
            T_st = float(interp_T(self.Z_st))
            
            # Collect solution data
            T_max = max(self.flame.T)
            T_mid = 0.5 * (min(self.flame.T) + max(self.flame.T))
            s = np.where(self.flame.T > T_mid)[0][[0, -1]]
            width = self.flame.grid[s[1]] - self.flame.grid[s[0]]
            strain_rate = self.flame.strain_rate('max')
            a_max = max(strain_rate, a_max)
            
            step_data = {
                'T_max': T_max,
                'T_st': T_st,
                'strain_rate': strain_rate,
                'chi_st': chi_st,
                'total_heat_release_rate': np.trapz(self.flame.heat_release_rate, self.flame.grid),
                'n_points': len(self.flame.grid),
                'flame_width': width,
                'Tc_increment': temperature_increment,
                'time_steps': sum(self.flame.time_step_stats),
                'eval_count': sum(self.flame.eval_count_stats),
                'cpu_time': sum(self.flame.jacobian_time_stats + self.flame.eval_time_stats),
                'errors': error_count
            }
            data.append(step_data)
            self.solutions.append({
                'state': self.flame.to_array(),
                'Z': Z,
                'chi': chi,
                'metadata': step_data
            })
            
            if output_path:
                self.save_solution(output_path, len(self.solutions) - 1)
            
            # Logging after successful solution
            self.logger.info(
                f"Iteration {i} completed: T_max = {T_max:.2f}, T_st = {T_st:.2f}, "
                f"chi_st = {chi_st:.4e}, strain_rate = {strain_rate:.2f}"
            )

        self.logger.info(f'Stopped after {i} iterations')
        return data

    def compute_extinct_branch(
        self,
        chi_st_min: float,
        chi_st_max: float,
        n_points: int = 10,
        output_path: Optional[Path] = None,
        loglevel: int = 0
    ) -> List[Dict]:
        """Compute the extinction (lower) branch of the S-curve.
        
        This method calculates solutions along the lower (extinction) branch of the S-curve
        by starting from a cold mixing solution and gradually increasing the scalar
        dissipation rate from chi_st_min to chi_st_max. The solutions are computed at
        geometrically spaced intervals of scalar dissipation rate.
        
        Args:
            chi_st_min: Minimum scalar dissipation rate at stoichiometric mixture fraction [1/s]
            chi_st_max: Maximum scalar dissipation rate at stoichiometric mixture fraction [1/s]
            n_points: Number of points to compute along the extinction branch
            output_path: Optional directory path to save solution files
            loglevel: Cantera solver log level (0-3)
        
        Returns:
            List[Dict]: List of solution metadata dictionaries containing properties
                of each computed solution point. Each dictionary includes:
                - T_max: Maximum temperature [K]
                - T_st: Temperature at stoichiometric mixture fraction [K]
                - strain_rate: Maximum strain rate [1/s]
                - chi_st: Scalar dissipation rate at stoichiometric mixture fraction [1/s]
                - total_heat_release_rate: Integrated heat release rate [W/m³]
                - n_points: Number of grid points
                - flame_width: Width of the flame [m]
                - branch: 'extinction' to identify the branch
        
        Note:
            The extinction branch is computed by starting from a cold mixing solution
            and using strain rate as a control parameter to achieve target scalar
            dissipation rates. Solutions are saved to HDF5 files if output_path
            is provided.
        """
        chi_st_values = np.geomspace(chi_st_min, chi_st_max, n_points)
        data = []

        self.gas.TPX = (self.fuel_inlet.temperature,
                        self.pressure,
                        self.fuel_inlet.composition)
        Y_fuel = self.gas.Y

        self.gas.TPX = (self.oxidizer_inlet.temperature,
                        self.pressure,
                        self.oxidizer_inlet.composition)
        Y_ox = self.gas.Y

        # Initialize with cold mixing profile
        T_profile = np.ones_like(self.flame.grid) * self.fuel_inlet.temperature
        Y_profile = np.zeros((self.gas.n_species, len(self.flame.grid)))
        normalized_grid = (self.flame.grid - self.flame.grid[0]) / (self.flame.grid[-1] - self.flame.grid[0])
        for i in range(len(self.flame.grid)):
            x = normalized_grid[i]
            T_profile[i] = self.fuel_inlet.temperature + \
                x * (self.oxidizer_inlet.temperature - self.fuel_inlet.temperature)
            for k, species in enumerate(self.gas.species_names):
                Y_profile[k,i] = Y_fuel[k] + x * (Y_ox[k] - Y_fuel[k])
        
        self.flame.set_profile('T', normalized_grid, T_profile)
        for k, species in enumerate(self.gas.species_names):
            self.flame.set_profile(species, normalized_grid, Y_profile[k,:])
        
        for chi_st in chi_st_values:
            self.logger.info(f"Computing extinction branch solution at chi_st = {chi_st:.2e}")
            
            mdot_fuel, mdot_oxidizer = self._mdots_from_chi_st(chi_st)
            self.flame.fuel_inlet.mdot = mdot_fuel
            self.flame.oxidizer_inlet.mdot = mdot_oxidizer
            
            try:
                self.flame.solve(loglevel=loglevel, auto=True)
                
                # Compute solution properties
                Z = self._compute_mixture_fraction()
                chi, chi_st_actual = self._compute_scalar_dissipation(Z)
                
                # Get temperature at stoichiometric mixture fraction
                interp_T = interpolate.interp1d(Z, self.flame.T)
                T_st = float(interp_T(self.Z_st))
                
                # Collect solution data
                step_data = {
                    'T_max': max(self.flame.T),
                    'T_st': T_st,
                    'strain_rate': self.flame.strain_rate('max'),
                    'chi_st': chi_st_actual,
                    'total_heat_release_rate': np.trapz(self.flame.heat_release_rate, self.flame.grid),
                    'n_points': len(self.flame.grid),
                    'flame_width': self.width,
                    'branch': 'extinction'
                }
                
                data.append(step_data)
                self.solutions.append({
                    'state': self.flame.to_array(),
                    'Z': Z,
                    'chi': chi,
                    'metadata': step_data
                })
                
                if output_path:
                    self.save_solution(output_path, len(self.solutions) - 1)
                    
            except ct.CanteraError as err:
                self.logger.warning(f"Failed to compute solution at chi_st = {chi_st:.2e}: {err}")
                continue
        
        return data

    def save_solution(self, output_path: Path, solution_index: int):
        """Save a single solution to the HDF5 files.
        
        Saves both the flame profiles and associated metadata for a single solution.
        
        Args:
            output_path: Directory path where files will be saved
            solution_index: Index of the solution being saved
        """
        solutions_file = output_path / 'solutions.h5'
        solution = self.solutions[solution_index]
        meta_name = f'meta_{solution_index:04d}'
        state_name = f'solution_state_{solution_index:04d}'

        # If this is the first solution, create the file, overwriting if necessary
        if solution_index == 0:
            # Write condition parameters
            with h5py.File(solutions_file, 'w') as f:
                params = {
                    'mechanism_file': self.mechanism_file,
                    'pressure': self.pressure,
                    'width': self.width,
                    'Z_st': float(self.Z_st),
                    'fuel_inlet': {
                        'composition': self.fuel_inlet.composition,
                        'temperature': self.fuel_inlet.temperature,
                        'mass_flux': self.fuel_inlet.mass_flux
                    },
                    'oxidizer_inlet': {
                        'composition': self.oxidizer_inlet.composition,
                        'temperature': self.oxidizer_inlet.temperature,
                        'mass_flux': self.oxidizer_inlet.mass_flux
                    }
                }
                f.attrs['parameters'] = json.dumps(params)
                f.create_group('solutions_meta')

        # Write the solution metadata
        with h5py.File(solutions_file, 'a') as f:
            # Delete existing solution group if it exists
            meta_group = f['solutions_meta']
            if meta_name in meta_group:
                del meta_group[meta_name]
            
            # Create new meta group
            sol_group = meta_group.create_group(meta_name)
            
            # Save mixture fraction and scalar dissipation
            sol_group.create_dataset('Z', data=solution['Z'])
            sol_group.create_dataset('chi', data=solution['chi'])
            
            # Convert numpy values in metadata to native Python types
            metadata = {}
            for key, value in solution['metadata'].items():
                if isinstance(value, np.number):
                    metadata[key] = value.item()
                else:
                    metadata[key] = value
            
            # Save solution metadata
            sol_group.attrs['metadata'] = json.dumps(metadata)

        # Write flame profile
        solution['state'].save(
            solutions_file,
            name=state_name,
            overwrite=True
        )

    def save_all_solutions(self, output_path: Path):
        """Save all computed solutions to HDF5 files.
        
        Args:
            output_path: Directory path where files will be saved
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i in range(len(self.solutions)):
            self.save_solution(output_path, i)

    @classmethod
    def load_solutions(
        cls,
        filename: str,
    ) -> 'FlameletTableGenerator':
        """Load a complete solution set from the HDF5 file.
        
        Args:
            filename: Path to the solutions HDF5 file
        
        Returns:
            FlameletTableGenerator: New instance with loaded solutions
        
        Raises:
            ValueError: If files cannot be read or are invalid
        """
        # Load the metadata
        with h5py.File(filename, 'r') as f:
            # Load parameters
            params = json.loads(f.attrs['parameters'])
            
            # Create inlet conditions
            fuel_inlet = InletCondition(**params['fuel_inlet'])
            oxidizer_inlet = InletCondition(**params['oxidizer_inlet'])
            
            # Create generator instance
            generator = cls(
                mechanism_file=params['mechanism_file'],
                fuel_inlet=fuel_inlet,
                oxidizer_inlet=oxidizer_inlet,
                pressure=params['pressure'],
                width=params['width']
            )
            
            # Load solutions
            generator.solutions = []
            meta_group = f['solutions_meta']
            for meta_name in sorted(meta_group.keys()):
                sol_group = meta_group[meta_name]
                state_array = ct.SolutionArray(generator.gas)
                solution = {
                    'state': state_array,
                    'Z': sol_group['Z'][:],
                    'chi': sol_group['chi'][:],
                    'metadata': json.loads(sol_group.attrs['metadata'])
                }
                generator.solutions.append(solution)
        
        # Load the states
        for solution_index, solution in enumerate(generator.solutions):
            state_name = f'solution_state_{solution_index:04d}'
            generator.solutions[solution_index]['state'].restore(filename, state_name)

        return generator

    def plot_s_curve(
        self,
        x_quantity: Literal['strain_rate', 'chi_st'] = 'chi_st',
        y_quantity: Literal['T_max', 'T_st'] = 'T_max',
        output_file: Optional[str] = None
    ):
        """Plot the S-curve with configurable axes.
        
        Args:
            x_quantity: Which quantity to plot on x-axis ('strain_rate' or 'chi_st')
            y_quantity: Which quantity to plot on y-axis ('T_max' or 'T_st')
            output_file: Optional file path to save the figure
        
        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes objects
        """
        import matplotlib.pyplot as plt
        plt.rcParams.update(pyplot_params)
        
        x_values = [d['metadata'][x_quantity] for d in self.solutions]
        y_values = [d['metadata'][y_quantity] for d in self.solutions]
        
        # Set up axis labels
        x_labels = {
            'strain_rate': r"$\alpha$ [1/s]",
            'chi_st': r"$\chi_{st}$ [1/s]"
        }
        
        y_labels = {
            'T_max': r"$T_\textrm{max}$ [K]",
            'T_st': r"$T_{st}$ [K]"
        }
        
        fig, ax = plt.subplots()
        ax.semilogx(x_values, y_values, 'o-')
        ax.set_xlabel(x_labels[x_quantity])
        ax.set_ylabel(y_labels[y_quantity])
        ax.set_xmargin(0.1)
        ax.set_ymargin(0.1)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        if output_file:
            fig.savefig(output_file, bbox_inches='tight', dpi=300)
        
        return fig, ax
    
    def plot_temperature_profiles(
        self,
        output_file: Optional[str] = None,
        num_profiles: Optional[int] = None,
        colormap: str = 'viridis'
    ):
        """Plot temperature vs mixture fraction profiles for all solutions.
        
        Creates a plot showing temperature profiles colored by scalar dissipation rate.
        
        Args:
            output_file: Optional file path to save the figure
            num_profiles: Optional number of profiles to plot (will sample evenly)
            colormap: Name of colormap to use for the profiles
        
        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes objects
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        plt.rcParams.update(pyplot_params)
        
        # Select which solutions to plot
        if num_profiles is None:
            solutions_to_plot = self.solutions
        else:
            indices = np.linspace(0, len(self.solutions)-1, num_profiles, dtype=int)
            solutions_to_plot = [self.solutions[i] for i in indices]
        
        fig, ax = plt.subplots()
        
        # Create log-scaled colormap based on chi_st values
        chi_st_values = [sol['metadata']['chi_st'] for sol in solutions_to_plot]
        norm = LogNorm(vmin=min(chi_st_values), vmax=max(chi_st_values))
        cmap = plt.get_cmap(colormap)
        
        # Plot each profile
        for solution in solutions_to_plot:
            color = cmap(norm(solution['metadata']['chi_st']))
            ax.plot(solution['Z'], solution['state'].T, color=color, alpha=0.7)
        
        # Add colorbar with log scale
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$\chi_{st}$ [1/s]")
        
        # Add stoichiometric mixture fraction line
        ax.axvline(self.Z_st, color='k', linestyle='--', alpha=0.3)
        
        ax.set_xlabel(r"$Z$ [-]")
        ax.set_ylabel(r"$T$ [K]")
        ax.grid(True, alpha=0.2)
        
        if output_file:
            fig.savefig(output_file, bbox_inches='tight', dpi=300)
        
        return fig, ax

