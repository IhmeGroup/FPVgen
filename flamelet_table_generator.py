import logging
import datetime
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

class FlameletTableGenerator:
    """Generator for flamelet calculations including unstable branch traversal.
    
    This class handles the generation of flamelet solutions for counterflow diffusion flames,
    including the computation of complete S-curves with stable and unstable branches.
    
    Attributes:
        mechanism_file (str): Path to the chemical mechanism file
        transport_model (str): Transport model to use in Cantera
        fuel_inlet (InletCondition): Fuel stream inlet conditions
        oxidizer_inlet (InletCondition): Oxidizer stream inlet conditions
        pressure (float): Operating pressure in Pa
        prog_def (Dict): Progress variable definition
        width_ratio (float): Ratio of the domain width to the flame thickness
        width_change_enable (bool): Enable domain width changes
        width_change_max (float): Maximum domain width change
        width_change_min (float): Minimum domain width change
        initial_chi_st (float): Initial scalar dissipation rate at stoichiometric mixture fraction
        gas (ct.Solution): Cantera Solution object for the mechanism
        flame (ct.CounterflowDiffusionFlame): Cantera flame object
        solutions (List[Dict]): List of computed solutions and their metadata
        Z_st (float): Stoichiometric mixture fraction
        solver_loglevel (int): Cantera solver log level (0-3)
    """

    def __init__(
        self,
        mechanism_file: str,
        transport_model: str,
        fuel_inlet: InletCondition,
        oxidizer_inlet: InletCondition,
        pressure: float,
        prog_def: Optional[Dict] = None,
        width_ratio: Optional[float] = 10.0,
        width_change_enable: Optional[bool] = False,
        width_change_max: Optional[float] = 0.2,
        width_change_min: Optional[float] = 0.05,
        initial_chi_st: Optional[float] = 1.0e-2,
        solver_loglevel: Optional[int] = 0,
        strain_chi_st_model_param_file: Optional[str] = None
    ):
        """Initialize the flamelet generator with mechanism and conditions.
        
        Args:
            mechanism_file: Path to the chemical mechanism file
            transport_model: Transport model to use in Cantera
            fuel_inlet: Fuel stream inlet conditions
            oxidizer_inlet: Oxidizer stream inlet conditions
            pressure: Operating pressure in Pa
            prog_def: Progress variable definition
            width_ratio: Ratio of the domain width to the flame thickness
            width_change_enable: Enable domain width changes
            width_change_max: Maximum domain width change
            width_change_min: Minimum domain width change
            initial_chi_st: Initial scalar dissipation rate at stoichiometric mixture fraction
            solver_loglevel: Cantera solver log level (0-3)
            strain_chi_st_model_param_file: Path to JSON file with strain vs chi_st model parameters
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Store input parameters
        self.mechanism_file = mechanism_file
        self.transport_model = transport_model
        self.fuel_inlet = fuel_inlet
        self.oxidizer_inlet = oxidizer_inlet
        self.pressure = pressure
        self.width_ratio = width_ratio
        self.width_change_enable = width_change_enable
        self.width_change_max = width_change_max
        self.width_change_min = width_change_min
        self.initial_chi_st = initial_chi_st
        self.solver_loglevel = solver_loglevel

        if prog_def is None:
            self.prog_def = {
                "CO" : 1.0,
                "H2" : 1.0,
                "CO2" : 1.0,
                "H2O" : 1.0
            }
        else:
            self.prog_def = prog_def

        if strain_chi_st_model_param_file is not None:
            with open(strain_chi_st_model_param_file, 'r') as f:
                self.strain_chi_st_model_params = json.load(f)
        else:
            self.strain_chi_st_model_params = None
        
        # Initialization
        self.gas = ct.Solution(self.mechanism_file)
        self.Z_st = self._compute_stoichiometric_mixture_fraction()
        self.width = 1.0 # Needed for initial flame construction but will be overridden
        self.flame = None
        self.solutions = []
        self._update_flame_width()
    
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
    
    def _compute_progress_variable(self) -> np.ndarray:
        """Compute the progress variable field for the current flame solution.
        
        Returns:
            np.ndarray: Progress variable field
        """
        prog_var = np.zeros_like(self.flame.grid)
        for species, value in self.prog_def.items():
            idx = self.gas.species_index(species)
            prog_var += value * self.flame.Y[idx, :]
        return prog_var
    
    def _compute_progress_variable_production(self) -> np.ndarray:
        """Compute the progress variable production rate field for the current flame solution.
        
        Returns:
            np.ndarray: Progress variable production rate field
        """
        prog_var_prod = np.zeros_like(self.flame.grid)
        for species, value in self.prog_def.items():
            idx = self.gas.species_index(species)
            prog_var_prod += (value *
                              self.flame.net_production_rates[idx, :] *
                              self.gas.molecular_weights[idx])
        return prog_var_prod

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
            mixture_fraction = self.flame.mixture_fraction("Bilger")
        
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
    
    def _compute_flame_loc_Z(self) -> float:
        """Compute the flame location in mixture fraction space.
        
        Returns:
            float: Mixture fraction at the flame location
        """
        Z = self.flame.mixture_fraction("Bilger")
        T = self.flame.T
        return Z[np.argmax(T)]
    
    def _compute_dLnCpDZ(self) -> np.ndarray:
        """Compute the derivative of the natural logarithm of the heat capacity with respect to Z.
        
        Returns:
            np.ndarray: Derivative of ln(Cp) with respect to Z
        """
        Z = self.flame.mixture_fraction("Bilger")[::-1]
        cp = self.flame.cp_mass[::-1]
        return np.gradient(np.log(cp), Z)
    
    def _estimate_chi_st_from_strain(self, strain_rate: float) -> float:
        """Estimate the stoichiometric scalar dissipation rate from a target strain rate.

        Args:
            strain_rate: Target strain rate [1/s]
        
        Returns:
            float: Estimated stoichiometric scalar dissipation rate [1/s]

        Note:
            Uses regression model if available, otherwise uses Peters' relationship (Peters, PECS 1984)
        """
        if self.strain_chi_st_model_params is not None:
            m = self.strain_chi_st_model_params['slope']
            b = self.strain_chi_st_model_params['intercept']
            log_chi_st = m * np.log10(strain_rate) + b
            chi_st = 10**log_chi_st
        else:
            chi_st = (strain_rate / np.pi) * np.exp(-2 * erfcinv(2 * self.Z_st)**2)
        return chi_st

    def _estimate_strain_from_chi_st(self, chi_st: float) -> float:
        """Estimate the strain rate from a given stoichiometric scalar dissipation rate.

        Args:
            chi_st: Scalar dissipation rate at stoichiometric mixture fraction [1/s]
        
        Returns:
            float: Estimated strain rate [1/s]
        
        Note:
            Uses regression model if available, otherwise uses Peters' relationship (Peters, PECS 1984)
        """
        if self.strain_chi_st_model_params is not None:
            m = self.strain_chi_st_model_params['slope']
            b = self.strain_chi_st_model_params['intercept']
            log_strain_rate = (np.log10(chi_st) - b) / m
            strain_rate = 10**log_strain_rate
        else:
            strain_rate = chi_st * np.pi / np.exp(-2 * erfcinv(2 * self.Z_st)**2)
        return strain_rate

    def _estimate_flame_thickness(
        self,
        strain_rate: Optional[float] = None,
        chi_st: Optional[float] = None
    ) -> float:
        """Estimate the flame thickness based on strain rate or chi_st.
        
        Args:
            strain_rate: Optional strain rate [1/s]
            chi_st: Optional scalar dissipation rate at stoichiometric mixture fraction [1/s]
        
        Returns:
            float: Estimated flame thickness [m]
        """
        if strain_rate is not None:
            chi_st = self._estimate_chi_st_from_strain(strain_rate)
            return self._estimate_flame_thickness(chi_st=chi_st)
        elif chi_st is not None:
            self.gas.TPX = (0.5 * (self.fuel_inlet.temperature + self.oxidizer_inlet.temperature),
                            self.pressure,
                            self.fuel_inlet.composition)
            thermal_diffusivity = self.gas.thermal_conductivity / (self.gas.density * self.gas.cp_mass)
            return np.sqrt(thermal_diffusivity / chi_st)
        else:
            raise ValueError("Either strain_rate or chi_st must be provided.")
    
    def _measure_flame_thickness(self) -> float:
        """Measure the flame thickness in the current state.

        Returns:
            float: Measured flame thickness [m]
        """
        if self.flame.extinct():
            # Placeholder to prevent mesh update
            return self.width / self.width_ratio
            # Alternative - use estimate
            strain_rate = self.flame.strain_rate('max')
            return self._estimate_flame_thickness(strain_rate=strain_rate)
        else:
            T_max = np.max(self.flame.T)
            T_threshold = 0.5 * T_max
            indices = np.where(self.flame.T >= T_threshold)[0]
            flame_thickness = self.flame.grid[indices[-1]] - self.flame.grid[indices[0]]
        return flame_thickness

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
        v_tot = target_strain * self.width / 2
        v_fuel = v_tot / (1 + np.sqrt(rho_fuel / rho_ox))
        v_ox = v_tot - v_fuel

        # Convert to mass fluxes
        mdot_fuel = rho_fuel * v_fuel
        mdot_oxidizer = rho_ox * v_ox

        return mdot_fuel, mdot_oxidizer
    
    def _strain_rate_nominal(self) -> float:
        """Compute the nominal strain rate based on the input velocities.

        Returns:
            float: Nominal strain rate [1/s]
        """
        self.gas.TPX = (self.fuel_inlet.temperature,
                        self.pressure,
                        self.fuel_inlet.composition)
        rho_fuel = self.gas.density
        self.gas.TPX = (self.oxidizer_inlet.temperature,
                        self.pressure,
                        self.oxidizer_inlet.composition)
        rho_ox = self.gas.density
        v_fuel = self.flame.fuel_inlet.mdot / rho_fuel
        v_ox = self.flame.oxidizer_inlet.mdot / rho_ox
        return 2 * (v_fuel + v_ox) / self.width

    def _initialize_flame(
            self,
            chi_st: float,
            grid: Optional[np.ndarray] = None):
        """Set up the initial counterflow diffusion flame configuration.

        Args:
            chi_st: The scalar dissipation rate at the stoichiometric mixture fraction
            grid: The grid for the flame object
        
        Initializes the Cantera flame object with appropriate grid, inlet conditions,
        and refinement criteria. Estimates appropriate strain rate based on target
        scalar dissipation rate.
        """
        if grid is not None:
            self.flame = ct.CounterflowDiffusionFlame(self.gas, grid=grid)
        else:
            self.flame = ct.CounterflowDiffusionFlame(self.gas, width=self.width)
        self.flame.transport_model = self.transport_model
        
        # Set operating conditions
        self.flame.P = self.pressure
        
        # Set inlet conditions
        mdot_fuel, mdot_ox = self._mdots_from_chi_st(chi_st)
        self.flame.fuel_inlet.mdot = mdot_fuel
        self.flame.fuel_inlet.X = self.fuel_inlet.composition
        self.flame.fuel_inlet.T = self.fuel_inlet.temperature
        self.flame.oxidizer_inlet.mdot = mdot_ox
        self.flame.oxidizer_inlet.X = self.oxidizer_inlet.composition
        self.flame.oxidizer_inlet.T = self.oxidizer_inlet.temperature

        # Set refinement parameters
        self.flame.set_refine_criteria(ratio=4.0, slope=0.1, curve=0.2, prune=0.05)
    
    def _enable_two_point_control(self):
        self.flame.two_point_control_enabled = True
        self.flame.flame.set_bounds(spread_rate=(-1e-5, 1e20))
        self.flame.max_time_step_count = 100
    
    def _update_flame_width(self, solve: Optional[bool] = True):
        """Update the flame width and reinitialize the flame object.

        Args:
            solve: Whether to solve to steady state after update
        """
        if self.flame is None:
            flame_thickness = self._estimate_flame_thickness(chi_st=self.initial_chi_st)
        else:
            flame_thickness = self._measure_flame_thickness()

        # Compute the new width
        old_width = self.width
        target_width = self.width_ratio * flame_thickness
        if self.flame is None:
            self.width = target_width
            self._initialize_flame(self.initial_chi_st)
            return
        if not self.width_change_enable:
            return
        if np.abs(target_width - old_width) / old_width <= self.width_change_min:
            return
        self.width = np.clip(target_width,
                               (1+self.width_change_max) * old_width,
                             1/(1+self.width_change_max) * old_width)
        if self.width == old_width:
            return
        
        width_increasing = (self.width >= old_width)
        self.logger.info(f"Updating domain width from {old_width:.3e} m to {self.width:.3e} m")
        
        # Save current state
        old_solution = self.flame.to_array()
        old_mdots = (self.flame.fuel_inlet.mdot,
                     self.flame.oxidizer_inlet.mdot)
        old_grid = self.flame.grid
        
        # Find approximate flame location (using peak temperature)
        old_grid_norm = old_grid / old_width
        flame_idx = np.argmax(self.flame.T)
        flame_loc_old = old_grid[flame_idx]
        flame_loc_normalized = old_grid_norm[flame_idx]
        flame_loc_new = flame_loc_normalized * self.width
        
        # Construct new grid maintaining resolution and flame position
        if width_increasing:
            # For width increase, extend the existing grid
            # Create new grid, keeping flame_loc_normalized and absolute old spacing
            new_grid = old_grid + (flame_loc_new - flame_loc_old)

            # Fill in the gaps at the sides
            dx_l = old_grid[ 1] - old_grid[ 0]
            dx_r = old_grid[-1] - old_grid[-2]
            grid_l = np.arange(0, new_grid[0], dx_l)
            grid_r = np.arange(new_grid[-1], self.width, dx_r)
            if len(grid_r) > 0 and grid_r[-1] < self.width:
                grid_r = np.append(grid_r[1:], self.width)
            new_grid = np.concatenate((grid_l, new_grid, grid_r))
        else:
            # For width decrease, trim existing grid
            # Create new grid, keeping flame_loc_normalized and absolute old spacing
            new_grid = old_grid + (flame_loc_new - flame_loc_old)

            # Find points that fall within new domain
            interior_mask = (new_grid > 0) & (new_grid < self.width)
            new_grid = new_grid[interior_mask]
            
            # Add boundary points
            new_grid = np.concatenate(([0.0], new_grid, [self.width]))
        
        new_grid_norm = new_grid / self.width
        self._initialize_flame(chi_st=self.initial_chi_st,
                               grid=new_grid)
        # ^ uses initial_chi_st but mdots will be overwritten below
        
        # Interpolate solution onto new grid
        scale_factor = self.width / old_width
        
        var_names = ['velocity', 'spread_rate', 'lambda', 'T']
        var_names += self.gas.species_names
        if self.flame.two_point_control_enabled:
            var_names += ['Uo']
        
        for var_name in var_names:
            old_values = getattr(old_solution, var_name)
            interp = interpolate.interp1d(
                old_grid_norm, old_values,
                kind='cubic', bounds_error=False,
                fill_value=(old_values[0], old_values[-1]))
            new_values = interp(new_grid_norm)

            # Handle special cases
            if var_name in ['velocity', 'spread_rate', 'Uo']:
                new_values *= scale_factor
            # elif var_name in self.gas.species_names:
            #     new_values = np.clip(new_values, 0, 1)
            
            # Update the flame solution
            self.flame.set_profile(var_name, new_grid_norm, new_values)
        
        # # Normalize mass fractions
        # Y_sum = np.zeros_like(new_grid)
        # for k in range(self.gas.n_species):
        #     Y_sum += self.flame.Y[k, :]
        # for k in range(self.gas.n_species):
        #     Y = self.flame.Y[k, :]
        #     Y_normalized = np.where(Y_sum > 0, Y / Y_sum, 0)
        #     self.flame.set_profile(self.gas.species_names[k],
        #                            new_grid_norm,
        #                            Y_normalized)

        # Update inlet mass flow rates
        self.flame.fuel_inlet.mdot     = old_mdots[0] * scale_factor
        self.flame.oxidizer_inlet.mdot = old_mdots[1] * scale_factor

        # Solve to steady state
        if solve:
            self.logger.info("Computing the solution in the new domain")
            self.flame.solve(loglevel=self.solver_loglevel, auto=True)
    
    def compute_s_curve(
        self,
        output_dir: Optional[Path] = None,
        n_extinction_points: int = 10,
        write_FlameMaster: bool = False,
        **kwargs
    ) -> List[Dict]:
        """Compute complete S-curve including ignited branches and extinction branch.
        
        Computes the full S-curve by first traversing the upper (stable) and middle
        (unstable) branches, then computing the lower (extinction) branch.
        
        Args:
            output_dir: Directory to save solution files
            n_extinction_points: Number of points to compute along extinction branch
            write_FlameMaster: Whether to write FlameMaster output files for each solution
            **kwargs: Additional arguments passed to compute_ignited_branches
        
        Returns:
            List[Dict]: List of solution metadata dictionaries containing properties
                of each computed solution point
        """
        # First compute the ignited and unstable branches
        self.logger.info("Computing ignited and unstable branches")
        ignited_data = self.compute_ignited_branches(
            output_dir=output_dir,
            write_FlameMaster=write_FlameMaster,
            **kwargs)
        
        # Get chi_st bounds for extinction branch
        chi_st_values = [sol['chi_st'] for sol in ignited_data]
        chi_st_max = max(chi_st_values) * 1.0e1
        chi_st_min = ignited_data[-1]['chi_st']  # Last point on unstable branch
        
        # Compute extinction branch
        self.logger.info("Computing extinction branch")
        extinct_data = self.compute_extinct_branch(
            chi_st_min=chi_st_min,
            chi_st_max=chi_st_max,
            n_points=n_extinction_points,
            output_dir=output_dir,
            write_FlameMaster=write_FlameMaster)
        
        return ignited_data + extinct_data

    def compute_ignited_branches(
        self,
        output_dir: Optional[Path] = None,
        restart_from: Optional[int] = None,
        n_max: int = 5000,
        initial_spacing: float = 0.6,
        unstable_spacing: float = 0.95,
        temperature_increment: float = 20.0,
        max_increment: float = 100.0,
        target_delta_T_max: float = 20.0,
        max_error_count: int = 3,
        strain_rate_tol: float = 0.10,
        write_FlameMaster: bool = False
    ) -> List[Dict]:
        """Compute upper (stable) and middle (unstable) branches of the S-curve.
        
        This method traverses both the upper (stable) and middle (unstable) branches of the
        S-curve using temperature as a control parameter. It employs a two-point continuation
        method to track solutions through the turning point and down the unstable branch.
        
        Args:
            output_dir: Directory to save solution files
            restart_from: Optional solution index to restart from if continuing a previous calculation
            n_max: Maximum number of iterations before stopping
            initial_spacing: Initial control point spacing for stable branch (0-1)
            unstable_spacing: Control point spacing for unstable branch (0-1)
            temperature_increment: Initial temperature increment between solutions [K]
            max_increment: Maximum allowed temperature increment [K]
            target_delta_T_max: Target maximum temperature change per step [K]
            max_error_count: Maximum number of successive solver errors before stopping
            strain_rate_tol: Tolerance for minimum strain rate relative to maximum
            write_FlameMaster: Whether to write FlameMaster output files for each solution
        
        Returns:
            List[Dict]: List of solution metadata dictionaries containing properties
                of each computed solution point. Each dictionary includes:
                - T_max: Maximum temperature [K]
                - T_st: Temperature at stoichiometric mixture fraction [K]
                - strain_rate_max: Maximum strain rate [1/s]
                - strain_rate_nom: Nominal strain rate [1/s]
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
            Solutions are saved to HDF5 files if output_dir is provided.
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle restart case
        if restart_from is not None:
            if restart_from >= len(self.solutions):
                raise ValueError(f"Restart index {restart_from} exceeds number of available solutions")
            
            # Restore flame state and get previous temperature increment
            restart_solution = self.solutions[restart_from]
            self.flame.from_array(restart_solution['state'])
            temperature_increment = restart_solution['metadata']['Tc_increment']
            
            # Get maximum strain rate from previous solutions
            a_max = max(sol['metadata']['strain_rate_max'] for sol in self.solutions[:restart_from+1])
            
            # Initialize data with previous solutions
            data = [sol['metadata'] for sol in self.solutions[:restart_from+1]]
            
            self.logger.info(f'Restarting from solution {restart_from} with T_increment = {temperature_increment:.2f}')
            start_iteration = restart_from + 1
            
        else:
            self.logger.info('Computing the initial solution')
            self.flame.solve(loglevel=self.solver_loglevel, auto=True)
            T_max = np.max(self.flame.T)
            strain_rate_max = self.flame.strain_rate('max')
            strain_rate_max_glob = strain_rate_max
            data = []
            start_iteration = 0
        
        self.logger.info('Beginning s-curve computation')
        error_count = 0
        for i in range(start_iteration, n_max):
            # Update flame width if we are attempting a new point
            if error_count == 0:
                self._update_flame_width(solve=True)
                self._enable_two_point_control()
                T_max = np.max(self.flame.T)
            
            backup_state = self.flame.to_array()

            # Update control temperatures
            spacing = unstable_spacing if strain_rate_max <= 0.98 * strain_rate_max_glob else initial_spacing
            control_temperature = (np.min(self.flame.T) + 
                                   spacing * (np.max(self.flame.T) - np.min(self.flame.T)))
            self.logger.debug(f'Iteration {i}: Control temperature = {control_temperature:.2f} K')
            self.flame.set_left_control_point(control_temperature)
            self.flame.set_right_control_point(control_temperature)
            self.flame.left_control_point_temperature -= temperature_increment
            self.flame.right_control_point_temperature -= temperature_increment
            self.flame.clear_stats()
            if (self.flame.left_control_point_temperature < self.flame.fuel_inlet.T + 100 or
                self.flame.right_control_point_temperature < self.flame.oxidizer_inlet.T + 100):
                self.logger.info("SUCCESS! Control point temperature near inlet temperature.")
                break
            
            try:
                self.flame.solve(loglevel=self.solver_loglevel)
                
                # Adjust temperature increment based on convergence
                if abs(max(self.flame.T) - T_max) < 0.8 * target_delta_T_max:
                    temperature_increment = min(temperature_increment + 3, max_increment)
                elif abs(max(self.flame.T) - T_max) > target_delta_T_max:
                    temperature_increment *= (0.9 * target_delta_T_max / 
                                           abs(max(self.flame.T) - T_max))
                error_count = 0
                
            except ct.CanteraError as err:
                self.logger.debug(err)
                
                # Restore previous solution and reduce increment
                self.flame.from_array(backup_state)
                temperature_increment = 0.7 * temperature_increment
                error_count += 1
                if error_count > max_error_count:
                    if strain_rate_max / strain_rate_max_glob < strain_rate_tol:
                        self.logger.info(
                            'SUCCESS! Traversed unstable branch down to '
                            f'{100 * strain_rate_max / strain_rate_max_glob:.4f}% of the maximum strain rate.'
                        )
                    else:
                        self.logger.warning(
                            f'FAILURE! Stopping after {error_count} successive solver errors.'
                        )
                    break
                self.logger.warning(
                    f"Solver did not converge on iteration {i}. "
                    f"Trying again with dT = {temperature_increment:.2f}"
                )
                continue
            
            # Compute postprocessing data
            Z = self.flame.mixture_fraction("Bilger")
            chi, chi_st = self._compute_scalar_dissipation(Z)
            interp_T = interpolate.interp1d(Z, self.flame.T)
            T_st = float(interp_T(self.Z_st))
            T_max = max(self.flame.T)
            T_mid = 0.5 * (min(self.flame.T) + max(self.flame.T))
            s = np.where(self.flame.T > T_mid)[0][[0, -1]]
            width = self.flame.grid[s[1]] - self.flame.grid[s[0]]
            strain_rate_max = self.flame.strain_rate('max')
            strain_rate_nom = self._strain_rate_nominal()
            strain_rate_max_glob = max(strain_rate_max, strain_rate_max_glob)
            step_data = {
                'T_max': T_max,
                'T_st': T_st,
                'strain_rate_max': strain_rate_max,
                'strain_rate_nom': strain_rate_nom,
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
            
            if output_dir:
                self.save_solution(output_dir, len(self.solutions) - 1)
                if write_FlameMaster:
                    self.write_FlameMaster(output_dir)
            
            # Logging after successful solution
            self.logger.info(
                f"Iteration {i} completed: T_max = {T_max:.2f}, T_st = {T_st:.2f}, "
                f"chi_st = {chi_st:.4e}, strain_rate_max = {strain_rate_max:.2f}, "
                f"strain_rate_nom = {strain_rate_nom:.2f}"
            )

            if chi_st < self.solutions[0]['metadata']['chi_st'] and not self.width_change_enable:
                self.logger.info("SUCCESS! Traversed unstable branch down to initial chi_st.")
                self.logger.info("Stopping because width changes are disabled. (Flame will start to grow beyond domain.)")
                break

        self.logger.info(f'Stopped after {i} iterations')
        return data

    def compute_extinct_branch(
        self,
        chi_st_min: float,
        chi_st_max: float,
        n_points: int = 10,
        output_dir: Optional[Path] = None,
        write_FlameMaster: bool = False
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
            output_dir: Optional directory path to save solution files
            write_FlameMaster: Whether to write FlameMaster output files for each solution
        
        Returns:
            List[Dict]: List of solution metadata dictionaries containing properties
                of each computed solution point. Each dictionary includes:
                - T_max: Maximum temperature [K]
                - T_st: Temperature at stoichiometric mixture fraction [K]
                - strain_rate_max: Maximum strain rate [1/s]
                - strain_rate_nom: Nominal strain rate [1/s]
                - chi_st: Scalar dissipation rate at stoichiometric mixture fraction [1/s]
                - total_heat_release_rate: Integrated heat release rate [W/m³]
                - n_points: Number of grid points
                - flame_width: Width of the flame [m]
                - branch: 'extinction' to identify the branch
        
        Note:
            The extinction branch is computed by starting from a cold mixing solution
            and using strain rate as a control parameter to achieve target scalar
            dissipation rates. Solutions are saved to HDF5 files if output_dir
            is provided.
        """
        breakpoint()
        chi_st_values = np.logspace(np.log10(chi_st_min), np.log10(chi_st_max), n_points)
        chi_st_values = chi_st_values[::-1]
        data = []

        iter = 0
        while iter < 3:
            self._initialize_flame(chi_st=chi_st_max)
            try:
                self.flame.solve(loglevel=self.solver_loglevel)
            except:
                self.logger.error(f"Failed to compute initial solution at chi_st = {chi_st_max:.2e}")
                return data
            
            if self.flame.extinct():
                self.logger.info(f"Computed initial solution at chi_st = {chi_st_max:.2e} as cold mixing")
                break
            
            self.logger.warning(f"Failed to compute initial solution at chi_st = {chi_st_max:.2e} (autoignited)")
            chi_st_max = 1.0e1 * chi_st_max
            chi_st_values = np.geomspace(chi_st_min, chi_st_max, n_points)
            chi_st_values = chi_st_values[::-1]
            self.logger.info(f"Retrying with chi_st_max = {chi_st_max:.2e}")
        
        for i, chi_st in enumerate(chi_st_values):
            mdot_fuel, mdot_ox = self._mdots_from_chi_st(chi_st)
            self.flame.fuel_inlet.mdot = mdot_fuel
            self.flame.oxidizer_inlet.mdot = mdot_ox
            
            try:
                self.flame.solve(loglevel=self.solver_loglevel)
            except:
                self.logger.warning(f"Failed to compute solution at chi_st = {chi_st:.2e}")
                continue

            if not self.flame.extinct():
                self.logger.warning(f"Failed to compute solution at chi_st = {chi_st:.2e} (autoignited)")
                continue
            
            # Compute postprocessing data
            Z = self.flame.mixture_fraction("Bilger")
            chi, chi_st = self._compute_scalar_dissipation(Z)
            interp_T = interpolate.interp1d(Z, self.flame.T)
            T_st = float(interp_T(self.Z_st))
            T_max = max(self.flame.T)
            T_mid = 0.5 * (min(self.flame.T) + max(self.flame.T))
            s = np.where(self.flame.T > T_mid)[0][[0, -1]]
            width = self.flame.grid[s[1]] - self.flame.grid[s[0]]
            strain_rate_max = self.flame.strain_rate('max')
            strain_rate_nom = self._strain_rate_nominal()
            step_data = {
                'T_max': T_max,
                'T_st': T_st,
                'strain_rate_max': strain_rate_max,
                'strain_rate_nom': strain_rate_nom,
                'chi_st': chi_st,
                'total_heat_release_rate': np.trapz(self.flame.heat_release_rate, self.flame.grid),
                'n_points': len(self.flame.grid),
                'flame_width': width,
                'time_steps': sum(self.flame.time_step_stats),
                'eval_count': sum(self.flame.eval_count_stats),
                'cpu_time': sum(self.flame.jacobian_time_stats + self.flame.eval_time_stats),
            }
            data.append(step_data)
            self.solutions.append({
                'state': self.flame.to_array(),
                'Z': Z,
                'chi': chi,
                'metadata': step_data
            })

            if output_dir:
                self.save_solution(output_dir, len(self.solutions) - 1)
                if write_FlameMaster:
                    self.write_FlameMaster(output_dir)
            
            # Logging after successful solution
            self.logger.info(
                f"Iteration {i} completed: T_max = {T_max:.2f}, T_st = {T_st:.2f}, "
                f"chi_st = {chi_st:.4e}, strain_rate_max = {strain_rate_max:.2f}, "
                f"strain_rate_nom = {strain_rate_nom:.2f}"
            )
        
        self.logger.info(f'Completed {len(data)} points on the extinction branch')
        return data

    def save_solution(self, output_dir: Path, solution_index: int):
        """Save a single solution to the HDF5 files.
        
        Saves both the flame profiles and associated metadata for a single solution.
        
        Args:
            output_dir: Directory path where files will be saved
            solution_index: Index of the solution being saved
        """
        solutions_file = output_dir / 'solutions.h5'
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
                    },
                    'oxidizer_inlet': {
                        'composition': self.oxidizer_inlet.composition,
                        'temperature': self.oxidizer_inlet.temperature,
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

    def save_all_solutions(self, output_dir: Path):
        """Save all computed solutions to HDF5 files.
        
        Args:
            output_dir: Directory path where files will be saved
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(len(self.solutions)):
            self.save_solution(output_dir, i)
    
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
                pressure=params['pressure']
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
    
    def write_FlameMaster(self, output_dir: Path):
        """Write FlameMaster input files for a single solution.
        
        Args:
            output_dir: Directory path where files will be saved
        """
        solution = self.solutions[-1]

        # Build filename
        fuel_name = [key for key in self.fuel_inlet.composition.keys()][0]
        pressure_str = f"p{(self.pressure/1e5):04.1f}".replace('.', '_')
        chi_str = f"chi{solution['metadata']['chi_st']:0.4e}"
        tf_str = f"tf{self.fuel_inlet.temperature:04.0f}"
        to_str = f"to{self.oxidizer_inlet.temperature:04.0f}"
        Tst_str = f"Tst{solution['metadata']['T_st']:04.0f}"
        filename = output_dir / f"{fuel_name}_{pressure_str}{chi_str}{tf_str}{to_str}{Tst_str}"

        def write_array(f, name, data, n_cols=5):
            f.write(f"{name}\n")
            i_col = 0
            for i in range(len(data)):
                if i_col == n_cols:
                    f.write("\n")
                    i_col = 0
                f.write(f"\t{data[i]:0.6e}")
                i_col += 1
            f.write("\n")

        # Write the FlameMaster output file
        with open(filename, 'w') as f:
            # Write header
            f.write(f"header\n")
            f.write(f"\n")
            f.write(f"title = \"planar counterflow diffusion flame\"\n")
            f.write(f"mechanism = \"{self.mechanism_file}\"\n")
            f.write(f"author = FPVgen\n")
            f.write(f"date = {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n")
            f.write(f"fuel = {fuel_name}\n")
            f.write(f"pressure = {self.pressure/1e5} [bar]\n")
            f.write(f"Z_st = {self.Z_st} [1/s]\n")
            f.write(f"chi_st = {solution['metadata']['chi_st']} [1/s]\n")
            f.write(f"ConstantLewisNumbers = \"False\"\n")
            f.write(f"FlameLoc = {self._compute_flame_loc_Z():0.5e}")
            f.write(f"Tmax = {solution['metadata']['T_max']} [K]\n")
            f.write(f"\n")
            f.write(f"FuelSide\n")
            f.write(f"begin\n")
            f.write(f"\tTemperature = {self.fuel_inlet.temperature} [K]\n")
            for key, value in self.fuel_inlet.composition.items():
                f.write(f"\tMassFraction-{key} = {value}\n")
            f.write(f"end\n")
            f.write(f"\n")
            f.write(f"OxidizerSide\n")
            f.write(f"begin\n")
            f.write(f"\tTemperature = {self.oxidizer_inlet.temperature} [K]\n")
            for key, value in self.oxidizer_inlet.composition.items():
                f.write(f"\tMassFraction-{key} = {value}\n")
            f.write(f"end\n")
            f.write(f"\n")
            f.write(f"numOfSpecies = {self.gas.n_species}\n")
            f.write(f"gridPoints = {len(self.flame.grid)}\n")
            f.write(f"\n")

            # Write flame profiles
            f.write(f"body\n")
            write_array(f, "Z", self.flame.mixture_fraction("Bilger")[::-1])
            write_array(f, "temperature [K]", self.flame.T[::-1])
            for k in range(self.gas.n_species):
                write_array(f, f"massfraction-{self.gas.species_names[k]}", self.flame.Y[k, ::-1])
            write_array(f, "W", self.flame.mean_molecular_weight[::-1])
            write_array(f, "ZBilger", self.flame.mixture_fraction("Bilger")[::-1])
            write_array(f, "chi [1/s]", solution['chi'][::-1])
            write_array(f, "density", self.flame.density[::-1])
            write_array(f, "lambda [W/m K]", self.flame.thermal_conductivity[::-1])
            write_array(f, "cp [J/kg K]", self.flame.cp_mass[::-1])
            write_array(f, "lambdaOverCp [kg/ms]", (self.flame.thermal_conductivity[::-1] / self.flame.cp_mass[::-1]))
            write_array(f, "mu [kg/sm]", self.flame.viscosity[::-1])
            for k in range(self.gas.n_species):
                write_array(f, f"ProdRate-{self.gas.species_names[k]} [kg/m^3s]",
                            self.flame.net_production_rates[k, ::-1] * self.gas.molecular_weights[k])
            write_array(f, "ProgVar", self._compute_progress_variable()[::-1])
            write_array(f, "ProdRateProgVar [kg/m^3s]", self._compute_progress_variable_production()[::-1])
            write_array(f, "TotalEnthalpy [J/kg]", self.flame.enthalpy_mass[::-1])
            write_array(f, "HeatRelease [J/m^3 s]", self.flame.heat_release_rate[::-1])
            write_array(f, "Diffusivity [m/s]", np.zeros_like(self.flame.grid)) # Real gas thing, zeros are fine for now
            write_array(f, "rho_Diffusivity [kg/(m^2 s)]", np.zeros_like(self.flame.grid)) # Real gas thing, zeros are fine for now
            write_array(f, "TotalEnthalpy_rg [J/kg]", self.flame.enthalpy_mass[::-1] + self.flame.cp_mass[::-1]) # Real gas thing, zeros are fine for now
            write_array(f, "IsoCompressibility [m^2/N]", np.zeros_like(self.flame.grid)) # Real gas thing, zeros are fine for now
            write_array(f, "IsenCompressibility [m^2/N]", np.zeros_like(self.flame.grid)) # Real gas thing, zeros are fine for now
            write_array(f, "SpeedOfSound [m/s]", self.flame.sound_speed[::-1])
            write_array(f, "dLnCpDZ", self._compute_dLnCpDZ())
            write_array(f, "SumCpiOverCpMixDYiDZ", np.zeros_like(self.flame.grid)) # Real gas thing, zeros are fine for now

            # Write footer
            # (These are the lewis numbers, which we'll set to 1 here regardless of self.transport_model)
            f.write(f"trailer\n")
            for k in range(self.gas.n_species):
                f.write(f"{self.gas.species_names[k]}\t1")
    
    def assemble_FPV_table_CharlesX(
        self,
        output_dir: str,
        dims: Tuple[int, int, int] = (100, 2, 100),
        force_monotonicity: bool = False,
        igniting_table: bool = False,
        include_species_mass_fractions: List[str] = [],
        include_species_production_rates: List[str] = []
    ):
        raise NotImplementedError("CharlesX FPV table assembly is not yet implemented")
    
    def learn_strain_chi_st_mapping(self, output_file: Optional[str] = None):
        """Learn a mapping between strain rate and scalar dissipation rate at stoichiometry.

        Args:
            output_file: Optional file path to save the mapping as a JSON file
        
        Note:
            This method uses the computed solutions to learn a mapping between strain rate
            and scalar dissipation rate at the stoichiometric mixture fraction. The mapping
            is learned using a simple linear regression model.
        """
        from scipy.stats import linregress
        X = np.array([np.log10(sol['metadata']['strain_rate_nom']) for sol in self.solutions]).reshape(-1, 1)
        y = np.array([np.log10(sol['metadata']['chi_st']) for sol in self.solutions])
        slope, intercept, r_value, p_value, std_err = linregress(X.ravel(), y)
        self.strain_chi_st_model_params = {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err
        }

        if output_file is not None:
            with open(output_file, 'w') as f:
                json.dump(self.strain_chi_st_model_params, f)

    def plot_s_curve(
        self,
        x_quantity: Literal['strain_rate_max', 'chi_st'] = 'chi_st',
        y_quantity: Literal['T_max', 'T_st'] = 'T_max',
        output_file: Optional[str] = None
    ):
        """Plot the S-curve with configurable axes.
        
        Args:
            x_quantity: Which quantity to plot on x-axis ('strain_rate_max' or 'chi_st')
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
            'strain_rate_max': r"$\alpha$ [1/s]",
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

    def plot_current_state(self, output_file: Optional[str] = None):
        """Plot the current state of the flame in physical space.
        
        Args:
            output_file: Optional file path to save the figure

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes objects
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        plt.rcParams.update(pyplot_params)
        
        if self.flame is None:
            raise ValueError("Flame has not been initialized. Please initialize the flame first.")
        
        fig, ax = plt.subplots()
        ax.plot(self.flame.grid, self.flame.T, color='red')
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$T$ [K]")
        ax.grid(True, alpha=0.2)
        
        if output_file:
            fig.savefig(output_file, bbox_inches='tight', dpi=300)
        
        return fig, ax

    def plot_current_state_composition_space(self, output_file: Optional[str] = None):
        """Plot the current state of the flame in composition space.
        
        Args:
            output_file: Optional file path to save the figure

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes objects
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        plt.rcParams.update(pyplot_params)
        
        if self.flame is None:
            raise ValueError("Flame has not been initialized. Please initialize the flame first.")
        
        Z = self.flame.mixture_fraction("Bilger")
        T = self.flame.T
        
        fig, ax = plt.subplots()
        ax.plot(Z, T, color='red')
        ax.axvline(self.Z_st, color='k', linestyle='--')
        ax.set_xlabel(r"$Z$ [-]")
        ax.set_ylabel(r"$T$ [K]")
        ax.grid(True, alpha=0.2)
        
        if output_file:
            fig.savefig(output_file, bbox_inches='tight', dpi=300)
        
        return fig, ax

    def plot_strain_chi_st(
        self,
        strain_rate_type: Literal['max', 'nom'] = 'nom',
        output_file: Optional[str] = None
    ):
        """Plot the strain rate vs scalar dissipation rate at stoichiometric mixture fraction.
        
        Args:
            output_file: Optional file path to save the figure

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes objects
        """
        import matplotlib.pyplot as plt
        plt.rcParams.update(pyplot_params)

        strain_rates = [sol['metadata'][f'strain_rate_{strain_rate_type}'] for sol in self.solutions]
        chi_st_values = [sol['metadata']['chi_st'] for sol in self.solutions]
        
        fig, ax = plt.subplots()
        ax.scatter(chi_st_values, strain_rates)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"$\chi_{st}$ [1/s]")
        ax.set_ylabel(r"$\alpha$ [1/s]")
        ax.set_xmargin(0.1)
        ax.set_ymargin(0.1)
        ax.set_axisbelow(True)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        if output_file:
            fig.savefig(output_file, bbox_inches='tight', dpi=300)
        
        return fig, ax
