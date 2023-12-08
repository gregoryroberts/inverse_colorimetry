import numpy as npf
import lumapi

import autograd.numpy as np
from autograd import grad

from scipy import interpolate


import matplotlib.pyplot as plt




class SimulationEngine():

	def __init__( self, parameters, create_geometry_fn, create_forward_sources_fn, create_adjoint_sources_fn, create_monitors_fn ):

		self.fdtd_hook = lumapi.FDTD()

		self.fdtd_hook.new_project()

		self.project_path = parameters[ 'project_path' ]

		self.save_project()

		self.create_geometry_fn = create_geometry_fn
		self.create_forward_sources_fn = create_forward_sources_fn
		self.create_adjoint_sources_fn = create_adjoint_sources_fn
		self.create_monitors_fn = create_monitors_fn


	def save_project( self ):
		self.fdtd_hook.save( self.project_path )

	def init_geometry( self, parameters ):
		self.gemoetry_dictionary = self.create_geometry_fn( self.fdtd_hook, parameters )
		self.monitor_dictionary = self.create_monitors_fn( self.fdtd_hook, parameters )

	def import_device( self, device ):
		self.fdtd_hook.switchtolayout()
		self.fdtd_hook.select( self.gemoetry_dictionary[ 'device_import' ] )
		self.fdtd_hook.importnk2( device.reinterpolate_device(), device.device_region_reinterpolate_x_um, device.device_region_reinterpolate_y_um, device.device_region_reinterpolate_z_um )

	def init_forward_sources( self, parameters ):
		self.forward_source_dictionary = self.create_forward_sources_fn( self.fdtd_hook, parameters )

	def init_adjoint_sources( self, parameters ):
		self.adjoint_source_dictionary = self.create_adjoint_sources_fn( self.fdtd_hook, parameters )

	def init_all_sources( self, parameters ):
		self.init_forward_sources( parameters )
		self.init_adjoint_sources( parameters )

	def disable_source_dictionary( self, source_dictionary_ ):
		for key, value in self.source_dictionary_.items():
			value.enabled = 0

	def disable_all_sources( self ):
		self.fdtd_hook_.switchtolayout()

		self.disable_source_dictionary( self.forward_source_dictionary )
		self.disable_source_dictionary( self.adjoint_source_dictionary )

	def enable_forward_source( self, key, disable_other_sources=True ):
		self.fdtd_hook_.switchtolayout()

		if disable_other_sources:
			self.disable_all_sources()

		self.forward_source_dictionary[ key ].enabled = 1

	def enable_adjoint_source( self, key, disable_other_sources=True ):
		self.fdtd_hook_.switchtolayout()

		if disable_other_sources:
			self.disable_all_sources()

		self.adjoint_source_dictionary[ key ].enabled = 1


	def get_afield( self, monitor_name_, field_indicator_ ):
		field_polariations = [ field_indicator_ + 'x', field_indicator_ + 'y', field_indicator_ + 'z' ]

		field_pol_0 = self.fdtd_hook.getdata( monitor_name_, field_polariations[ 0 ] )

		total_field = np.zeros( [ len (field_polariations ) ] + list( field_pol_0.shape ), dtype=np.complex )
		total_field[ 0 ] = field_pol_0

		for pol_idx in range( 1, len( field_polariations ) ):
			field_pol = self.fdtd_hook.getdata( monitor_name_, field_polariations[ pol_idx ] )

			total_field[ pol_idx ] = field_pol

		return total_field

	def get_hfield( self, monitor_name_ ):
		return get_afield( monitor_name_, 'H' )

	def get_efield( self, monitor_name_ ):
		return get_afield( monitor_name_, 'E' )

	def run_forward_simulations( self ):

		volumetric_efield_dictionary = {}

		forward_observation_efield_dictionary = {}

		for key, value in self.forward_source_dictionary:
			self.enable_forward_source( key )

			self.fdtd_hook.run()

			volumetric_electric_field_dictionary[ key ] = get_efield( self.monitor_dictionary[ 'volumetric_device_monitor' ] )

			forward_observation_efield_dictionary[ key ] = {}

			for idx in range( 0, self.monitor_dictionary[ 'adjoint_focal_monitors' ] ):
				forward_observation_efield_dictionary[ key ][ name ] = get_efield( name )


		return volumetric_efield_dictionary, forward_observation_efield_dictionary


	def run_adjoint_simulations( self ):

		volumetric_efield_dictionary = {}

		for key, value in self.adjoint_source_dictionary:
			self.enable_forward_source( key )

			self.fdtd_hook.run()

			volumetric_electric_field_dictionary[ key ] = get_efield( self.monitor_dictionary[ 'volumetric_device_monitor' ] )

		return volumetric_efield_dictionary


	def run_simulations( self ):

		forward_volumetric_efields, forward_observation_efields = self.run_forward_simulations()
		adjoint_volumetric_efields = self.run_adjoint_simulations()

		return forward_volumetric_efields, forward_observation_efields, adjoint_volumetric_efields




class Device2D():

	def __init__( self, parameters, fabrication_filter ):

		self.parameters = parameters

		self.device_height_um = parameters[ 'device_max_y_um' ] - parameters[ 'device_min_y_um' ]

		self.width_voxels = int( parameters[ 'device_width_um' ] / parameters[ 'minimum_feature_size_um' ] )
		self.height_voxels = int( self.device_height_um / parameters[ 'layer_height_um' ] )

		self.fabrication_filter = fabrication_filter

		self.design = init_design_uniform( 0.5 )
		self.device = self.fabrication_filter.forward( self.design )

		self.reinterpolate_factor = parameters[ 'reinterpolate_permittivity_factor' ]

		self.width_voxels_reinterpolate = self.reinterpolate_factor * self.width_voxels
		self.height_voxels_reinterpolate = self.reinterpolate_factor * self.height_voxels

		self.device_region_x_um = np.linspace( -0.5 * parameters[ 'device_width_um' ], 0.5 * parameters[ 'device_width_um' ], self.width_voxels )
		self.device_region_y_um = np.linspace( parameters[ 'device_min_y_um' ], parameters[ 'device_max_y_um' ], self.height_voxels )
		self.device_region_z_um = np.linspace( -0.51, 0.51, 2 )

		self.device_region_reinterpolate_x_um = np.linspace( -0.5 * parameters[ 'device_width_um' ], 0.5 * parameters[ 'device_width_um' ], self.width_voxels_reinterpolate )
		self.device_region_reinterpolate_y_um = np.linspace( parameters[ 'device_min_y_um' ], parameters[ 'device_max_y_um' ], self.height_voxels_reinterpolate )
		self.device_region_reinterpolate_z_um = np.linspace( -0.51, 0.51, 2 )

		self.device_gradient_region = None

	def reinterpolate_device( self ):
		return np.repeat( np.repeat( np.repeat( self.get_device(), self.reinterpolate_factor, axis=0 ), self.reinterpolate_factor, axis=1 ) )

	def init_design_uniform( self, uniform_value=0.5 ):
		assert ( uniform_value >= 0 ) and ( uniform_value <= 1 ), "invalid uniform initialization!"

		self.design = uniform_value * np.ones( ( self.width_voxels, self.height_voxels ) )
		self.device = self.fabrication_filter.forward( self.design, self.parameters )

	def init_design_with_design( self, design_ ):
		self.design = design_.copy()
		self.device = self.fabrication_filter.forward( self.design, self.parameters )

	def get_design( self ):
		return self.design.copy()

	def get_device( self ):
		self.device = self.fabrication_filter.forward( self.design, self.parameters )

		return self.device.copy()

	def backpropagate_gradient( self, gradient_ ):

		if self.device_gradient_reinterpolate_region is None:

			input_gradient_shape = gradient_.shape

			self.gradient_region_x_um = np.linspace( -0.5 * self.parameters[ 'device_width_um' ], 0.5 * self.parameters[ 'device_width_um' ], input_gradient_shape[ 0 ] )
			self.gradient_region_y_um = np.linspace( self.parameters[ 'device_min_y_um' ], self.parameters[ 'device_max_y_um' ], input_gradient_shape[ 1 ] )

			self.device_gradient_region = np.zeros( ( self.width_voxels, self.height_voxels, 2 ) )
			for x_idx in range( 0, self.width_voxels ):
				for y_idx in range( 0, self.height_voxels ):
						self.device_gradient_region[ x_idx, y_idx, : ] = [ self.device_region_x_um[ x_idx ], self.device_region_y_um[ y_idx ] ]

		gradient_interpolated = interpolate.interpn( ( gradient_region_x, gradient_region_y, gradient_region_z ), gradient_, self.device_gradient_region, method='linear' )

		def fabrication_filter_for_autograd( gradient_interpolated_ ):
			return np.sum( self.fabrication_filter.forward( np.reshape( self.device, ( self.width_voxels, self.height_voxels ) ) ) * gradient_interpolated_, self.parameters )

		fabrication_grad = grad( fabrication_filter_for_autograd )

		backpropagate_gradient_ = np.reshape( fabrication_grad( gradient_interpolated ), ( self.width_voxels, self.height_voxels ) )

		return backpropagate_gradient_

	def evolve( self, gradient_, step_size_ ):
		backpropagate_gradient_ = self.backpropagate_gradient( gradient_ )

		self.design += step_size_ * backpropagate_gradient


def softplus( x_in, kappa_=2 ):
	return np.log( 1 + np.exp( kappa_ * x_in ) ) / kappa_

def softplus_prime( x_in, kappa_=2 ):
	return ( 1. / ( 1 + np.exp( -kappa_ * x_in ) ) )


def compute_fom_and_gradient( forward_volumetric_efields_, forward_observation_efields_, adjoint_volumetric_efields_, parameters_, do_performance_weighting=True, just_fom=False ):
	wl_min_um = parameters_[ 'wavelength_min_um' ]
	wl_max_um = parameters_[ 'wavelength_max_um' ]
	wl_range_um = wl_max_um - wl_min_um
	wl_mid_um = 0.5 * ( wl_min_um + wl_max_um )

	wavelengths_um = np.linspace( wl_min_um, wl_max_um, parameters_[ "monitor_num_wavelength_points" ] )

	wavelength_centers_relative = [ 0.25, 0.75 ]
	wavelength_centers_um = [ wl_min_um + wavelength_centers_relative[ idx ] * wl_range_um for idx in range( 0, len( wavelength_centers_relative ) ) ]

	min_value_filter = -0.5
	max_value_filter = 1.0

	filter_function_width_um = 0.15 * wl_range_um

	filter_functions_by_pol = [
		min_value_filter + ( max_value_filter - min_value_filter ) * np.exp( -( wavelengths_um - wavelength_centers_um[ idx ] )**2 / ( 2 * filter_function_width_um**2 ) ) for idx in range( 0, len( wavelength_centers_relative ) )
	]

	#
	# 2d scaling - only confinement in one direction
	#
	normalized_intensities = [
		np.squeeze( np.sum( np.abs( forward_observation_efields_[ 'x' ][ 0 ][ :, 0, 0, 0, : ] )**2, axis=0 ) * wavelengths_um / wl_mid_um ),
		np.squeeze( np.sum( np.abs( forward_observation_efields_[ 'y' ][ 0 ][ :, 0, 0, 0, : ] )**2, axis=0 ) * wavelengths_um / wl_mid_um )
	]

	fom_by_pol = []
	fom_by_pol = np.array( [ softplus( np.sum( filter_functions_by_pol[ idx ] * normalized_intensities[ idx ] ) ) for idx in range( 0, len( normalized_intensities ) ) ] )

	fom = np.sum( fom_by_pol )

	if just_fom:
		return fom


	softplus_grad_by_pol = np.array( [ softplus_prime( np.sum( filter_functions_by_pol[ idx ] * normalized_intensities[ idx ] ) ) for idx in range( 0, len( normalized_intensities ) ) ] )


	if do_performance_weighting:
		weight_by_pol = ( 2. / len( fom_by_pol ) ) - ( fom_by_pol**2 / np.sum( fom_by_pol**2 ) )

		weight_by_pol = np.maximum( weight_by_pol, 0 )
		weight_by_pol /= np.sum( weight_by_pol )
	else:
		weight_by_pol = np.ones( len( fom_by_pol ) )


	gradient_dimension = np.squeeze( forward_volumetric_efields_[ 'x' ][ 0, :, 0, :, 0 ] ).shape
	gradient = np.zeros( gradient_dimension, dtype=complex )

	for forward_key_idx in range( 0, len( forward_volumetric_efields_.keys() ) ):
		forward_key = forward_volumetric_efields_.keys()[ forward_key_idx ]

		for pol_idx in adjoint_volumetric_efields_.keys():

			weight_adjoint = np.conj(
				forward_observation_efields_[ forward_key ][ 0 ][ pol_idx, 0, 0, 0, : ] )

			forward_efields = forward_volumetric_efields_[ forward_key ]
			adjoint_efields = adjoint_volumetric_efields_[ pol_idx ]

			gradient += 2 * np.real( np.sum( np.sum( weight_by_pol[ forward_key_idx ] * softplus_grad_by_pol[ forward_key_idx ] * weight_adjoint, axis=0 ), axis=-1 ) )


	return fom, gradient


def create_geometry_transmission( fdtd_hook_, parameters_ ):

	simulation_geometry = {}

	simulation_geometry[ 'fdtd' ] = fdtd_hook_.addfdtd( {
		'name' : 'fdtd',
		'dimension' : '2D',
		'x span' : parameters_[ 'simulation_width_um' ] * 1e-6,
		'y min' : parameters_[ 'simulation_min_y_um' ] * 1e-6,
		'y max' : parameters_[ 'simulation_max_y_um' ] * 1e-6,
		'simulation time' : parameters_[ 'simulation_time_fs' ] * 1e-15
		'background_index' : parameters_[ 'background_index' ]
	} )


	simulation_geometry[ 'device_mesh' ] = fdtd_hook_.addmesh( {
		'name' : 'device_mesh',
		'x' : 0,
		'x span' : parameters_[ 'simulation_width_um' ] * 1e-6,
		'y min' : parameters_[ 'device_min_y_um' ] * 1e-6,
		'y max' : parameters_[ 'device_max_y_um' ] * 1e-6,
		'dx' : parameters_[ 'mesh_spacing_um' ] * 1e-6,
		'dy' : parameters_[ 'mesh_spacing_um' ] * 1e-6
	} )


	simulation_geometry[ 'device_import' ] = fdtd_hook_.addimport( {
		'name' : 'device_import',
		'x span' : parameters_[ 'device_width_um' ] * 1e-6,
		'x' : 0,
		'y min' : parameters_[ 'device_min_y_um' ] * 1e-6,
		'y max' : parameters_[ 'device_max_y_um' ] * 1e-6
	} )

	return simulation_geometry



def create_forward_sources_transmission( fdtd_hook_, parameters_ ):

	forward_sources = {}

	for polarization_idx in range( 0, len( parameters_[ 'forward_polarizations' ] ) ):

		forward_sources[ polarization_idx ] = fdtd_hook_.addgaussian( {
			'name' : 'forward_src_' + str( polarization_idx ),
			'angle theta' : 0,
			'polarization angle' : parameters_[ 'forward_polarizations' ][ polarization_idx ],
			'direction' : 'Backward',
			'x span' : 2 * parameters_[ 'fdtd_simulation_width_um' ] * 1e-6,
			'x' : 0,
			'y' : parameters_[ 'forward_src_vertical_um' ] * 1e-6,
			'wavelength start' : parameters_[ 'wavelength_min_um' ] * 1e-6,
			'wavelength stop' : parameters_[ 'wavelength_max_um' ] * 1e-6,
			'waist radius w0' : parameters_[ 'forward_waist_radius_um' ] * 1e-6,
			'distance from waist' : ( parameters_[ 'device_vertical_maximum_um' ] - parameters_[ 'forward_src_vertical_um' ] ) * 1e-6
		} )

	return forward_sources



def create_adjoint_sources_transmission( fdtd_hook_, parameters_ ):

	adjoint_sources = {}

	for polarization_idx in ( 0, len( parameters_[ 'adjoint_polarizations' ] ) ):

		adjoint_sources[ polarization_idx ] = fdtd_hook_.adddipole( {
			'name' : 'forward_src_' + str( polarization_idx ),
			'angle theta' : 0,
			'polarization angle' : parameters_[ 'adjoint_polarizations' ][ polarization_idx ],
			'theta' : parameters_[ 'adjoint_thetas' ][ polarization_idx ],
			'phi' : 0,
			'direction' : 'Backward',
			'x span' : 2 * parameters_[ 'fdtd_simulation_width_um' ] * 1e-6,
			'x' : 0,
			'y' : parameters_[ 'adjoint_src_vertical_um' ] * 1e-6,
			'wavelength start' : parameters_[ 'wavelength_min_um' ] * 1e-6,
			'wavelength stop' : parameters_[ 'wavelength_max_um' ] * 1e-6,
		} )

	return adjoint_sources


def create_monitors_transmission( fdtd_hook_, parameters_ ):

	simulation_monitors = {}

	simulation_monitors[ 'adjoint_focal_monitors' ] = [ fdtd_hook_.addpower( {
		'name' : 'adjoint_focal_monitor',
		'x' : 0,
		'y' : parameters_[ 'adjoint_src_vertical_um' ] * 1e-6,
		'override global monitor settings' : 1,
		'use linear wavelength spacing' : parameters_[ 'evenly_space_in_wavelength' ],
		'wavelength start' : parameters_[ 'wavelength_min_um' ] * 1e-6,
		'wavelength stop' : parameters_[ 'wavelength_max_um' ] * 1e-6,
		'frequency points' : parameters_[ 'monitor_num_wavelength_points' ]
	} ) ]

	simulation_monitors[ 'volumetric_device_monitor' ] = fdtd_hook.addprofile( {
		'name' : 'volumetric_device_monitor',
		'x span' : parameters_[ 'device_width_um' ] * 1e-6,
		'x' : 0,
		'y min' : parameters_[ 'device_min_y_um' ] * 1e-6,
		'y max' : parameters_[ 'device_max_y_um' ] * 1e-6,
		'override global monitor settings' : 1,
		'use linear wavelength spacing' : parameters_[ 'evenly_space_in_wavelength' ],
		'wavelength start' : parameters_[ 'wavelength_min_um' ] * 1e-6,
		'wavelength stop' : parameters_[ 'wavelength_max_um' ] * 1e-6,
		'frequency points' : parameters_[ 'monitor_num_wavelength_points' ],
		'output Hx' : 0,
		'output Hy' : 0,
		'output Hz' : 0
	} )

	simulation_monitors[ 'device_index_monitor' ] = fdtd_hook.addindex( {
		'name' : 'device_index_monitor',
		'x span' : parameters_[ 'device_width_um' ] * 1e-6,
		'y min' : parameters_[ 'device_min_y_um' ] * 1e-6,
		'y max' : parameters_[ 'device_max_y_um' ] * 1e-6
	} )


	return simulation_monitors



def scale_density( rho, parameters_ ):
	return parameters_[ "minimum_device_index" ] + ( parameters_[ "maximum_device_index" ] - parameters_[ "minimum_device_index" ] ) * rho

transmission_continuous_parameters = {
	"wavelength_min_um" : 0.4,
	"wavelength_max_um" : 0.7,
	"device_width_um" : 4 * 0.6,
	"device_min_y_um" : 0,
	"device_max_y_um" : 3 * 0.6,
	"evenly_space_in_wavelength" : 1,
	"monitor_num_wavelength_points" : 60,
	"adjoint_src_vertical_um" : -3 * 0.55,
	"forward_waist_radius_um" : 1.5 * 0.55,
	"forward_src_vertical_um" : 3 * 0.55 + 2 * 0.55,
	"simulation_width_um" : 4 * 0.6 + 4 * 0.55,
	"simulation_min_y_um" : -3 * 0.55 - 2 * 0.55,
	"simulation_max_y_um" : 3 * 0.6 + 2 * 0.55 + 2 * 0.55,
	"simulation_time_fs" : 900,
	"background_index" : 1,
	"mesh_spacing_um" : 0.4 * 0.05,
	"minimum_feature_size_um" : 0.4 * 0.05 * 3,
	"layer_height_um" : 0.4 * 0.05 * 3,
	"minimum_device_index" : 1.5,
	"maximum_device_index" : 2.0
}



simulation_engine = SimulationEngine( transmission_parameters, create_geometry_transmission, create_forward_sources_transmission, create_adjoint_sources_transmission, create_monitors_transmission )
simulation_engine.init_geometry( transmission_parameters )
simulation_engine.init_all_sources( transmission_parameters )

	# def run_simulations( self, volumetric_monitor_name_, forward_observation_keys_ ):


device = Device2D.Device2D( transmission_parameters, scale_density )


#
# Run finite difference test
#


	# def import_device( self, fdtd_hook_, simulation_geometry ):

device.init_design_uniform( 0.5 )

get_design = device.get_design()

simulation_engine.import_device( device )


		return 


forward_volumetric_efields, forward_observation_efields, adjoint_volumetric_efields = simulation_engine.run_simulations()

# def compute_fom_and_gradient( forward_volumetric_efields_, forward_observation_efields_, adjoint_volumetric_efields_, parameters_, do_performance_weighting=True, just_fom=False ):


fom, gradient = compute_fom_and_gradient( forward_volumetric_efields, forward_observation_efields, adjoint_volumetric_efields, transmission_parameters, False, False )

adj_grad = device.backpropagate_gradient( gradient )


num_fd_points = 20
trace_x_idx = get_design.shape[ 0 ] // 2
trace_y_idxs = np.arange( 0, num_fd_points )
h = 1e-3

fd_grad = np.zeros( num_fd_points )

for y_idx in range( 0, len( trace_y_idxs ) ):

	get_design_up = get_design.copy()
	get_design_up[ trace_x_idx, trace_y_idxs[ y_idx ] ] += h

	device.init_design_with_design( get_design_up )

	forward_volumetric_efields, forward_observation_efields, adjoint_volumetric_efields = simulation_engine.run_simulations()
	fom_up = compute_fom_and_gradient( forward_volumetric_efields, forward_observation_efields, adjoint_volumetric_efields, transmission_parameters, False, True )


	get_design_down = get_design.copy()
	get_design_down[ trace_x_idx, trace_y_idxs[ y_idx ] ] -= h

	device.init_design_with_design( get_design_down )

	forward_volumetric_efields, forward_observation_efields, adjoint_volumetric_efields = simulation_engine.run_simulations()
	fom_down = compute_fom_and_gradient( forward_volumetric_efields, forward_observation_efields, adjoint_volumetric_efields, transmission_parameters, False, True )


	fd_grad = ( fom_up - fom_down ) / ( 2 * h )




np.save( "adj_grad.npy", adj_grad )
np.save( "fd_grad.npy", fd_grad )





