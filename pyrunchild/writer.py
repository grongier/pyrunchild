################################################################################
# Imports
################################################################################

import os
from collections import OrderedDict
import numpy as np
from scipy import stats

from pyrunchild.manager import DataManager
from pyrunchild.utils import divide_line, rename_old_file, RangeModel, MixtureModel, MemoryModel, LinearTimeSeries, FloodplainTimeSeries

################################################################################
# InputWriter
################################################################################

class InputWriter(DataManager):
    
    def __init__(self,
                 base_directory,
                 base_name,
                 nb_realizations=1,
                 init_realization_nb=None,
                 preset_parameters=True,
                 seed=100):

        DataManager.__init__(self, base_directory, base_name)
        self.out_file_name = base_name
        
        self.nb_realizations = nb_realizations
        self.init_realization_nb = init_realization_nb
        self.parameters = OrderedDict()
        if preset_parameters == True:
            self.set_run_control()
            self.set_mesh()
            self.set_boundaries()
            self.set_bedrock()
            self.set_lithology()
            self.set_layers()
            self.set_stratigraphic_grid()
            self.set_tectonics()
            self.set_rainfall()
            self.set_runoff()
            self.set_hydraulic_geometry()
            self.set_meandering()
            self.set_materials()
            self.set_grainsize()
            self.set_fluvial_transport()
            self.set_overbank_deposition()
            self.set_hillslope_transport()
            self.set_landsliding()
            self.set_eolian_deposition()
            self.set_weathering()
            self.set_vegetation()
            self.set_forest()
            self.set_fire()
            self.set_various()
        self.parameter_values = [OrderedDict() for r in range(self.nb_realizations)]
        self.parameter_descriptions = self.build_parameter_descriptions()

        self.seed = seed

        self.base_names = [None for r in range(self.nb_realizations)]
        
    def build_parameter_descriptions(self):
        
        parameter_descriptions = dict()
        # Run control
        parameter_descriptions['OUTFILENAME'] = 'Base name for output files'
        parameter_descriptions['RUNTIME'] = '(yr) Duration of run'
        parameter_descriptions['OPINTRVL'] = '(yr) Frequency of output to files'
        parameter_descriptions['SEED'] = 'Seed for random number generation. Must be an integer'
        parameter_descriptions['FSEED'] = 'Seed for random number generation. Must be an integer'
        # Mesh setup
        parameter_descriptions['OPTREADINPUT'] = 'Option for initial mesh input or generation. Options include creating a mesh from scratch (10), reading an existing mesh (1), reading in a set of (x,y,z,b) points (where b is a boundary code) (12), and reading from an ArcInfo grid (3 or 4). If OPTREADINPUT=10, additional required parameters are X GRID_SIZE, Y GRID_SIZE, OPT_PT_PLACE, GRID_SPACING. If OPTREADINPUT=1, additional required parameters are INPUTDATAFILE, INPUTTIME, and OPTINITMESHDENS. If OPTREADINPUT=12, the parameter POINTFILENAME must also be included'
        parameter_descriptions['OPTINITMESHDENS'] = 'Option for densifying the initial mesh by inserting a new node at the circumcenter of each triangle. The value of this parameter is the number of successive densification passes (for example, if 2, then the mesh is densified twice)'
        parameter_descriptions['X_GRID_SIZE'] = '(m) Total length of model domain in x direction'
        parameter_descriptions['Y_GRID_SIZE'] = '(m) Total length of model domain in y direction'
        parameter_descriptions['OPT_PT_PLACE'] = 'Method of placing points when generating a new mesh: 0 = uniform hexagonal mesh; 1 = regular staggered (hexagonal) mesh with small random offsets in (x, y) positions; 2 = random placement'
        parameter_descriptions['GRID_SPACING'] = 'mean distance between grid nodes, meters'
        parameter_descriptions['NUM_PTS'] = 'Number of points in grid interior, if random point positions are used'
        parameter_descriptions['INPUTDATAFILE'] = 'Base name of files from which input data will be read, if option for reading input from a previous run is selected'
        parameter_descriptions['INPUTTIME'] = 'Time for which to read input, when re-starting from a previous run'
        parameter_descriptions['OPTREADLAYER'] = 'Option for reading layers from input file when generating new mesh. If set to zero, each node will be assigned a single bedrock layer and a single regolith layer, with thicknesses determined by REGINIT and BEDROCKDEPTH'
        parameter_descriptions['POINTFILENAME'] = 'Name of file containing (x,y,z,b) values for a series of points. Used when OPTREADINPUT = 2'
        parameter_descriptions['ARCGRIDFILENAME'] = 'Name of ascii file in ArcInfo format containing initial DEM'
        parameter_descriptions['TILE_INPUT_PATH'] = 'make irregular mesh from point tiles (files of x,y,z coords) for node coordinates and a regular Arc grid for masking a custom area'
        parameter_descriptions['OPT_TILES_OR_SINGLE_FILE'] = ''
        parameter_descriptions['LOWER_LEFT_EASTING'] = ''
        parameter_descriptions['LOWER_LEFT_NORTHING'] = ''
        parameter_descriptions['NUM_TILES_EAST'] = ''
        parameter_descriptions['NUM_TILES_NORTH'] = ''
        parameter_descriptions['OPTMESHADAPTDZ'] = 'If adaptive re-meshing is used, this option tells the model to add nodes at locations where the local volumetric erosion rate exceeds MESHADAPT_MAXNODEFLUX'
        parameter_descriptions['MESHADAPT_MAXNODEFLUX'] = 'For dynamic point addition: max ero flux rate'
        parameter_descriptions['OPTMESHADAPTAREA'] = 'Option for increasing mesh density around areas of large drainage area'
        parameter_descriptions['MESHADAPTAREA_MINAREA'] = 'For dynamic re-meshing based on drainage area: minimum drainage area for adaptive re-meshing'
        parameter_descriptions['MESHADAPTAREA_MAXVAREA'] = 'For dynamic re-meshing based on drainagearea: maximum Voronoi area for nodes meeting the minimum area criterion' 
        # Boundaries
        parameter_descriptions['TYP_BOUND'] = 'Configuration of boundaries with a rectangular mesh: 0 = open boundary in one corner; 1 = open boundary along x = 0; 2 = open boundaries along x = 0 and x = xmax; 3 = open boundaries along all four sides; 4 = single open boundary node at specified coordinates'
        parameter_descriptions['NUMBER_OUTLETS'] = ''
        parameter_descriptions['OUTLET_X_COORD'] = '(m) x coordinate of single-node outlet (open boundary)'
        parameter_descriptions['OUTLET_Y_COORD'] = '(m) y coordinate of single-node outlet (open boundary)'
        parameter_descriptions['MEAN_ELEV'] = '(m) Mean elevation of initial surface'
        parameter_descriptions['RAND_ELEV'] = '(m) Maximum amplitude of random variations in initial node elevations'
        parameter_descriptions['SLOPED_SURF'] = 'Option for initial sloping surface (downward toward y = 0)'
        parameter_descriptions['UPPER_BOUND_Z'] = '(m) If sloping initial surface is applied, this sets the slope by setting the altitude of the model edge at y = ymax'
        parameter_descriptions['OPTINLET'] = 'Option for an external water and sediment input at an inlet point'
        parameter_descriptions['INDRAREA'] = '(m2) For runs with an inlet: drainage area of inlet stream'
        parameter_descriptions['INSEDLOADi'] = '(m3/yr) For runs with an inlet and specified sediment influx: input sediment discharge of size fraction i'
        parameter_descriptions['INLET_X'] = '(m) For runs with an inlet: x position of the inlet'
        parameter_descriptions['INLET_Y'] = '(m) For runs with an inlet: y position of the inlet'
        parameter_descriptions['INLET_OPTCALCSEDFEED'] = 'For runs with an inlet: option for calculating sediment input at inlet based on specified slope (INLETSLOPE) and bed grain-size distribution'
        parameter_descriptions['INLET_SLOPE'] = 'For runs with an inlet: if option for calculating rather than specifying sediment discharge is chosen, this is the slope that is used to calculate sediment discharge'
        # Bedrock and regolith
        parameter_descriptions['BEDROCKDEPTH'] = '(m) Starting thickness of bedrock layer'
        parameter_descriptions['REGINIT'] = '(m) Starting thickness of regolith layer'
        parameter_descriptions['MAXREGDEPTH'] = '(m) Depth of active layer, and maximum thickness of a deposited layer'
        # Lithology
        parameter_descriptions['OPT_READ_LAYFILE'] = 'start with an existing .lay file'
        parameter_descriptions['INPUT_LAY_FILE'] = '.lay file'
        parameter_descriptions['OPT_READ_ETCHFILE'] = 'modify layers according to an Etch File. An Etch File specifies one or more layers, with given properties, to be "etched in" to the current topography and lithology'
        parameter_descriptions['ETCHFILE_NAME'] = 'Etch file'
        parameter_descriptions['OPT_SET_ERODY_FROM_FILE'] = 'set initial rock erodibility values at all depths based on values in a file'
        parameter_descriptions['ERODYFILE_NAME'] = 'Erodibility file'
        parameter_descriptions['OPT_NEW_LAYERSINPUT'] = 'Hack: make layers input backwards compatible for simulations without bulk density'
        # Layers
        parameter_descriptions['OPTLAYEROUTPUT'] = 'Option for output of layer data'
        parameter_descriptions['OPT_NEW_LAYERSOUTPUT'] = 'Hack: make backward compatible for sims without bulk density'
        parameter_descriptions['OPTINTERPLAYER'] = 'Option for layer interpolation when points are moved or added'
        # Stratigraphic grid
        parameter_descriptions['OPTSTRATGRID'] = 'Option for tracking stratigraphy using subjacent raster grid (only relevant when meandering and floodplain modules are activated; see Clevis et al., 2006b)'
        parameter_descriptions['XCORNER'] = 'Corner of stratigraphy grid in StratGrid module'
        parameter_descriptions['YCORNER'] = 'Corner of stratigraphy grid in StratGrid module'
        parameter_descriptions['GRIDDX'] = '(m) Grid spacing for StratGrid module'
        parameter_descriptions['GR_WIDTH'] = '(m) Stratigraphy grid width in StratGrid module'
        parameter_descriptions['GR_LENGTH'] = '(m) Stratigraphy grid length in StratGrid module'
        parameter_descriptions['SG_MAXREGDEPTH'] = '(m) Layer thickness in StratGrid module'
        # Tectonics and baselevel
        parameter_descriptions['OPTNOUPLIFT'] = 'Option to turn off uplift (default to false)'
        parameter_descriptions['UPTYPE'] = 'Type of uplift/baselevel change to be applied: 0 = None; 1 = Spatially and temporally uniform uplift; 2 = Uniform uplift at Y >= fault location, zero elsewhere; 3 = Block uplift with strike-slip motion along given Y coord; 4 = Propagating fold modeled w/ simple error function curve; 5 = 2D cosine-based uplift-subsidence pattern; 6 = Block, fault, and foreland sinusoidal fold; 7 = Two-sided differential uplift; 8 = Fault bend fold; 9 = Back-tilting normal fault block; 10 = Linear change in uplift rate; 11 = Power law change in uplift rate in the y-direction; 12 = Uplift rate maps in separate files; 13 = Propagating horizontal front; 14 = Baselevel fall at open boundaries; 15 = Moving block; 16 = Moving sinusoid; 17 = Uplift with crustal thickening; 18 = Uplift and whole-landscape tilting; 19 = Migrating Gaussian bump'
        parameter_descriptions['UPDUR'] = '(yr) Duration of uplift / baselevel change'
        parameter_descriptions['UPRATE'] = '(m/yr) Rate parameter for uplift routines (usage differs among different uplift functions)'
        parameter_descriptions['FAULTPOS'] = '(m) y location of a fault perpendicular to the x-axis'
        parameter_descriptions['SUBSRATE'] = '(m/yr) Subsidence rate (used for some uplift functions)'
        parameter_descriptions['SLIPRATE'] = '(m/yr) Tectonic parameter: rate of strike-slip motion (option 3), dip-slip motion (option 8)'
        parameter_descriptions['SS_OPT_WRAP_BOUNDARIES'] = ''
        parameter_descriptions['SS_BUFFER_WIDTH'] = ''
        parameter_descriptions['FOLDPROPRATE'] = '(m/yr) Uplift option 4: propagation rate of a fold'
        parameter_descriptions['FOLDWAVELEN'] = '(m) Uplift options 4, 5, 6: fold wavelength'
        parameter_descriptions['TIGHTENINGRATE'] = 'Uplift option 5: rate at which fold tightens'
        parameter_descriptions['ANTICLINEXCOORD'] = '(m) Uplift option 5: xcoordinate of anticline crest'
        parameter_descriptions['ANTICLINEYCOORD'] = '(m) Uplift option 5: ycoordinate of anticline crest'
        parameter_descriptions['YFOLDINGSTART'] = '(yr) Uplift option 5: starting time of fold deformation'
        parameter_descriptions['UPSUBRATIO'] = 'Uplift option 5: uplift-subsidence ratio'
        parameter_descriptions['FOLDLATRATE'] = 'Uplift option 6: lateral propagation rate of fold'
        parameter_descriptions['FOLDUPRATE'] = '(m/yr) Uplift option 6: uplift rate of fold axis'
        parameter_descriptions['FOLDPOSITION'] = '(m) Uplift option 6: position coordinate for fold'
        parameter_descriptions['BLFALL_UPPER'] = '(m/yr) Uplift option 7: rate of baselevel fall at upper (y=ymax) boundary'
        parameter_descriptions['BLDIVIDINGLINE'] = '''(m) Uplift option 7: ycoordinate that separates the two zones of baselevel fall. Open boundary nodes with y greater than this value are given the "upper" rate'''
        parameter_descriptions['FLATDEPTH'] = '(m) Uplift option 8: depth to flat portion of fault plane'
        parameter_descriptions['RAMPDIP'] = 'Uplift option 8: dip of fault ramp'
        parameter_descriptions['KINKDIP'] = 'Uplift option 8: dip of fault kink in fault-bend fold model'
        parameter_descriptions['UPPERKINKDIP'] = ''
        parameter_descriptions['ACCEL_REL_UPTIME'] = 'Uplift option 9: fraction of total time that fault motion has been accelerated'
        parameter_descriptions['VERTICAL_THROW'] = '(m) Uplift option 9: total fault throw'
        parameter_descriptions['FAULT_PIVOT_DISTANCE'] = '(m) Uplift option 9: distance from normal fault to pivot point'
        parameter_descriptions['MINIMUM_UPRATE'] = '(m/yr) Uplift option 10: minimum uplift rate'
        parameter_descriptions['OPT_INCREASE_TO_FRONT'] = 'Uplift option 10: option for having uplift rate increase (rather than decrease) toward y = 0'
        parameter_descriptions['DECAY_PARAM_UPLIFT'] = 'Uplift option 11: decay parameter for power-law uplift function'
        parameter_descriptions['NUMUPLIFTMAPS'] = 'Uplift option 12: number of uplift rate maps to read from file'
        parameter_descriptions['UPMAPFILENAME'] = 'Uplift option 12: base name of files containing uplift rate fields'
        parameter_descriptions['UPTIMEFILENAME'] = 'Uplift option 12: name of file containing times corresponding to each uplift rate map'
        parameter_descriptions['FRONT_PROP_RATE'] = '(m/yr) Uplift option 13: rate of horizontal propagation of deformation front'
        parameter_descriptions['UPLIFT_FRONT_GRADIENT'] = 'Uplift option 13: this defines the azimuth of the uplift front. If zero, the front is parallel to the x-axis. If positive, it angles away from the open boundary (if there is one). The idea is that this captures (crudely) the north-to-south propagation of wedge growth in Taiwan'
        parameter_descriptions['STARTING_YCOORD'] = '(m) Uplift option 13: y coordinate at which propagating deformation front starts'
        parameter_descriptions['BLOCKEDGEPOSX'] = ''
        parameter_descriptions['BLOCKWIDTHX'] = ''
        parameter_descriptions['BLOCKEDGEPOSY'] = ''
        parameter_descriptions['BLOCKWIDTHY'] = ''
        parameter_descriptions['BLOCKMOVERATE'] = ''
        parameter_descriptions['TILT_RATE'] = ''
        parameter_descriptions['TILT_ORIENTATION'] = ''
        parameter_descriptions['BUMP_MIGRATION_RATE'] = ''
        parameter_descriptions['BUMP_INITIAL_POSITION'] = ''
        parameter_descriptions['BUMP_AMPLITUDE'] = ''
        parameter_descriptions['BUMP_WAVELENGTH'] = ''
        parameter_descriptions['OPT_INITIAL_BUMP'] = ''
        # Rainfall
        parameter_descriptions['OPTVAR'] = 'Option for random rainfall variation'
        parameter_descriptions['ST_PMEAN'] = '(m/yr) Mean storm rainfall intensity (16.4 m/yr = Atlanta, GA)'
        parameter_descriptions['ST_STDUR'] = '(yr) Mean storm duration (Denver July = 0.00057yrs = 5 hrs)'
        parameter_descriptions['ST_ISTDUR'] = '(yr) Mean time between storms (Denver July = 0.01yr = 88hrs)'
        parameter_descriptions['ST_OPTSINVAR'] = 'option for sinusoidal variations'
        parameter_descriptions['OPTSINVARINFILT'] = 'Option for sinusoidal variations through time in soil in- filtration capacity'
        # Runoff and infiltration
        parameter_descriptions['FLOWGEN'] = '''Runoff generation option: 0. Hortonian (uniform infilt-excess runoff); 1. Saturated flow 1 (sat-excess runoff w/ return flow); 2. Saturated flow 2 (sat-excess runoff w/o return flow); 3. Constant soil store ("bucket"-type flow generation); 4. 2D kinematic wave (2D steady kinematic wave multi-flow); 5. Hydrograph peak method; 6 Subsurface 2D kinematic wave (kinematic wave with Darcy's Law)'''
        parameter_descriptions['TRANSMISSIVITY'] = '(m2/yr) For subsurface flow options: soil hydraulic transmissivity.'
        parameter_descriptions['OPTVAR_TRANSMISSIVITY'] = ''
        parameter_descriptions['INFILTRATION'] = '(Ic, m/yr) Soil infiltration capacity'
        parameter_descriptions['OPTSINVARINFILT'] = ''
        parameter_descriptions['PERIOD_INFILT'] = '(yr) Period for sinusoidal variations in soil infiltration capacity'
        parameter_descriptions['MAXICMEAN'] = 'Maximum value of sinusoidally varying soil infiltration capacity'
        parameter_descriptions['SOILSTORE'] = '''(m) For "bucket" hydrology sub-model: soil water storage capacity'''
        parameter_descriptions['KINWAVE_HQEXP'] = 'For kinematic wave water-routing module: exponent on depth-discharge relationship'
        parameter_descriptions['FLOWVELOCITY'] = 'For peak hydrograph method of flow calculation: speed of channel flow (used to compute travel time; see Solyom and Tucker, 2004)'
        parameter_descriptions['HYDROSHAPEFAC'] = 'For hydrograph peak flow-calculation method: hydrograph shape factor (see Solyom and Tucker, 2004)'
        parameter_descriptions['LAKEFILL'] = 'Option for computing inundated area and drainage pathways in closed depressions (see Tucker et al.,  2001b). If not selected, any water entering a closed depression is assumed to evaporate'
        # Hydraulic geometry
        parameter_descriptions['CHAN_GEOM_MODEL'] = 'Type of channel geometry model to be used. Option 1 is standard empirical hydraulic geometry. Other options are experimental: 1. Regime theory (empirical power-law scaling); 2. Parker-Paola self-formed channel theory; 3. Finnegan slope-dependent channel width model'
        parameter_descriptions['HYDR_WID_COEFF_DS'] = 'Coefficient in bankfull width-discharge relation'
        parameter_descriptions['HYDR_WID_EXP_DS'] = 'Exponent in bankfull width-discharge relation'
        parameter_descriptions['HYDR_WID_EXP_STN'] = 'Exponent in at-a-station width-discharge relation'
        parameter_descriptions['HYDR_DEP_COEFF_DS'] = 'Coefficient in bankfull depth-discharge relation'
        parameter_descriptions['HYDR_DEP_EXP_DS'] = 'Exponent in bankfull depth-discharge relation'
        parameter_descriptions['HYDR_DEP_EXP_STN'] = 'Exponent in at-a-station depth-discharge relation'
        parameter_descriptions['HYDR_ROUGH_COEFF_DS'] = 'Coefficient in bankfull roughness-discharge relation'
        parameter_descriptions['HYDR_ROUGH_EXP_DS'] = 'Exponent in bankfull roughness-discharge relation'
        parameter_descriptions['HYDR_ROUGH_EXP_STN'] = 'Exponent in at-a-station roughness-discharge relation'
        parameter_descriptions['HYDR_SLOPE_EXP'] = ''
        parameter_descriptions['THETAC'] = '''For "Parker" channel geometry option: critical Shields stress'''
        parameter_descriptions['SHEAR_RATIO'] = '''For "Parker" channel geometry option: ratio of actual to threshold shear stress'''
        parameter_descriptions['BANK_ROUGH_COEFF'] = 'Coefficient in bank roughness-discharge relation'
        parameter_descriptions['BANK_ROUGH_EXP'] = 'Exponent in bank roughness-discharge relation'
        parameter_descriptions['BANKFULLEVENT'] = 'Runoff rate associated with bankfull flood event. Used to compute hydraulic geometry'
        # Meandering
        parameter_descriptions['OPTMEANDER'] = 'Option for stream meandering'
        parameter_descriptions['CRITICAL_AREA'] = '(m2) Minimum drainage area for a meandering channel in stream meander module'
        parameter_descriptions['CRITICAL_FLOW'] = '(m3/yr) Minimum flow for which we calculate meandering in stream meander module'
        parameter_descriptions['OPT_VAR_SIZE'] = 'Flag that indicates use of multiple grain sizes in stream meander module'
        parameter_descriptions['MEDIAN_DIAMETER'] = '(m) Median bed-sediment diameter for use in meander module'
        parameter_descriptions['BANKERO'] = 'Stream meander module: stream-bank erodibility coefficient'
        parameter_descriptions['BNKHTDEP'] = 'Stream meander module: degree to which bank erosion rate depends on bank height (0 to 1)'
        parameter_descriptions['DEF_CHAN_DISCR'] = '(m) Default channel node spacing in meander module'
        parameter_descriptions['FRAC_WID_MOVE'] = 'Stream meander module: maximum distance that a meandering channel point can migrate in one time step, in channel widths'
        parameter_descriptions['FRAC_WID_ADD'] = 'Stream meander module: maximum distance of a meandering channel point from a bank point, in channel widths. If exceeded, a new node is added'
        # Materials
        parameter_descriptions['ROCKDENSITYINIT'] = 'initial rock bulk density (kg/m3)'
        parameter_descriptions['SOILBULKDENSITY'] = 'bulk density of soil (constant) (kg/m3)'
        parameter_descriptions['WOODDENSITY'] = 'density of wood (kg/m3)'
        # Grain size
        parameter_descriptions['NUMGRNSIZE'] = 'Number of grain size classes used in run. Must be consistent with selected sediment transport law'
        parameter_descriptions['BRPROPORTIONi'] = 'Volumetric proportion of grain-size fraction i generated from eroded bedrock. Enter one per size fraction, starting with 1'
        parameter_descriptions['REGPROPORTIONi'] = 'Initial volumetric proportion of size i in regolith layers. Must specify one value for each grain size class. The range is zero to one'
        parameter_descriptions['GRAINDIAMi'] = '(Di, m) Diameter of grain size class i. There must be a value corresponding to each grain-size class used in the run. For example, a run with two grain-size classes must have GRAINDIAM1 and GRAINDIAM2'
        parameter_descriptions['HIDINGEXP'] = 'Exponent in equation for correcting critical shear stress to account for protrusion and hiding when multiple grain-size fractions are present on the bed'
        parameter_descriptions['GRAINDIAM0'] = 'representative d50 grain size (if NUMGRNSIZE=1) [m]'  
        # Fluvial transport
        parameter_descriptions['OPTNOFLUVIAL'] = 'Option to turn off fluvial processes (default to false)'
        parameter_descriptions['DETACHMENT_LAW'] = 'Code for detachment-capacity law to be applied: 0 = power law, form 1; 1 = power law, form 2; 2 = almost parabolic law; 3 = generalized f(Qs) detachment-rule; 4 = dummy law for no fluvial erosion'
        parameter_descriptions['KB'] = 'Erodibility coefficient for bedrock. If layers are read in from a previous run, values from layer file are used instead'
        parameter_descriptions['KR'] = 'Erodibility coefficient for regolith. If layers are read in from a previous run, values from layer file are used instead'
        parameter_descriptions['KT'] = '(Pa per (m2/s)M, where M is Mb for detachment and Mf for sediment transport) Coefficient relating shear stress to discharge and slope. Can be calculated from water density, gravitational acceleration, and roughness; see, e.g., Tucker and Slingerland (1997)'
        parameter_descriptions['MB'] = 'Discharge exponent in detachment capacity equation'
        parameter_descriptions['NB'] = 'Slope exponent in detachment capacity equation'
        parameter_descriptions['PB'] = 'Excess power/shear exponent in detachment capacity equation'
        parameter_descriptions['TAUCB'] = '(normally Pa) Detachment threshold for bedrock'
        parameter_descriptions['TAUCR'] = '(normally Pa) Detachment threshold for regolith'
        parameter_descriptions['BETA'] = 'Fraction of eroded sediment that forms bed load. Applies only to sediment-flux-dependent detachment laws'
        parameter_descriptions['OPTDETACHLIM'] = 'Option for detachment-limited fluvial erosion only'
        parameter_descriptions['TRANSPORT_LAW'] = 'Code for fluvial transport capacity law to be applied: 0 = power-law transport formula; 1 = power-law transport formula, form 2; 2 = Bridge-Dominic form of Bagnold bedload formula; 3 = Wilcock sand-gravel formula; 4 = multi-size power-law formula; 5 = Willgoose/Riley mine tailings formula; 6 = ultra-simplified power-law transport formula; 7 = dummy law for no fluvial transport'
        parameter_descriptions['KF'] = 'Fluvial sediment transport efficiency coefficient'
        parameter_descriptions['MF'] = 'Discharge exponent in fluvial transport capacity equation'
        parameter_descriptions['NF'] = 'Slope exponent in fluvial transport capacity equation'
        parameter_descriptions['PF'] = 'Excess power/shear exponent in fluvial transport capacity equation'
        # Overbank deposition
        parameter_descriptions['OPTFLOODPLAIN'] = 'Option for floodplain over-bank deposition'
        parameter_descriptions['FP_DRAREAMIN'] = '''(m2) In floodplain module, the minimum drainage area that defines a "major" channel that is subject to overbank flooding and sedimentation'''
        parameter_descriptions['FP_BANKFULLEVENT'] = '(m/yr) In floodplain module, the minimum runoff rate required to generate a flood'
        parameter_descriptions['FP_MU'] = '(μ, m/yr) In floodplain module, the rate coefficient for overbank sedimentation (see Clevis et al., 2006a)'
        parameter_descriptions['FP_LAMBDA'] = '(λ, m) In floodplain module, the distance decay coefficient for sedimentation rate (e-folding length for sedimentation rate as a function of distance from the main channel)'
        parameter_descriptions['FP_OPTCONTROLCHAN'] = 'When the floodplain module is used, setting this option tells the model to drive the altitude of the main channel as a boundary condition.  See Clevis et al. (2006a)'
        parameter_descriptions['FP_VALDROP'] = '(m) In floodplain module, the difference in altitude of the main channel between its inlet and its exit point'
        parameter_descriptions['FP_INLET_ELEVATION'] = '(m) In floodplain module, the altitude of the inlet of the main channel'
        # Hillslope transport
        parameter_descriptions['OPTNODIFFUSION'] = 'Option to turn off diffusive processes (default to false)'
        parameter_descriptions['KD'] = '(m2/yr) Hillslope diffusivity coefficient'
        parameter_descriptions['OPTDIFFDEP'] = 'Option to deactivate deposition by hillslope diffusion'
        parameter_descriptions['DIFFUSIONTHRESHOLD'] = 'When this parameter is greater than zero, it is the drainage area above which slope-dependent (“diffusive”) creep transport no longer takes place. Designed for use with sediment-flux-dependent transport functions; see Gasparini et al. (2007)'
        parameter_descriptions['OPT_NONLINEAR_DIFFUSION'] = 'Option for nonlinear diffusion model of soil creep'
        parameter_descriptions['OPT_DEPTH_DEPENDENT_DIFFUSION'] = 'Option for depth dependent creep transport'
        parameter_descriptions['DIFFDEPTHSCALE'] = 'depth scale for depth-dependent diffusion'
        parameter_descriptions['CRITICAL_SLOPE'] = 'Threshold slope gradient for nonlinear creep law'
        # Landsliding
        parameter_descriptions['OPT_LANDSLIDES'] = 'Option for landsliding'
        parameter_descriptions['OPT_3D_LANDSLIDES'] = 'Option for determining which landslide function to use'
        parameter_descriptions['FRICSLOPE'] = 'tangent of angle of repose for soil (unitless)'
        parameter_descriptions['DF_RUNOUT_RULE'] = 'set runout rules'
        parameter_descriptions['DF_SCOUR_RULE'] = 'set scour rules'
        parameter_descriptions['DF_DEPOSITION_RULE'] = 'set deposition rules'
        # Eolian
        parameter_descriptions['OPTLOESSDEP'] = 'space-time uniform surface accumulation of sediment (loess)'
        parameter_descriptions['LOESS_DEP_RATE'] = '(m/yr) Rate of accumulation of aeolian sediment across the landscape'
        # Chemical and physical weathering
        parameter_descriptions['CHEM_WEATHERING_LAW'] = 'possible values 0-1: 0 = None; 1 = Dissolution'
        parameter_descriptions['MAXDISSOLUTIONRATE'] = 'maximum dissolution rate (kg/m3/yr)'
        parameter_descriptions['CHEMDEPTH'] = 'depth scale for dissolution (m)'
        parameter_descriptions['PRODUCTION_LAW'] = 'possible values 0-2: 0 = None; 1 = exponential law; 2 = exp. with density dep.'
        parameter_descriptions['SOILPRODRATE'] = 'uniform and constant soil production rate for zero soil depth for exponential law (m/yr)'
        parameter_descriptions['SOILPRODRATEINTERCEPT'] = 'density-dependent soil production rate intercept (m/yr)'
        parameter_descriptions['SOILPRODRATESLOPE'] = 'density-dependent soil production rate slope ( (m/yr)/(kg/m3) )'
        parameter_descriptions['SOILPRODDEPTH'] = 'depth scale for soil production rate (m)'
        # Vegetation
        parameter_descriptions['OPTVEG'] = 'Option for dynamic vegetation layer (see Collins et al., 2004)'
        parameter_descriptions['OPTGRASS_SIMPLE'] = 'option for simple grass'
        parameter_descriptions['VEG_KVD'] = 'Vegetation erosion coefficient (dims LT/M)'
        parameter_descriptions['VEG_TV'] = 'Vegetation regrowth time scale (years)'
        parameter_descriptions['TAUC'] = 'Erosion threshold on bare soil'
        parameter_descriptions['VEG_TAUCVEG'] = 'Erosion threshold under 100% cover'
        # Forest
        parameter_descriptions['OPTFOREST'] = 'option for basic forest evolution'
        parameter_descriptions['OPTFOREST'] = ''
        parameter_descriptions['ROOTDECAY_K'] = ''
        parameter_descriptions['ROOTDECAY_N'] = ''
        parameter_descriptions['ROOTGROWTH_A'] = ''
        parameter_descriptions['ROOTGROWTH_B'] = ''
        parameter_descriptions['ROOTGROWTH_C'] = ''
        parameter_descriptions['ROOTGROWTH_F'] = ''
        parameter_descriptions['ROOTSTRENGTH_J'] = ''
        parameter_descriptions['MAXVERTROOTCOHESION'] = ''
        parameter_descriptions['MAXLATROOTCOHESION'] = ''
        parameter_descriptions['TREEHEIGHTINDEX'] = ''
        parameter_descriptions['VEGWEIGHT_MAX'] = ''
        parameter_descriptions['VEGWEIGHT_A'] = ''
        parameter_descriptions['VEGWEIGHT_B'] = ''
        parameter_descriptions['VEGWEIGHT_C'] = ''
        parameter_descriptions['VEGWEIGHT_K'] = ''
        parameter_descriptions['BLOWDOWNPARAM'] = ''
        parameter_descriptions['BLOW_SEED'] = ''
        parameter_descriptions['TREEDIAM_B0'] = ''
        parameter_descriptions['TREEDIAM_B1'] = ''
        parameter_descriptions['TREEDIAM_B2'] = ''
        parameter_descriptions['WOODDECAY_K'] = ''
        parameter_descriptions['INITSTANDAGE'] = ''
        # Fire
        parameter_descriptions['OPTFIRE'] = 'option for random fires assuming an exponential distribution of time to the next fire'
        parameter_descriptions['IFRDUR'] = 'Mean time between fires'
        parameter_descriptions['OPTRANDOMFIRES'] = 'random fires'
        # Various options
        parameter_descriptions['OPTTSOUTPUT'] = 'Option for output of quantities at each storm (time step)'
        parameter_descriptions['TSOPINTRVL'] = 'not currently operational'
        parameter_descriptions['SURFER'] = 'Option for output in a Surfer-compatible data format'
        parameter_descriptions['OPTEXPOSURETIME'] = 'option for tracking surface-layer exposure ages'
        parameter_descriptions['OPTFOLDDENS'] = 'Option for mesh densification around a growing fold'
        parameter_descriptions['OPT_TRACK_WATER_SED_TIMESERIES'] = 'Option to record timeseries Q and Qs'
        parameter_descriptions['OPT_FREEZE_ELEVATIONS'] = ''
        parameter_descriptions['OPTSTREAMLINEBNDY'] = 'Option for converting streamlines to open boundaries'
        
        return parameter_descriptions
    
    def print_parameter_descriptions(self):
        
        length = 0
        for parameter in self.parameter_descriptions:
            if len(parameter) > length:
                length = len(parameter)
                
        for parameter in self.parameter_descriptions:
            print(parameter,
                  (length - len(parameter))*' ',
                  ' : ',
                  self.parameter_descriptions[parameter],
                  sep='')
            
    def set_run_control(self,
                        OUTFILENAME=None,
                        DESCRIPTION='',
                        RUNTIME=100000,
                        OPINTRVL=1000,
                        SEED=100,
                        FSEED=100):
        
        if OUTFILENAME is None:
            OUTFILENAME = self.out_file_name
        self.parameters['OUTFILENAME'] = OUTFILENAME
        self.DESCRIPTION = DESCRIPTION
        self.parameters['RUNTIME'] = RUNTIME
        self.parameters['OPINTRVL'] = OPINTRVL
        self.parameters['SEED'] = SEED
        self.parameters['FSEED'] = FSEED
        
    def set_mesh(self,
                 OPTREADINPUT=10,
                 OPTINITMESHDENS=0,
                 X_GRID_SIZE=10000,
                 Y_GRID_SIZE=10000,
                 OPT_PT_PLACE=1,
                 GRID_SPACING=200,
                 NUM_PTS='n/a',
                 INPUTDATAFILE='n/a',
                 INPUTTIME='n/a',
                 OPTREADLAYER=0,
                 POINTFILENAME='n/a',
                 ARCGRIDFILENAME='n/a',
                 TILE_INPUT_PATH='n/a',
                 OPT_TILES_OR_SINGLE_FILE=0,
                 LOWER_LEFT_EASTING='n/a',
                 LOWER_LEFT_NORTHING='n/a',
                 NUM_TILES_EAST='n/a',
                 NUM_TILES_NORTH='n/a',
                 OPTMESHADAPTDZ=0,
                 MESHADAPT_MAXNODEFLUX='n/a',
                 OPTMESHADAPTAREA=0,
                 MESHADAPTAREA_MINAREA='n/a',
                 MESHADAPTAREA_MAXVAREA='n/a'):
        
        self.parameters['OPTREADINPUT'] = OPTREADINPUT
        self.parameters['OPTINITMESHDENS'] = OPTINITMESHDENS
        self.parameters['X_GRID_SIZE'] = X_GRID_SIZE
        self.parameters['Y_GRID_SIZE'] = Y_GRID_SIZE
        self.parameters['OPT_PT_PLACE'] = OPT_PT_PLACE
        self.parameters['GRID_SPACING'] = GRID_SPACING
        self.parameters['NUM_PTS'] = NUM_PTS
        self.parameters['INPUTDATAFILE'] = INPUTDATAFILE
        self.parameters['INPUTTIME'] = INPUTTIME
        self.parameters['OPTREADLAYER'] = OPTREADLAYER
        self.parameters['POINTFILENAME'] = POINTFILENAME
        self.parameters['ARCGRIDFILENAME'] = ARCGRIDFILENAME
        self.parameters['TILE_INPUT_PATH'] = TILE_INPUT_PATH
        self.parameters['OPT_TILES_OR_SINGLE_FILE'] = OPT_TILES_OR_SINGLE_FILE
        self.parameters['LOWER_LEFT_EASTING'] = LOWER_LEFT_EASTING
        self.parameters['LOWER_LEFT_NORTHING'] = LOWER_LEFT_NORTHING
        self.parameters['NUM_TILES_EAST'] = NUM_TILES_EAST
        self.parameters['NUM_TILES_NORTH'] = NUM_TILES_NORTH
        self.parameters['OPTMESHADAPTDZ'] = OPTMESHADAPTDZ
        self.parameters['MESHADAPT_MAXNODEFLUX'] = MESHADAPT_MAXNODEFLUX
        self.parameters['OPTMESHADAPTAREA'] = OPTMESHADAPTAREA
        self.parameters['MESHADAPTAREA_MINAREA'] = MESHADAPTAREA_MINAREA
        self.parameters['MESHADAPTAREA_MAXVAREA'] = MESHADAPTAREA_MAXVAREA
        
    def set_boundaries(self,
                       TYP_BOUND=1,
                       NUMBER_OUTLETS=0,
                       OUTLET_X_COORD='n/a',
                       OUTLET_Y_COORD='n/a',
                       MEAN_ELEV=0,
                       RAND_ELEV=1,
                       SLOPED_SURF=0,
                       UPPER_BOUND_Z=0,
                       OPTINLET=0,
                       INDRAREA='n/a',
                       INSEDLOADi=(0, 0, 0, 0, 0, 0, 0, 0, 0),
                       INLET_X='n/a',
                       INLET_Y='n/a',
                       INLET_OPTCALCSEDFEED='n/a',
                       INLET_SLOPE='n/a'):
        
        self.parameters['TYP_BOUND'] = TYP_BOUND
        self.parameters['NUMBER_OUTLETS'] = NUMBER_OUTLETS
        self.parameters['OUTLET_X_COORD'] = OUTLET_X_COORD
        self.parameters['OUTLET_Y_COORD'] = OUTLET_Y_COORD
        self.parameters['MEAN_ELEV'] = MEAN_ELEV
        self.parameters['RAND_ELEV'] = RAND_ELEV
        self.parameters['SLOPED_SURF'] = SLOPED_SURF
        self.parameters['UPPER_BOUND_Z'] = UPPER_BOUND_Z
        self.parameters['OPTINLET'] = OPTINLET
        self.parameters['INDRAREA'] = INDRAREA
        for i, in_sed_load in enumerate(INSEDLOADi):
            self.parameters['INSEDLOAD' + str(i + 1)] = in_sed_load
        self.parameters['INLET_X'] = INLET_X
        self.parameters['INLET_Y'] = INLET_Y
        self.parameters['INLET_OPTCALCSEDFEED'] = INLET_OPTCALCSEDFEED
        self.parameters['INLET_SLOPE'] = INLET_SLOPE
        
    def set_bedrock(self,
                    BEDROCKDEPTH=1e10,
                    REGINIT=0,
                    MAXREGDEPTH=100):
        
        self.parameters['BEDROCKDEPTH'] = BEDROCKDEPTH
        self.parameters['REGINIT'] = REGINIT
        self.parameters['MAXREGDEPTH'] = MAXREGDEPTH
        
    def set_lithology(self,
                      OPT_READ_LAYFILE=0,
                      INPUT_LAY_FILE='n/a',
                      OPT_READ_ETCHFILE=0,
                      ETCHFILE_NAME='n/a',
                      OPT_SET_ERODY_FROM_FILE=0,
                      ERODYFILE_NAME='n/a',
                      OPT_NEW_LAYERSINPUT=0):
        
#         print('Lithology also requires ROCKDENSITYINIT and SOILBULKDENSITY (see material_parameters)')
        
        self.parameters['OPT_READ_LAYFILE'] = OPT_READ_LAYFILE
        self.parameters['INPUT_LAY_FILE'] = INPUT_LAY_FILE
        self.parameters['OPT_READ_ETCHFILE'] = OPT_READ_ETCHFILE
        self.parameters['ETCHFILE_NAME'] = ETCHFILE_NAME
        self.parameters['OPT_SET_ERODY_FROM_FILE'] = OPT_SET_ERODY_FROM_FILE
        self.parameters['ERODYFILE_NAME'] = ERODYFILE_NAME
        self.parameters['OPT_NEW_LAYERSINPUT'] = OPT_NEW_LAYERSINPUT
        
    def set_layers(self,
                   OPTLAYEROUTPUT=0,
                   OPT_NEW_LAYERSOUTPUT=0,
                   OPTINTERPLAYER=0):
        
        self.parameters['OPTLAYEROUTPUT'] = OPTLAYEROUTPUT
        self.parameters['OPT_NEW_LAYERSOUTPUT'] = OPT_NEW_LAYERSOUTPUT
        self.parameters['OPTINTERPLAYER'] = OPTINTERPLAYER
        
    def set_stratigraphic_grid(self,
                               OPTSTRATGRID=0,
                               XCORNER=0,
                               YCORNER=0,
                               GRIDDX=200,
                               GR_WIDTH=10000,
                               GR_LENGTH=10000,
                               SG_MAXREGDEPTH=100):
        
        self.parameters['OPTSTRATGRID'] = OPTSTRATGRID
        self.parameters['XCORNER'] = XCORNER
        self.parameters['YCORNER'] = YCORNER
        self.parameters['GRIDDX'] = GRIDDX
        self.parameters['GR_WIDTH'] = GR_WIDTH
        self.parameters['GR_LENGTH'] = GR_LENGTH
        self.parameters['SG_MAXREGDEPTH'] = SG_MAXREGDEPTH
        
    def set_tectonics(self,
                      OPTNOUPLIFT=0,
                      UPTYPE=1,
                      UPDUR=10000000,
                      UPRATE=0.001,
                      FAULTPOS=10000,
                      SUBSRATE='n/a',
                      SLIPRATE='n/a',
                      SS_OPT_WRAP_BOUNDARIES='n/a',
                      SS_BUFFER_WIDTH='n/a',
                      FOLDPROPRATE='n/a',
                      FOLDWAVELEN='n/a',
                      TIGHTENINGRATE='n/a',
                      ANTICLINEXCOORD='n/a',
                      ANTICLINEYCOORD='n/a',
                      YFOLDINGSTART='n/a',
                      UPSUBRATIO='n/a',
                      FOLDLATRATE='n/a',
                      FOLDUPRATE='n/a',
                      FOLDPOSITION='n/a',
                      BLFALL_UPPER='n/a',
                      BLDIVIDINGLINE=2000,
                      FLATDEPTH='n/a',
                      RAMPDIP='n/a',
                      KINKDIP='n/a',
                      UPPERKINKDIP='n/a',
                      ACCEL_REL_UPTIME=0.5,
                      VERTICAL_THROW=1100.0,
                      FAULT_PIVOT_DISTANCE=15000,
                      MINIMUM_UPRATE='n/a',
                      OPT_INCREASE_TO_FRONT=0,
                      DECAY_PARAM_UPLIFT='n/a',
                      NUMUPLIFTMAPS=0,
                      UPMAPFILENAME='n/a',
                      UPTIMEFILENAME='n/a',
                      FRONT_PROP_RATE=1,
                      UPLIFT_FRONT_GRADIENT=0.5,
                      STARTING_YCOORD=100000,
                      BLOCKEDGEPOSX='n/a',
                      BLOCKWIDTHX='n/a',
                      BLOCKEDGEPOSY='n/a',
                      BLOCKWIDTHY='n/a',
                      BLOCKMOVERATE='n/a',
                      TILT_RATE='n/a',
                      TILT_ORIENTATION='n/a',
                      BUMP_MIGRATION_RATE='n/a',
                      BUMP_INITIAL_POSITION='n/a',
                      BUMP_AMPLITUDE='n/a',
                      BUMP_WAVELENGTH='n/a',
                      OPT_INITIAL_BUMP=0):
        ''' Set the parameters for tectonics

        @param OPTNOUPLIFT: Option to turn off tectonics (default to false)
        @param UPTYPE: Code for the type of uplift to be applied:
                       0. None
                       1. Spatially and temporally uniform uplift
                       2. Uniform uplift at Y >= fault location, zero elsewhere
                       3. Block uplift with strike-slip motion along given Y coord
                       4. Propagating fold modeled w/ simple error function curve
                       5. 2D cosine-based uplift-subsidence pattern
                       6. Block, fault, and foreland sinusoidal fold
                       7. Two-sided differential uplift
                       8. Fault bend fold
                       9. Back-tilting normal fault block
                       10. Linear change in uplift rate
                       11. Power law change in uplift rate in the y-direction
                       12. Uplift rate maps in separate files
                       13. Propagating horizontal front
                       14. Baselevel fall at open boundaries
                       15. Moving block
                       16. Moving sinusoid
                       17. Uplift with crustal thickening
                       18. Uplift and whole-landscape tilting
                       19. Migrating Gaussian bump
        '''
        self.parameters['OPTNOUPLIFT'] = OPTNOUPLIFT
        self.parameters['UPTYPE'] = UPTYPE
        self.parameters['UPDUR'] = UPDUR
        self.parameters['UPRATE'] = UPRATE
        self.parameters['FAULTPOS'] = FAULTPOS
        self.parameters['SUBSRATE'] = SUBSRATE
        self.parameters['SLIPRATE'] = SLIPRATE
        self.parameters['SS_OPT_WRAP_BOUNDARIES'] = SS_OPT_WRAP_BOUNDARIES
        self.parameters['SS_BUFFER_WIDTH'] = SS_BUFFER_WIDTH
        self.parameters['FOLDPROPRATE'] = FOLDPROPRATE
        self.parameters['FOLDWAVELEN'] = FOLDWAVELEN
        self.parameters['TIGHTENINGRATE'] = TIGHTENINGRATE
        self.parameters['ANTICLINEXCOORD'] = ANTICLINEXCOORD
        self.parameters['ANTICLINEYCOORD'] = ANTICLINEYCOORD
        self.parameters['YFOLDINGSTART'] = YFOLDINGSTART
        self.parameters['UPSUBRATIO'] = UPSUBRATIO
        self.parameters['FOLDLATRATE'] = FOLDLATRATE
        self.parameters['FOLDUPRATE'] = FOLDUPRATE
        self.parameters['FOLDPOSITION'] = FOLDPOSITION
        self.parameters['BLFALL_UPPER'] = BLFALL_UPPER
        self.parameters['BLDIVIDINGLINE'] = BLDIVIDINGLINE
        self.parameters['FLATDEPTH'] = FLATDEPTH
        self.parameters['RAMPDIP'] = RAMPDIP
        self.parameters['KINKDIP'] = KINKDIP
        self.parameters['UPPERKINKDIP'] = UPPERKINKDIP
        self.parameters['ACCEL_REL_UPTIME'] = ACCEL_REL_UPTIME
        self.parameters['VERTICAL_THROW'] = VERTICAL_THROW
        self.parameters['FAULT_PIVOT_DISTANCE'] = FAULT_PIVOT_DISTANCE
        self.parameters['MINIMUM_UPRATE'] = MINIMUM_UPRATE
        self.parameters['OPT_INCREASE_TO_FRONT'] = OPT_INCREASE_TO_FRONT
        self.parameters['DECAY_PARAM_UPLIFT'] = DECAY_PARAM_UPLIFT
        self.parameters['NUMUPLIFTMAPS'] = NUMUPLIFTMAPS
        self.parameters['UPMAPFILENAME'] = UPMAPFILENAME
        self.parameters['UPTIMEFILENAME'] = UPTIMEFILENAME
        self.parameters['FRONT_PROP_RATE'] = FRONT_PROP_RATE
        self.parameters['UPLIFT_FRONT_GRADIENT'] = UPLIFT_FRONT_GRADIENT
        self.parameters['STARTING_YCOORD'] = STARTING_YCOORD
        self.parameters['BLOCKEDGEPOSX'] = BLOCKEDGEPOSX
        self.parameters['BLOCKWIDTHX'] = BLOCKWIDTHX
        self.parameters['BLOCKEDGEPOSY'] = BLOCKEDGEPOSY
        self.parameters['BLOCKWIDTHY'] = BLOCKWIDTHY
        self.parameters['BLOCKMOVERATE'] = BLOCKMOVERATE
        self.parameters['TILT_RATE'] = TILT_RATE
        self.parameters['TILT_ORIENTATION'] = TILT_ORIENTATION
        self.parameters['BUMP_MIGRATION_RATE'] = BUMP_MIGRATION_RATE
        self.parameters['BUMP_INITIAL_POSITION'] = BUMP_INITIAL_POSITION
        self.parameters['BUMP_AMPLITUDE'] = BUMP_AMPLITUDE
        self.parameters['BUMP_WAVELENGTH'] = BUMP_WAVELENGTH
        self.parameters['OPT_INITIAL_BUMP'] = OPT_INITIAL_BUMP
        
    def set_uniform_uplift(self,
                           UPRATE=0.001,
                           UPDUR=10000000):
        
        self.parameters['OPTNOUPLIFT'] = 0
        self.parameters['UPTYPE'] = 1
        self.parameters['UPRATE'] = UPRATE
        self.parameters['UPDUR'] = UPDUR
        
    def set_block_uplift(self,
                         FAULTPOS,
                         UPRATE=0.001,
                         SUBSRATE=0,
                         UPDUR=10000000):
        
        self.parameters['OPTNOUPLIFT'] = 0
        self.parameters['UPTYPE'] = 2
        self.parameters['FAULTPOS'] = FAULTPOS
        self.parameters['UPRATE'] = UPRATE
        self.parameters['SUBSRATE'] = SUBSRATE
        self.parameters['UPDUR'] = UPDUR
        
    def set_uplift_maps(self,
                        NUMUPLIFTMAPS,
                        UPMAPFILENAME,
                        UPTIMEFILENAME='n/a',
                        UPDUR=10000000,):
        
        self.parameters['OPTNOUPLIFT'] = 0
        self.parameters['UPTYPE'] = 12
        self.parameters['UPRATE'] = 0
        self.parameters['NUMUPLIFTMAPS'] = NUMUPLIFTMAPS
        self.parameters['UPMAPFILENAME'] = UPMAPFILENAME
        self.parameters['UPTIMEFILENAME'] = UPTIMEFILENAME
        self.parameters['UPDUR'] = UPDUR
        
    def set_horizontal_propagating_front(self,
                                         STARTING_YCOORD,
                                         UPRATE=0.001,
                                         FRONT_PROP_RATE=0.01,
                                         UPLIFT_FRONT_GRADIENT=0,
                                         UPDUR=10000000):
        
        self.parameters['OPTNOUPLIFT'] = 0
        self.parameters['UPTYPE'] = 13
        self.parameters['STARTING_YCOORD'] = STARTING_YCOORD
        self.parameters['UPRATE'] = UPRATE
        self.parameters['FRONT_PROP_RATE'] = FRONT_PROP_RATE
        self.parameters['UPLIFT_FRONT_GRADIENT'] = UPLIFT_FRONT_GRADIENT
        self.parameters['UPDUR'] = UPDUR
        
    def set_rainfall(self,
                     OPTVAR=0,
                     ST_PMEAN=10.,
                     ST_STDUR=1.,
                     ST_ISTDUR=0.,
                     ST_OPTSINVAR=0,
                     OPTSINVARINFILT=0):
        
        self.parameters['OPTVAR'] = OPTVAR
        self.parameters['ST_PMEAN'] = ST_PMEAN
        self.parameters['ST_STDUR'] = ST_STDUR
        self.parameters['ST_ISTDUR'] = ST_ISTDUR
        self.parameters['ST_OPTSINVAR'] = ST_OPTSINVAR
        self.parameters['OPTSINVARINFILT'] = OPTSINVARINFILT
        
    def set_runoff(self,
                   FLOWGEN=0,
                   TRANSMISSIVITY='n/a',
                   OPTVAR_TRANSMISSIVITY=0,
                   INFILTRATION=0,
                   OPTSINVARINFILT=0,
                   PERIOD_INFILT=0,
                   MAXICMEAN=0,
                   SOILSTORE=0,
                   KINWAVE_HQEXP=1,
                   FLOWVELOCITY=31536000,
                   HYDROSHAPEFAC=1,
                   LAKEFILL=1):

        self.parameters['FLOWGEN'] = FLOWGEN
        self.parameters['TRANSMISSIVITY'] = TRANSMISSIVITY
        self.parameters['OPTVAR_TRANSMISSIVITY'] = OPTVAR_TRANSMISSIVITY
        self.parameters['INFILTRATION'] = INFILTRATION
        self.parameters['OPTSINVARINFILT'] = OPTSINVARINFILT
        self.parameters['PERIOD_INFILT'] = PERIOD_INFILT
        self.parameters['MAXICMEAN'] = MAXICMEAN
        self.parameters['SOILSTORE'] = SOILSTORE
        self.parameters['KINWAVE_HQEXP'] = KINWAVE_HQEXP
        self.parameters['FLOWVELOCITY'] = FLOWVELOCITY
        self.parameters['HYDROSHAPEFAC'] = HYDROSHAPEFAC
        self.parameters['LAKEFILL'] = LAKEFILL

    def set_hydraulic_geometry(self,
                               CHAN_GEOM_MODEL=1,
                               HYDR_WID_COEFF_DS=10.0,
                               HYDR_WID_EXP_DS=0.5,
                               HYDR_WID_EXP_STN=0.5,
                               HYDR_DEP_COEFF_DS=1.0,
                               HYDR_DEP_EXP_DS=0,
                               HYDR_DEP_EXP_STN=0,
                               HYDR_ROUGH_COEFF_DS=0.03,
                               HYDR_ROUGH_EXP_DS=0,
                               HYDR_ROUGH_EXP_STN=0,
                               HYDR_SLOPE_EXP=0,
                               THETAC=0.045,
                               SHEAR_RATIO=1.1,
                               BANK_ROUGH_COEFF=15.0,
                               BANK_ROUGH_EXP=0.80,
                               BANKFULLEVENT=10):
        ''' Set the hydraulic geometry

        @param CHAN_GEOM_MODEL: Type of channel geometry model to be used:
                                0. standard empirical hydraulic geometry
                                1. Regime theory (empirical power-law scaling) [experimental]
                                2. Parker-Paola self-formed channel theory [experimental]
                                3. Finnegan slope-dependent channel width model [experimental]
        @param HYDR_WID_COEFF_DS: Coefficient in bankfull width-discharge relation
        @param HYDR_WID_EXP_DS: Exponent in bankfull width-discharge relatio
        @param HYDR_WID_EXP_STN: Exponent in at-a-station width-discharge relation
        @param HYDR_DEP_COEFF_DS: Coefficient in bankfull depth-discharge relation
        @param HYDR_DEP_EXP_DS: Exponent in bankfull depth-discharge relation
        @param HYDR_DEP_EXP_STN: Exponent in at-a-station depth-discharge relation
        @param HYDR_ROUGH_COEFF_DS: Coefficient in bankfull roughness-discharge relation
        @param HYDR_ROUGH_EXP_DS: Exponent in bankfull roughness-discharge relation
        @param HYDR_ROUGH_EXP_STN: Exponent in at-a-station roughness-discharge relation
        @param HYDR_SLOPE_EXP:
        @param THETAC: For "Parker" channel geometry option: critical Shields stress
        @param SHEAR_RATIO: For "Parker" channel geometry option: ratio of 
                            actual to threshold shear stress
        @param BANK_ROUGH_COEFF: Coefficient in bank roughness-discharge relation
        @param BANK_ROUGH_EXP: Exponent in bank roughness-discharge relation
        @param BANKFULLEVENT: Runoff rate associated with bankfull flood event.
                              Used to compute hydraulic geometry
        '''
        self.parameters['CHAN_GEOM_MODEL'] = CHAN_GEOM_MODEL
        self.parameters['HYDR_WID_COEFF_DS'] = HYDR_WID_COEFF_DS
        self.parameters['HYDR_WID_EXP_DS'] = HYDR_WID_EXP_DS
        self.parameters['HYDR_WID_EXP_STN'] = HYDR_WID_EXP_STN
        self.parameters['HYDR_DEP_COEFF_DS'] = HYDR_DEP_COEFF_DS
        self.parameters['HYDR_DEP_EXP_DS'] = HYDR_DEP_EXP_DS
        self.parameters['HYDR_DEP_EXP_STN'] = HYDR_DEP_EXP_STN
        self.parameters['HYDR_ROUGH_COEFF_DS'] = HYDR_ROUGH_COEFF_DS
        self.parameters['HYDR_ROUGH_EXP_DS'] = HYDR_ROUGH_EXP_DS
        self.parameters['HYDR_ROUGH_EXP_STN'] = HYDR_ROUGH_EXP_STN
        self.parameters['HYDR_SLOPE_EXP'] = HYDR_SLOPE_EXP
        self.parameters['THETAC'] = THETAC
        self.parameters['SHEAR_RATIO'] = SHEAR_RATIO
        self.parameters['BANK_ROUGH_COEFF'] = BANK_ROUGH_COEFF
        self.parameters['BANK_ROUGH_EXP'] = BANK_ROUGH_EXP
        self.parameters['BANKFULLEVENT'] = BANKFULLEVENT
        
    def set_meandering(self,
                       OPTMEANDER=0,
                       CRITICAL_AREA=1e8,
                       # CRITICAL_FLOW=1e8,
                       OPT_VAR_SIZE=0,
                       MEDIAN_DIAMETER=0.0007,
                       BANKERO=0,
                       BNKHTDEP=0,
                       DEF_CHAN_DISCR=1,
                       FRAC_WID_MOVE=0.1,
                       FRAC_WID_ADD=0.7):
        
        self.parameters['OPTMEANDER'] = OPTMEANDER
        self.parameters['CRITICAL_AREA'] = CRITICAL_AREA
        # self.parameters['CRITICAL_FLOW'] = CRITICAL_FLOW
        self.parameters['OPT_VAR_SIZE'] = OPT_VAR_SIZE
        self.parameters['MEDIAN_DIAMETER'] = MEDIAN_DIAMETER
        self.parameters['BANKERO'] = BANKERO
        self.parameters['BNKHTDEP'] = BNKHTDEP
        self.parameters['DEF_CHAN_DISCR'] = DEF_CHAN_DISCR
        self.parameters['FRAC_WID_MOVE'] = FRAC_WID_MOVE
        self.parameters['FRAC_WID_ADD'] = FRAC_WID_ADD
        
    def set_materials(self,
                      ROCKDENSITYINIT=2270,
                      SOILBULKDENSITY=740,
                      WOODDENSITY=450):
        
        self.parameters['ROCKDENSITYINIT'] = ROCKDENSITYINIT
        self.parameters['SOILBULKDENSITY'] = SOILBULKDENSITY
        self.parameters['WOODDENSITY'] = WOODDENSITY
        
    def set_grainsize(self,
                      NUMGRNSIZE=1,
                      BRPROPORTIONi=(1, 0, 0, 0, 0, 0, 0, 0, 0),
                      REGPROPORTIONi=(1, 0, 0, 0, 0, 0, 0, 0, 0),
                      GRAINDIAMi=(0.001, 0, 0, 0, 0, 0, 0, 0, 0),
                      HIDINGEXP=0.75,
                      GRAINDIAM0=0.007):
        
        self.parameters['NUMGRNSIZE'] = NUMGRNSIZE
        for i, br_proportion in enumerate(BRPROPORTIONi):
            self.parameters['BRPROPORTION' + str(i + 1)] = br_proportion
        for i, reg_proportion in enumerate(REGPROPORTIONi):
            self.parameters['REGPROPORTION' + str(i + 1)] = reg_proportion
        for i, grain_diam in enumerate(GRAINDIAMi):
            self.parameters['GRAINDIAM' + str(i + 1)] = grain_diam
        self.parameters['HIDINGEXP'] = HIDINGEXP
        self.parameters['GRAINDIAM0'] = GRAINDIAM0

    def set_fluvial_transport(self,
                              OPTNOFLUVIAL=0,
                              DETACHMENT_LAW=1,
                              KB=0.0005,
                              KR=0.0005,
                              KT=1000.,
                              MB=0.66667,
                              NB=0.66667,
                              PB=1.5,
                              TAUCB=30,
                              TAUCR=30,
                              BETA=1,
                              OPTDETACHLIM=0,
                              TRANSPORT_LAW=1,
                              KF=617.,
                              MF=0.66667,
                              NF=0.66667,
                              PF=1.5):
        ''' Set the parameters for fluvial erosion and deposition

        @param OPTNOFLUVIAL: Option to turn off fluvial processes (default to false)
        @param DETACHMENT_LAW: Code for detachment-capacity law to be applied: 
                               0. Power law, form 1
                               1. Power law, form 2
                               2. Almost parabolic law
                               3. Generalized f(Qs) detachment-rule
                               4. Dummy law for no fluvial erosion
        @param KB: Erodibility coefficient for bedrock. If layers are read in
                   from a previous run, values from layer file are used instead
        @param KR: Erodibility coefficient for regolith. If layers are read in
                   from a previous run, values from layer file are used instead
        @param KT: (Pa per (m2/s)M, where M is Mb for detachment and Mf for
                    sediment transport) Coefficient relating shear stress to
                    discharge and slope. Can be calculated from water density,
                    gravitational acceleration, and roughness; see, e.g., Tucker
                    and Slingerland (1997)
        @param MB: Discharge exponent in detachment capacity equation
        @param NB: Slope exponent in detachment capacity equation
        @param PB: Excess power/shear exponent in detachment capacity equation
        @param TAUCB: (normally Pa) Detachment threshold for bedrock
        @param TAUCR: (normally Pa) Detachment threshold for regolith
        @param BETA: Fraction of eroded sediment that forms bed load. Applies
                     only to sediment-flux-dependent detachment laws
        @param OPTDETACHLIM: Option for detachment-limited fluvial erosion only
        @param TRANSPORT_LAW: Code for fluvial transport capacity law to be applied:
                              0. Power-law transport formula
                              1. Power-law transport formula, form 2
                              2. Bridge-Dominic form of Bagnold bedload formula
                              3. Wilcock sand-gravel formula
                              4. Multi-size power-law formula
                              5. Willgoose/Riley mine tailings formula
                              6. Ultra-simplified power-law transport formula
                              7. Dummy law for no fluvial transport
        @param KF: Fluvial sediment transport efficiency coefficient
        @param MF: Discharge exponent in fluvial transport capacity equation
        @param NF: Slope exponent in fluvial transport capacity equation
        @param PF: Excess power/shear exponent in fluvial transport capacity equation
        '''
        self.parameters['OPTNOFLUVIAL'] = OPTNOFLUVIAL
        self.parameters['DETACHMENT_LAW'] = DETACHMENT_LAW
        self.parameters['KB'] = KB
        self.parameters['KR'] = KR
        self.parameters['KT'] = KT
        self.parameters['MB'] = MB
        self.parameters['NB'] = NB
        self.parameters['PB'] = PB
        self.parameters['TAUCB'] = TAUCB
        self.parameters['TAUCR'] = TAUCR
        self.parameters['BETA'] = BETA
        self.parameters['OPTDETACHLIM'] = OPTDETACHLIM
        self.parameters['TRANSPORT_LAW'] = TRANSPORT_LAW
        self.parameters['KF'] = KF
        self.parameters['MF'] = MF
        self.parameters['NF'] = NF
        self.parameters['PF'] = PF

    def set_detachment_power_law_form_1(self,
                                        KB=0.0005,
                                        KR=0.0005,
                                        KT=1000.,
                                        MB=0.66667,
                                        NB=0.66667,
                                        PB=1.5,
                                        TAUCB=30,
                                        TAUCR=30):
        
        self.parameters['OPTNOFLUVIAL'] = 0
        self.parameters['DETACHMENT_LAW'] = 0
        self.parameters['KB'] = KB
        self.parameters['KR'] = KR
        self.parameters['KT'] = KT
        self.parameters['MB'] = MB
        self.parameters['NB'] = NB
        self.parameters['PB'] = PB
        self.parameters['TAUCB'] = TAUCB
        self.parameters['TAUCR'] = TAUCR

    def set_detachment_power_law_form_2(self,
                                        KB=0.0005,
                                        KR=0.0005,
                                        KT=1000.,
                                        MB=0.66667,
                                        NB=0.66667,
                                        PB=1.5,
                                        TAUCB=30,
                                        TAUCR=30):
        
        self.parameters['OPTNOFLUVIAL'] = 0
        self.parameters['DETACHMENT_LAW'] = 1
        self.parameters['KB'] = KB
        self.parameters['KR'] = KR
        self.parameters['KT'] = KT
        self.parameters['MB'] = MB
        self.parameters['NB'] = NB
        self.parameters['PB'] = PB
        self.parameters['TAUCB'] = TAUCB
        self.parameters['TAUCR'] = TAUCR

    def set_detachment_almost_parabolic_law(self,
                                            KB=1e-4,
                                            KR=1e-4,
                                            MB=0.5,
                                            NB=1,
                                            BETA=1):
        
        self.parameters['OPTNOFLUVIAL'] = 0
        self.parameters['DETACHMENT_LAW'] = 2
        self.parameters['KB'] = KB
        self.parameters['KR'] = KR
        self.parameters['MB'] = MB
        self.parameters['NB'] = NB
        self.parameters['BETA'] = BETA

    def set_detachment_generalized_fqs_law(self,
                                           KB=1e-4,
                                           KR=1e-4,
                                           MB=0.5,
                                           NB=1,
                                           BETA=1):
        
        self.parameters['OPTNOFLUVIAL'] = 0
        self.parameters['DETACHMENT_LAW'] = 3
        self.parameters['KB'] = KB
        self.parameters['KR'] = KR
        self.parameters['MB'] = MB
        self.parameters['NB'] = NB
        self.parameters['BETA'] = BETA

    def set_detachment_dummy_law(self):
        
        self.parameters['OPTNOFLUVIAL'] = 0
        self.parameters['DETACHMENT_LAW'] = 4

    def set_transport_law(self,
                          OPTDETACHLIM=0,
                          TRANSPORT_LAW=1,
                          KF=617.,
                          MF=0.66667,
                          NF=0.66667,
                          PF=1.5):
        
        self.parameters['OPTNOFLUVIAL'] = 0
        self.parameters['OPTDETACHLIM'] = OPTDETACHLIM
        self.parameters['TRANSPORT_LAW'] = TRANSPORT_LAW
        self.parameters['KF'] = KF
        self.parameters['MF'] = MF
        self.parameters['NF'] = NF
        self.parameters['PF'] = PF
        
    def set_overbank_deposition(self,
                                OPTFLOODPLAIN=0,
                                FP_DRAREAMIN=1e8,
                                FP_BANKFULLEVENT=20,
                                FP_MU=1,
                                FP_LAMBDA=200,
                                FP_OPTCONTROLCHAN=0,
                                FP_VALDROP=1,
                                FP_INLET_ELEVATION=1):
        
        # HYDR_DEP_COEFF_DS, HYDR_DEP_EXP_STN, HYDR_DEP_EXP_DS, NUMGRNSIZE
        
        self.parameters['OPTFLOODPLAIN'] = OPTFLOODPLAIN
        self.parameters['FP_DRAREAMIN'] = FP_DRAREAMIN
        self.parameters['FP_BANKFULLEVENT'] = FP_BANKFULLEVENT
        self.parameters['FP_MU'] = FP_MU
        self.parameters['FP_LAMBDA'] = FP_LAMBDA
        self.parameters['FP_OPTCONTROLCHAN'] = FP_OPTCONTROLCHAN
        self.parameters['FP_VALDROP'] = FP_VALDROP
        self.parameters['FP_INLET_ELEVATION'] = FP_INLET_ELEVATION
        
    def set_hillslope_transport(self,
                                OPTNODIFFUSION=0,
                                KD=0.01,
                                OPTDIFFDEP=0,
                                DIFFUSIONTHRESHOLD=0,
                                OPT_NONLINEAR_DIFFUSION=0,
                                OPT_DEPTH_DEPENDENT_DIFFUSION=0,
                                DIFFDEPTHSCALE=1,
                                CRITICAL_SLOPE=0.5774):
        
        self.parameters['OPTNODIFFUSION'] = OPTNODIFFUSION
        self.parameters['KD'] = KD
        self.parameters['OPTDIFFDEP'] = OPTDIFFDEP
        self.parameters['DIFFUSIONTHRESHOLD'] = DIFFUSIONTHRESHOLD
        self.parameters['OPT_NONLINEAR_DIFFUSION'] = OPT_NONLINEAR_DIFFUSION
        self.parameters['OPT_DEPTH_DEPENDENT_DIFFUSION'] = OPT_DEPTH_DEPENDENT_DIFFUSION
        self.parameters['DIFFDEPTHSCALE'] = DIFFDEPTHSCALE
        self.parameters['CRITICAL_SLOPE'] = CRITICAL_SLOPE
        
    def set_landsliding(self,
                        OPT_LANDSLIDES=0,
                        OPT_3D_LANDSLIDES=0,
                        FRICSLOPE=1,
                        DF_RUNOUT_RULE=0,
                        DF_SCOUR_RULE=0,
                        DF_DEPOSITION_RULE=0):
        
#         print('Landsliding also requires ROCKDENSITYINIT and WOODDENSITY (see material_parameters)')
        
        self.parameters['OPT_LANDSLIDES'] = OPT_LANDSLIDES
        self.parameters['OPT_3D_LANDSLIDES'] = OPT_3D_LANDSLIDES
        self.parameters['FRICSLOPE'] = FRICSLOPE
        self.parameters['DF_RUNOUT_RULE'] = DF_RUNOUT_RULE
        self.parameters['DF_SCOUR_RULE'] = DF_SCOUR_RULE
        self.parameters['DF_DEPOSITION_RULE'] = DF_DEPOSITION_RULE
        
    def set_eolian_deposition(self,
                              OPTLOESSDEP=0,
                              LOESS_DEP_RATE=0):
        
        self.parameters['OPTLOESSDEP'] = OPTLOESSDEP
        self.parameters['LOESS_DEP_RATE'] = LOESS_DEP_RATE

    def set_weathering(self,
                       CHEM_WEATHERING_LAW=0,
                       MAXDISSOLUTIONRATE=0.099,
                       CHEMDEPTH=0.18,
                       PRODUCTION_LAW=0,
                       SOILPRODRATE=0.00055,
                       SOILPRODRATEINTERCEPT=0.00055,
                       SOILPRODRATESLOPE=0.00000017,
                       SOILPRODDEPTH=0.30):
        
#         print('Weathering also requires ROCKDENSITYINIT and SOILBULKDENSITY (see material_parameters)')
        
        self.parameters['CHEM_WEATHERING_LAW'] = CHEM_WEATHERING_LAW
        self.parameters['MAXDISSOLUTIONRATE'] = MAXDISSOLUTIONRATE
        self.parameters['CHEMDEPTH'] = CHEMDEPTH
        self.parameters['PRODUCTION_LAW'] = PRODUCTION_LAW
        self.parameters['SOILPRODRATE'] = SOILPRODRATE
        self.parameters['SOILPRODRATEINTERCEPT'] = SOILPRODRATEINTERCEPT
        self.parameters['SOILPRODRATESLOPE'] = SOILPRODRATESLOPE
        self.parameters['SOILPRODDEPTH'] = SOILPRODDEPTH
        
    def set_vegetation(self,
                       OPTVEG=0,
                       OPTGRASS_SIMPLE=1,
                       VEG_KVD=0,
                       VEG_TV=1,
                       TAUC=0,
                       VEG_TAUCVEG=0):
        
#         print('Forest also requires OPTFOREST (see forest_parameters) and OPTFIRE (see fire_parameters)')
                
        self.parameters['OPTVEG'] = OPTVEG
        self.parameters['OPTGRASS_SIMPLE'] = OPTGRASS_SIMPLE
        self.parameters['VEG_KVD'] = VEG_KVD
        self.parameters['VEG_TV'] = VEG_TV
        self.parameters['TAUC'] = TAUC
        self.parameters['VEG_TAUCVEG'] = VEG_TAUCVEG
        
    def set_forest(self,
                   OPTFOREST=0,
                   ROOTDECAY_K=0,
                   ROOTDECAY_N=0,
                   ROOTGROWTH_A=1,
                   ROOTGROWTH_B=1,
                   ROOTGROWTH_C=1,
                   ROOTGROWTH_F=1,
                   ROOTSTRENGTH_J=0,
                   MAXVERTROOTCOHESION=0,
                   MAXLATROOTCOHESION=0,
                   TREEHEIGHTINDEX=0,
                   VEGWEIGHT_MAX=0,
                   VEGWEIGHT_A=0,
                   VEGWEIGHT_B=0,
                   VEGWEIGHT_C=0,
                   VEGWEIGHT_K=0,
                   BLOWDOWNPARAM=0,
                   BLOW_SEED=0,
                   TREEDIAM_B0=0,
                   TREEDIAM_B1=0,
                   TREEDIAM_B2=0,
                   WOODDECAY_K=0,
                   INITSTANDAGE=0):
        
        # TODO: Find default values
        
#         print('Forest also requires WOODDENSITY (see material_parameters) and FSEED (see run_control_parameters)')
        
        self.parameters['OPTFOREST'] = OPTFOREST
        self.parameters['ROOTDECAY_K'] = ROOTDECAY_K
        self.parameters['ROOTDECAY_N'] = ROOTDECAY_N
        self.parameters['ROOTGROWTH_A'] = ROOTGROWTH_A
        self.parameters['ROOTGROWTH_B'] = ROOTGROWTH_B
        self.parameters['ROOTGROWTH_C'] = ROOTGROWTH_C
        self.parameters['ROOTGROWTH_F'] = ROOTGROWTH_F
        self.parameters['ROOTSTRENGTH_J'] = ROOTSTRENGTH_J
        self.parameters['MAXVERTROOTCOHESION'] = MAXVERTROOTCOHESION
        self.parameters['MAXLATROOTCOHESION'] = MAXLATROOTCOHESION
        self.parameters['TREEHEIGHTINDEX'] = TREEHEIGHTINDEX
        self.parameters['VEGWEIGHT_MAX'] = VEGWEIGHT_MAX
        self.parameters['VEGWEIGHT_A'] = VEGWEIGHT_A
        self.parameters['VEGWEIGHT_B'] = VEGWEIGHT_B
        self.parameters['VEGWEIGHT_C'] = VEGWEIGHT_C
        self.parameters['VEGWEIGHT_K'] = VEGWEIGHT_K
        self.parameters['BLOWDOWNPARAM'] = BLOWDOWNPARAM
        self.parameters['BLOW_SEED'] = BLOW_SEED
        self.parameters['TREEDIAM_B0'] = TREEDIAM_B0
        self.parameters['TREEDIAM_B1'] = TREEDIAM_B1
        self.parameters['TREEDIAM_B2'] = TREEDIAM_B2
        self.parameters['WOODDECAY_K'] = WOODDECAY_K
        self.parameters['INITSTANDAGE'] = INITSTANDAGE

    def set_fire(self,
                 OPTFIRE=0,
                 IFRDUR=1,
                 OPTRANDOMFIRES=0):
        
#         print('Fire also requires FSEED (see run_control_parameters)')
        
        self.parameters['OPTFIRE'] = OPTFIRE
        self.parameters['IFRDUR'] = IFRDUR
        self.parameters['OPTRANDOMFIRES'] = OPTRANDOMFIRES

    def set_various(self,
                    OPTTSOUTPUT=1,
                    TSOPINTRVL=100,
                    SURFER=0,
                    OPTEXPOSURETIME=0,
                    OPTFOLDDENS=0,
                    OPT_TRACK_WATER_SED_TIMESERIES=0,
                    OPT_FREEZE_ELEVATIONS=0,
                    OPTSTREAMLINEBNDY=0):
        
        self.parameters['OPTTSOUTPUT'] = OPTTSOUTPUT
        self.parameters['TSOPINTRVL'] = TSOPINTRVL
        self.parameters['SURFER'] = SURFER
        self.parameters['OPTEXPOSURETIME'] = OPTEXPOSURETIME
        self.parameters['OPTFOLDDENS'] = OPTFOLDDENS
        self.parameters['OPT_TRACK_WATER_SED_TIMESERIES'] = OPT_TRACK_WATER_SED_TIMESERIES
        self.parameters['OPT_FREEZE_ELEVATIONS'] = OPT_FREEZE_ELEVATIONS
        self.parameters['OPTSTREAMLINEBNDY'] = OPTSTREAMLINEBNDY

    def solve_parameter(self,
                        parameter,
                        realization,
                        value,
                        save_previous_file=True,
                        random_state=None):
        
        if parameter == 'OUTFILENAME':
            if self.init_realization_nb is not None:
                value += '_' + str(self.init_realization_nb + realization)
            self.parameter_values[realization][parameter] = value
        elif isinstance(value, RangeModel) == True:
            self.parameter_values[realization][parameter] = value.rvs()
        elif isinstance(value, (stats._distn_infrastructure.rv_frozen, MixtureModel, MemoryModel)) == True:
            self.parameter_values[realization][parameter] = value.rvs(random_state=random_state)
        # elif isinstance(value, LinearTimeSeries) == True:
        #     self.parameter_values[realization][parameter] = value.write()
        elif isinstance(value, FloodplainTimeSeries) == True:
            self.parameter_values[realization][parameter] = value.write(parameter,
                                                                        base_name=self.parameter_values[realization]['OUTFILENAME'],
                                                                        save_previous_file=save_previous_file,
                                                                        random_state=random_state)
        else:
            self.parameter_values[realization][parameter] = value

    def solve_parameters(self,
                         realization,
                         save_previous_file=True,
                         random_state=None):

        if random_state is None:
            random_state = np.random.RandomState(self.seed + realization)

        for parameter in self.parameters:
            self.solve_parameter(parameter,
                                 realization,
                                 self.parameters[parameter],
                                 save_previous_file=save_previous_file,
                                 random_state=random_state)

    def write_parameter(self, input_file, parameter, value, parameter_name=None):
        
        if parameter_name is None:
            parameter_name = parameter
        input_file.write(parameter_name + ': ' + self.parameter_descriptions[parameter] + '\n')
        input_file.write(str(value) + '\n')
        
    def write_header(self, input_file, outfile_name, description, line_size):
        
        input_file.write('#' + (line_size - 1)*'-' + '\n')
        input_file.write('#\n')
        input_file.write(divide_line(outfile_name + '.in: ' + description, line_size))
        input_file.write('#' + (line_size - 1)*'-' + '\n')

    def write_input_file(self,
                         realization,
                         COMMENTS=None,
                         resolve_parameters=False,
                         save_previous_file=True,
                         random_state=None):

        line_size = 68

        if not self.parameter_values[realization] or resolve_parameters == True:
            self.solve_parameters(realization,
                                  save_previous_file=save_previous_file,
                                  random_state=random_state)

        if save_previous_file == True:
            rename_old_file(self.locate_input_file(realization))

        with open(self.locate_input_file(realization), 'w') as input_file:
            
            self.write_header(input_file,
                              self.parameter_values[realization]['OUTFILENAME'],
                              self.DESCRIPTION,
                              line_size)
            for parameter in self.parameter_values[realization]:
                description_key = parameter
                if description_key[-1].isdigit() == True and description_key[:-1] != 'TREEDIAM_B':
                    description_key = description_key[:-1] + 'i'
                self.write_parameter(input_file,
                                     description_key,
                                     self.parameter_values[realization][parameter],
                                     parameter_name=parameter)
            input_file.write('\n')
            input_file.write('Comments here:\n')
            input_file.write('\n')
            if COMMENTS is not None:
                input_file.write(divide_line(COMMENTS, line_size))

            self.base_names[realization] = self.parameter_values[realization]['OUTFILENAME']

    def write_input_files(self,
                          COMMENTS=None,
                          resolve_parameters=False,
                          random_state=None):

        for r in range(self.nb_realizations):

            self.write_input_file(r,
                                  COMMENTS=COMMENTS,
                                  resolve_parameters=resolve_parameters,
                                  random_state=random_state)

    def locate_input_file(self, realization=0):
        
        return os.path.join(self.base_directory,
                            self.parameter_values[realization]['OUTFILENAME'] + '.in')
    
    def get_base_name(self, realization=0):
        
        return os.path.join(self.base_directory,
                            self.parameter_values[realization]['OUTFILENAME'])
    
    def delete_input_file(self, realization=0):
        
        if os.path.isfile(self.locate_input_file(realization)) == True:
            subprocess.call('rm ' + self.locate_input_file(realization), shell=True)

    def extract_input_file_parameters(self,
                                      parameters,
                                      realization=0,
                                      input_file_path=None):
        
        if input_file_path is None:
            input_file_path = self.base_name + '_' + str(realization) + '.in'
            
        parameter_values = dict()
        found_parameters = [False]*len(parameters)

        with open(input_file_path) as file:
            
            line = file.readline().strip('\n')
            while False in found_parameters and line:
                file_parameter = line.split(':')[0]
                if file_parameter in parameters:
                    line = file.readline().strip('\n')
                    value = line.split(':')[0]
                    try:
                        value = float(value)
                    finally:
                        parameter_values[file_parameter] = value
                    found_parameters[parameters.index(file_parameter)] = True
                else:
                    line = file.readline().strip('\n')
                line = file.readline().strip('\n')
                
        return parameter_values
            