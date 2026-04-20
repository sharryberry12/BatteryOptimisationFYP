<?xml version="1.0" encoding="ISO-8859-1"?>
<!-- output by GridLAB-D -->
<html xsl:version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns="http://www.w3.org/1999/xhtml">
<xsl:for-each select="/gridlabd">
<head>
<title>GridLAB-D <xsl:value-of select="version.major"/>.<xsl:value-of select="version.minor"/> - <xsl:value-of select="modelname"/></title>
<link rel="stylesheet" href="./gridlabd-2_3.css" type="text/css"/>
</head>
<body>
<H1><xsl:value-of select="modelname"/></H1>
<H2>Table of Contents</H2>
<OL TYPE="1">
<LI><A HREF="#global_variables">Global variables</A></LI>
<LI><A HREF="#solver_ranks">Solver ranks</A></LI>
<LI><A HREF="#modules">Modules</A></LI><OL TYPE="a">
<LI><A HREF="#modules_climate">climate</A></LI>
<LI><A HREF="#modules_commercial">commercial</A></LI>
<LI><A HREF="#modules_generators">generators</A></LI>
<LI><A HREF="#modules_market">market</A></LI>
<LI><A HREF="#modules_plc">plc</A></LI>
<LI><A HREF="#modules_powerflow">powerflow</A></LI>
<LI><A HREF="#modules_residential">residential</A></LI>
<LI><A HREF="#modules_tape">tape</A></LI>
</OL>
<LI><A HREF="#output">Output</A></LI>
</OL>
<H2><A NAME="global_variables">GridLAB-D system variables</A></H2>
<TABLE BORDER="1">
<TR><TH>version.major</TH><TD><xsl:value-of select="version.major"/></TD></TR>
<TR><TH>version.minor</TH><TD><xsl:value-of select="version.minor"/></TD></TR>
<TR><TH>command_line</TH><TD><xsl:value-of select="command_line"/></TD></TR>
<TR><TH>environment</TH><TD><xsl:value-of select="environment"/></TD></TR>
<TR><TH>quiet</TH><TD><xsl:value-of select="quiet"/></TD></TR>
<TR><TH>warn</TH><TD><xsl:value-of select="warn"/></TD></TR>
<TR><TH>debugger</TH><TD><xsl:value-of select="debugger"/></TD></TR>
<TR><TH>gdb</TH><TD><xsl:value-of select="gdb"/></TD></TR>
<TR><TH>debug</TH><TD><xsl:value-of select="debug"/></TD></TR>
<TR><TH>test</TH><TD><xsl:value-of select="test"/></TD></TR>
<TR><TH>verbose</TH><TD><xsl:value-of select="verbose"/></TD></TR>
<TR><TH>iteration_limit</TH><TD><xsl:value-of select="iteration_limit"/></TD></TR>
<TR><TH>workdir</TH><TD><xsl:value-of select="workdir"/></TD></TR>
<TR><TH>dumpfile</TH><TD><xsl:value-of select="dumpfile"/></TD></TR>
<TR><TH>savefile</TH><TD><xsl:value-of select="savefile"/></TD></TR>
<TR><TH>dumpall</TH><TD><xsl:value-of select="dumpall"/></TD></TR>
<TR><TH>runchecks</TH><TD><xsl:value-of select="runchecks"/></TD></TR>
<TR><TH>threadcount</TH><TD><xsl:value-of select="threadcount"/></TD></TR>
<TR><TH>profiler</TH><TD><xsl:value-of select="profiler"/></TD></TR>
<TR><TH>pauseatexit</TH><TD><xsl:value-of select="pauseatexit"/></TD></TR>
<TR><TH>testoutputfile</TH><TD><xsl:value-of select="testoutputfile"/></TD></TR>
<TR><TH>xml_encoding</TH><TD><xsl:value-of select="xml_encoding"/></TD></TR>
<TR><TH>clock</TH><TD><xsl:value-of select="clock"/></TD></TR>
<TR><TH>starttime</TH><TD><xsl:value-of select="starttime"/></TD></TR>
<TR><TH>stoptime</TH><TD><xsl:value-of select="stoptime"/></TD></TR>
<TR><TH>double_format</TH><TD><xsl:value-of select="double_format"/></TD></TR>
<TR><TH>complex_format</TH><TD><xsl:value-of select="complex_format"/></TD></TR>
<TR><TH>object_format</TH><TD><xsl:value-of select="object_format"/></TD></TR>
<TR><TH>object_scan</TH><TD><xsl:value-of select="object_scan"/></TD></TR>
<TR><TH>object_tree_balance</TH><TD><xsl:value-of select="object_tree_balance"/></TD></TR>
<TR><TH>kmlfile</TH><TD><xsl:value-of select="kmlfile"/></TD></TR>
<TR><TH>modelname</TH><TD><xsl:value-of select="modelname"/></TD></TR>
<TR><TH>execdir</TH><TD><xsl:value-of select="execdir"/></TD></TR>
<TR><TH>strictnames</TH><TD><xsl:value-of select="strictnames"/></TD></TR>
<TR><TH>website</TH><TD><xsl:value-of select="website"/></TD></TR>
<TR><TH>urlbase</TH><TD><xsl:value-of select="urlbase"/></TD></TR>
<TR><TH>randomseed</TH><TD><xsl:value-of select="randomseed"/></TD></TR>
<TR><TH>include</TH><TD><xsl:value-of select="include"/></TD></TR>
<TR><TH>trace</TH><TD><xsl:value-of select="trace"/></TD></TR>
<TR><TH>gdb_window</TH><TD><xsl:value-of select="gdb_window"/></TD></TR>
<TR><TH>tmp</TH><TD><xsl:value-of select="tmp"/></TD></TR>
<TR><TH>force_compile</TH><TD><xsl:value-of select="force_compile"/></TD></TR>
<TR><TH>nolocks</TH><TD><xsl:value-of select="nolocks"/></TD></TR>
<TR><TH>skipsafe</TH><TD><xsl:value-of select="skipsafe"/></TD></TR>
<TR><TH>dateformat</TH><TD><xsl:value-of select="dateformat"/></TD></TR>
<TR><TH>minimum_timestep</TH><TD><xsl:value-of select="minimum_timestep"/></TD></TR>
<TR><TH>platform</TH><TD><xsl:value-of select="platform"/></TD></TR>
<TR><TH>suppress_repeat_messages</TH><TD><xsl:value-of select="suppress_repeat_messages"/></TD></TR>
<TR><TH>maximum_synctime</TH><TD><xsl:value-of select="maximum_synctime"/></TD></TR>
<TR><TH>run_realtime</TH><TD><xsl:value-of select="run_realtime"/></TD></TR>
<TR><TH>no_deprecate</TH><TD><xsl:value-of select="no_deprecate"/></TD></TR>
<TR><TH>sync_dumpfile</TH><TD><xsl:value-of select="sync_dumpfile"/></TD></TR>
<TR><TH>streaming_io</TH><TD><xsl:value-of select="streaming_io"/></TD></TR>
<TR><TH>compileonly</TH><TD><xsl:value-of select="compileonly"/></TD></TR>
<TR><TH>relax_naming_rules</TH><TD><xsl:value-of select="relax_naming_rules"/></TD></TR>
<TR><TH>server_portnum</TH><TD><xsl:value-of select="server_portnum"/></TD></TR>
<TR><TH>show_progress</TH><TD><xsl:value-of select="show_progress"/></TD></TR>
<TR><TH>simulation_mode</TH><TD><xsl:value-of select="simulation_mode"/></TD></TR>
<TR><TH>deltamode_timestep</TH><TD><xsl:value-of select="deltamode_timestep"/></TD></TR>
<TR><TH>deltamode_maximumtime</TH><TD><xsl:value-of select="deltamode_maximumtime"/></TD></TR>
<TR><TH>global_deltaclock</TH><TD><xsl:value-of select="global_deltaclock"/></TD></TR>
</TABLE>
<H2><A NAME="solver_ranks">Solver ranks</A></H2>
<TABLE BORDER="1">
<TR><xsl:for-each select="sync-order/pass">
<TH>Pass <xsl:value-of select="name"/></TH>
</xsl:for-each>
</TR>
<TR>
<xsl:for-each select="sync-order/pass"><TD><DL>
<xsl:for-each select="rank">
<xsl:sort select="ordinal" data-type="number" order="descending"/>
<DT>Rank <xsl:value-of select="ordinal"/></DT><xsl:for-each select="object">
<DD><a href="#{name}"><xsl:value-of select="name"/></a></DD>
</xsl:for-each>
</xsl:for-each>
</DL></TD></xsl:for-each>
</TR>
</TABLE>
<H2><A NAME="modules">Modules</A></H2>
<H3><A NAME="modules_climate">climate</A></H3><TABLE BORDER="1">
<TR><TH>version.major</TH><TD><xsl:value-of select="climate/version.major"/></TD></TR><TR><TH>version.minor</TH><TD><xsl:value-of select="climate/version.minor"/></TD></TR></TABLE>
<H4>climate objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>solar_elevation</TH><TH>solar_azimuth</TH><TH>city</TH><TH>tmyfile</TH><TH>temperature</TH><TH>humidity</TH><TH>solar_flux</TH><TH>solar_direct</TH><TH>solar_diffuse</TH><TH>solar_global</TH><TH>extraterrestrial_direct_normal</TH><TH>pressure</TH><TH>wind_speed</TH><TH>wind_dir</TH><TH>wind_gust</TH><TH>record.low</TH><TH>record.low_day</TH><TH>record.high</TH><TH>record.high_day</TH><TH>record.solar</TH><TH>rainfall</TH><TH>snowdepth</TH><TH>interpolate</TH><TH>solar_horiz</TH><TH>solar_north</TH><TH>solar_northeast</TH><TH>solar_east</TH><TH>solar_southeast</TH><TH>solar_south</TH><TH>solar_southwest</TH><TH>solar_west</TH><TH>solar_northwest</TH><TH>solar_raw</TH><TH>ground_reflectivity</TH><TH>reader</TH><TH>forecast</TH></TR>
<xsl:for-each select="climate/climate_list/climate"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="solar_elevation"/></TD><TD><xsl:value-of select="solar_azimuth"/></TD><TD><xsl:value-of select="city"/></TD><TD><xsl:value-of select="tmyfile"/></TD><TD><xsl:value-of select="temperature"/></TD><TD><xsl:value-of select="humidity"/></TD><TD><xsl:value-of select="solar_flux"/></TD><TD><xsl:value-of select="solar_direct"/></TD><TD><xsl:value-of select="solar_diffuse"/></TD><TD><xsl:value-of select="solar_global"/></TD><TD><xsl:value-of select="extraterrestrial_direct_normal"/></TD><TD><xsl:value-of select="pressure"/></TD><TD><xsl:value-of select="wind_speed"/></TD><TD><xsl:value-of select="wind_dir"/></TD><TD><xsl:value-of select="wind_gust"/></TD><TD><xsl:value-of select="record.low"/></TD><TD><xsl:value-of select="record.low_day"/></TD><TD><xsl:value-of select="record.high"/></TD><TD><xsl:value-of select="record.high_day"/></TD><TD><xsl:value-of select="record.solar"/></TD><TD><xsl:value-of select="rainfall"/></TD><TD><xsl:value-of select="snowdepth"/></TD><TD><xsl:value-of select="interpolate"/></TD><TD><xsl:value-of select="solar_horiz"/></TD><TD><xsl:value-of select="solar_north"/></TD><TD><xsl:value-of select="solar_northeast"/></TD><TD><xsl:value-of select="solar_east"/></TD><TD><xsl:value-of select="solar_southeast"/></TD><TD><xsl:value-of select="solar_south"/></TD><TD><xsl:value-of select="solar_southwest"/></TD><TD><xsl:value-of select="solar_west"/></TD><TD><xsl:value-of select="solar_northwest"/></TD><TD><xsl:value-of select="solar_raw"/></TD><TD><xsl:value-of select="ground_reflectivity"/></TD><TD><a href="#{reader}"><xsl:value-of select="reader"/></a></TD><TD><xsl:value-of select="forecast"/></TD></TR>
</xsl:for-each></TABLE>
<H4>weather objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>temperature</TH><TH>humidity</TH><TH>solar_dir</TH><TH>solar_diff</TH><TH>solar_global</TH><TH>wind_speed</TH><TH>rainfall</TH><TH>snowdepth</TH><TH>month</TH><TH>day</TH><TH>hour</TH><TH>minute</TH><TH>second</TH></TR>
<xsl:for-each select="climate/weather_list/weather"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="temperature"/></TD><TD><xsl:value-of select="humidity"/></TD><TD><xsl:value-of select="solar_dir"/></TD><TD><xsl:value-of select="solar_diff"/></TD><TD><xsl:value-of select="solar_global"/></TD><TD><xsl:value-of select="wind_speed"/></TD><TD><xsl:value-of select="rainfall"/></TD><TD><xsl:value-of select="snowdepth"/></TD><TD><xsl:value-of select="month"/></TD><TD><xsl:value-of select="day"/></TD><TD><xsl:value-of select="hour"/></TD><TD><xsl:value-of select="minute"/></TD><TD><xsl:value-of select="second"/></TD></TR>
</xsl:for-each></TABLE>
<H4>csv_reader objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>index</TH><TH>city_name</TH><TH>state_name</TH><TH>lat_deg</TH><TH>lat_min</TH><TH>long_deg</TH><TH>long_min</TH><TH>low_temp</TH><TH>high_temp</TH><TH>peak_solar</TH><TH>status</TH><TH>timefmt</TH><TH>timezone</TH><TH>timezone_offset</TH><TH>columns</TH><TH>filename</TH></TR>
<xsl:for-each select="climate/csv_reader_list/csv_reader"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="index"/></TD><TD><xsl:value-of select="city_name"/></TD><TD><xsl:value-of select="state_name"/></TD><TD><xsl:value-of select="lat_deg"/></TD><TD><xsl:value-of select="lat_min"/></TD><TD><xsl:value-of select="long_deg"/></TD><TD><xsl:value-of select="long_min"/></TD><TD><xsl:value-of select="low_temp"/></TD><TD><xsl:value-of select="high_temp"/></TD><TD><xsl:value-of select="peak_solar"/></TD><TD><xsl:value-of select="status"/></TD><TD><xsl:value-of select="timefmt"/></TD><TD><xsl:value-of select="timezone"/></TD><TD><xsl:value-of select="timezone_offset"/></TD><TD><xsl:value-of select="columns"/></TD><TD><xsl:value-of select="filename"/></TD></TR>
</xsl:for-each></TABLE>
<H3><A NAME="modules_commercial">commercial</A></H3><TABLE BORDER="1">
<TR><TH>version.major</TH><TD><xsl:value-of select="commercial/version.major"/></TD></TR><TR><TH>version.minor</TH><TD><xsl:value-of select="commercial/version.minor"/></TD></TR><TR><TH>warn_control</TH><TD><xsl:value-of select="commercial/warn_control"/></TD></TR><TR><TH>warn_low_temp</TH><TD><xsl:value-of select="commercial/warn_low_temp"/></TD></TR><TR><TH>warn_high_temp</TH><TD><xsl:value-of select="commercial/warn_high_temp"/></TD></TR></TABLE>
<H4>office objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>floor_area</TH><TH>floor_height</TH><TH>exterior_ua</TH><TH>interior_ua</TH><TH>interior_mass</TH><TH>glazing</TH><TH>glazing.north</TH><TH>glazing.northeast</TH><TH>glazing.east</TH><TH>glazing.southeast</TH><TH>glazing.south</TH><TH>glazing.southwest</TH><TH>glazing.west</TH><TH>glazing.northwest</TH><TH>glazing.horizontal</TH><TH>glazing.coefficient</TH><TH>occupancy</TH><TH>occupants</TH><TH>schedule</TH><TH>air_temperature</TH><TH>mass_temperature</TH><TH>temperature_change</TH><TH>outdoor_temperature</TH><TH>Qh</TH><TH>Qs</TH><TH>Qi</TH><TH>Qz</TH><TH>hvac_mode</TH><TH>hvac.cooling.balance_temperature</TH><TH>hvac.cooling.capacity</TH><TH>hvac.cooling.capacity_perF</TH><TH>hvac.cooling.design_temperature</TH><TH>hvac.cooling.efficiency</TH><TH>hvac.cooling.cop</TH><TH>hvac.heating.balance_temperature</TH><TH>hvac.heating.capacity</TH><TH>hvac.heating.capacity_perF</TH><TH>hvac.heating.design_temperature</TH><TH>hvac.heating.efficiency</TH><TH>hvac.heating.cop</TH><TH>lights.capacity</TH><TH>lights.fraction</TH><TH>plugs.capacity</TH><TH>plugs.fraction</TH><TH>demand</TH><TH>total_load</TH><TH>energy</TH><TH>power_factor</TH><TH>power</TH><TH>current</TH><TH>admittance</TH><TH>hvac.demand</TH><TH>hvac.load</TH><TH>hvac.energy</TH><TH>hvac.power_factor</TH><TH>lights.demand</TH><TH>lights.load</TH><TH>lights.energy</TH><TH>lights.power_factor</TH><TH>lights.heatgain_fraction</TH><TH>lights.heatgain</TH><TH>plugs.demand</TH><TH>plugs.load</TH><TH>plugs.energy</TH><TH>plugs.power_factor</TH><TH>plugs.heatgain_fraction</TH><TH>plugs.heatgain</TH><TH>cooling_setpoint</TH><TH>heating_setpoint</TH><TH>thermostat_deadband</TH><TH>control.ventilation_fraction</TH><TH>control.lighting_fraction</TH><TH>ACH</TH></TR>
<xsl:for-each select="commercial/office_list/office"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="floor_area"/></TD><TD><xsl:value-of select="floor_height"/></TD><TD><xsl:value-of select="exterior_ua"/></TD><TD><xsl:value-of select="interior_ua"/></TD><TD><xsl:value-of select="interior_mass"/></TD><TD><xsl:value-of select="glazing"/></TD><TD><xsl:value-of select="glazing.north"/></TD><TD><xsl:value-of select="glazing.northeast"/></TD><TD><xsl:value-of select="glazing.east"/></TD><TD><xsl:value-of select="glazing.southeast"/></TD><TD><xsl:value-of select="glazing.south"/></TD><TD><xsl:value-of select="glazing.southwest"/></TD><TD><xsl:value-of select="glazing.west"/></TD><TD><xsl:value-of select="glazing.northwest"/></TD><TD><xsl:value-of select="glazing.horizontal"/></TD><TD><xsl:value-of select="glazing.coefficient"/></TD><TD><xsl:value-of select="occupancy"/></TD><TD><xsl:value-of select="occupants"/></TD><TD><xsl:value-of select="schedule"/></TD><TD><xsl:value-of select="air_temperature"/></TD><TD><xsl:value-of select="mass_temperature"/></TD><TD><xsl:value-of select="temperature_change"/></TD><TD><xsl:value-of select="outdoor_temperature"/></TD><TD><xsl:value-of select="Qh"/></TD><TD><xsl:value-of select="Qs"/></TD><TD><xsl:value-of select="Qi"/></TD><TD><xsl:value-of select="Qz"/></TD><TD><xsl:value-of select="hvac_mode"/></TD><TD><xsl:value-of select="hvac.cooling.balance_temperature"/></TD><TD><xsl:value-of select="hvac.cooling.capacity"/></TD><TD><xsl:value-of select="hvac.cooling.capacity_perF"/></TD><TD><xsl:value-of select="hvac.cooling.design_temperature"/></TD><TD><xsl:value-of select="hvac.cooling.efficiency"/></TD><TD><xsl:value-of select="hvac.cooling.cop"/></TD><TD><xsl:value-of select="hvac.heating.balance_temperature"/></TD><TD><xsl:value-of select="hvac.heating.capacity"/></TD><TD><xsl:value-of select="hvac.heating.capacity_perF"/></TD><TD><xsl:value-of select="hvac.heating.design_temperature"/></TD><TD><xsl:value-of select="hvac.heating.efficiency"/></TD><TD><xsl:value-of select="hvac.heating.cop"/></TD><TD><xsl:value-of select="lights.capacity"/></TD><TD><xsl:value-of select="lights.fraction"/></TD><TD><xsl:value-of select="plugs.capacity"/></TD><TD><xsl:value-of select="plugs.fraction"/></TD><TD><xsl:value-of select="demand"/></TD><TD><xsl:value-of select="total_load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="current"/></TD><TD><xsl:value-of select="admittance"/></TD><TD><xsl:value-of select="hvac.demand"/></TD><TD><xsl:value-of select="hvac.load"/></TD><TD><xsl:value-of select="hvac.energy"/></TD><TD><xsl:value-of select="hvac.power_factor"/></TD><TD><xsl:value-of select="lights.demand"/></TD><TD><xsl:value-of select="lights.load"/></TD><TD><xsl:value-of select="lights.energy"/></TD><TD><xsl:value-of select="lights.power_factor"/></TD><TD><xsl:value-of select="lights.heatgain_fraction"/></TD><TD><xsl:value-of select="lights.heatgain"/></TD><TD><xsl:value-of select="plugs.demand"/></TD><TD><xsl:value-of select="plugs.load"/></TD><TD><xsl:value-of select="plugs.energy"/></TD><TD><xsl:value-of select="plugs.power_factor"/></TD><TD><xsl:value-of select="plugs.heatgain_fraction"/></TD><TD><xsl:value-of select="plugs.heatgain"/></TD><TD><xsl:value-of select="cooling_setpoint"/></TD><TD><xsl:value-of select="heating_setpoint"/></TD><TD><xsl:value-of select="thermostat_deadband"/></TD><TD><xsl:value-of select="control.ventilation_fraction"/></TD><TD><xsl:value-of select="control.lighting_fraction"/></TD><TD><xsl:value-of select="ACH"/></TD></TR>
</xsl:for-each></TABLE>
<H4>multizone objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>from</TH><TH>to</TH><TH>ua</TH></TR>
<xsl:for-each select="commercial/multizone_list/multizone"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><a href="#{from}"><xsl:value-of select="from"/></a></TD><TD><a href="#{to}"><xsl:value-of select="to"/></a></TD><TD><xsl:value-of select="ua"/></TD></TR>
</xsl:for-each></TABLE>
<H3><A NAME="modules_generators">generators</A></H3><TABLE BORDER="1">
<TR><TH>version.major</TH><TD><xsl:value-of select="generators/version.major"/></TD></TR><TR><TH>version.minor</TH><TD><xsl:value-of select="generators/version.minor"/></TD></TR><TR><TH>enable_subsecond_models</TH><TD><xsl:value-of select="generators/enable_subsecond_models"/></TD></TR><TR><TH>deltamode_timestep</TH><TD><xsl:value-of select="generators/deltamode_timestep"/></TD></TR><TR><TH>maximum_deltamode_time</TH><TD><xsl:value-of select="generators/maximum_deltamode_time"/></TD></TR><TR><TH>deltamode_update_count</TH><TD><xsl:value-of select="generators/deltamode_update_count"/></TD></TR></TABLE>
<H4>diesel_dg objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>Gen_mode</TH><TH>Gen_status</TH><TH>Gen_type</TH><TH>pf</TH><TH>GenElecEff</TH><TH>TotalRealPow</TH><TH>TotalReacPow</TH><TH>speed</TH><TH>cylinders</TH><TH>stroke</TH><TH>torque</TH><TH>pressure</TH><TH>time_operation</TH><TH>fuel</TH><TH>w_coolingwater</TH><TH>inlet_temperature</TH><TH>outlet_temperature</TH><TH>air_fuel</TH><TH>room_temperature</TH><TH>exhaust_temperature</TH><TH>cylinder_length</TH><TH>cylinder_radius</TH><TH>brake_diameter</TH><TH>calotific_fuel</TH><TH>steam_exhaust</TH><TH>specific_heat_steam</TH><TH>specific_heat_dry</TH><TH>indicated_hp</TH><TH>brake_hp</TH><TH>thermal_efficiency</TH><TH>energy_supplied</TH><TH>heat_equivalent_ip</TH><TH>energy_coolingwater</TH><TH>mass_exhaustgas</TH><TH>energy_exhaustgas</TH><TH>energy_steam</TH><TH>total_energy_exhaustgas</TH><TH>unaccounted_energyloss</TH><TH>f</TH><TH>poles</TH><TH>wm</TH><TH>Pconv</TH><TH>Tind</TH><TH>EA</TH><TH>Vo</TH><TH>Rs1</TH><TH>Xs1</TH><TH>delta</TH><TH>IA</TH><TH>Ploss</TH><TH>Pout</TH><TH>effe</TH><TH>effo</TH><TH>Rated_V</TH><TH>Rated_VA</TH><TH>Rs</TH><TH>Xs</TH><TH>Rg</TH><TH>Xg</TH><TH>voltage_A</TH><TH>voltage_B</TH><TH>voltage_C</TH><TH>current_A</TH><TH>current_B</TH><TH>current_C</TH><TH>EfA</TH><TH>EfB</TH><TH>EfC</TH><TH>power_A</TH><TH>power_B</TH><TH>power_C</TH><TH>power_A_sch</TH><TH>power_B_sch</TH><TH>power_C_sch</TH><TH>EfA_sch</TH><TH>EfB_sch</TH><TH>EfC_sch</TH><TH>SlackBus</TH><TH>phases</TH></TR>
<xsl:for-each select="generators/diesel_dg_list/diesel_dg"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="Gen_mode"/></TD><TD><xsl:value-of select="Gen_status"/></TD><TD><xsl:value-of select="Gen_type"/></TD><TD><xsl:value-of select="pf"/></TD><TD><xsl:value-of select="GenElecEff"/></TD><TD><xsl:value-of select="TotalRealPow"/></TD><TD><xsl:value-of select="TotalReacPow"/></TD><TD><xsl:value-of select="speed"/></TD><TD><xsl:value-of select="cylinders"/></TD><TD><xsl:value-of select="stroke"/></TD><TD><xsl:value-of select="torque"/></TD><TD><xsl:value-of select="pressure"/></TD><TD><xsl:value-of select="time_operation"/></TD><TD><xsl:value-of select="fuel"/></TD><TD><xsl:value-of select="w_coolingwater"/></TD><TD><xsl:value-of select="inlet_temperature"/></TD><TD><xsl:value-of select="outlet_temperature"/></TD><TD><xsl:value-of select="air_fuel"/></TD><TD><xsl:value-of select="room_temperature"/></TD><TD><xsl:value-of select="exhaust_temperature"/></TD><TD><xsl:value-of select="cylinder_length"/></TD><TD><xsl:value-of select="cylinder_radius"/></TD><TD><xsl:value-of select="brake_diameter"/></TD><TD><xsl:value-of select="calotific_fuel"/></TD><TD><xsl:value-of select="steam_exhaust"/></TD><TD><xsl:value-of select="specific_heat_steam"/></TD><TD><xsl:value-of select="specific_heat_dry"/></TD><TD><xsl:value-of select="indicated_hp"/></TD><TD><xsl:value-of select="brake_hp"/></TD><TD><xsl:value-of select="thermal_efficiency"/></TD><TD><xsl:value-of select="energy_supplied"/></TD><TD><xsl:value-of select="heat_equivalent_ip"/></TD><TD><xsl:value-of select="energy_coolingwater"/></TD><TD><xsl:value-of select="mass_exhaustgas"/></TD><TD><xsl:value-of select="energy_exhaustgas"/></TD><TD><xsl:value-of select="energy_steam"/></TD><TD><xsl:value-of select="total_energy_exhaustgas"/></TD><TD><xsl:value-of select="unaccounted_energyloss"/></TD><TD><xsl:value-of select="f"/></TD><TD><xsl:value-of select="poles"/></TD><TD><xsl:value-of select="wm"/></TD><TD><xsl:value-of select="Pconv"/></TD><TD><xsl:value-of select="Tind"/></TD><TD><xsl:value-of select="EA"/></TD><TD><xsl:value-of select="Vo"/></TD><TD><xsl:value-of select="Rs1"/></TD><TD><xsl:value-of select="Xs1"/></TD><TD><xsl:value-of select="delta"/></TD><TD><xsl:value-of select="IA"/></TD><TD><xsl:value-of select="Ploss"/></TD><TD><xsl:value-of select="Pout"/></TD><TD><xsl:value-of select="effe"/></TD><TD><xsl:value-of select="effo"/></TD><TD><xsl:value-of select="Rated_V"/></TD><TD><xsl:value-of select="Rated_VA"/></TD><TD><xsl:value-of select="Rs"/></TD><TD><xsl:value-of select="Xs"/></TD><TD><xsl:value-of select="Rg"/></TD><TD><xsl:value-of select="Xg"/></TD><TD><xsl:value-of select="voltage_A"/></TD><TD><xsl:value-of select="voltage_B"/></TD><TD><xsl:value-of select="voltage_C"/></TD><TD><xsl:value-of select="current_A"/></TD><TD><xsl:value-of select="current_B"/></TD><TD><xsl:value-of select="current_C"/></TD><TD><xsl:value-of select="EfA"/></TD><TD><xsl:value-of select="EfB"/></TD><TD><xsl:value-of select="EfC"/></TD><TD><xsl:value-of select="power_A"/></TD><TD><xsl:value-of select="power_B"/></TD><TD><xsl:value-of select="power_C"/></TD><TD><xsl:value-of select="power_A_sch"/></TD><TD><xsl:value-of select="power_B_sch"/></TD><TD><xsl:value-of select="power_C_sch"/></TD><TD><xsl:value-of select="EfA_sch"/></TD><TD><xsl:value-of select="EfB_sch"/></TD><TD><xsl:value-of select="EfC_sch"/></TD><TD><xsl:value-of select="SlackBus"/></TD><TD><xsl:value-of select="phases"/></TD></TR>
</xsl:for-each></TABLE>
<H4>windturb_dg objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>Gen_status</TH><TH>Gen_type</TH><TH>Gen_mode</TH><TH>Turbine_Model</TH><TH>turbine_height</TH><TH>blade_diam</TH><TH>cut_in_ws</TH><TH>cut_out_ws</TH><TH>ws_rated</TH><TH>ws_maxcp</TH><TH>Cp_max</TH><TH>Cp_rated</TH><TH>Rated_VA</TH><TH>Rated_V</TH><TH>Pconv</TH><TH>WSadj</TH><TH>Wind_Speed</TH><TH>R_stator</TH><TH>X_stator</TH><TH>R_rotor</TH><TH>X_rotor</TH><TH>R_core</TH><TH>X_magnetic</TH><TH>Max_Vrotor</TH><TH>Min_Vrotor</TH><TH>Max_Ef</TH><TH>Min_Ef</TH><TH>Rs</TH><TH>Xs</TH><TH>Rg</TH><TH>Xg</TH><TH>pf</TH><TH>GenElecEff</TH><TH>TotalRealPow</TH><TH>TotalReacPow</TH><TH>voltage_A</TH><TH>voltage_B</TH><TH>voltage_C</TH><TH>current_A</TH><TH>current_B</TH><TH>current_C</TH><TH>EfA</TH><TH>EfB</TH><TH>EfC</TH><TH>Vrotor_A</TH><TH>Vrotor_B</TH><TH>Vrotor_C</TH><TH>Irotor_A</TH><TH>Irotor_B</TH><TH>Irotor_C</TH><TH>phases</TH></TR>
<xsl:for-each select="generators/windturb_dg_list/windturb_dg"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="Gen_status"/></TD><TD><xsl:value-of select="Gen_type"/></TD><TD><xsl:value-of select="Gen_mode"/></TD><TD><xsl:value-of select="Turbine_Model"/></TD><TD><xsl:value-of select="turbine_height"/></TD><TD><xsl:value-of select="blade_diam"/></TD><TD><xsl:value-of select="cut_in_ws"/></TD><TD><xsl:value-of select="cut_out_ws"/></TD><TD><xsl:value-of select="ws_rated"/></TD><TD><xsl:value-of select="ws_maxcp"/></TD><TD><xsl:value-of select="Cp_max"/></TD><TD><xsl:value-of select="Cp_rated"/></TD><TD><xsl:value-of select="Rated_VA"/></TD><TD><xsl:value-of select="Rated_V"/></TD><TD><xsl:value-of select="Pconv"/></TD><TD><xsl:value-of select="WSadj"/></TD><TD><xsl:value-of select="Wind_Speed"/></TD><TD><xsl:value-of select="R_stator"/></TD><TD><xsl:value-of select="X_stator"/></TD><TD><xsl:value-of select="R_rotor"/></TD><TD><xsl:value-of select="X_rotor"/></TD><TD><xsl:value-of select="R_core"/></TD><TD><xsl:value-of select="X_magnetic"/></TD><TD><xsl:value-of select="Max_Vrotor"/></TD><TD><xsl:value-of select="Min_Vrotor"/></TD><TD><xsl:value-of select="Max_Ef"/></TD><TD><xsl:value-of select="Min_Ef"/></TD><TD><xsl:value-of select="Rs"/></TD><TD><xsl:value-of select="Xs"/></TD><TD><xsl:value-of select="Rg"/></TD><TD><xsl:value-of select="Xg"/></TD><TD><xsl:value-of select="pf"/></TD><TD><xsl:value-of select="GenElecEff"/></TD><TD><xsl:value-of select="TotalRealPow"/></TD><TD><xsl:value-of select="TotalReacPow"/></TD><TD><xsl:value-of select="voltage_A"/></TD><TD><xsl:value-of select="voltage_B"/></TD><TD><xsl:value-of select="voltage_C"/></TD><TD><xsl:value-of select="current_A"/></TD><TD><xsl:value-of select="current_B"/></TD><TD><xsl:value-of select="current_C"/></TD><TD><xsl:value-of select="EfA"/></TD><TD><xsl:value-of select="EfB"/></TD><TD><xsl:value-of select="EfC"/></TD><TD><xsl:value-of select="Vrotor_A"/></TD><TD><xsl:value-of select="Vrotor_B"/></TD><TD><xsl:value-of select="Vrotor_C"/></TD><TD><xsl:value-of select="Irotor_A"/></TD><TD><xsl:value-of select="Irotor_B"/></TD><TD><xsl:value-of select="Irotor_C"/></TD><TD><xsl:value-of select="phases"/></TD></TR>
</xsl:for-each></TABLE>
<H4>power_electronics objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>generator_mode</TH><TH>generator_status</TH><TH>converter_type</TH><TH>switch_type</TH><TH>filter_type</TH><TH>filter_implementation</TH><TH>filter_frequency</TH><TH>power_type</TH><TH>Rated_kW</TH><TH>Max_P</TH><TH>Min_P</TH><TH>Rated_kVA</TH><TH>Rated_kV</TH><TH>phases</TH></TR>
<xsl:for-each select="generators/power_electronics_list/power_electronics"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="generator_mode"/></TD><TD><xsl:value-of select="generator_status"/></TD><TD><xsl:value-of select="converter_type"/></TD><TD><xsl:value-of select="switch_type"/></TD><TD><xsl:value-of select="filter_type"/></TD><TD><xsl:value-of select="filter_implementation"/></TD><TD><xsl:value-of select="filter_frequency"/></TD><TD><xsl:value-of select="power_type"/></TD><TD><xsl:value-of select="Rated_kW"/></TD><TD><xsl:value-of select="Max_P"/></TD><TD><xsl:value-of select="Min_P"/></TD><TD><xsl:value-of select="Rated_kVA"/></TD><TD><xsl:value-of select="Rated_kV"/></TD><TD><xsl:value-of select="phases"/></TD></TR>
</xsl:for-each></TABLE>
<H4>energy_storage objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>generator_mode</TH><TH>generator_status</TH><TH>power_type</TH><TH>Rinternal</TH><TH>V_Max</TH><TH>I_Max</TH><TH>E_Max</TH><TH>Energy</TH><TH>efficiency</TH><TH>Rated_kVA</TH><TH>V_Out</TH><TH>I_Out</TH><TH>VA_Out</TH><TH>V_In</TH><TH>I_In</TH><TH>V_Internal</TH><TH>I_Internal</TH><TH>I_Prev</TH><TH>phases</TH></TR>
<xsl:for-each select="generators/energy_storage_list/energy_storage"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="generator_mode"/></TD><TD><xsl:value-of select="generator_status"/></TD><TD><xsl:value-of select="power_type"/></TD><TD><xsl:value-of select="Rinternal"/></TD><TD><xsl:value-of select="V_Max"/></TD><TD><xsl:value-of select="I_Max"/></TD><TD><xsl:value-of select="E_Max"/></TD><TD><xsl:value-of select="Energy"/></TD><TD><xsl:value-of select="efficiency"/></TD><TD><xsl:value-of select="Rated_kVA"/></TD><TD><xsl:value-of select="V_Out"/></TD><TD><xsl:value-of select="I_Out"/></TD><TD><xsl:value-of select="VA_Out"/></TD><TD><xsl:value-of select="V_In"/></TD><TD><xsl:value-of select="I_In"/></TD><TD><xsl:value-of select="V_Internal"/></TD><TD><xsl:value-of select="I_Internal"/></TD><TD><xsl:value-of select="I_Prev"/></TD><TD><xsl:value-of select="phases"/></TD></TR>
</xsl:for-each></TABLE>
<H4>inverter objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>inverter_type</TH><TH>four_quadrant_control_mode</TH><TH>generator_status</TH><TH>generator_mode</TH><TH>V_In</TH><TH>I_In</TH><TH>VA_In</TH><TH>VA_Out</TH><TH>Vdc</TH><TH>phaseA_V_Out</TH><TH>phaseB_V_Out</TH><TH>phaseC_V_Out</TH><TH>phaseA_I_Out</TH><TH>phaseB_I_Out</TH><TH>phaseC_I_Out</TH><TH>power_A</TH><TH>power_B</TH><TH>power_C</TH><TH>P_Out</TH><TH>Q_Out</TH><TH>power_in</TH><TH>rated_power</TH><TH>rated_battery_power</TH><TH>inverter_efficiency</TH><TH>battery_soc</TH><TH>soc_reserve</TH><TH>power_factor</TH><TH>phases</TH><TH>use_multipoint_efficiency</TH><TH>inverter_manufacturer</TH><TH>maximum_dc_power</TH><TH>maximum_dc_voltage</TH><TH>minimum_dc_power</TH><TH>c_0</TH><TH>c_1</TH><TH>c_2</TH><TH>c_3</TH></TR>
<xsl:for-each select="generators/inverter_list/inverter"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="inverter_type"/></TD><TD><xsl:value-of select="four_quadrant_control_mode"/></TD><TD><xsl:value-of select="generator_status"/></TD><TD><xsl:value-of select="generator_mode"/></TD><TD><xsl:value-of select="V_In"/></TD><TD><xsl:value-of select="I_In"/></TD><TD><xsl:value-of select="VA_In"/></TD><TD><xsl:value-of select="VA_Out"/></TD><TD><xsl:value-of select="Vdc"/></TD><TD><xsl:value-of select="phaseA_V_Out"/></TD><TD><xsl:value-of select="phaseB_V_Out"/></TD><TD><xsl:value-of select="phaseC_V_Out"/></TD><TD><xsl:value-of select="phaseA_I_Out"/></TD><TD><xsl:value-of select="phaseB_I_Out"/></TD><TD><xsl:value-of select="phaseC_I_Out"/></TD><TD><xsl:value-of select="power_A"/></TD><TD><xsl:value-of select="power_B"/></TD><TD><xsl:value-of select="power_C"/></TD><TD><xsl:value-of select="P_Out"/></TD><TD><xsl:value-of select="Q_Out"/></TD><TD><xsl:value-of select="power_in"/></TD><TD><xsl:value-of select="rated_power"/></TD><TD><xsl:value-of select="rated_battery_power"/></TD><TD><xsl:value-of select="inverter_efficiency"/></TD><TD><xsl:value-of select="battery_soc"/></TD><TD><xsl:value-of select="soc_reserve"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="use_multipoint_efficiency"/></TD><TD><xsl:value-of select="inverter_manufacturer"/></TD><TD><xsl:value-of select="maximum_dc_power"/></TD><TD><xsl:value-of select="maximum_dc_voltage"/></TD><TD><xsl:value-of select="minimum_dc_power"/></TD><TD><xsl:value-of select="c_0"/></TD><TD><xsl:value-of select="c_1"/></TD><TD><xsl:value-of select="c_2"/></TD><TD><xsl:value-of select="c_3"/></TD></TR>
</xsl:for-each></TABLE>
<H4>dc_dc_converter objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>dc_dc_converter_type</TH><TH>generator_mode</TH><TH>V_Out</TH><TH>I_Out</TH><TH>Vdc</TH><TH>VA_Out</TH><TH>P_Out</TH><TH>Q_Out</TH><TH>service_ratio</TH><TH>V_In</TH><TH>I_In</TH><TH>VA_In</TH><TH>phases</TH></TR>
<xsl:for-each select="generators/dc_dc_converter_list/dc_dc_converter"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="dc_dc_converter_type"/></TD><TD><xsl:value-of select="generator_mode"/></TD><TD><xsl:value-of select="V_Out"/></TD><TD><xsl:value-of select="I_Out"/></TD><TD><xsl:value-of select="Vdc"/></TD><TD><xsl:value-of select="VA_Out"/></TD><TD><xsl:value-of select="P_Out"/></TD><TD><xsl:value-of select="Q_Out"/></TD><TD><xsl:value-of select="service_ratio"/></TD><TD><xsl:value-of select="V_In"/></TD><TD><xsl:value-of select="I_In"/></TD><TD><xsl:value-of select="VA_In"/></TD><TD><xsl:value-of select="phases"/></TD></TR>
</xsl:for-each></TABLE>
<H4>rectifier objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>rectifier_type</TH><TH>generator_mode</TH><TH>V_Out</TH><TH>I_Out</TH><TH>VA_Out</TH><TH>P_Out</TH><TH>Q_Out</TH><TH>Vdc</TH><TH>phaseA_V_In</TH><TH>phaseB_V_In</TH><TH>phaseC_V_In</TH><TH>phaseA_I_In</TH><TH>phaseB_I_In</TH><TH>phaseC_I_In</TH><TH>power_A_In</TH><TH>power_B_In</TH><TH>power_C_In</TH><TH>phases</TH></TR>
<xsl:for-each select="generators/rectifier_list/rectifier"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="rectifier_type"/></TD><TD><xsl:value-of select="generator_mode"/></TD><TD><xsl:value-of select="V_Out"/></TD><TD><xsl:value-of select="I_Out"/></TD><TD><xsl:value-of select="VA_Out"/></TD><TD><xsl:value-of select="P_Out"/></TD><TD><xsl:value-of select="Q_Out"/></TD><TD><xsl:value-of select="Vdc"/></TD><TD><xsl:value-of select="phaseA_V_In"/></TD><TD><xsl:value-of select="phaseB_V_In"/></TD><TD><xsl:value-of select="phaseC_V_In"/></TD><TD><xsl:value-of select="phaseA_I_In"/></TD><TD><xsl:value-of select="phaseB_I_In"/></TD><TD><xsl:value-of select="phaseC_I_In"/></TD><TD><xsl:value-of select="power_A_In"/></TD><TD><xsl:value-of select="power_B_In"/></TD><TD><xsl:value-of select="power_C_In"/></TD><TD><xsl:value-of select="phases"/></TD></TR>
</xsl:for-each></TABLE>
<H4>microturbine objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>generator_mode</TH><TH>generator_status</TH><TH>power_type</TH><TH>Rinternal</TH><TH>Rload</TH><TH>V_Max</TH><TH>I_Max</TH><TH>frequency</TH><TH>Max_Frequency</TH><TH>Min_Frequency</TH><TH>Fuel_Used</TH><TH>Heat_Out</TH><TH>KV</TH><TH>Power_Angle</TH><TH>Max_P</TH><TH>Min_P</TH><TH>phaseA_V_Out</TH><TH>phaseB_V_Out</TH><TH>phaseC_V_Out</TH><TH>phaseA_I_Out</TH><TH>phaseB_I_Out</TH><TH>phaseC_I_Out</TH><TH>power_A_Out</TH><TH>power_B_Out</TH><TH>power_C_Out</TH><TH>VA_Out</TH><TH>pf_Out</TH><TH>E_A_Internal</TH><TH>E_B_Internal</TH><TH>E_C_Internal</TH><TH>efficiency</TH><TH>Rated_kVA</TH><TH>phases</TH></TR>
<xsl:for-each select="generators/microturbine_list/microturbine"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="generator_mode"/></TD><TD><xsl:value-of select="generator_status"/></TD><TD><xsl:value-of select="power_type"/></TD><TD><xsl:value-of select="Rinternal"/></TD><TD><xsl:value-of select="Rload"/></TD><TD><xsl:value-of select="V_Max"/></TD><TD><xsl:value-of select="I_Max"/></TD><TD><xsl:value-of select="frequency"/></TD><TD><xsl:value-of select="Max_Frequency"/></TD><TD><xsl:value-of select="Min_Frequency"/></TD><TD><xsl:value-of select="Fuel_Used"/></TD><TD><xsl:value-of select="Heat_Out"/></TD><TD><xsl:value-of select="KV"/></TD><TD><xsl:value-of select="Power_Angle"/></TD><TD><xsl:value-of select="Max_P"/></TD><TD><xsl:value-of select="Min_P"/></TD><TD><xsl:value-of select="phaseA_V_Out"/></TD><TD><xsl:value-of select="phaseB_V_Out"/></TD><TD><xsl:value-of select="phaseC_V_Out"/></TD><TD><xsl:value-of select="phaseA_I_Out"/></TD><TD><xsl:value-of select="phaseB_I_Out"/></TD><TD><xsl:value-of select="phaseC_I_Out"/></TD><TD><xsl:value-of select="power_A_Out"/></TD><TD><xsl:value-of select="power_B_Out"/></TD><TD><xsl:value-of select="power_C_Out"/></TD><TD><xsl:value-of select="VA_Out"/></TD><TD><xsl:value-of select="pf_Out"/></TD><TD><xsl:value-of select="E_A_Internal"/></TD><TD><xsl:value-of select="E_B_Internal"/></TD><TD><xsl:value-of select="E_C_Internal"/></TD><TD><xsl:value-of select="efficiency"/></TD><TD><xsl:value-of select="Rated_kVA"/></TD><TD><xsl:value-of select="phases"/></TD></TR>
</xsl:for-each></TABLE>
<H4>battery objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>generator_mode</TH><TH>additional_controls</TH><TH>generator_status</TH><TH>rfb_size</TH><TH>power_type</TH><TH>battery_state</TH><TH>number_battery_state_changes</TH><TH>monitored_power</TH><TH>power_set_high</TH><TH>power_set_low</TH><TH>power_set_high_highT</TH><TH>power_set_low_highT</TH><TH>check_power_low</TH><TH>check_power_high</TH><TH>voltage_set_high</TH><TH>voltage_set_low</TH><TH>deadband</TH><TH>sensitivity</TH><TH>high_temperature</TH><TH>midpoint_temperature</TH><TH>low_temperature</TH><TH>scheduled_power</TH><TH>Rinternal</TH><TH>V_Max</TH><TH>I_Max</TH><TH>E_Max</TH><TH>P_Max</TH><TH>power_factor</TH><TH>Energy</TH><TH>efficiency</TH><TH>base_efficiency</TH><TH>parasitic_power_draw</TH><TH>Rated_kVA</TH><TH>V_Out</TH><TH>I_Out</TH><TH>VA_Out</TH><TH>V_In</TH><TH>I_In</TH><TH>V_Internal</TH><TH>I_Internal</TH><TH>I_Prev</TH><TH>power_transferred</TH><TH>phases</TH><TH>use_internal_battery_model</TH><TH>battery_type</TH><TH>nominal_voltage</TH><TH>rated_power</TH><TH>battery_capacity</TH><TH>round_trip_efficiency</TH><TH>state_of_charge</TH><TH>battery_load</TH><TH>reserve_state_of_charge</TH></TR>
<xsl:for-each select="generators/battery_list/battery"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="generator_mode"/></TD><TD><xsl:value-of select="additional_controls"/></TD><TD><xsl:value-of select="generator_status"/></TD><TD><xsl:value-of select="rfb_size"/></TD><TD><xsl:value-of select="power_type"/></TD><TD><xsl:value-of select="battery_state"/></TD><TD><xsl:value-of select="number_battery_state_changes"/></TD><TD><xsl:value-of select="monitored_power"/></TD><TD><xsl:value-of select="power_set_high"/></TD><TD><xsl:value-of select="power_set_low"/></TD><TD><xsl:value-of select="power_set_high_highT"/></TD><TD><xsl:value-of select="power_set_low_highT"/></TD><TD><xsl:value-of select="check_power_low"/></TD><TD><xsl:value-of select="check_power_high"/></TD><TD><xsl:value-of select="voltage_set_high"/></TD><TD><xsl:value-of select="voltage_set_low"/></TD><TD><xsl:value-of select="deadband"/></TD><TD><xsl:value-of select="sensitivity"/></TD><TD><xsl:value-of select="high_temperature"/></TD><TD><xsl:value-of select="midpoint_temperature"/></TD><TD><xsl:value-of select="low_temperature"/></TD><TD><xsl:value-of select="scheduled_power"/></TD><TD><xsl:value-of select="Rinternal"/></TD><TD><xsl:value-of select="V_Max"/></TD><TD><xsl:value-of select="I_Max"/></TD><TD><xsl:value-of select="E_Max"/></TD><TD><xsl:value-of select="P_Max"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="Energy"/></TD><TD><xsl:value-of select="efficiency"/></TD><TD><xsl:value-of select="base_efficiency"/></TD><TD><xsl:value-of select="parasitic_power_draw"/></TD><TD><xsl:value-of select="Rated_kVA"/></TD><TD><xsl:value-of select="V_Out"/></TD><TD><xsl:value-of select="I_Out"/></TD><TD><xsl:value-of select="VA_Out"/></TD><TD><xsl:value-of select="V_In"/></TD><TD><xsl:value-of select="I_In"/></TD><TD><xsl:value-of select="V_Internal"/></TD><TD><xsl:value-of select="I_Internal"/></TD><TD><xsl:value-of select="I_Prev"/></TD><TD><xsl:value-of select="power_transferred"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="use_internal_battery_model"/></TD><TD><xsl:value-of select="battery_type"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD><TD><xsl:value-of select="rated_power"/></TD><TD><xsl:value-of select="battery_capacity"/></TD><TD><xsl:value-of select="round_trip_efficiency"/></TD><TD><xsl:value-of select="state_of_charge"/></TD><TD><xsl:value-of select="battery_load"/></TD><TD><xsl:value-of select="reserve_state_of_charge"/></TD></TR>
</xsl:for-each></TABLE>
<H4>solar objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>generator_mode</TH><TH>generator_status</TH><TH>panel_type</TH><TH>power_type</TH><TH>INSTALLATION_TYPE</TH><TH>SOLAR_TILT_MODEL</TH><TH>SOLAR_POWER_MODEL</TH><TH>a_coeff</TH><TH>b_coeff</TH><TH>dT_coeff</TH><TH>T_coeff</TH><TH>NOCT</TH><TH>Tmodule</TH><TH>Tambient</TH><TH>wind_speed</TH><TH>ambient_temperature</TH><TH>Insolation</TH><TH>Rinternal</TH><TH>Rated_Insolation</TH><TH>Pmax_temp_coeff</TH><TH>Voc_temp_coeff</TH><TH>V_Max</TH><TH>Voc_Max</TH><TH>Voc</TH><TH>efficiency</TH><TH>area</TH><TH>soiling</TH><TH>derating</TH><TH>Tcell</TH><TH>Rated_kVA</TH><TH>P_Out</TH><TH>V_Out</TH><TH>I_Out</TH><TH>VA_Out</TH><TH>weather</TH><TH>shading_factor</TH><TH>tilt_angle</TH><TH>orientation_azimuth</TH><TH>latitude_angle_fix</TH><TH>orientation</TH><TH>phases</TH></TR>
<xsl:for-each select="generators/solar_list/solar"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="generator_mode"/></TD><TD><xsl:value-of select="generator_status"/></TD><TD><xsl:value-of select="panel_type"/></TD><TD><xsl:value-of select="power_type"/></TD><TD><xsl:value-of select="INSTALLATION_TYPE"/></TD><TD><xsl:value-of select="SOLAR_TILT_MODEL"/></TD><TD><xsl:value-of select="SOLAR_POWER_MODEL"/></TD><TD><xsl:value-of select="a_coeff"/></TD><TD><xsl:value-of select="b_coeff"/></TD><TD><xsl:value-of select="dT_coeff"/></TD><TD><xsl:value-of select="T_coeff"/></TD><TD><xsl:value-of select="NOCT"/></TD><TD><xsl:value-of select="Tmodule"/></TD><TD><xsl:value-of select="Tambient"/></TD><TD><xsl:value-of select="wind_speed"/></TD><TD><xsl:value-of select="ambient_temperature"/></TD><TD><xsl:value-of select="Insolation"/></TD><TD><xsl:value-of select="Rinternal"/></TD><TD><xsl:value-of select="Rated_Insolation"/></TD><TD><xsl:value-of select="Pmax_temp_coeff"/></TD><TD><xsl:value-of select="Voc_temp_coeff"/></TD><TD><xsl:value-of select="V_Max"/></TD><TD><xsl:value-of select="Voc_Max"/></TD><TD><xsl:value-of select="Voc"/></TD><TD><xsl:value-of select="efficiency"/></TD><TD><xsl:value-of select="area"/></TD><TD><xsl:value-of select="soiling"/></TD><TD><xsl:value-of select="derating"/></TD><TD><xsl:value-of select="Tcell"/></TD><TD><xsl:value-of select="Rated_kVA"/></TD><TD><xsl:value-of select="P_Out"/></TD><TD><xsl:value-of select="V_Out"/></TD><TD><xsl:value-of select="I_Out"/></TD><TD><xsl:value-of select="VA_Out"/></TD><TD><a href="#{weather}"><xsl:value-of select="weather"/></a></TD><TD><xsl:value-of select="shading_factor"/></TD><TD><xsl:value-of select="tilt_angle"/></TD><TD><xsl:value-of select="orientation_azimuth"/></TD><TD><xsl:value-of select="latitude_angle_fix"/></TD><TD><xsl:value-of select="orientation"/></TD><TD><xsl:value-of select="phases"/></TD></TR>
</xsl:for-each></TABLE>
<H3><A NAME="modules_market">market</A></H3><TABLE BORDER="1">
<TR><TH>version.major</TH><TD><xsl:value-of select="market/version.major"/></TD></TR><TR><TH>version.minor</TH><TD><xsl:value-of select="market/version.minor"/></TD></TR><TR><TH>bid_offset</TH><TD><xsl:value-of select="market/bid_offset"/></TD></TR></TABLE>
<H4>auction objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>unit</TH><TH>period</TH><TH>latency</TH><TH>market_id</TH><TH>network</TH><TH>verbose</TH><TH>linkref</TH><TH>pricecap</TH><TH>price_cap</TH><TH>special_mode</TH><TH>statistic_mode</TH><TH>fixed_price</TH><TH>fixed_quantity</TH><TH>capacity_reference_object</TH><TH>capacity_reference_property</TH><TH>capacity_reference_bid_price</TH><TH>max_capacity_reference_bid_quantity</TH><TH>capacity_reference_bid_quantity</TH><TH>init_price</TH><TH>init_stdev</TH><TH>future_mean_price</TH><TH>use_future_mean_price</TH><TH>current_market.start_time</TH><TH>current_market.end_time</TH><TH>current_market.clearing_price</TH><TH>current_market.clearing_quantity</TH><TH>current_market.clearing_type</TH><TH>current_market.marginal_quantity_load</TH><TH>current_market.marginal_quantity</TH><TH>current_market.marginal_quantity_bid</TH><TH>current_market.marginal_quantity_frac</TH><TH>current_market.seller_total_quantity</TH><TH>current_market.buyer_total_quantity</TH><TH>current_market.seller_min_price</TH><TH>current_market.buyer_total_unrep</TH><TH>current_market.cap_ref_unrep</TH><TH>next_market.start_time</TH><TH>next_market.end_time</TH><TH>next_market.clearing_price</TH><TH>next_market.clearing_quantity</TH><TH>next_market.clearing_type</TH><TH>next_market.marginal_quantity_load</TH><TH>next_market.marginal_quantity_bid</TH><TH>next_market.marginal_quantity_frac</TH><TH>next_market.seller_total_quantity</TH><TH>next_market.buyer_total_quantity</TH><TH>next_market.seller_min_price</TH><TH>next_market.cap_ref_unrep</TH><TH>past_market.start_time</TH><TH>past_market.end_time</TH><TH>past_market.clearing_price</TH><TH>past_market.clearing_quantity</TH><TH>past_market.clearing_type</TH><TH>past_market.marginal_quantity_load</TH><TH>past_market.marginal_quantity_bid</TH><TH>past_market.marginal_quantity_frac</TH><TH>past_market.seller_total_quantity</TH><TH>past_market.buyer_total_quantity</TH><TH>past_market.seller_min_price</TH><TH>past_market.cap_ref_unrep</TH><TH>margin_mode</TH><TH>warmup</TH><TH>ignore_pricecap</TH><TH>transaction_log_file</TH><TH>transaction_log_limit</TH><TH>curve_log_file</TH><TH>curve_log_limit</TH><TH>curve_log_info</TH></TR>
<xsl:for-each select="market/auction_list/auction"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="unit"/></TD><TD><xsl:value-of select="period"/></TD><TD><xsl:value-of select="latency"/></TD><TD><xsl:value-of select="market_id"/></TD><TD><a href="#{network}"><xsl:value-of select="network"/></a></TD><TD><xsl:value-of select="verbose"/></TD><TD><a href="#{linkref}"><xsl:value-of select="linkref"/></a></TD><TD><xsl:value-of select="pricecap"/></TD><TD><xsl:value-of select="price_cap"/></TD><TD><xsl:value-of select="special_mode"/></TD><TD><xsl:value-of select="statistic_mode"/></TD><TD><xsl:value-of select="fixed_price"/></TD><TD><xsl:value-of select="fixed_quantity"/></TD><TD><a href="#{capacity_reference_object}"><xsl:value-of select="capacity_reference_object"/></a></TD><TD><xsl:value-of select="capacity_reference_property"/></TD><TD><xsl:value-of select="capacity_reference_bid_price"/></TD><TD><xsl:value-of select="max_capacity_reference_bid_quantity"/></TD><TD><xsl:value-of select="capacity_reference_bid_quantity"/></TD><TD><xsl:value-of select="init_price"/></TD><TD><xsl:value-of select="init_stdev"/></TD><TD><xsl:value-of select="future_mean_price"/></TD><TD><xsl:value-of select="use_future_mean_price"/></TD><TD><xsl:value-of select="current_market.start_time"/></TD><TD><xsl:value-of select="current_market.end_time"/></TD><TD><xsl:value-of select="current_market.clearing_price"/></TD><TD><xsl:value-of select="current_market.clearing_quantity"/></TD><TD><xsl:value-of select="current_market.clearing_type"/></TD><TD><xsl:value-of select="current_market.marginal_quantity_load"/></TD><TD><xsl:value-of select="current_market.marginal_quantity"/></TD><TD><xsl:value-of select="current_market.marginal_quantity_bid"/></TD><TD><xsl:value-of select="current_market.marginal_quantity_frac"/></TD><TD><xsl:value-of select="current_market.seller_total_quantity"/></TD><TD><xsl:value-of select="current_market.buyer_total_quantity"/></TD><TD><xsl:value-of select="current_market.seller_min_price"/></TD><TD><xsl:value-of select="current_market.buyer_total_unrep"/></TD><TD><xsl:value-of select="current_market.cap_ref_unrep"/></TD><TD><xsl:value-of select="next_market.start_time"/></TD><TD><xsl:value-of select="next_market.end_time"/></TD><TD><xsl:value-of select="next_market.clearing_price"/></TD><TD><xsl:value-of select="next_market.clearing_quantity"/></TD><TD><xsl:value-of select="next_market.clearing_type"/></TD><TD><xsl:value-of select="next_market.marginal_quantity_load"/></TD><TD><xsl:value-of select="next_market.marginal_quantity_bid"/></TD><TD><xsl:value-of select="next_market.marginal_quantity_frac"/></TD><TD><xsl:value-of select="next_market.seller_total_quantity"/></TD><TD><xsl:value-of select="next_market.buyer_total_quantity"/></TD><TD><xsl:value-of select="next_market.seller_min_price"/></TD><TD><xsl:value-of select="next_market.cap_ref_unrep"/></TD><TD><xsl:value-of select="past_market.start_time"/></TD><TD><xsl:value-of select="past_market.end_time"/></TD><TD><xsl:value-of select="past_market.clearing_price"/></TD><TD><xsl:value-of select="past_market.clearing_quantity"/></TD><TD><xsl:value-of select="past_market.clearing_type"/></TD><TD><xsl:value-of select="past_market.marginal_quantity_load"/></TD><TD><xsl:value-of select="past_market.marginal_quantity_bid"/></TD><TD><xsl:value-of select="past_market.marginal_quantity_frac"/></TD><TD><xsl:value-of select="past_market.seller_total_quantity"/></TD><TD><xsl:value-of select="past_market.buyer_total_quantity"/></TD><TD><xsl:value-of select="past_market.seller_min_price"/></TD><TD><xsl:value-of select="past_market.cap_ref_unrep"/></TD><TD><xsl:value-of select="margin_mode"/></TD><TD><xsl:value-of select="warmup"/></TD><TD><xsl:value-of select="ignore_pricecap"/></TD><TD><xsl:value-of select="transaction_log_file"/></TD><TD><xsl:value-of select="transaction_log_limit"/></TD><TD><xsl:value-of select="curve_log_file"/></TD><TD><xsl:value-of select="curve_log_limit"/></TD><TD><xsl:value-of select="curve_log_info"/></TD></TR>
</xsl:for-each></TABLE>
<H4>controller objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>simple_mode</TH><TH>bid_mode</TH><TH>use_override</TH><TH>ramp_low</TH><TH>ramp_high</TH><TH>range_low</TH><TH>range_high</TH><TH>target</TH><TH>setpoint</TH><TH>demand</TH><TH>load</TH><TH>total</TH><TH>market</TH><TH>state</TH><TH>avg_target</TH><TH>std_target</TH><TH>bid_price</TH><TH>bid_quantity</TH><TH>set_temp</TH><TH>base_setpoint</TH><TH>period</TH><TH>control_mode</TH><TH>resolve_mode</TH><TH>slider_setting</TH><TH>slider_setting_heat</TH><TH>slider_setting_cool</TH><TH>override</TH><TH>heating_range_high</TH><TH>heating_range_low</TH><TH>heating_ramp_high</TH><TH>heating_ramp_low</TH><TH>cooling_range_high</TH><TH>cooling_range_low</TH><TH>cooling_ramp_high</TH><TH>cooling_ramp_low</TH><TH>heating_base_setpoint</TH><TH>cooling_base_setpoint</TH><TH>deadband</TH><TH>heating_setpoint</TH><TH>heating_demand</TH><TH>cooling_setpoint</TH><TH>cooling_demand</TH><TH>sliding_time_delay</TH><TH>average_target</TH><TH>standard_deviation_target</TH><TH>current_mode</TH><TH>dominant_mode</TH><TH>previous_mode</TH><TH>heat_max</TH><TH>cool_min</TH><TH>bid_delay</TH></TR>
<xsl:for-each select="market/controller_list/controller"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="simple_mode"/></TD><TD><xsl:value-of select="bid_mode"/></TD><TD><xsl:value-of select="use_override"/></TD><TD><xsl:value-of select="ramp_low"/></TD><TD><xsl:value-of select="ramp_high"/></TD><TD><xsl:value-of select="range_low"/></TD><TD><xsl:value-of select="range_high"/></TD><TD><xsl:value-of select="target"/></TD><TD><xsl:value-of select="setpoint"/></TD><TD><xsl:value-of select="demand"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="total"/></TD><TD><a href="#{market}"><xsl:value-of select="market"/></a></TD><TD><xsl:value-of select="state"/></TD><TD><xsl:value-of select="avg_target"/></TD><TD><xsl:value-of select="std_target"/></TD><TD><xsl:value-of select="bid_price"/></TD><TD><xsl:value-of select="bid_quantity"/></TD><TD><xsl:value-of select="set_temp"/></TD><TD><xsl:value-of select="base_setpoint"/></TD><TD><xsl:value-of select="period"/></TD><TD><xsl:value-of select="control_mode"/></TD><TD><xsl:value-of select="resolve_mode"/></TD><TD><xsl:value-of select="slider_setting"/></TD><TD><xsl:value-of select="slider_setting_heat"/></TD><TD><xsl:value-of select="slider_setting_cool"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="heating_range_high"/></TD><TD><xsl:value-of select="heating_range_low"/></TD><TD><xsl:value-of select="heating_ramp_high"/></TD><TD><xsl:value-of select="heating_ramp_low"/></TD><TD><xsl:value-of select="cooling_range_high"/></TD><TD><xsl:value-of select="cooling_range_low"/></TD><TD><xsl:value-of select="cooling_ramp_high"/></TD><TD><xsl:value-of select="cooling_ramp_low"/></TD><TD><xsl:value-of select="heating_base_setpoint"/></TD><TD><xsl:value-of select="cooling_base_setpoint"/></TD><TD><xsl:value-of select="deadband"/></TD><TD><xsl:value-of select="heating_setpoint"/></TD><TD><xsl:value-of select="heating_demand"/></TD><TD><xsl:value-of select="cooling_setpoint"/></TD><TD><xsl:value-of select="cooling_demand"/></TD><TD><xsl:value-of select="sliding_time_delay"/></TD><TD><xsl:value-of select="average_target"/></TD><TD><xsl:value-of select="standard_deviation_target"/></TD><TD><xsl:value-of select="current_mode"/></TD><TD><xsl:value-of select="dominant_mode"/></TD><TD><xsl:value-of select="previous_mode"/></TD><TD><xsl:value-of select="heat_max"/></TD><TD><xsl:value-of select="cool_min"/></TD><TD><xsl:value-of select="bid_delay"/></TD></TR>
</xsl:for-each></TABLE>
<H4>stubauction objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>unit</TH><TH>period</TH><TH>last.P</TH><TH>current_market.clearing_price</TH><TH>past_market.clearing_price</TH><TH>next.P</TH><TH>avg24</TH><TH>std24</TH><TH>avg72</TH><TH>std72</TH><TH>avg168</TH><TH>std168</TH><TH>market_id</TH><TH>verbose</TH><TH>control_mode</TH></TR>
<xsl:for-each select="market/stubauction_list/stubauction"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="unit"/></TD><TD><xsl:value-of select="period"/></TD><TD><xsl:value-of select="last.P"/></TD><TD><xsl:value-of select="current_market.clearing_price"/></TD><TD><xsl:value-of select="past_market.clearing_price"/></TD><TD><xsl:value-of select="next.P"/></TD><TD><xsl:value-of select="avg24"/></TD><TD><xsl:value-of select="std24"/></TD><TD><xsl:value-of select="avg72"/></TD><TD><xsl:value-of select="std72"/></TD><TD><xsl:value-of select="avg168"/></TD><TD><xsl:value-of select="std168"/></TD><TD><xsl:value-of select="market_id"/></TD><TD><xsl:value-of select="verbose"/></TD><TD><xsl:value-of select="control_mode"/></TD></TR>
</xsl:for-each></TABLE>
<H4>passive_controller objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>input_state</TH><TH>input_setpoint</TH><TH>input_chained</TH><TH>observation</TH><TH>mean_observation</TH><TH>stdev_observation</TH><TH>expectation</TH><TH>sensitivity</TH><TH>period</TH><TH>expectation_prop</TH><TH>expectation_obj</TH><TH>expectation_property</TH><TH>expectation_object</TH><TH>setpoint_prop</TH><TH>setpoint</TH><TH>state_prop</TH><TH>state_property</TH><TH>observation_obj</TH><TH>observation_prop</TH><TH>observation_object</TH><TH>observation_property</TH><TH>mean_observation_prop</TH><TH>stdev_observation_prop</TH><TH>stdev_observation_property</TH><TH>cycle_length</TH><TH>base_setpoint</TH><TH>critical_day</TH><TH>two_tier_cpp</TH><TH>daily_elasticity</TH><TH>sub_elasticity_first_second</TH><TH>sub_elasticity_first_third</TH><TH>second_tier_hours</TH><TH>third_tier_hours</TH><TH>first_tier_hours</TH><TH>first_tier_price</TH><TH>second_tier_price</TH><TH>third_tier_price</TH><TH>old_first_tier_price</TH><TH>old_second_tier_price</TH><TH>old_third_tier_price</TH><TH>Percent_change_in_price</TH><TH>Percent_change_in_peakoffpeak_ratio</TH><TH>Percent_change_in_Criticalpeakoffpeak_ratio</TH><TH>linearize_elasticity</TH><TH>price_offset</TH><TH>pool_pump_model</TH><TH>base_duty_cycle</TH><TH>distribution_type</TH><TH>comfort_level</TH><TH>range_high</TH><TH>range_low</TH><TH>ramp_high</TH><TH>ramp_low</TH><TH>prob_off</TH><TH>output_state</TH><TH>output_setpoint</TH><TH>control_mode</TH><TH>dlc_mode</TH><TH>cycle_length_off</TH><TH>cycle_length_on</TH></TR>
<xsl:for-each select="market/passive_controller_list/passive_controller"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="input_state"/></TD><TD><xsl:value-of select="input_setpoint"/></TD><TD><xsl:value-of select="input_chained"/></TD><TD><xsl:value-of select="observation"/></TD><TD><xsl:value-of select="mean_observation"/></TD><TD><xsl:value-of select="stdev_observation"/></TD><TD><xsl:value-of select="expectation"/></TD><TD><xsl:value-of select="sensitivity"/></TD><TD><xsl:value-of select="period"/></TD><TD><xsl:value-of select="expectation_prop"/></TD><TD><a href="#{expectation_obj}"><xsl:value-of select="expectation_obj"/></a></TD><TD><xsl:value-of select="expectation_property"/></TD><TD><a href="#{expectation_object}"><xsl:value-of select="expectation_object"/></a></TD><TD><xsl:value-of select="setpoint_prop"/></TD><TD><xsl:value-of select="setpoint"/></TD><TD><xsl:value-of select="state_prop"/></TD><TD><xsl:value-of select="state_property"/></TD><TD><a href="#{observation_obj}"><xsl:value-of select="observation_obj"/></a></TD><TD><xsl:value-of select="observation_prop"/></TD><TD><a href="#{observation_object}"><xsl:value-of select="observation_object"/></a></TD><TD><xsl:value-of select="observation_property"/></TD><TD><xsl:value-of select="mean_observation_prop"/></TD><TD><xsl:value-of select="stdev_observation_prop"/></TD><TD><xsl:value-of select="stdev_observation_property"/></TD><TD><xsl:value-of select="cycle_length"/></TD><TD><xsl:value-of select="base_setpoint"/></TD><TD><xsl:value-of select="critical_day"/></TD><TD><xsl:value-of select="two_tier_cpp"/></TD><TD><xsl:value-of select="daily_elasticity"/></TD><TD><xsl:value-of select="sub_elasticity_first_second"/></TD><TD><xsl:value-of select="sub_elasticity_first_third"/></TD><TD><xsl:value-of select="second_tier_hours"/></TD><TD><xsl:value-of select="third_tier_hours"/></TD><TD><xsl:value-of select="first_tier_hours"/></TD><TD><xsl:value-of select="first_tier_price"/></TD><TD><xsl:value-of select="second_tier_price"/></TD><TD><xsl:value-of select="third_tier_price"/></TD><TD><xsl:value-of select="old_first_tier_price"/></TD><TD><xsl:value-of select="old_second_tier_price"/></TD><TD><xsl:value-of select="old_third_tier_price"/></TD><TD><xsl:value-of select="Percent_change_in_price"/></TD><TD><xsl:value-of select="Percent_change_in_peakoffpeak_ratio"/></TD><TD><xsl:value-of select="Percent_change_in_Criticalpeakoffpeak_ratio"/></TD><TD><xsl:value-of select="linearize_elasticity"/></TD><TD><xsl:value-of select="price_offset"/></TD><TD><xsl:value-of select="pool_pump_model"/></TD><TD><xsl:value-of select="base_duty_cycle"/></TD><TD><xsl:value-of select="distribution_type"/></TD><TD><xsl:value-of select="comfort_level"/></TD><TD><xsl:value-of select="range_high"/></TD><TD><xsl:value-of select="range_low"/></TD><TD><xsl:value-of select="ramp_high"/></TD><TD><xsl:value-of select="ramp_low"/></TD><TD><xsl:value-of select="prob_off"/></TD><TD><xsl:value-of select="output_state"/></TD><TD><xsl:value-of select="output_setpoint"/></TD><TD><xsl:value-of select="control_mode"/></TD><TD><xsl:value-of select="dlc_mode"/></TD><TD><xsl:value-of select="cycle_length_off"/></TD><TD><xsl:value-of select="cycle_length_on"/></TD></TR>
</xsl:for-each></TABLE>
<H4>double_controller objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>thermostat_mode</TH><TH>last_mode</TH><TH>resolve_mode</TH><TH>setup_mode</TH><TH>bid_mode</TH><TH>last_mode_timer</TH><TH>cool_ramp_low</TH><TH>cool_ramp_high</TH><TH>cool_range_low</TH><TH>cool_range_high</TH><TH>cool_set_base</TH><TH>cool_setpoint</TH><TH>heat_ramp_low</TH><TH>heat_ramp_high</TH><TH>heat_range_low</TH><TH>heat_range_high</TH><TH>heat_set_base</TH><TH>heat_setpoint</TH><TH>temperature_name</TH><TH>cool_setpoint_name</TH><TH>cool_demand_name</TH><TH>heat_setpoint_name</TH><TH>heat_demand_name</TH><TH>load_name</TH><TH>total_load_name</TH><TH>deadband_name</TH><TH>state_name</TH><TH>market</TH><TH>market_period</TH><TH>bid_price</TH><TH>bid_quant</TH><TH>load</TH><TH>total</TH><TH>last_price</TH><TH>temperature</TH><TH>cool_bid</TH><TH>heat_bid</TH><TH>cool_demand</TH><TH>heat_demand</TH><TH>price</TH><TH>avg_price</TH><TH>stdev_price</TH></TR>
<xsl:for-each select="market/double_controller_list/double_controller"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="thermostat_mode"/></TD><TD><xsl:value-of select="last_mode"/></TD><TD><xsl:value-of select="resolve_mode"/></TD><TD><xsl:value-of select="setup_mode"/></TD><TD><xsl:value-of select="bid_mode"/></TD><TD><xsl:value-of select="last_mode_timer"/></TD><TD><xsl:value-of select="cool_ramp_low"/></TD><TD><xsl:value-of select="cool_ramp_high"/></TD><TD><xsl:value-of select="cool_range_low"/></TD><TD><xsl:value-of select="cool_range_high"/></TD><TD><xsl:value-of select="cool_set_base"/></TD><TD><xsl:value-of select="cool_setpoint"/></TD><TD><xsl:value-of select="heat_ramp_low"/></TD><TD><xsl:value-of select="heat_ramp_high"/></TD><TD><xsl:value-of select="heat_range_low"/></TD><TD><xsl:value-of select="heat_range_high"/></TD><TD><xsl:value-of select="heat_set_base"/></TD><TD><xsl:value-of select="heat_setpoint"/></TD><TD><xsl:value-of select="temperature_name"/></TD><TD><xsl:value-of select="cool_setpoint_name"/></TD><TD><xsl:value-of select="cool_demand_name"/></TD><TD><xsl:value-of select="heat_setpoint_name"/></TD><TD><xsl:value-of select="heat_demand_name"/></TD><TD><xsl:value-of select="load_name"/></TD><TD><xsl:value-of select="total_load_name"/></TD><TD><xsl:value-of select="deadband_name"/></TD><TD><xsl:value-of select="state_name"/></TD><TD><a href="#{market}"><xsl:value-of select="market"/></a></TD><TD><xsl:value-of select="market_period"/></TD><TD><xsl:value-of select="bid_price"/></TD><TD><xsl:value-of select="bid_quant"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="total"/></TD><TD><xsl:value-of select="last_price"/></TD><TD><xsl:value-of select="temperature"/></TD><TD><xsl:value-of select="cool_bid"/></TD><TD><xsl:value-of select="heat_bid"/></TD><TD><xsl:value-of select="cool_demand"/></TD><TD><xsl:value-of select="heat_demand"/></TD><TD><xsl:value-of select="price"/></TD><TD><xsl:value-of select="avg_price"/></TD><TD><xsl:value-of select="stdev_price"/></TD></TR>
</xsl:for-each></TABLE>
<H4>stub_bidder objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>bid_period</TH><TH>count</TH><TH>market</TH><TH>role</TH><TH>price</TH><TH>quantity</TH></TR>
<xsl:for-each select="market/stub_bidder_list/stub_bidder"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="bid_period"/></TD><TD><xsl:value-of select="count"/></TD><TD><a href="#{market}"><xsl:value-of select="market"/></a></TD><TD><xsl:value-of select="role"/></TD><TD><xsl:value-of select="price"/></TD><TD><xsl:value-of select="quantity"/></TD></TR>
</xsl:for-each></TABLE>
<H4>generator_controller objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>generator_state</TH><TH>amortization_type</TH><TH>generator_state_number</TH><TH>market</TH><TH>bid_curve</TH><TH>bid_curve_file</TH><TH>bid_generator_rating</TH><TH>update_curve</TH><TH>is_marginal_gen</TH><TH>generator_rating</TH><TH>generator_output</TH><TH>input_unit_base</TH><TH>startup_cost</TH><TH>shutdown_cost</TH><TH>minimum_runtime</TH><TH>minimum_downtime</TH><TH>capacity_factor</TH><TH>amortization_factor</TH><TH>bid_delay</TH><TH>generator_attachment</TH><TH>building_load_curr</TH><TH>building_load_bid</TH><TH>year_runtime_limit</TH><TH>current_year_runtime</TH><TH>runtime_year_end</TH><TH>scaling_factor</TH><TH>license_premium</TH><TH>hours_in_year</TH><TH>op_and_maint_cost</TH></TR>
<xsl:for-each select="market/generator_controller_list/generator_controller"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="generator_state"/></TD><TD><xsl:value-of select="amortization_type"/></TD><TD><xsl:value-of select="generator_state_number"/></TD><TD><a href="#{market}"><xsl:value-of select="market"/></a></TD><TD><xsl:value-of select="bid_curve"/></TD><TD><xsl:value-of select="bid_curve_file"/></TD><TD><xsl:value-of select="bid_generator_rating"/></TD><TD><xsl:value-of select="update_curve"/></TD><TD><xsl:value-of select="is_marginal_gen"/></TD><TD><xsl:value-of select="generator_rating"/></TD><TD><xsl:value-of select="generator_output"/></TD><TD><xsl:value-of select="input_unit_base"/></TD><TD><xsl:value-of select="startup_cost"/></TD><TD><xsl:value-of select="shutdown_cost"/></TD><TD><xsl:value-of select="minimum_runtime"/></TD><TD><xsl:value-of select="minimum_downtime"/></TD><TD><xsl:value-of select="capacity_factor"/></TD><TD><xsl:value-of select="amortization_factor"/></TD><TD><xsl:value-of select="bid_delay"/></TD><TD><xsl:value-of select="generator_attachment"/></TD><TD><xsl:value-of select="building_load_curr"/></TD><TD><xsl:value-of select="building_load_bid"/></TD><TD><xsl:value-of select="year_runtime_limit"/></TD><TD><xsl:value-of select="current_year_runtime"/></TD><TD><xsl:value-of select="runtime_year_end"/></TD><TD><xsl:value-of select="scaling_factor"/></TD><TD><xsl:value-of select="license_premium"/></TD><TD><xsl:value-of select="hours_in_year"/></TD><TD><xsl:value-of select="op_and_maint_cost"/></TD></TR>
</xsl:for-each></TABLE>
<H3><A NAME="modules_plc">plc</A></H3><TABLE BORDER="1">
<TR><TH>version.major</TH><TD><xsl:value-of select="plc/version.major"/></TD></TR><TR><TH>version.minor</TH><TD><xsl:value-of select="plc/version.minor"/></TD></TR><TR><TH>libpath</TH><TD><xsl:value-of select="plc/libpath"/></TD></TR><TR><TH>incpath</TH><TD><xsl:value-of select="plc/incpath"/></TD></TR></TABLE>
<H4>plc objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>source</TH><TH>network</TH></TR>
<xsl:for-each select="plc/plc_list/plc"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="source"/></TD><TD><a href="#{network}"><xsl:value-of select="network"/></a></TD></TR>
</xsl:for-each></TABLE>
<H4>comm objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>latency</TH><TH>reliability</TH><TH>bitrate</TH><TH>timeout</TH></TR>
<xsl:for-each select="plc/comm_list/comm"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="latency"/></TD><TD><xsl:value-of select="reliability"/></TD><TD><xsl:value-of select="bitrate"/></TD><TD><xsl:value-of select="timeout"/></TD></TR>
</xsl:for-each></TABLE>
<H3><A NAME="modules_powerflow">powerflow</A></H3><TABLE BORDER="1">
<TR><TH>version.major</TH><TD><xsl:value-of select="powerflow/version.major"/></TD></TR><TR><TH>version.minor</TH><TD><xsl:value-of select="powerflow/version.minor"/></TD></TR><TR><TH>show_matrix_values</TH><TD><xsl:value-of select="powerflow/show_matrix_values"/></TD></TR><TR><TH>primary_voltage_ratio</TH><TD><xsl:value-of select="powerflow/primary_voltage_ratio"/></TD></TR><TR><TH>nominal_frequency</TH><TD><xsl:value-of select="powerflow/nominal_frequency"/></TD></TR><TR><TH>require_voltage_control</TH><TD><xsl:value-of select="powerflow/require_voltage_control"/></TD></TR><TR><TH>geographic_degree</TH><TD><xsl:value-of select="powerflow/geographic_degree"/></TD></TR><TR><TH>fault_impedance</TH><TD><xsl:value-of select="powerflow/fault_impedance"/></TD></TR><TR><TH>warning_underfrequency</TH><TD><xsl:value-of select="powerflow/warning_underfrequency"/></TD></TR><TR><TH>warning_overfrequency</TH><TD><xsl:value-of select="powerflow/warning_overfrequency"/></TD></TR><TR><TH>warning_undervoltage</TH><TD><xsl:value-of select="powerflow/warning_undervoltage"/></TD></TR><TR><TH>warning_overvoltage</TH><TD><xsl:value-of select="powerflow/warning_overvoltage"/></TD></TR><TR><TH>warning_voltageangle</TH><TD><xsl:value-of select="powerflow/warning_voltageangle"/></TD></TR><TR><TH>maximum_voltage_error</TH><TD><xsl:value-of select="powerflow/maximum_voltage_error"/></TD></TR><TR><TH>solver_method</TH><TD><xsl:value-of select="powerflow/solver_method"/></TD></TR><TR><TH>line_capacitance</TH><TD><xsl:value-of select="powerflow/line_capacitance"/></TD></TR><TR><TH>line_limits</TH><TD><xsl:value-of select="powerflow/line_limits"/></TD></TR><TR><TH>lu_solver</TH><TD><xsl:value-of select="powerflow/lu_solver"/></TD></TR><TR><TH>acceleration_factor</TH><TD><xsl:value-of select="powerflow/acceleration_factor"/></TD></TR><TR><TH>NR_iteration_limit</TH><TD><xsl:value-of select="powerflow/NR_iteration_limit"/></TD></TR><TR><TH>NR_superLU_procs</TH><TD><xsl:value-of select="powerflow/NR_superLU_procs"/></TD></TR><TR><TH>default_maximum_voltage_error</TH><TD><xsl:value-of select="powerflow/default_maximum_voltage_error"/></TD></TR></TABLE>
<H4>node objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>bustype</TH><TH>busflags</TH><TH>reference_bus</TH><TH>maximum_voltage_error</TH><TH>voltage_A</TH><TH>voltage_B</TH><TH>voltage_C</TH><TH>voltage_AB</TH><TH>voltage_BC</TH><TH>voltage_CA</TH><TH>current_A</TH><TH>current_B</TH><TH>current_C</TH><TH>power_A</TH><TH>power_B</TH><TH>power_C</TH><TH>shunt_A</TH><TH>shunt_B</TH><TH>shunt_C</TH><TH>NR_mode</TH><TH>mean_repair_time</TH><TH>service_status</TH><TH>service_status_double</TH><TH>previous_uptime</TH><TH>current_uptime</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/node_list/node"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="bustype"/></TD><TD><xsl:value-of select="busflags"/></TD><TD><a href="#{reference_bus}"><xsl:value-of select="reference_bus"/></a></TD><TD><xsl:value-of select="maximum_voltage_error"/></TD><TD><xsl:value-of select="voltage_A"/></TD><TD><xsl:value-of select="voltage_B"/></TD><TD><xsl:value-of select="voltage_C"/></TD><TD><xsl:value-of select="voltage_AB"/></TD><TD><xsl:value-of select="voltage_BC"/></TD><TD><xsl:value-of select="voltage_CA"/></TD><TD><xsl:value-of select="current_A"/></TD><TD><xsl:value-of select="current_B"/></TD><TD><xsl:value-of select="current_C"/></TD><TD><xsl:value-of select="power_A"/></TD><TD><xsl:value-of select="power_B"/></TD><TD><xsl:value-of select="power_C"/></TD><TD><xsl:value-of select="shunt_A"/></TD><TD><xsl:value-of select="shunt_B"/></TD><TD><xsl:value-of select="shunt_C"/></TD><TD><xsl:value-of select="NR_mode"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="service_status"/></TD><TD><xsl:value-of select="service_status_double"/></TD><TD><xsl:value-of select="previous_uptime"/></TD><TD><xsl:value-of select="current_uptime"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>link objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>status</TH><TH>from</TH><TH>to</TH><TH>power_in</TH><TH>power_out</TH><TH>power_out_real</TH><TH>power_losses</TH><TH>power_in_A</TH><TH>power_in_B</TH><TH>power_in_C</TH><TH>power_out_A</TH><TH>power_out_B</TH><TH>power_out_C</TH><TH>power_losses_A</TH><TH>power_losses_B</TH><TH>power_losses_C</TH><TH>current_out_A</TH><TH>current_out_B</TH><TH>current_out_C</TH><TH>current_in_A</TH><TH>current_in_B</TH><TH>current_in_C</TH><TH>fault_current_in_A</TH><TH>fault_current_in_B</TH><TH>fault_current_in_C</TH><TH>fault_current_out_A</TH><TH>fault_current_out_B</TH><TH>fault_current_out_C</TH><TH>flow_direction</TH><TH>mean_repair_time</TH><TH>continuous_rating</TH><TH>emergency_rating</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/link_list/link"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="status"/></TD><TD><a href="#{from}"><xsl:value-of select="from"/></a></TD><TD><a href="#{to}"><xsl:value-of select="to"/></a></TD><TD><xsl:value-of select="power_in"/></TD><TD><xsl:value-of select="power_out"/></TD><TD><xsl:value-of select="power_out_real"/></TD><TD><xsl:value-of select="power_losses"/></TD><TD><xsl:value-of select="power_in_A"/></TD><TD><xsl:value-of select="power_in_B"/></TD><TD><xsl:value-of select="power_in_C"/></TD><TD><xsl:value-of select="power_out_A"/></TD><TD><xsl:value-of select="power_out_B"/></TD><TD><xsl:value-of select="power_out_C"/></TD><TD><xsl:value-of select="power_losses_A"/></TD><TD><xsl:value-of select="power_losses_B"/></TD><TD><xsl:value-of select="power_losses_C"/></TD><TD><xsl:value-of select="current_out_A"/></TD><TD><xsl:value-of select="current_out_B"/></TD><TD><xsl:value-of select="current_out_C"/></TD><TD><xsl:value-of select="current_in_A"/></TD><TD><xsl:value-of select="current_in_B"/></TD><TD><xsl:value-of select="current_in_C"/></TD><TD><xsl:value-of select="fault_current_in_A"/></TD><TD><xsl:value-of select="fault_current_in_B"/></TD><TD><xsl:value-of select="fault_current_in_C"/></TD><TD><xsl:value-of select="fault_current_out_A"/></TD><TD><xsl:value-of select="fault_current_out_B"/></TD><TD><xsl:value-of select="fault_current_out_C"/></TD><TD><xsl:value-of select="flow_direction"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="continuous_rating"/></TD><TD><xsl:value-of select="emergency_rating"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>capacitor objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>pt_phase</TH><TH>phases_connected</TH><TH>switchA</TH><TH>switchB</TH><TH>switchC</TH><TH>control</TH><TH>cap_A_switch_count</TH><TH>cap_B_switch_count</TH><TH>cap_C_switch_count</TH><TH>voltage_set_high</TH><TH>voltage_set_low</TH><TH>VAr_set_high</TH><TH>VAr_set_low</TH><TH>current_set_low</TH><TH>current_set_high</TH><TH>capacitor_A</TH><TH>capacitor_B</TH><TH>capacitor_C</TH><TH>cap_nominal_voltage</TH><TH>time_delay</TH><TH>dwell_time</TH><TH>lockout_time</TH><TH>remote_sense</TH><TH>remote_sense_B</TH><TH>control_level</TH><TH>bustype</TH><TH>busflags</TH><TH>reference_bus</TH><TH>maximum_voltage_error</TH><TH>voltage_A</TH><TH>voltage_B</TH><TH>voltage_C</TH><TH>voltage_AB</TH><TH>voltage_BC</TH><TH>voltage_CA</TH><TH>current_A</TH><TH>current_B</TH><TH>current_C</TH><TH>power_A</TH><TH>power_B</TH><TH>power_C</TH><TH>shunt_A</TH><TH>shunt_B</TH><TH>shunt_C</TH><TH>NR_mode</TH><TH>mean_repair_time</TH><TH>service_status</TH><TH>service_status_double</TH><TH>previous_uptime</TH><TH>current_uptime</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/capacitor_list/capacitor"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="pt_phase"/></TD><TD><xsl:value-of select="phases_connected"/></TD><TD><xsl:value-of select="switchA"/></TD><TD><xsl:value-of select="switchB"/></TD><TD><xsl:value-of select="switchC"/></TD><TD><xsl:value-of select="control"/></TD><TD><xsl:value-of select="cap_A_switch_count"/></TD><TD><xsl:value-of select="cap_B_switch_count"/></TD><TD><xsl:value-of select="cap_C_switch_count"/></TD><TD><xsl:value-of select="voltage_set_high"/></TD><TD><xsl:value-of select="voltage_set_low"/></TD><TD><xsl:value-of select="VAr_set_high"/></TD><TD><xsl:value-of select="VAr_set_low"/></TD><TD><xsl:value-of select="current_set_low"/></TD><TD><xsl:value-of select="current_set_high"/></TD><TD><xsl:value-of select="capacitor_A"/></TD><TD><xsl:value-of select="capacitor_B"/></TD><TD><xsl:value-of select="capacitor_C"/></TD><TD><xsl:value-of select="cap_nominal_voltage"/></TD><TD><xsl:value-of select="time_delay"/></TD><TD><xsl:value-of select="dwell_time"/></TD><TD><xsl:value-of select="lockout_time"/></TD><TD><a href="#{remote_sense}"><xsl:value-of select="remote_sense"/></a></TD><TD><a href="#{remote_sense_B}"><xsl:value-of select="remote_sense_B"/></a></TD><TD><xsl:value-of select="control_level"/></TD><TD><xsl:value-of select="bustype"/></TD><TD><xsl:value-of select="busflags"/></TD><TD><a href="#{reference_bus}"><xsl:value-of select="reference_bus"/></a></TD><TD><xsl:value-of select="maximum_voltage_error"/></TD><TD><xsl:value-of select="voltage_A"/></TD><TD><xsl:value-of select="voltage_B"/></TD><TD><xsl:value-of select="voltage_C"/></TD><TD><xsl:value-of select="voltage_AB"/></TD><TD><xsl:value-of select="voltage_BC"/></TD><TD><xsl:value-of select="voltage_CA"/></TD><TD><xsl:value-of select="current_A"/></TD><TD><xsl:value-of select="current_B"/></TD><TD><xsl:value-of select="current_C"/></TD><TD><xsl:value-of select="power_A"/></TD><TD><xsl:value-of select="power_B"/></TD><TD><xsl:value-of select="power_C"/></TD><TD><xsl:value-of select="shunt_A"/></TD><TD><xsl:value-of select="shunt_B"/></TD><TD><xsl:value-of select="shunt_C"/></TD><TD><xsl:value-of select="NR_mode"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="service_status"/></TD><TD><xsl:value-of select="service_status_double"/></TD><TD><xsl:value-of select="previous_uptime"/></TD><TD><xsl:value-of select="current_uptime"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>fuse objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>phase_A_status</TH><TH>phase_B_status</TH><TH>phase_C_status</TH><TH>repair_dist_type</TH><TH>current_limit</TH><TH>mean_replacement_time</TH><TH>status</TH><TH>from</TH><TH>to</TH><TH>power_in</TH><TH>power_out</TH><TH>power_out_real</TH><TH>power_losses</TH><TH>power_in_A</TH><TH>power_in_B</TH><TH>power_in_C</TH><TH>power_out_A</TH><TH>power_out_B</TH><TH>power_out_C</TH><TH>power_losses_A</TH><TH>power_losses_B</TH><TH>power_losses_C</TH><TH>current_out_A</TH><TH>current_out_B</TH><TH>current_out_C</TH><TH>current_in_A</TH><TH>current_in_B</TH><TH>current_in_C</TH><TH>fault_current_in_A</TH><TH>fault_current_in_B</TH><TH>fault_current_in_C</TH><TH>fault_current_out_A</TH><TH>fault_current_out_B</TH><TH>fault_current_out_C</TH><TH>flow_direction</TH><TH>mean_repair_time</TH><TH>continuous_rating</TH><TH>emergency_rating</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/fuse_list/fuse"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="phase_A_status"/></TD><TD><xsl:value-of select="phase_B_status"/></TD><TD><xsl:value-of select="phase_C_status"/></TD><TD><xsl:value-of select="repair_dist_type"/></TD><TD><xsl:value-of select="current_limit"/></TD><TD><xsl:value-of select="mean_replacement_time"/></TD><TD><xsl:value-of select="status"/></TD><TD><a href="#{from}"><xsl:value-of select="from"/></a></TD><TD><a href="#{to}"><xsl:value-of select="to"/></a></TD><TD><xsl:value-of select="power_in"/></TD><TD><xsl:value-of select="power_out"/></TD><TD><xsl:value-of select="power_out_real"/></TD><TD><xsl:value-of select="power_losses"/></TD><TD><xsl:value-of select="power_in_A"/></TD><TD><xsl:value-of select="power_in_B"/></TD><TD><xsl:value-of select="power_in_C"/></TD><TD><xsl:value-of select="power_out_A"/></TD><TD><xsl:value-of select="power_out_B"/></TD><TD><xsl:value-of select="power_out_C"/></TD><TD><xsl:value-of select="power_losses_A"/></TD><TD><xsl:value-of select="power_losses_B"/></TD><TD><xsl:value-of select="power_losses_C"/></TD><TD><xsl:value-of select="current_out_A"/></TD><TD><xsl:value-of select="current_out_B"/></TD><TD><xsl:value-of select="current_out_C"/></TD><TD><xsl:value-of select="current_in_A"/></TD><TD><xsl:value-of select="current_in_B"/></TD><TD><xsl:value-of select="current_in_C"/></TD><TD><xsl:value-of select="fault_current_in_A"/></TD><TD><xsl:value-of select="fault_current_in_B"/></TD><TD><xsl:value-of select="fault_current_in_C"/></TD><TD><xsl:value-of select="fault_current_out_A"/></TD><TD><xsl:value-of select="fault_current_out_B"/></TD><TD><xsl:value-of select="fault_current_out_C"/></TD><TD><xsl:value-of select="flow_direction"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="continuous_rating"/></TD><TD><xsl:value-of select="emergency_rating"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>meter objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>measured_real_energy</TH><TH>measured_reactive_energy</TH><TH>measured_power</TH><TH>measured_power_A</TH><TH>measured_power_B</TH><TH>measured_power_C</TH><TH>measured_demand</TH><TH>measured_real_power</TH><TH>measured_reactive_power</TH><TH>meter_power_consumption</TH><TH>measured_voltage_A</TH><TH>measured_voltage_B</TH><TH>measured_voltage_C</TH><TH>measured_voltage_AB</TH><TH>measured_voltage_BC</TH><TH>measured_voltage_CA</TH><TH>measured_current_A</TH><TH>measured_current_B</TH><TH>measured_current_C</TH><TH>customer_interrupted</TH><TH>customer_interrupted_secondary</TH><TH>monthly_bill</TH><TH>previous_monthly_bill</TH><TH>previous_monthly_energy</TH><TH>monthly_fee</TH><TH>monthly_energy</TH><TH>bill_mode</TH><TH>power_market</TH><TH>bill_day</TH><TH>price</TH><TH>price_base</TH><TH>first_tier_price</TH><TH>first_tier_energy</TH><TH>second_tier_price</TH><TH>second_tier_energy</TH><TH>third_tier_price</TH><TH>third_tier_energy</TH><TH>bustype</TH><TH>busflags</TH><TH>reference_bus</TH><TH>maximum_voltage_error</TH><TH>voltage_A</TH><TH>voltage_B</TH><TH>voltage_C</TH><TH>voltage_AB</TH><TH>voltage_BC</TH><TH>voltage_CA</TH><TH>current_A</TH><TH>current_B</TH><TH>current_C</TH><TH>power_A</TH><TH>power_B</TH><TH>power_C</TH><TH>shunt_A</TH><TH>shunt_B</TH><TH>shunt_C</TH><TH>NR_mode</TH><TH>mean_repair_time</TH><TH>service_status</TH><TH>service_status_double</TH><TH>previous_uptime</TH><TH>current_uptime</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/meter_list/meter"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="measured_real_energy"/></TD><TD><xsl:value-of select="measured_reactive_energy"/></TD><TD><xsl:value-of select="measured_power"/></TD><TD><xsl:value-of select="measured_power_A"/></TD><TD><xsl:value-of select="measured_power_B"/></TD><TD><xsl:value-of select="measured_power_C"/></TD><TD><xsl:value-of select="measured_demand"/></TD><TD><xsl:value-of select="measured_real_power"/></TD><TD><xsl:value-of select="measured_reactive_power"/></TD><TD><xsl:value-of select="meter_power_consumption"/></TD><TD><xsl:value-of select="measured_voltage_A"/></TD><TD><xsl:value-of select="measured_voltage_B"/></TD><TD><xsl:value-of select="measured_voltage_C"/></TD><TD><xsl:value-of select="measured_voltage_AB"/></TD><TD><xsl:value-of select="measured_voltage_BC"/></TD><TD><xsl:value-of select="measured_voltage_CA"/></TD><TD><xsl:value-of select="measured_current_A"/></TD><TD><xsl:value-of select="measured_current_B"/></TD><TD><xsl:value-of select="measured_current_C"/></TD><TD><xsl:value-of select="customer_interrupted"/></TD><TD><xsl:value-of select="customer_interrupted_secondary"/></TD><TD><xsl:value-of select="monthly_bill"/></TD><TD><xsl:value-of select="previous_monthly_bill"/></TD><TD><xsl:value-of select="previous_monthly_energy"/></TD><TD><xsl:value-of select="monthly_fee"/></TD><TD><xsl:value-of select="monthly_energy"/></TD><TD><xsl:value-of select="bill_mode"/></TD><TD><a href="#{power_market}"><xsl:value-of select="power_market"/></a></TD><TD><xsl:value-of select="bill_day"/></TD><TD><xsl:value-of select="price"/></TD><TD><xsl:value-of select="price_base"/></TD><TD><xsl:value-of select="first_tier_price"/></TD><TD><xsl:value-of select="first_tier_energy"/></TD><TD><xsl:value-of select="second_tier_price"/></TD><TD><xsl:value-of select="second_tier_energy"/></TD><TD><xsl:value-of select="third_tier_price"/></TD><TD><xsl:value-of select="third_tier_energy"/></TD><TD><xsl:value-of select="bustype"/></TD><TD><xsl:value-of select="busflags"/></TD><TD><a href="#{reference_bus}"><xsl:value-of select="reference_bus"/></a></TD><TD><xsl:value-of select="maximum_voltage_error"/></TD><TD><xsl:value-of select="voltage_A"/></TD><TD><xsl:value-of select="voltage_B"/></TD><TD><xsl:value-of select="voltage_C"/></TD><TD><xsl:value-of select="voltage_AB"/></TD><TD><xsl:value-of select="voltage_BC"/></TD><TD><xsl:value-of select="voltage_CA"/></TD><TD><xsl:value-of select="current_A"/></TD><TD><xsl:value-of select="current_B"/></TD><TD><xsl:value-of select="current_C"/></TD><TD><xsl:value-of select="power_A"/></TD><TD><xsl:value-of select="power_B"/></TD><TD><xsl:value-of select="power_C"/></TD><TD><xsl:value-of select="shunt_A"/></TD><TD><xsl:value-of select="shunt_B"/></TD><TD><xsl:value-of select="shunt_C"/></TD><TD><xsl:value-of select="NR_mode"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="service_status"/></TD><TD><xsl:value-of select="service_status_double"/></TD><TD><xsl:value-of select="previous_uptime"/></TD><TD><xsl:value-of select="current_uptime"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>line objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>configuration</TH><TH>length</TH><TH>status</TH><TH>from</TH><TH>to</TH><TH>power_in</TH><TH>power_out</TH><TH>power_out_real</TH><TH>power_losses</TH><TH>power_in_A</TH><TH>power_in_B</TH><TH>power_in_C</TH><TH>power_out_A</TH><TH>power_out_B</TH><TH>power_out_C</TH><TH>power_losses_A</TH><TH>power_losses_B</TH><TH>power_losses_C</TH><TH>current_out_A</TH><TH>current_out_B</TH><TH>current_out_C</TH><TH>current_in_A</TH><TH>current_in_B</TH><TH>current_in_C</TH><TH>fault_current_in_A</TH><TH>fault_current_in_B</TH><TH>fault_current_in_C</TH><TH>fault_current_out_A</TH><TH>fault_current_out_B</TH><TH>fault_current_out_C</TH><TH>flow_direction</TH><TH>mean_repair_time</TH><TH>continuous_rating</TH><TH>emergency_rating</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/line_list/line"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><a href="#{configuration}"><xsl:value-of select="configuration"/></a></TD><TD><xsl:value-of select="length"/></TD><TD><xsl:value-of select="status"/></TD><TD><a href="#{from}"><xsl:value-of select="from"/></a></TD><TD><a href="#{to}"><xsl:value-of select="to"/></a></TD><TD><xsl:value-of select="power_in"/></TD><TD><xsl:value-of select="power_out"/></TD><TD><xsl:value-of select="power_out_real"/></TD><TD><xsl:value-of select="power_losses"/></TD><TD><xsl:value-of select="power_in_A"/></TD><TD><xsl:value-of select="power_in_B"/></TD><TD><xsl:value-of select="power_in_C"/></TD><TD><xsl:value-of select="power_out_A"/></TD><TD><xsl:value-of select="power_out_B"/></TD><TD><xsl:value-of select="power_out_C"/></TD><TD><xsl:value-of select="power_losses_A"/></TD><TD><xsl:value-of select="power_losses_B"/></TD><TD><xsl:value-of select="power_losses_C"/></TD><TD><xsl:value-of select="current_out_A"/></TD><TD><xsl:value-of select="current_out_B"/></TD><TD><xsl:value-of select="current_out_C"/></TD><TD><xsl:value-of select="current_in_A"/></TD><TD><xsl:value-of select="current_in_B"/></TD><TD><xsl:value-of select="current_in_C"/></TD><TD><xsl:value-of select="fault_current_in_A"/></TD><TD><xsl:value-of select="fault_current_in_B"/></TD><TD><xsl:value-of select="fault_current_in_C"/></TD><TD><xsl:value-of select="fault_current_out_A"/></TD><TD><xsl:value-of select="fault_current_out_B"/></TD><TD><xsl:value-of select="fault_current_out_C"/></TD><TD><xsl:value-of select="flow_direction"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="continuous_rating"/></TD><TD><xsl:value-of select="emergency_rating"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>line_spacing objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>distance_AB</TH><TH>distance_BC</TH><TH>distance_AC</TH><TH>distance_AN</TH><TH>distance_BN</TH><TH>distance_CN</TH><TH>distance_AE</TH><TH>distance_BE</TH><TH>distance_CE</TH><TH>distance_NE</TH></TR>
<xsl:for-each select="powerflow/line_spacing_list/line_spacing"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="distance_AB"/></TD><TD><xsl:value-of select="distance_BC"/></TD><TD><xsl:value-of select="distance_AC"/></TD><TD><xsl:value-of select="distance_AN"/></TD><TD><xsl:value-of select="distance_BN"/></TD><TD><xsl:value-of select="distance_CN"/></TD><TD><xsl:value-of select="distance_AE"/></TD><TD><xsl:value-of select="distance_BE"/></TD><TD><xsl:value-of select="distance_CE"/></TD><TD><xsl:value-of select="distance_NE"/></TD></TR>
</xsl:for-each></TABLE>
<H4>overhead_line objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>configuration</TH><TH>length</TH><TH>status</TH><TH>from</TH><TH>to</TH><TH>power_in</TH><TH>power_out</TH><TH>power_out_real</TH><TH>power_losses</TH><TH>power_in_A</TH><TH>power_in_B</TH><TH>power_in_C</TH><TH>power_out_A</TH><TH>power_out_B</TH><TH>power_out_C</TH><TH>power_losses_A</TH><TH>power_losses_B</TH><TH>power_losses_C</TH><TH>current_out_A</TH><TH>current_out_B</TH><TH>current_out_C</TH><TH>current_in_A</TH><TH>current_in_B</TH><TH>current_in_C</TH><TH>fault_current_in_A</TH><TH>fault_current_in_B</TH><TH>fault_current_in_C</TH><TH>fault_current_out_A</TH><TH>fault_current_out_B</TH><TH>fault_current_out_C</TH><TH>flow_direction</TH><TH>mean_repair_time</TH><TH>continuous_rating</TH><TH>emergency_rating</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/overhead_line_list/overhead_line"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><a href="#{configuration}"><xsl:value-of select="configuration"/></a></TD><TD><xsl:value-of select="length"/></TD><TD><xsl:value-of select="status"/></TD><TD><a href="#{from}"><xsl:value-of select="from"/></a></TD><TD><a href="#{to}"><xsl:value-of select="to"/></a></TD><TD><xsl:value-of select="power_in"/></TD><TD><xsl:value-of select="power_out"/></TD><TD><xsl:value-of select="power_out_real"/></TD><TD><xsl:value-of select="power_losses"/></TD><TD><xsl:value-of select="power_in_A"/></TD><TD><xsl:value-of select="power_in_B"/></TD><TD><xsl:value-of select="power_in_C"/></TD><TD><xsl:value-of select="power_out_A"/></TD><TD><xsl:value-of select="power_out_B"/></TD><TD><xsl:value-of select="power_out_C"/></TD><TD><xsl:value-of select="power_losses_A"/></TD><TD><xsl:value-of select="power_losses_B"/></TD><TD><xsl:value-of select="power_losses_C"/></TD><TD><xsl:value-of select="current_out_A"/></TD><TD><xsl:value-of select="current_out_B"/></TD><TD><xsl:value-of select="current_out_C"/></TD><TD><xsl:value-of select="current_in_A"/></TD><TD><xsl:value-of select="current_in_B"/></TD><TD><xsl:value-of select="current_in_C"/></TD><TD><xsl:value-of select="fault_current_in_A"/></TD><TD><xsl:value-of select="fault_current_in_B"/></TD><TD><xsl:value-of select="fault_current_in_C"/></TD><TD><xsl:value-of select="fault_current_out_A"/></TD><TD><xsl:value-of select="fault_current_out_B"/></TD><TD><xsl:value-of select="fault_current_out_C"/></TD><TD><xsl:value-of select="flow_direction"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="continuous_rating"/></TD><TD><xsl:value-of select="emergency_rating"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>underground_line objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>configuration</TH><TH>length</TH><TH>status</TH><TH>from</TH><TH>to</TH><TH>power_in</TH><TH>power_out</TH><TH>power_out_real</TH><TH>power_losses</TH><TH>power_in_A</TH><TH>power_in_B</TH><TH>power_in_C</TH><TH>power_out_A</TH><TH>power_out_B</TH><TH>power_out_C</TH><TH>power_losses_A</TH><TH>power_losses_B</TH><TH>power_losses_C</TH><TH>current_out_A</TH><TH>current_out_B</TH><TH>current_out_C</TH><TH>current_in_A</TH><TH>current_in_B</TH><TH>current_in_C</TH><TH>fault_current_in_A</TH><TH>fault_current_in_B</TH><TH>fault_current_in_C</TH><TH>fault_current_out_A</TH><TH>fault_current_out_B</TH><TH>fault_current_out_C</TH><TH>flow_direction</TH><TH>mean_repair_time</TH><TH>continuous_rating</TH><TH>emergency_rating</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/underground_line_list/underground_line"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><a href="#{configuration}"><xsl:value-of select="configuration"/></a></TD><TD><xsl:value-of select="length"/></TD><TD><xsl:value-of select="status"/></TD><TD><a href="#{from}"><xsl:value-of select="from"/></a></TD><TD><a href="#{to}"><xsl:value-of select="to"/></a></TD><TD><xsl:value-of select="power_in"/></TD><TD><xsl:value-of select="power_out"/></TD><TD><xsl:value-of select="power_out_real"/></TD><TD><xsl:value-of select="power_losses"/></TD><TD><xsl:value-of select="power_in_A"/></TD><TD><xsl:value-of select="power_in_B"/></TD><TD><xsl:value-of select="power_in_C"/></TD><TD><xsl:value-of select="power_out_A"/></TD><TD><xsl:value-of select="power_out_B"/></TD><TD><xsl:value-of select="power_out_C"/></TD><TD><xsl:value-of select="power_losses_A"/></TD><TD><xsl:value-of select="power_losses_B"/></TD><TD><xsl:value-of select="power_losses_C"/></TD><TD><xsl:value-of select="current_out_A"/></TD><TD><xsl:value-of select="current_out_B"/></TD><TD><xsl:value-of select="current_out_C"/></TD><TD><xsl:value-of select="current_in_A"/></TD><TD><xsl:value-of select="current_in_B"/></TD><TD><xsl:value-of select="current_in_C"/></TD><TD><xsl:value-of select="fault_current_in_A"/></TD><TD><xsl:value-of select="fault_current_in_B"/></TD><TD><xsl:value-of select="fault_current_in_C"/></TD><TD><xsl:value-of select="fault_current_out_A"/></TD><TD><xsl:value-of select="fault_current_out_B"/></TD><TD><xsl:value-of select="fault_current_out_C"/></TD><TD><xsl:value-of select="flow_direction"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="continuous_rating"/></TD><TD><xsl:value-of select="emergency_rating"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>overhead_line_conductor objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>geometric_mean_radius</TH><TH>resistance</TH><TH>diameter</TH><TH>rating.summer.continuous</TH><TH>rating.summer.emergency</TH><TH>rating.winter.continuous</TH><TH>rating.winter.emergency</TH></TR>
<xsl:for-each select="powerflow/overhead_line_conductor_list/overhead_line_conductor"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="geometric_mean_radius"/></TD><TD><xsl:value-of select="resistance"/></TD><TD><xsl:value-of select="diameter"/></TD><TD><xsl:value-of select="rating.summer.continuous"/></TD><TD><xsl:value-of select="rating.summer.emergency"/></TD><TD><xsl:value-of select="rating.winter.continuous"/></TD><TD><xsl:value-of select="rating.winter.emergency"/></TD></TR>
</xsl:for-each></TABLE>
<H4>underground_line_conductor objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>outer_diameter</TH><TH>conductor_gmr</TH><TH>conductor_diameter</TH><TH>conductor_resistance</TH><TH>neutral_gmr</TH><TH>neutral_diameter</TH><TH>neutral_resistance</TH><TH>neutral_strands</TH><TH>insulation_relative_permitivitty</TH><TH>shield_gmr</TH><TH>shield_resistance</TH><TH>rating.summer.continuous</TH><TH>rating.summer.emergency</TH><TH>rating.winter.continuous</TH><TH>rating.winter.emergency</TH></TR>
<xsl:for-each select="powerflow/underground_line_conductor_list/underground_line_conductor"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="outer_diameter"/></TD><TD><xsl:value-of select="conductor_gmr"/></TD><TD><xsl:value-of select="conductor_diameter"/></TD><TD><xsl:value-of select="conductor_resistance"/></TD><TD><xsl:value-of select="neutral_gmr"/></TD><TD><xsl:value-of select="neutral_diameter"/></TD><TD><xsl:value-of select="neutral_resistance"/></TD><TD><xsl:value-of select="neutral_strands"/></TD><TD><xsl:value-of select="insulation_relative_permitivitty"/></TD><TD><xsl:value-of select="shield_gmr"/></TD><TD><xsl:value-of select="shield_resistance"/></TD><TD><xsl:value-of select="rating.summer.continuous"/></TD><TD><xsl:value-of select="rating.summer.emergency"/></TD><TD><xsl:value-of select="rating.winter.continuous"/></TD><TD><xsl:value-of select="rating.winter.emergency"/></TD></TR>
</xsl:for-each></TABLE>
<H4>line_configuration objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>conductor_A</TH><TH>conductor_B</TH><TH>conductor_C</TH><TH>conductor_N</TH><TH>spacing</TH><TH>z11</TH><TH>z12</TH><TH>z13</TH><TH>z21</TH><TH>z22</TH><TH>z23</TH><TH>z31</TH><TH>z32</TH><TH>z33</TH></TR>
<xsl:for-each select="powerflow/line_configuration_list/line_configuration"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><a href="#{conductor_A}"><xsl:value-of select="conductor_A"/></a></TD><TD><a href="#{conductor_B}"><xsl:value-of select="conductor_B"/></a></TD><TD><a href="#{conductor_C}"><xsl:value-of select="conductor_C"/></a></TD><TD><a href="#{conductor_N}"><xsl:value-of select="conductor_N"/></a></TD><TD><a href="#{spacing}"><xsl:value-of select="spacing"/></a></TD><TD><xsl:value-of select="z11"/></TD><TD><xsl:value-of select="z12"/></TD><TD><xsl:value-of select="z13"/></TD><TD><xsl:value-of select="z21"/></TD><TD><xsl:value-of select="z22"/></TD><TD><xsl:value-of select="z23"/></TD><TD><xsl:value-of select="z31"/></TD><TD><xsl:value-of select="z32"/></TD><TD><xsl:value-of select="z33"/></TD></TR>
</xsl:for-each></TABLE>
<H4>relay objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>time_to_change</TH><TH>recloser_delay</TH><TH>recloser_tries</TH><TH>recloser_limit</TH><TH>recloser_event</TH><TH>status</TH><TH>from</TH><TH>to</TH><TH>power_in</TH><TH>power_out</TH><TH>power_out_real</TH><TH>power_losses</TH><TH>power_in_A</TH><TH>power_in_B</TH><TH>power_in_C</TH><TH>power_out_A</TH><TH>power_out_B</TH><TH>power_out_C</TH><TH>power_losses_A</TH><TH>power_losses_B</TH><TH>power_losses_C</TH><TH>current_out_A</TH><TH>current_out_B</TH><TH>current_out_C</TH><TH>current_in_A</TH><TH>current_in_B</TH><TH>current_in_C</TH><TH>fault_current_in_A</TH><TH>fault_current_in_B</TH><TH>fault_current_in_C</TH><TH>fault_current_out_A</TH><TH>fault_current_out_B</TH><TH>fault_current_out_C</TH><TH>flow_direction</TH><TH>mean_repair_time</TH><TH>continuous_rating</TH><TH>emergency_rating</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/relay_list/relay"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="time_to_change"/></TD><TD><xsl:value-of select="recloser_delay"/></TD><TD><xsl:value-of select="recloser_tries"/></TD><TD><xsl:value-of select="recloser_limit"/></TD><TD><xsl:value-of select="recloser_event"/></TD><TD><xsl:value-of select="status"/></TD><TD><a href="#{from}"><xsl:value-of select="from"/></a></TD><TD><a href="#{to}"><xsl:value-of select="to"/></a></TD><TD><xsl:value-of select="power_in"/></TD><TD><xsl:value-of select="power_out"/></TD><TD><xsl:value-of select="power_out_real"/></TD><TD><xsl:value-of select="power_losses"/></TD><TD><xsl:value-of select="power_in_A"/></TD><TD><xsl:value-of select="power_in_B"/></TD><TD><xsl:value-of select="power_in_C"/></TD><TD><xsl:value-of select="power_out_A"/></TD><TD><xsl:value-of select="power_out_B"/></TD><TD><xsl:value-of select="power_out_C"/></TD><TD><xsl:value-of select="power_losses_A"/></TD><TD><xsl:value-of select="power_losses_B"/></TD><TD><xsl:value-of select="power_losses_C"/></TD><TD><xsl:value-of select="current_out_A"/></TD><TD><xsl:value-of select="current_out_B"/></TD><TD><xsl:value-of select="current_out_C"/></TD><TD><xsl:value-of select="current_in_A"/></TD><TD><xsl:value-of select="current_in_B"/></TD><TD><xsl:value-of select="current_in_C"/></TD><TD><xsl:value-of select="fault_current_in_A"/></TD><TD><xsl:value-of select="fault_current_in_B"/></TD><TD><xsl:value-of select="fault_current_in_C"/></TD><TD><xsl:value-of select="fault_current_out_A"/></TD><TD><xsl:value-of select="fault_current_out_B"/></TD><TD><xsl:value-of select="fault_current_out_C"/></TD><TD><xsl:value-of select="flow_direction"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="continuous_rating"/></TD><TD><xsl:value-of select="emergency_rating"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>transformer_configuration objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>connect_type</TH><TH>install_type</TH><TH>coolant_type</TH><TH>cooling_type</TH><TH>primary_voltage</TH><TH>secondary_voltage</TH><TH>power_rating</TH><TH>powerA_rating</TH><TH>powerB_rating</TH><TH>powerC_rating</TH><TH>resistance</TH><TH>reactance</TH><TH>impedance</TH><TH>resistance1</TH><TH>reactance1</TH><TH>impedance1</TH><TH>resistance2</TH><TH>reactance2</TH><TH>impedance2</TH><TH>shunt_resistance</TH><TH>shunt_reactance</TH><TH>shunt_impedance</TH><TH>core_coil_weight</TH><TH>tank_fittings_weight</TH><TH>oil_volume</TH><TH>rated_winding_time_constant</TH><TH>rated_winding_hot_spot_rise</TH><TH>rated_top_oil_rise</TH><TH>no_load_loss</TH><TH>full_load_loss</TH><TH>reactance_resistance_ratio</TH><TH>installed_insulation_life</TH></TR>
<xsl:for-each select="powerflow/transformer_configuration_list/transformer_configuration"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="connect_type"/></TD><TD><xsl:value-of select="install_type"/></TD><TD><xsl:value-of select="coolant_type"/></TD><TD><xsl:value-of select="cooling_type"/></TD><TD><xsl:value-of select="primary_voltage"/></TD><TD><xsl:value-of select="secondary_voltage"/></TD><TD><xsl:value-of select="power_rating"/></TD><TD><xsl:value-of select="powerA_rating"/></TD><TD><xsl:value-of select="powerB_rating"/></TD><TD><xsl:value-of select="powerC_rating"/></TD><TD><xsl:value-of select="resistance"/></TD><TD><xsl:value-of select="reactance"/></TD><TD><xsl:value-of select="impedance"/></TD><TD><xsl:value-of select="resistance1"/></TD><TD><xsl:value-of select="reactance1"/></TD><TD><xsl:value-of select="impedance1"/></TD><TD><xsl:value-of select="resistance2"/></TD><TD><xsl:value-of select="reactance2"/></TD><TD><xsl:value-of select="impedance2"/></TD><TD><xsl:value-of select="shunt_resistance"/></TD><TD><xsl:value-of select="shunt_reactance"/></TD><TD><xsl:value-of select="shunt_impedance"/></TD><TD><xsl:value-of select="core_coil_weight"/></TD><TD><xsl:value-of select="tank_fittings_weight"/></TD><TD><xsl:value-of select="oil_volume"/></TD><TD><xsl:value-of select="rated_winding_time_constant"/></TD><TD><xsl:value-of select="rated_winding_hot_spot_rise"/></TD><TD><xsl:value-of select="rated_top_oil_rise"/></TD><TD><xsl:value-of select="no_load_loss"/></TD><TD><xsl:value-of select="full_load_loss"/></TD><TD><xsl:value-of select="reactance_resistance_ratio"/></TD><TD><xsl:value-of select="installed_insulation_life"/></TD></TR>
</xsl:for-each></TABLE>
<H4>transformer objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>configuration</TH><TH>climate</TH><TH>ambient_temperature</TH><TH>top_oil_hot_spot_temperature</TH><TH>winding_hot_spot_temperature</TH><TH>percent_loss_of_life</TH><TH>aging_constant</TH><TH>use_thermal_model</TH><TH>transformer_replacement_count</TH><TH>aging_granularity</TH><TH>status</TH><TH>from</TH><TH>to</TH><TH>power_in</TH><TH>power_out</TH><TH>power_out_real</TH><TH>power_losses</TH><TH>power_in_A</TH><TH>power_in_B</TH><TH>power_in_C</TH><TH>power_out_A</TH><TH>power_out_B</TH><TH>power_out_C</TH><TH>power_losses_A</TH><TH>power_losses_B</TH><TH>power_losses_C</TH><TH>current_out_A</TH><TH>current_out_B</TH><TH>current_out_C</TH><TH>current_in_A</TH><TH>current_in_B</TH><TH>current_in_C</TH><TH>fault_current_in_A</TH><TH>fault_current_in_B</TH><TH>fault_current_in_C</TH><TH>fault_current_out_A</TH><TH>fault_current_out_B</TH><TH>fault_current_out_C</TH><TH>flow_direction</TH><TH>mean_repair_time</TH><TH>continuous_rating</TH><TH>emergency_rating</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/transformer_list/transformer"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><a href="#{configuration}"><xsl:value-of select="configuration"/></a></TD><TD><a href="#{climate}"><xsl:value-of select="climate"/></a></TD><TD><xsl:value-of select="ambient_temperature"/></TD><TD><xsl:value-of select="top_oil_hot_spot_temperature"/></TD><TD><xsl:value-of select="winding_hot_spot_temperature"/></TD><TD><xsl:value-of select="percent_loss_of_life"/></TD><TD><xsl:value-of select="aging_constant"/></TD><TD><xsl:value-of select="use_thermal_model"/></TD><TD><xsl:value-of select="transformer_replacement_count"/></TD><TD><xsl:value-of select="aging_granularity"/></TD><TD><xsl:value-of select="status"/></TD><TD><a href="#{from}"><xsl:value-of select="from"/></a></TD><TD><a href="#{to}"><xsl:value-of select="to"/></a></TD><TD><xsl:value-of select="power_in"/></TD><TD><xsl:value-of select="power_out"/></TD><TD><xsl:value-of select="power_out_real"/></TD><TD><xsl:value-of select="power_losses"/></TD><TD><xsl:value-of select="power_in_A"/></TD><TD><xsl:value-of select="power_in_B"/></TD><TD><xsl:value-of select="power_in_C"/></TD><TD><xsl:value-of select="power_out_A"/></TD><TD><xsl:value-of select="power_out_B"/></TD><TD><xsl:value-of select="power_out_C"/></TD><TD><xsl:value-of select="power_losses_A"/></TD><TD><xsl:value-of select="power_losses_B"/></TD><TD><xsl:value-of select="power_losses_C"/></TD><TD><xsl:value-of select="current_out_A"/></TD><TD><xsl:value-of select="current_out_B"/></TD><TD><xsl:value-of select="current_out_C"/></TD><TD><xsl:value-of select="current_in_A"/></TD><TD><xsl:value-of select="current_in_B"/></TD><TD><xsl:value-of select="current_in_C"/></TD><TD><xsl:value-of select="fault_current_in_A"/></TD><TD><xsl:value-of select="fault_current_in_B"/></TD><TD><xsl:value-of select="fault_current_in_C"/></TD><TD><xsl:value-of select="fault_current_out_A"/></TD><TD><xsl:value-of select="fault_current_out_B"/></TD><TD><xsl:value-of select="fault_current_out_C"/></TD><TD><xsl:value-of select="flow_direction"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="continuous_rating"/></TD><TD><xsl:value-of select="emergency_rating"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>load objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>load_class</TH><TH>constant_power_A</TH><TH>constant_power_B</TH><TH>constant_power_C</TH><TH>constant_power_A_real</TH><TH>constant_power_B_real</TH><TH>constant_power_C_real</TH><TH>constant_power_A_reac</TH><TH>constant_power_B_reac</TH><TH>constant_power_C_reac</TH><TH>constant_current_A</TH><TH>constant_current_B</TH><TH>constant_current_C</TH><TH>constant_current_A_real</TH><TH>constant_current_B_real</TH><TH>constant_current_C_real</TH><TH>constant_current_A_reac</TH><TH>constant_current_B_reac</TH><TH>constant_current_C_reac</TH><TH>constant_impedance_A</TH><TH>constant_impedance_B</TH><TH>constant_impedance_C</TH><TH>constant_impedance_A_real</TH><TH>constant_impedance_B_real</TH><TH>constant_impedance_C_real</TH><TH>constant_impedance_A_reac</TH><TH>constant_impedance_B_reac</TH><TH>constant_impedance_C_reac</TH><TH>measured_voltage_A</TH><TH>measured_voltage_B</TH><TH>measured_voltage_C</TH><TH>measured_voltage_AB</TH><TH>measured_voltage_BC</TH><TH>measured_voltage_CA</TH><TH>phase_loss_protection</TH><TH>base_power_A</TH><TH>base_power_B</TH><TH>base_power_C</TH><TH>power_pf_A</TH><TH>current_pf_A</TH><TH>impedance_pf_A</TH><TH>power_pf_B</TH><TH>current_pf_B</TH><TH>impedance_pf_B</TH><TH>power_pf_C</TH><TH>current_pf_C</TH><TH>impedance_pf_C</TH><TH>power_fraction_A</TH><TH>current_fraction_A</TH><TH>impedance_fraction_A</TH><TH>power_fraction_B</TH><TH>current_fraction_B</TH><TH>impedance_fraction_B</TH><TH>power_fraction_C</TH><TH>current_fraction_C</TH><TH>impedance_fraction_C</TH><TH>bustype</TH><TH>busflags</TH><TH>reference_bus</TH><TH>maximum_voltage_error</TH><TH>voltage_A</TH><TH>voltage_B</TH><TH>voltage_C</TH><TH>voltage_AB</TH><TH>voltage_BC</TH><TH>voltage_CA</TH><TH>current_A</TH><TH>current_B</TH><TH>current_C</TH><TH>power_A</TH><TH>power_B</TH><TH>power_C</TH><TH>shunt_A</TH><TH>shunt_B</TH><TH>shunt_C</TH><TH>NR_mode</TH><TH>mean_repair_time</TH><TH>service_status</TH><TH>service_status_double</TH><TH>previous_uptime</TH><TH>current_uptime</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/load_list/load"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="load_class"/></TD><TD><xsl:value-of select="constant_power_A"/></TD><TD><xsl:value-of select="constant_power_B"/></TD><TD><xsl:value-of select="constant_power_C"/></TD><TD><xsl:value-of select="constant_power_A_real"/></TD><TD><xsl:value-of select="constant_power_B_real"/></TD><TD><xsl:value-of select="constant_power_C_real"/></TD><TD><xsl:value-of select="constant_power_A_reac"/></TD><TD><xsl:value-of select="constant_power_B_reac"/></TD><TD><xsl:value-of select="constant_power_C_reac"/></TD><TD><xsl:value-of select="constant_current_A"/></TD><TD><xsl:value-of select="constant_current_B"/></TD><TD><xsl:value-of select="constant_current_C"/></TD><TD><xsl:value-of select="constant_current_A_real"/></TD><TD><xsl:value-of select="constant_current_B_real"/></TD><TD><xsl:value-of select="constant_current_C_real"/></TD><TD><xsl:value-of select="constant_current_A_reac"/></TD><TD><xsl:value-of select="constant_current_B_reac"/></TD><TD><xsl:value-of select="constant_current_C_reac"/></TD><TD><xsl:value-of select="constant_impedance_A"/></TD><TD><xsl:value-of select="constant_impedance_B"/></TD><TD><xsl:value-of select="constant_impedance_C"/></TD><TD><xsl:value-of select="constant_impedance_A_real"/></TD><TD><xsl:value-of select="constant_impedance_B_real"/></TD><TD><xsl:value-of select="constant_impedance_C_real"/></TD><TD><xsl:value-of select="constant_impedance_A_reac"/></TD><TD><xsl:value-of select="constant_impedance_B_reac"/></TD><TD><xsl:value-of select="constant_impedance_C_reac"/></TD><TD><xsl:value-of select="measured_voltage_A"/></TD><TD><xsl:value-of select="measured_voltage_B"/></TD><TD><xsl:value-of select="measured_voltage_C"/></TD><TD><xsl:value-of select="measured_voltage_AB"/></TD><TD><xsl:value-of select="measured_voltage_BC"/></TD><TD><xsl:value-of select="measured_voltage_CA"/></TD><TD><xsl:value-of select="phase_loss_protection"/></TD><TD><xsl:value-of select="base_power_A"/></TD><TD><xsl:value-of select="base_power_B"/></TD><TD><xsl:value-of select="base_power_C"/></TD><TD><xsl:value-of select="power_pf_A"/></TD><TD><xsl:value-of select="current_pf_A"/></TD><TD><xsl:value-of select="impedance_pf_A"/></TD><TD><xsl:value-of select="power_pf_B"/></TD><TD><xsl:value-of select="current_pf_B"/></TD><TD><xsl:value-of select="impedance_pf_B"/></TD><TD><xsl:value-of select="power_pf_C"/></TD><TD><xsl:value-of select="current_pf_C"/></TD><TD><xsl:value-of select="impedance_pf_C"/></TD><TD><xsl:value-of select="power_fraction_A"/></TD><TD><xsl:value-of select="current_fraction_A"/></TD><TD><xsl:value-of select="impedance_fraction_A"/></TD><TD><xsl:value-of select="power_fraction_B"/></TD><TD><xsl:value-of select="current_fraction_B"/></TD><TD><xsl:value-of select="impedance_fraction_B"/></TD><TD><xsl:value-of select="power_fraction_C"/></TD><TD><xsl:value-of select="current_fraction_C"/></TD><TD><xsl:value-of select="impedance_fraction_C"/></TD><TD><xsl:value-of select="bustype"/></TD><TD><xsl:value-of select="busflags"/></TD><TD><a href="#{reference_bus}"><xsl:value-of select="reference_bus"/></a></TD><TD><xsl:value-of select="maximum_voltage_error"/></TD><TD><xsl:value-of select="voltage_A"/></TD><TD><xsl:value-of select="voltage_B"/></TD><TD><xsl:value-of select="voltage_C"/></TD><TD><xsl:value-of select="voltage_AB"/></TD><TD><xsl:value-of select="voltage_BC"/></TD><TD><xsl:value-of select="voltage_CA"/></TD><TD><xsl:value-of select="current_A"/></TD><TD><xsl:value-of select="current_B"/></TD><TD><xsl:value-of select="current_C"/></TD><TD><xsl:value-of select="power_A"/></TD><TD><xsl:value-of select="power_B"/></TD><TD><xsl:value-of select="power_C"/></TD><TD><xsl:value-of select="shunt_A"/></TD><TD><xsl:value-of select="shunt_B"/></TD><TD><xsl:value-of select="shunt_C"/></TD><TD><xsl:value-of select="NR_mode"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="service_status"/></TD><TD><xsl:value-of select="service_status_double"/></TD><TD><xsl:value-of select="previous_uptime"/></TD><TD><xsl:value-of select="current_uptime"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>regulator_configuration objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>connect_type</TH><TH>band_center</TH><TH>band_width</TH><TH>time_delay</TH><TH>dwell_time</TH><TH>raise_taps</TH><TH>lower_taps</TH><TH>current_transducer_ratio</TH><TH>power_transducer_ratio</TH><TH>compensator_r_setting_A</TH><TH>compensator_r_setting_B</TH><TH>compensator_r_setting_C</TH><TH>compensator_x_setting_A</TH><TH>compensator_x_setting_B</TH><TH>compensator_x_setting_C</TH><TH>CT_phase</TH><TH>PT_phase</TH><TH>regulation</TH><TH>control_level</TH><TH>Control</TH><TH>Type</TH><TH>tap_pos_A</TH><TH>tap_pos_B</TH><TH>tap_pos_C</TH></TR>
<xsl:for-each select="powerflow/regulator_configuration_list/regulator_configuration"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="connect_type"/></TD><TD><xsl:value-of select="band_center"/></TD><TD><xsl:value-of select="band_width"/></TD><TD><xsl:value-of select="time_delay"/></TD><TD><xsl:value-of select="dwell_time"/></TD><TD><xsl:value-of select="raise_taps"/></TD><TD><xsl:value-of select="lower_taps"/></TD><TD><xsl:value-of select="current_transducer_ratio"/></TD><TD><xsl:value-of select="power_transducer_ratio"/></TD><TD><xsl:value-of select="compensator_r_setting_A"/></TD><TD><xsl:value-of select="compensator_r_setting_B"/></TD><TD><xsl:value-of select="compensator_r_setting_C"/></TD><TD><xsl:value-of select="compensator_x_setting_A"/></TD><TD><xsl:value-of select="compensator_x_setting_B"/></TD><TD><xsl:value-of select="compensator_x_setting_C"/></TD><TD><xsl:value-of select="CT_phase"/></TD><TD><xsl:value-of select="PT_phase"/></TD><TD><xsl:value-of select="regulation"/></TD><TD><xsl:value-of select="control_level"/></TD><TD><xsl:value-of select="Control"/></TD><TD><xsl:value-of select="Type"/></TD><TD><xsl:value-of select="tap_pos_A"/></TD><TD><xsl:value-of select="tap_pos_B"/></TD><TD><xsl:value-of select="tap_pos_C"/></TD></TR>
</xsl:for-each></TABLE>
<H4>regulator objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>configuration</TH><TH>tap_A</TH><TH>tap_B</TH><TH>tap_C</TH><TH>tap_A_change_count</TH><TH>tap_B_change_count</TH><TH>tap_C_change_count</TH><TH>sense_node</TH><TH>status</TH><TH>from</TH><TH>to</TH><TH>power_in</TH><TH>power_out</TH><TH>power_out_real</TH><TH>power_losses</TH><TH>power_in_A</TH><TH>power_in_B</TH><TH>power_in_C</TH><TH>power_out_A</TH><TH>power_out_B</TH><TH>power_out_C</TH><TH>power_losses_A</TH><TH>power_losses_B</TH><TH>power_losses_C</TH><TH>current_out_A</TH><TH>current_out_B</TH><TH>current_out_C</TH><TH>current_in_A</TH><TH>current_in_B</TH><TH>current_in_C</TH><TH>fault_current_in_A</TH><TH>fault_current_in_B</TH><TH>fault_current_in_C</TH><TH>fault_current_out_A</TH><TH>fault_current_out_B</TH><TH>fault_current_out_C</TH><TH>flow_direction</TH><TH>mean_repair_time</TH><TH>continuous_rating</TH><TH>emergency_rating</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/regulator_list/regulator"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><a href="#{configuration}"><xsl:value-of select="configuration"/></a></TD><TD><xsl:value-of select="tap_A"/></TD><TD><xsl:value-of select="tap_B"/></TD><TD><xsl:value-of select="tap_C"/></TD><TD><xsl:value-of select="tap_A_change_count"/></TD><TD><xsl:value-of select="tap_B_change_count"/></TD><TD><xsl:value-of select="tap_C_change_count"/></TD><TD><a href="#{sense_node}"><xsl:value-of select="sense_node"/></a></TD><TD><xsl:value-of select="status"/></TD><TD><a href="#{from}"><xsl:value-of select="from"/></a></TD><TD><a href="#{to}"><xsl:value-of select="to"/></a></TD><TD><xsl:value-of select="power_in"/></TD><TD><xsl:value-of select="power_out"/></TD><TD><xsl:value-of select="power_out_real"/></TD><TD><xsl:value-of select="power_losses"/></TD><TD><xsl:value-of select="power_in_A"/></TD><TD><xsl:value-of select="power_in_B"/></TD><TD><xsl:value-of select="power_in_C"/></TD><TD><xsl:value-of select="power_out_A"/></TD><TD><xsl:value-of select="power_out_B"/></TD><TD><xsl:value-of select="power_out_C"/></TD><TD><xsl:value-of select="power_losses_A"/></TD><TD><xsl:value-of select="power_losses_B"/></TD><TD><xsl:value-of select="power_losses_C"/></TD><TD><xsl:value-of select="current_out_A"/></TD><TD><xsl:value-of select="current_out_B"/></TD><TD><xsl:value-of select="current_out_C"/></TD><TD><xsl:value-of select="current_in_A"/></TD><TD><xsl:value-of select="current_in_B"/></TD><TD><xsl:value-of select="current_in_C"/></TD><TD><xsl:value-of select="fault_current_in_A"/></TD><TD><xsl:value-of select="fault_current_in_B"/></TD><TD><xsl:value-of select="fault_current_in_C"/></TD><TD><xsl:value-of select="fault_current_out_A"/></TD><TD><xsl:value-of select="fault_current_out_B"/></TD><TD><xsl:value-of select="fault_current_out_C"/></TD><TD><xsl:value-of select="flow_direction"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="continuous_rating"/></TD><TD><xsl:value-of select="emergency_rating"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>triplex_node objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>bustype</TH><TH>busflags</TH><TH>reference_bus</TH><TH>maximum_voltage_error</TH><TH>voltage_1</TH><TH>voltage_2</TH><TH>voltage_N</TH><TH>voltage_12</TH><TH>voltage_1N</TH><TH>voltage_2N</TH><TH>current_1</TH><TH>current_2</TH><TH>current_N</TH><TH>current_1_real</TH><TH>current_2_real</TH><TH>current_N_real</TH><TH>current_1_reac</TH><TH>current_2_reac</TH><TH>current_N_reac</TH><TH>current_12</TH><TH>current_12_real</TH><TH>current_12_reac</TH><TH>residential_nominal_current_1</TH><TH>residential_nominal_current_2</TH><TH>residential_nominal_current_12</TH><TH>residential_nominal_current_1_real</TH><TH>residential_nominal_current_1_imag</TH><TH>residential_nominal_current_2_real</TH><TH>residential_nominal_current_2_imag</TH><TH>residential_nominal_current_12_real</TH><TH>residential_nominal_current_12_imag</TH><TH>power_1</TH><TH>power_2</TH><TH>power_12</TH><TH>power_1_real</TH><TH>power_2_real</TH><TH>power_12_real</TH><TH>power_1_reac</TH><TH>power_2_reac</TH><TH>power_12_reac</TH><TH>shunt_1</TH><TH>shunt_2</TH><TH>shunt_12</TH><TH>impedance_1</TH><TH>impedance_2</TH><TH>impedance_12</TH><TH>impedance_1_real</TH><TH>impedance_2_real</TH><TH>impedance_12_real</TH><TH>impedance_1_reac</TH><TH>impedance_2_reac</TH><TH>impedance_12_reac</TH><TH>house_present</TH><TH>NR_mode</TH><TH>service_status</TH><TH>service_status_double</TH><TH>previous_uptime</TH><TH>current_uptime</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/triplex_node_list/triplex_node"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="bustype"/></TD><TD><xsl:value-of select="busflags"/></TD><TD><a href="#{reference_bus}"><xsl:value-of select="reference_bus"/></a></TD><TD><xsl:value-of select="maximum_voltage_error"/></TD><TD><xsl:value-of select="voltage_1"/></TD><TD><xsl:value-of select="voltage_2"/></TD><TD><xsl:value-of select="voltage_N"/></TD><TD><xsl:value-of select="voltage_12"/></TD><TD><xsl:value-of select="voltage_1N"/></TD><TD><xsl:value-of select="voltage_2N"/></TD><TD><xsl:value-of select="current_1"/></TD><TD><xsl:value-of select="current_2"/></TD><TD><xsl:value-of select="current_N"/></TD><TD><xsl:value-of select="current_1_real"/></TD><TD><xsl:value-of select="current_2_real"/></TD><TD><xsl:value-of select="current_N_real"/></TD><TD><xsl:value-of select="current_1_reac"/></TD><TD><xsl:value-of select="current_2_reac"/></TD><TD><xsl:value-of select="current_N_reac"/></TD><TD><xsl:value-of select="current_12"/></TD><TD><xsl:value-of select="current_12_real"/></TD><TD><xsl:value-of select="current_12_reac"/></TD><TD><xsl:value-of select="residential_nominal_current_1"/></TD><TD><xsl:value-of select="residential_nominal_current_2"/></TD><TD><xsl:value-of select="residential_nominal_current_12"/></TD><TD><xsl:value-of select="residential_nominal_current_1_real"/></TD><TD><xsl:value-of select="residential_nominal_current_1_imag"/></TD><TD><xsl:value-of select="residential_nominal_current_2_real"/></TD><TD><xsl:value-of select="residential_nominal_current_2_imag"/></TD><TD><xsl:value-of select="residential_nominal_current_12_real"/></TD><TD><xsl:value-of select="residential_nominal_current_12_imag"/></TD><TD><xsl:value-of select="power_1"/></TD><TD><xsl:value-of select="power_2"/></TD><TD><xsl:value-of select="power_12"/></TD><TD><xsl:value-of select="power_1_real"/></TD><TD><xsl:value-of select="power_2_real"/></TD><TD><xsl:value-of select="power_12_real"/></TD><TD><xsl:value-of select="power_1_reac"/></TD><TD><xsl:value-of select="power_2_reac"/></TD><TD><xsl:value-of select="power_12_reac"/></TD><TD><xsl:value-of select="shunt_1"/></TD><TD><xsl:value-of select="shunt_2"/></TD><TD><xsl:value-of select="shunt_12"/></TD><TD><xsl:value-of select="impedance_1"/></TD><TD><xsl:value-of select="impedance_2"/></TD><TD><xsl:value-of select="impedance_12"/></TD><TD><xsl:value-of select="impedance_1_real"/></TD><TD><xsl:value-of select="impedance_2_real"/></TD><TD><xsl:value-of select="impedance_12_real"/></TD><TD><xsl:value-of select="impedance_1_reac"/></TD><TD><xsl:value-of select="impedance_2_reac"/></TD><TD><xsl:value-of select="impedance_12_reac"/></TD><TD><xsl:value-of select="house_present"/></TD><TD><xsl:value-of select="NR_mode"/></TD><TD><xsl:value-of select="service_status"/></TD><TD><xsl:value-of select="service_status_double"/></TD><TD><xsl:value-of select="previous_uptime"/></TD><TD><xsl:value-of select="current_uptime"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>triplex_meter objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>measured_real_energy</TH><TH>measured_reactive_energy</TH><TH>measured_power</TH><TH>indiv_measured_power_1</TH><TH>indiv_measured_power_2</TH><TH>indiv_measured_power_N</TH><TH>measured_demand</TH><TH>measured_real_power</TH><TH>measured_reactive_power</TH><TH>meter_power_consumption</TH><TH>measured_voltage_1</TH><TH>measured_voltage_2</TH><TH>measured_voltage_N</TH><TH>measured_current_1</TH><TH>measured_current_2</TH><TH>measured_current_N</TH><TH>customer_interrupted</TH><TH>customer_interrupted_secondary</TH><TH>monthly_bill</TH><TH>previous_monthly_bill</TH><TH>previous_monthly_energy</TH><TH>monthly_fee</TH><TH>monthly_energy</TH><TH>bill_mode</TH><TH>power_market</TH><TH>bill_day</TH><TH>price</TH><TH>price_base</TH><TH>first_tier_price</TH><TH>first_tier_energy</TH><TH>second_tier_price</TH><TH>second_tier_energy</TH><TH>third_tier_price</TH><TH>third_tier_energy</TH><TH>bustype</TH><TH>busflags</TH><TH>reference_bus</TH><TH>maximum_voltage_error</TH><TH>voltage_1</TH><TH>voltage_2</TH><TH>voltage_N</TH><TH>voltage_12</TH><TH>voltage_1N</TH><TH>voltage_2N</TH><TH>current_1</TH><TH>current_2</TH><TH>current_N</TH><TH>current_1_real</TH><TH>current_2_real</TH><TH>current_N_real</TH><TH>current_1_reac</TH><TH>current_2_reac</TH><TH>current_N_reac</TH><TH>current_12</TH><TH>current_12_real</TH><TH>current_12_reac</TH><TH>residential_nominal_current_1</TH><TH>residential_nominal_current_2</TH><TH>residential_nominal_current_12</TH><TH>residential_nominal_current_1_real</TH><TH>residential_nominal_current_1_imag</TH><TH>residential_nominal_current_2_real</TH><TH>residential_nominal_current_2_imag</TH><TH>residential_nominal_current_12_real</TH><TH>residential_nominal_current_12_imag</TH><TH>power_1</TH><TH>power_2</TH><TH>power_12</TH><TH>power_1_real</TH><TH>power_2_real</TH><TH>power_12_real</TH><TH>power_1_reac</TH><TH>power_2_reac</TH><TH>power_12_reac</TH><TH>shunt_1</TH><TH>shunt_2</TH><TH>shunt_12</TH><TH>impedance_1</TH><TH>impedance_2</TH><TH>impedance_12</TH><TH>impedance_1_real</TH><TH>impedance_2_real</TH><TH>impedance_12_real</TH><TH>impedance_1_reac</TH><TH>impedance_2_reac</TH><TH>impedance_12_reac</TH><TH>house_present</TH><TH>NR_mode</TH><TH>service_status</TH><TH>service_status_double</TH><TH>previous_uptime</TH><TH>current_uptime</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/triplex_meter_list/triplex_meter"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="measured_real_energy"/></TD><TD><xsl:value-of select="measured_reactive_energy"/></TD><TD><xsl:value-of select="measured_power"/></TD><TD><xsl:value-of select="indiv_measured_power_1"/></TD><TD><xsl:value-of select="indiv_measured_power_2"/></TD><TD><xsl:value-of select="indiv_measured_power_N"/></TD><TD><xsl:value-of select="measured_demand"/></TD><TD><xsl:value-of select="measured_real_power"/></TD><TD><xsl:value-of select="measured_reactive_power"/></TD><TD><xsl:value-of select="meter_power_consumption"/></TD><TD><xsl:value-of select="measured_voltage_1"/></TD><TD><xsl:value-of select="measured_voltage_2"/></TD><TD><xsl:value-of select="measured_voltage_N"/></TD><TD><xsl:value-of select="measured_current_1"/></TD><TD><xsl:value-of select="measured_current_2"/></TD><TD><xsl:value-of select="measured_current_N"/></TD><TD><xsl:value-of select="customer_interrupted"/></TD><TD><xsl:value-of select="customer_interrupted_secondary"/></TD><TD><xsl:value-of select="monthly_bill"/></TD><TD><xsl:value-of select="previous_monthly_bill"/></TD><TD><xsl:value-of select="previous_monthly_energy"/></TD><TD><xsl:value-of select="monthly_fee"/></TD><TD><xsl:value-of select="monthly_energy"/></TD><TD><xsl:value-of select="bill_mode"/></TD><TD><a href="#{power_market}"><xsl:value-of select="power_market"/></a></TD><TD><xsl:value-of select="bill_day"/></TD><TD><xsl:value-of select="price"/></TD><TD><xsl:value-of select="price_base"/></TD><TD><xsl:value-of select="first_tier_price"/></TD><TD><xsl:value-of select="first_tier_energy"/></TD><TD><xsl:value-of select="second_tier_price"/></TD><TD><xsl:value-of select="second_tier_energy"/></TD><TD><xsl:value-of select="third_tier_price"/></TD><TD><xsl:value-of select="third_tier_energy"/></TD><TD><xsl:value-of select="bustype"/></TD><TD><xsl:value-of select="busflags"/></TD><TD><a href="#{reference_bus}"><xsl:value-of select="reference_bus"/></a></TD><TD><xsl:value-of select="maximum_voltage_error"/></TD><TD><xsl:value-of select="voltage_1"/></TD><TD><xsl:value-of select="voltage_2"/></TD><TD><xsl:value-of select="voltage_N"/></TD><TD><xsl:value-of select="voltage_12"/></TD><TD><xsl:value-of select="voltage_1N"/></TD><TD><xsl:value-of select="voltage_2N"/></TD><TD><xsl:value-of select="current_1"/></TD><TD><xsl:value-of select="current_2"/></TD><TD><xsl:value-of select="current_N"/></TD><TD><xsl:value-of select="current_1_real"/></TD><TD><xsl:value-of select="current_2_real"/></TD><TD><xsl:value-of select="current_N_real"/></TD><TD><xsl:value-of select="current_1_reac"/></TD><TD><xsl:value-of select="current_2_reac"/></TD><TD><xsl:value-of select="current_N_reac"/></TD><TD><xsl:value-of select="current_12"/></TD><TD><xsl:value-of select="current_12_real"/></TD><TD><xsl:value-of select="current_12_reac"/></TD><TD><xsl:value-of select="residential_nominal_current_1"/></TD><TD><xsl:value-of select="residential_nominal_current_2"/></TD><TD><xsl:value-of select="residential_nominal_current_12"/></TD><TD><xsl:value-of select="residential_nominal_current_1_real"/></TD><TD><xsl:value-of select="residential_nominal_current_1_imag"/></TD><TD><xsl:value-of select="residential_nominal_current_2_real"/></TD><TD><xsl:value-of select="residential_nominal_current_2_imag"/></TD><TD><xsl:value-of select="residential_nominal_current_12_real"/></TD><TD><xsl:value-of select="residential_nominal_current_12_imag"/></TD><TD><xsl:value-of select="power_1"/></TD><TD><xsl:value-of select="power_2"/></TD><TD><xsl:value-of select="power_12"/></TD><TD><xsl:value-of select="power_1_real"/></TD><TD><xsl:value-of select="power_2_real"/></TD><TD><xsl:value-of select="power_12_real"/></TD><TD><xsl:value-of select="power_1_reac"/></TD><TD><xsl:value-of select="power_2_reac"/></TD><TD><xsl:value-of select="power_12_reac"/></TD><TD><xsl:value-of select="shunt_1"/></TD><TD><xsl:value-of select="shunt_2"/></TD><TD><xsl:value-of select="shunt_12"/></TD><TD><xsl:value-of select="impedance_1"/></TD><TD><xsl:value-of select="impedance_2"/></TD><TD><xsl:value-of select="impedance_12"/></TD><TD><xsl:value-of select="impedance_1_real"/></TD><TD><xsl:value-of select="impedance_2_real"/></TD><TD><xsl:value-of select="impedance_12_real"/></TD><TD><xsl:value-of select="impedance_1_reac"/></TD><TD><xsl:value-of select="impedance_2_reac"/></TD><TD><xsl:value-of select="impedance_12_reac"/></TD><TD><xsl:value-of select="house_present"/></TD><TD><xsl:value-of select="NR_mode"/></TD><TD><xsl:value-of select="service_status"/></TD><TD><xsl:value-of select="service_status_double"/></TD><TD><xsl:value-of select="previous_uptime"/></TD><TD><xsl:value-of select="current_uptime"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>triplex_line objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>configuration</TH><TH>length</TH><TH>status</TH><TH>from</TH><TH>to</TH><TH>power_in</TH><TH>power_out</TH><TH>power_out_real</TH><TH>power_losses</TH><TH>power_in_A</TH><TH>power_in_B</TH><TH>power_in_C</TH><TH>power_out_A</TH><TH>power_out_B</TH><TH>power_out_C</TH><TH>power_losses_A</TH><TH>power_losses_B</TH><TH>power_losses_C</TH><TH>current_out_A</TH><TH>current_out_B</TH><TH>current_out_C</TH><TH>current_in_A</TH><TH>current_in_B</TH><TH>current_in_C</TH><TH>fault_current_in_A</TH><TH>fault_current_in_B</TH><TH>fault_current_in_C</TH><TH>fault_current_out_A</TH><TH>fault_current_out_B</TH><TH>fault_current_out_C</TH><TH>flow_direction</TH><TH>mean_repair_time</TH><TH>continuous_rating</TH><TH>emergency_rating</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/triplex_line_list/triplex_line"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><a href="#{configuration}"><xsl:value-of select="configuration"/></a></TD><TD><xsl:value-of select="length"/></TD><TD><xsl:value-of select="status"/></TD><TD><a href="#{from}"><xsl:value-of select="from"/></a></TD><TD><a href="#{to}"><xsl:value-of select="to"/></a></TD><TD><xsl:value-of select="power_in"/></TD><TD><xsl:value-of select="power_out"/></TD><TD><xsl:value-of select="power_out_real"/></TD><TD><xsl:value-of select="power_losses"/></TD><TD><xsl:value-of select="power_in_A"/></TD><TD><xsl:value-of select="power_in_B"/></TD><TD><xsl:value-of select="power_in_C"/></TD><TD><xsl:value-of select="power_out_A"/></TD><TD><xsl:value-of select="power_out_B"/></TD><TD><xsl:value-of select="power_out_C"/></TD><TD><xsl:value-of select="power_losses_A"/></TD><TD><xsl:value-of select="power_losses_B"/></TD><TD><xsl:value-of select="power_losses_C"/></TD><TD><xsl:value-of select="current_out_A"/></TD><TD><xsl:value-of select="current_out_B"/></TD><TD><xsl:value-of select="current_out_C"/></TD><TD><xsl:value-of select="current_in_A"/></TD><TD><xsl:value-of select="current_in_B"/></TD><TD><xsl:value-of select="current_in_C"/></TD><TD><xsl:value-of select="fault_current_in_A"/></TD><TD><xsl:value-of select="fault_current_in_B"/></TD><TD><xsl:value-of select="fault_current_in_C"/></TD><TD><xsl:value-of select="fault_current_out_A"/></TD><TD><xsl:value-of select="fault_current_out_B"/></TD><TD><xsl:value-of select="fault_current_out_C"/></TD><TD><xsl:value-of select="flow_direction"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="continuous_rating"/></TD><TD><xsl:value-of select="emergency_rating"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>triplex_line_configuration objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>conductor_1</TH><TH>conductor_2</TH><TH>conductor_N</TH><TH>insulation_thickness</TH><TH>diameter</TH><TH>spacing</TH><TH>z11</TH><TH>z12</TH><TH>z21</TH><TH>z22</TH></TR>
<xsl:for-each select="powerflow/triplex_line_configuration_list/triplex_line_configuration"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><a href="#{conductor_1}"><xsl:value-of select="conductor_1"/></a></TD><TD><a href="#{conductor_2}"><xsl:value-of select="conductor_2"/></a></TD><TD><a href="#{conductor_N}"><xsl:value-of select="conductor_N"/></a></TD><TD><xsl:value-of select="insulation_thickness"/></TD><TD><xsl:value-of select="diameter"/></TD><TD><a href="#{spacing}"><xsl:value-of select="spacing"/></a></TD><TD><xsl:value-of select="z11"/></TD><TD><xsl:value-of select="z12"/></TD><TD><xsl:value-of select="z21"/></TD><TD><xsl:value-of select="z22"/></TD></TR>
</xsl:for-each></TABLE>
<H4>triplex_line_conductor objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>resistance</TH><TH>geometric_mean_radius</TH><TH>rating.summer.continuous</TH><TH>rating.summer.emergency</TH><TH>rating.winter.continuous</TH><TH>rating.winter.emergency</TH></TR>
<xsl:for-each select="powerflow/triplex_line_conductor_list/triplex_line_conductor"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="resistance"/></TD><TD><xsl:value-of select="geometric_mean_radius"/></TD><TD><xsl:value-of select="rating.summer.continuous"/></TD><TD><xsl:value-of select="rating.summer.emergency"/></TD><TD><xsl:value-of select="rating.winter.continuous"/></TD><TD><xsl:value-of select="rating.winter.emergency"/></TD></TR>
</xsl:for-each></TABLE>
<H4>switch objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>phase_A_state</TH><TH>phase_B_state</TH><TH>phase_C_state</TH><TH>operating_mode</TH><TH>status</TH><TH>from</TH><TH>to</TH><TH>power_in</TH><TH>power_out</TH><TH>power_out_real</TH><TH>power_losses</TH><TH>power_in_A</TH><TH>power_in_B</TH><TH>power_in_C</TH><TH>power_out_A</TH><TH>power_out_B</TH><TH>power_out_C</TH><TH>power_losses_A</TH><TH>power_losses_B</TH><TH>power_losses_C</TH><TH>current_out_A</TH><TH>current_out_B</TH><TH>current_out_C</TH><TH>current_in_A</TH><TH>current_in_B</TH><TH>current_in_C</TH><TH>fault_current_in_A</TH><TH>fault_current_in_B</TH><TH>fault_current_in_C</TH><TH>fault_current_out_A</TH><TH>fault_current_out_B</TH><TH>fault_current_out_C</TH><TH>flow_direction</TH><TH>mean_repair_time</TH><TH>continuous_rating</TH><TH>emergency_rating</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/switch_list/switch"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="phase_A_state"/></TD><TD><xsl:value-of select="phase_B_state"/></TD><TD><xsl:value-of select="phase_C_state"/></TD><TD><xsl:value-of select="operating_mode"/></TD><TD><xsl:value-of select="status"/></TD><TD><a href="#{from}"><xsl:value-of select="from"/></a></TD><TD><a href="#{to}"><xsl:value-of select="to"/></a></TD><TD><xsl:value-of select="power_in"/></TD><TD><xsl:value-of select="power_out"/></TD><TD><xsl:value-of select="power_out_real"/></TD><TD><xsl:value-of select="power_losses"/></TD><TD><xsl:value-of select="power_in_A"/></TD><TD><xsl:value-of select="power_in_B"/></TD><TD><xsl:value-of select="power_in_C"/></TD><TD><xsl:value-of select="power_out_A"/></TD><TD><xsl:value-of select="power_out_B"/></TD><TD><xsl:value-of select="power_out_C"/></TD><TD><xsl:value-of select="power_losses_A"/></TD><TD><xsl:value-of select="power_losses_B"/></TD><TD><xsl:value-of select="power_losses_C"/></TD><TD><xsl:value-of select="current_out_A"/></TD><TD><xsl:value-of select="current_out_B"/></TD><TD><xsl:value-of select="current_out_C"/></TD><TD><xsl:value-of select="current_in_A"/></TD><TD><xsl:value-of select="current_in_B"/></TD><TD><xsl:value-of select="current_in_C"/></TD><TD><xsl:value-of select="fault_current_in_A"/></TD><TD><xsl:value-of select="fault_current_in_B"/></TD><TD><xsl:value-of select="fault_current_in_C"/></TD><TD><xsl:value-of select="fault_current_out_A"/></TD><TD><xsl:value-of select="fault_current_out_B"/></TD><TD><xsl:value-of select="fault_current_out_C"/></TD><TD><xsl:value-of select="flow_direction"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="continuous_rating"/></TD><TD><xsl:value-of select="emergency_rating"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>substation objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>distribution_energy</TH><TH>distribution_power</TH><TH>distribution_demand</TH><TH>distribution_voltage_A</TH><TH>distribution_voltage_B</TH><TH>distribution_voltage_C</TH><TH>distribution_current_A</TH><TH>distribution_current_B</TH><TH>distribution_current_C</TH><TH>Network_Node_Base_Power</TH><TH>Network_Node_Base_Voltage</TH><TH>bustype</TH><TH>busflags</TH><TH>reference_bus</TH><TH>maximum_voltage_error</TH><TH>voltage_A</TH><TH>voltage_B</TH><TH>voltage_C</TH><TH>voltage_AB</TH><TH>voltage_BC</TH><TH>voltage_CA</TH><TH>current_A</TH><TH>current_B</TH><TH>current_C</TH><TH>power_A</TH><TH>power_B</TH><TH>power_C</TH><TH>shunt_A</TH><TH>shunt_B</TH><TH>shunt_C</TH><TH>NR_mode</TH><TH>mean_repair_time</TH><TH>service_status</TH><TH>service_status_double</TH><TH>previous_uptime</TH><TH>current_uptime</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/substation_list/substation"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="distribution_energy"/></TD><TD><xsl:value-of select="distribution_power"/></TD><TD><xsl:value-of select="distribution_demand"/></TD><TD><xsl:value-of select="distribution_voltage_A"/></TD><TD><xsl:value-of select="distribution_voltage_B"/></TD><TD><xsl:value-of select="distribution_voltage_C"/></TD><TD><xsl:value-of select="distribution_current_A"/></TD><TD><xsl:value-of select="distribution_current_B"/></TD><TD><xsl:value-of select="distribution_current_C"/></TD><TD><xsl:value-of select="Network_Node_Base_Power"/></TD><TD><xsl:value-of select="Network_Node_Base_Voltage"/></TD><TD><xsl:value-of select="bustype"/></TD><TD><xsl:value-of select="busflags"/></TD><TD><a href="#{reference_bus}"><xsl:value-of select="reference_bus"/></a></TD><TD><xsl:value-of select="maximum_voltage_error"/></TD><TD><xsl:value-of select="voltage_A"/></TD><TD><xsl:value-of select="voltage_B"/></TD><TD><xsl:value-of select="voltage_C"/></TD><TD><xsl:value-of select="voltage_AB"/></TD><TD><xsl:value-of select="voltage_BC"/></TD><TD><xsl:value-of select="voltage_CA"/></TD><TD><xsl:value-of select="current_A"/></TD><TD><xsl:value-of select="current_B"/></TD><TD><xsl:value-of select="current_C"/></TD><TD><xsl:value-of select="power_A"/></TD><TD><xsl:value-of select="power_B"/></TD><TD><xsl:value-of select="power_C"/></TD><TD><xsl:value-of select="shunt_A"/></TD><TD><xsl:value-of select="shunt_B"/></TD><TD><xsl:value-of select="shunt_C"/></TD><TD><xsl:value-of select="NR_mode"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="service_status"/></TD><TD><xsl:value-of select="service_status_double"/></TD><TD><xsl:value-of select="previous_uptime"/></TD><TD><xsl:value-of select="current_uptime"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>pqload objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>weather</TH><TH>T_nominal</TH><TH>Zp_T</TH><TH>Zp_H</TH><TH>Zp_S</TH><TH>Zp_W</TH><TH>Zp_R</TH><TH>Zp</TH><TH>Zq_T</TH><TH>Zq_H</TH><TH>Zq_S</TH><TH>Zq_W</TH><TH>Zq_R</TH><TH>Zq</TH><TH>Im_T</TH><TH>Im_H</TH><TH>Im_S</TH><TH>Im_W</TH><TH>Im_R</TH><TH>Im</TH><TH>Ia_T</TH><TH>Ia_H</TH><TH>Ia_S</TH><TH>Ia_W</TH><TH>Ia_R</TH><TH>Ia</TH><TH>Pp_T</TH><TH>Pp_H</TH><TH>Pp_S</TH><TH>Pp_W</TH><TH>Pp_R</TH><TH>Pp</TH><TH>Pq_T</TH><TH>Pq_H</TH><TH>Pq_S</TH><TH>Pq_W</TH><TH>Pq_R</TH><TH>Pq</TH><TH>input_temp</TH><TH>input_humid</TH><TH>input_solar</TH><TH>input_wind</TH><TH>input_rain</TH><TH>output_imped_p</TH><TH>output_imped_q</TH><TH>output_current_m</TH><TH>output_current_a</TH><TH>output_power_p</TH><TH>output_power_q</TH><TH>output_impedance</TH><TH>output_current</TH><TH>output_power</TH><TH>load_class</TH><TH>constant_power_A</TH><TH>constant_power_B</TH><TH>constant_power_C</TH><TH>constant_power_A_real</TH><TH>constant_power_B_real</TH><TH>constant_power_C_real</TH><TH>constant_power_A_reac</TH><TH>constant_power_B_reac</TH><TH>constant_power_C_reac</TH><TH>constant_current_A</TH><TH>constant_current_B</TH><TH>constant_current_C</TH><TH>constant_current_A_real</TH><TH>constant_current_B_real</TH><TH>constant_current_C_real</TH><TH>constant_current_A_reac</TH><TH>constant_current_B_reac</TH><TH>constant_current_C_reac</TH><TH>constant_impedance_A</TH><TH>constant_impedance_B</TH><TH>constant_impedance_C</TH><TH>constant_impedance_A_real</TH><TH>constant_impedance_B_real</TH><TH>constant_impedance_C_real</TH><TH>constant_impedance_A_reac</TH><TH>constant_impedance_B_reac</TH><TH>constant_impedance_C_reac</TH><TH>measured_voltage_A</TH><TH>measured_voltage_B</TH><TH>measured_voltage_C</TH><TH>measured_voltage_AB</TH><TH>measured_voltage_BC</TH><TH>measured_voltage_CA</TH><TH>phase_loss_protection</TH><TH>base_power_A</TH><TH>base_power_B</TH><TH>base_power_C</TH><TH>power_pf_A</TH><TH>current_pf_A</TH><TH>impedance_pf_A</TH><TH>power_pf_B</TH><TH>current_pf_B</TH><TH>impedance_pf_B</TH><TH>power_pf_C</TH><TH>current_pf_C</TH><TH>impedance_pf_C</TH><TH>power_fraction_A</TH><TH>current_fraction_A</TH><TH>impedance_fraction_A</TH><TH>power_fraction_B</TH><TH>current_fraction_B</TH><TH>impedance_fraction_B</TH><TH>power_fraction_C</TH><TH>current_fraction_C</TH><TH>impedance_fraction_C</TH><TH>bustype</TH><TH>busflags</TH><TH>reference_bus</TH><TH>maximum_voltage_error</TH><TH>voltage_A</TH><TH>voltage_B</TH><TH>voltage_C</TH><TH>voltage_AB</TH><TH>voltage_BC</TH><TH>voltage_CA</TH><TH>current_A</TH><TH>current_B</TH><TH>current_C</TH><TH>power_A</TH><TH>power_B</TH><TH>power_C</TH><TH>shunt_A</TH><TH>shunt_B</TH><TH>shunt_C</TH><TH>NR_mode</TH><TH>mean_repair_time</TH><TH>service_status</TH><TH>service_status_double</TH><TH>previous_uptime</TH><TH>current_uptime</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/pqload_list/pqload"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><a href="#{weather}"><xsl:value-of select="weather"/></a></TD><TD><xsl:value-of select="T_nominal"/></TD><TD><xsl:value-of select="Zp_T"/></TD><TD><xsl:value-of select="Zp_H"/></TD><TD><xsl:value-of select="Zp_S"/></TD><TD><xsl:value-of select="Zp_W"/></TD><TD><xsl:value-of select="Zp_R"/></TD><TD><xsl:value-of select="Zp"/></TD><TD><xsl:value-of select="Zq_T"/></TD><TD><xsl:value-of select="Zq_H"/></TD><TD><xsl:value-of select="Zq_S"/></TD><TD><xsl:value-of select="Zq_W"/></TD><TD><xsl:value-of select="Zq_R"/></TD><TD><xsl:value-of select="Zq"/></TD><TD><xsl:value-of select="Im_T"/></TD><TD><xsl:value-of select="Im_H"/></TD><TD><xsl:value-of select="Im_S"/></TD><TD><xsl:value-of select="Im_W"/></TD><TD><xsl:value-of select="Im_R"/></TD><TD><xsl:value-of select="Im"/></TD><TD><xsl:value-of select="Ia_T"/></TD><TD><xsl:value-of select="Ia_H"/></TD><TD><xsl:value-of select="Ia_S"/></TD><TD><xsl:value-of select="Ia_W"/></TD><TD><xsl:value-of select="Ia_R"/></TD><TD><xsl:value-of select="Ia"/></TD><TD><xsl:value-of select="Pp_T"/></TD><TD><xsl:value-of select="Pp_H"/></TD><TD><xsl:value-of select="Pp_S"/></TD><TD><xsl:value-of select="Pp_W"/></TD><TD><xsl:value-of select="Pp_R"/></TD><TD><xsl:value-of select="Pp"/></TD><TD><xsl:value-of select="Pq_T"/></TD><TD><xsl:value-of select="Pq_H"/></TD><TD><xsl:value-of select="Pq_S"/></TD><TD><xsl:value-of select="Pq_W"/></TD><TD><xsl:value-of select="Pq_R"/></TD><TD><xsl:value-of select="Pq"/></TD><TD><xsl:value-of select="input_temp"/></TD><TD><xsl:value-of select="input_humid"/></TD><TD><xsl:value-of select="input_solar"/></TD><TD><xsl:value-of select="input_wind"/></TD><TD><xsl:value-of select="input_rain"/></TD><TD><xsl:value-of select="output_imped_p"/></TD><TD><xsl:value-of select="output_imped_q"/></TD><TD><xsl:value-of select="output_current_m"/></TD><TD><xsl:value-of select="output_current_a"/></TD><TD><xsl:value-of select="output_power_p"/></TD><TD><xsl:value-of select="output_power_q"/></TD><TD><xsl:value-of select="output_impedance"/></TD><TD><xsl:value-of select="output_current"/></TD><TD><xsl:value-of select="output_power"/></TD><TD><xsl:value-of select="load_class"/></TD><TD><xsl:value-of select="constant_power_A"/></TD><TD><xsl:value-of select="constant_power_B"/></TD><TD><xsl:value-of select="constant_power_C"/></TD><TD><xsl:value-of select="constant_power_A_real"/></TD><TD><xsl:value-of select="constant_power_B_real"/></TD><TD><xsl:value-of select="constant_power_C_real"/></TD><TD><xsl:value-of select="constant_power_A_reac"/></TD><TD><xsl:value-of select="constant_power_B_reac"/></TD><TD><xsl:value-of select="constant_power_C_reac"/></TD><TD><xsl:value-of select="constant_current_A"/></TD><TD><xsl:value-of select="constant_current_B"/></TD><TD><xsl:value-of select="constant_current_C"/></TD><TD><xsl:value-of select="constant_current_A_real"/></TD><TD><xsl:value-of select="constant_current_B_real"/></TD><TD><xsl:value-of select="constant_current_C_real"/></TD><TD><xsl:value-of select="constant_current_A_reac"/></TD><TD><xsl:value-of select="constant_current_B_reac"/></TD><TD><xsl:value-of select="constant_current_C_reac"/></TD><TD><xsl:value-of select="constant_impedance_A"/></TD><TD><xsl:value-of select="constant_impedance_B"/></TD><TD><xsl:value-of select="constant_impedance_C"/></TD><TD><xsl:value-of select="constant_impedance_A_real"/></TD><TD><xsl:value-of select="constant_impedance_B_real"/></TD><TD><xsl:value-of select="constant_impedance_C_real"/></TD><TD><xsl:value-of select="constant_impedance_A_reac"/></TD><TD><xsl:value-of select="constant_impedance_B_reac"/></TD><TD><xsl:value-of select="constant_impedance_C_reac"/></TD><TD><xsl:value-of select="measured_voltage_A"/></TD><TD><xsl:value-of select="measured_voltage_B"/></TD><TD><xsl:value-of select="measured_voltage_C"/></TD><TD><xsl:value-of select="measured_voltage_AB"/></TD><TD><xsl:value-of select="measured_voltage_BC"/></TD><TD><xsl:value-of select="measured_voltage_CA"/></TD><TD><xsl:value-of select="phase_loss_protection"/></TD><TD><xsl:value-of select="base_power_A"/></TD><TD><xsl:value-of select="base_power_B"/></TD><TD><xsl:value-of select="base_power_C"/></TD><TD><xsl:value-of select="power_pf_A"/></TD><TD><xsl:value-of select="current_pf_A"/></TD><TD><xsl:value-of select="impedance_pf_A"/></TD><TD><xsl:value-of select="power_pf_B"/></TD><TD><xsl:value-of select="current_pf_B"/></TD><TD><xsl:value-of select="impedance_pf_B"/></TD><TD><xsl:value-of select="power_pf_C"/></TD><TD><xsl:value-of select="current_pf_C"/></TD><TD><xsl:value-of select="impedance_pf_C"/></TD><TD><xsl:value-of select="power_fraction_A"/></TD><TD><xsl:value-of select="current_fraction_A"/></TD><TD><xsl:value-of select="impedance_fraction_A"/></TD><TD><xsl:value-of select="power_fraction_B"/></TD><TD><xsl:value-of select="current_fraction_B"/></TD><TD><xsl:value-of select="impedance_fraction_B"/></TD><TD><xsl:value-of select="power_fraction_C"/></TD><TD><xsl:value-of select="current_fraction_C"/></TD><TD><xsl:value-of select="impedance_fraction_C"/></TD><TD><xsl:value-of select="bustype"/></TD><TD><xsl:value-of select="busflags"/></TD><TD><a href="#{reference_bus}"><xsl:value-of select="reference_bus"/></a></TD><TD><xsl:value-of select="maximum_voltage_error"/></TD><TD><xsl:value-of select="voltage_A"/></TD><TD><xsl:value-of select="voltage_B"/></TD><TD><xsl:value-of select="voltage_C"/></TD><TD><xsl:value-of select="voltage_AB"/></TD><TD><xsl:value-of select="voltage_BC"/></TD><TD><xsl:value-of select="voltage_CA"/></TD><TD><xsl:value-of select="current_A"/></TD><TD><xsl:value-of select="current_B"/></TD><TD><xsl:value-of select="current_C"/></TD><TD><xsl:value-of select="power_A"/></TD><TD><xsl:value-of select="power_B"/></TD><TD><xsl:value-of select="power_C"/></TD><TD><xsl:value-of select="shunt_A"/></TD><TD><xsl:value-of select="shunt_B"/></TD><TD><xsl:value-of select="shunt_C"/></TD><TD><xsl:value-of select="NR_mode"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="service_status"/></TD><TD><xsl:value-of select="service_status_double"/></TD><TD><xsl:value-of select="previous_uptime"/></TD><TD><xsl:value-of select="current_uptime"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>voltdump objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>group</TH><TH>runtime</TH><TH>filename</TH><TH>runcount</TH><TH>mode</TH></TR>
<xsl:for-each select="powerflow/voltdump_list/voltdump"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="group"/></TD><TD><xsl:value-of select="runtime"/></TD><TD><xsl:value-of select="filename"/></TD><TD><xsl:value-of select="runcount"/></TD><TD><xsl:value-of select="mode"/></TD></TR>
</xsl:for-each></TABLE>
<H4>series_reactor objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>phase_A_impedance</TH><TH>phase_A_resistance</TH><TH>phase_A_reactance</TH><TH>phase_B_impedance</TH><TH>phase_B_resistance</TH><TH>phase_B_reactance</TH><TH>phase_C_impedance</TH><TH>phase_C_resistance</TH><TH>phase_C_reactance</TH><TH>rated_current_limit</TH><TH>status</TH><TH>from</TH><TH>to</TH><TH>power_in</TH><TH>power_out</TH><TH>power_out_real</TH><TH>power_losses</TH><TH>power_in_A</TH><TH>power_in_B</TH><TH>power_in_C</TH><TH>power_out_A</TH><TH>power_out_B</TH><TH>power_out_C</TH><TH>power_losses_A</TH><TH>power_losses_B</TH><TH>power_losses_C</TH><TH>current_out_A</TH><TH>current_out_B</TH><TH>current_out_C</TH><TH>current_in_A</TH><TH>current_in_B</TH><TH>current_in_C</TH><TH>fault_current_in_A</TH><TH>fault_current_in_B</TH><TH>fault_current_in_C</TH><TH>fault_current_out_A</TH><TH>fault_current_out_B</TH><TH>fault_current_out_C</TH><TH>flow_direction</TH><TH>mean_repair_time</TH><TH>continuous_rating</TH><TH>emergency_rating</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/series_reactor_list/series_reactor"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="phase_A_impedance"/></TD><TD><xsl:value-of select="phase_A_resistance"/></TD><TD><xsl:value-of select="phase_A_reactance"/></TD><TD><xsl:value-of select="phase_B_impedance"/></TD><TD><xsl:value-of select="phase_B_resistance"/></TD><TD><xsl:value-of select="phase_B_reactance"/></TD><TD><xsl:value-of select="phase_C_impedance"/></TD><TD><xsl:value-of select="phase_C_resistance"/></TD><TD><xsl:value-of select="phase_C_reactance"/></TD><TD><xsl:value-of select="rated_current_limit"/></TD><TD><xsl:value-of select="status"/></TD><TD><a href="#{from}"><xsl:value-of select="from"/></a></TD><TD><a href="#{to}"><xsl:value-of select="to"/></a></TD><TD><xsl:value-of select="power_in"/></TD><TD><xsl:value-of select="power_out"/></TD><TD><xsl:value-of select="power_out_real"/></TD><TD><xsl:value-of select="power_losses"/></TD><TD><xsl:value-of select="power_in_A"/></TD><TD><xsl:value-of select="power_in_B"/></TD><TD><xsl:value-of select="power_in_C"/></TD><TD><xsl:value-of select="power_out_A"/></TD><TD><xsl:value-of select="power_out_B"/></TD><TD><xsl:value-of select="power_out_C"/></TD><TD><xsl:value-of select="power_losses_A"/></TD><TD><xsl:value-of select="power_losses_B"/></TD><TD><xsl:value-of select="power_losses_C"/></TD><TD><xsl:value-of select="current_out_A"/></TD><TD><xsl:value-of select="current_out_B"/></TD><TD><xsl:value-of select="current_out_C"/></TD><TD><xsl:value-of select="current_in_A"/></TD><TD><xsl:value-of select="current_in_B"/></TD><TD><xsl:value-of select="current_in_C"/></TD><TD><xsl:value-of select="fault_current_in_A"/></TD><TD><xsl:value-of select="fault_current_in_B"/></TD><TD><xsl:value-of select="fault_current_in_C"/></TD><TD><xsl:value-of select="fault_current_out_A"/></TD><TD><xsl:value-of select="fault_current_out_B"/></TD><TD><xsl:value-of select="fault_current_out_C"/></TD><TD><xsl:value-of select="flow_direction"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="continuous_rating"/></TD><TD><xsl:value-of select="emergency_rating"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>restoration objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>reconfig_attempts</TH><TH>reconfig_iteration_limit</TH><TH>populate_tree</TH></TR>
<xsl:for-each select="powerflow/restoration_list/restoration"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="reconfig_attempts"/></TD><TD><xsl:value-of select="reconfig_iteration_limit"/></TD><TD><xsl:value-of select="populate_tree"/></TD></TR>
</xsl:for-each></TABLE>
<H4>frequency_gen objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>Frequency_Mode</TH><TH>Frequency</TH><TH>FreqChange</TH><TH>Deadband</TH><TH>Tolerance</TH><TH>M</TH><TH>D</TH><TH>Rated_power</TH><TH>Gen_power</TH><TH>Load_power</TH><TH>Gov_delay</TH><TH>Ramp_rate</TH><TH>Low_Freq_OI</TH><TH>High_Freq_OI</TH><TH>avg24</TH><TH>std24</TH><TH>avg168</TH><TH>std168</TH><TH>Num_Resp_Eqs</TH></TR>
<xsl:for-each select="powerflow/frequency_gen_list/frequency_gen"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="Frequency_Mode"/></TD><TD><xsl:value-of select="Frequency"/></TD><TD><xsl:value-of select="FreqChange"/></TD><TD><xsl:value-of select="Deadband"/></TD><TD><xsl:value-of select="Tolerance"/></TD><TD><xsl:value-of select="M"/></TD><TD><xsl:value-of select="D"/></TD><TD><xsl:value-of select="Rated_power"/></TD><TD><xsl:value-of select="Gen_power"/></TD><TD><xsl:value-of select="Load_power"/></TD><TD><xsl:value-of select="Gov_delay"/></TD><TD><xsl:value-of select="Ramp_rate"/></TD><TD><xsl:value-of select="Low_Freq_OI"/></TD><TD><xsl:value-of select="High_Freq_OI"/></TD><TD><xsl:value-of select="avg24"/></TD><TD><xsl:value-of select="std24"/></TD><TD><xsl:value-of select="avg168"/></TD><TD><xsl:value-of select="std168"/></TD><TD><xsl:value-of select="Num_Resp_Eqs"/></TD></TR>
</xsl:for-each></TABLE>
<H4>volt_var_control objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>control_method</TH><TH>capacitor_delay</TH><TH>regulator_delay</TH><TH>desired_pf</TH><TH>d_max</TH><TH>d_min</TH><TH>substation_link</TH><TH>pf_phase</TH><TH>regulator_list</TH><TH>capacitor_list</TH><TH>voltage_measurements</TH><TH>minimum_voltages</TH><TH>maximum_voltages</TH><TH>desired_voltages</TH><TH>max_vdrop</TH><TH>high_load_deadband</TH><TH>low_load_deadband</TH><TH>pf_signed</TH></TR>
<xsl:for-each select="powerflow/volt_var_control_list/volt_var_control"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="control_method"/></TD><TD><xsl:value-of select="capacitor_delay"/></TD><TD><xsl:value-of select="regulator_delay"/></TD><TD><xsl:value-of select="desired_pf"/></TD><TD><xsl:value-of select="d_max"/></TD><TD><xsl:value-of select="d_min"/></TD><TD><a href="#{substation_link}"><xsl:value-of select="substation_link"/></a></TD><TD><xsl:value-of select="pf_phase"/></TD><TD><xsl:value-of select="regulator_list"/></TD><TD><xsl:value-of select="capacitor_list"/></TD><TD><xsl:value-of select="voltage_measurements"/></TD><TD><xsl:value-of select="minimum_voltages"/></TD><TD><xsl:value-of select="maximum_voltages"/></TD><TD><xsl:value-of select="desired_voltages"/></TD><TD><xsl:value-of select="max_vdrop"/></TD><TD><xsl:value-of select="high_load_deadband"/></TD><TD><xsl:value-of select="low_load_deadband"/></TD><TD><xsl:value-of select="pf_signed"/></TD></TR>
</xsl:for-each></TABLE>
<H4>fault_check objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>check_mode</TH><TH>output_filename</TH><TH>reliability_mode</TH><TH>eventgen_object</TH></TR>
<xsl:for-each select="powerflow/fault_check_list/fault_check"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="check_mode"/></TD><TD><xsl:value-of select="output_filename"/></TD><TD><xsl:value-of select="reliability_mode"/></TD><TD><a href="#{eventgen_object}"><xsl:value-of select="eventgen_object"/></a></TD></TR>
</xsl:for-each></TABLE>
<H4>motor objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>bustype</TH><TH>busflags</TH><TH>reference_bus</TH><TH>maximum_voltage_error</TH><TH>voltage_A</TH><TH>voltage_B</TH><TH>voltage_C</TH><TH>voltage_AB</TH><TH>voltage_BC</TH><TH>voltage_CA</TH><TH>current_A</TH><TH>current_B</TH><TH>current_C</TH><TH>power_A</TH><TH>power_B</TH><TH>power_C</TH><TH>shunt_A</TH><TH>shunt_B</TH><TH>shunt_C</TH><TH>NR_mode</TH><TH>mean_repair_time</TH><TH>service_status</TH><TH>service_status_double</TH><TH>previous_uptime</TH><TH>current_uptime</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/motor_list/motor"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="bustype"/></TD><TD><xsl:value-of select="busflags"/></TD><TD><a href="#{reference_bus}"><xsl:value-of select="reference_bus"/></a></TD><TD><xsl:value-of select="maximum_voltage_error"/></TD><TD><xsl:value-of select="voltage_A"/></TD><TD><xsl:value-of select="voltage_B"/></TD><TD><xsl:value-of select="voltage_C"/></TD><TD><xsl:value-of select="voltage_AB"/></TD><TD><xsl:value-of select="voltage_BC"/></TD><TD><xsl:value-of select="voltage_CA"/></TD><TD><xsl:value-of select="current_A"/></TD><TD><xsl:value-of select="current_B"/></TD><TD><xsl:value-of select="current_C"/></TD><TD><xsl:value-of select="power_A"/></TD><TD><xsl:value-of select="power_B"/></TD><TD><xsl:value-of select="power_C"/></TD><TD><xsl:value-of select="shunt_A"/></TD><TD><xsl:value-of select="shunt_B"/></TD><TD><xsl:value-of select="shunt_C"/></TD><TD><xsl:value-of select="NR_mode"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="service_status"/></TD><TD><xsl:value-of select="service_status_double"/></TD><TD><xsl:value-of select="previous_uptime"/></TD><TD><xsl:value-of select="current_uptime"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>billdump objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>group</TH><TH>runtime</TH><TH>filename</TH><TH>runcount</TH><TH>meter_type</TH></TR>
<xsl:for-each select="powerflow/billdump_list/billdump"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="group"/></TD><TD><xsl:value-of select="runtime"/></TD><TD><xsl:value-of select="filename"/></TD><TD><xsl:value-of select="runcount"/></TD><TD><xsl:value-of select="meter_type"/></TD></TR>
</xsl:for-each></TABLE>
<H4>power_metrics objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>SAIFI</TH><TH>SAIFI_int</TH><TH>SAIDI</TH><TH>SAIDI_int</TH><TH>CAIDI</TH><TH>CAIDI_int</TH><TH>ASAI</TH><TH>ASAI_int</TH><TH>MAIFI</TH><TH>MAIFI_int</TH><TH>base_time_value</TH></TR>
<xsl:for-each select="powerflow/power_metrics_list/power_metrics"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="SAIFI"/></TD><TD><xsl:value-of select="SAIFI_int"/></TD><TD><xsl:value-of select="SAIDI"/></TD><TD><xsl:value-of select="SAIDI_int"/></TD><TD><xsl:value-of select="CAIDI"/></TD><TD><xsl:value-of select="CAIDI_int"/></TD><TD><xsl:value-of select="ASAI"/></TD><TD><xsl:value-of select="ASAI_int"/></TD><TD><xsl:value-of select="MAIFI"/></TD><TD><xsl:value-of select="MAIFI_int"/></TD><TD><xsl:value-of select="base_time_value"/></TD></TR>
</xsl:for-each></TABLE>
<H4>currdump objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>group</TH><TH>runtime</TH><TH>filename</TH><TH>runcount</TH><TH>mode</TH></TR>
<xsl:for-each select="powerflow/currdump_list/currdump"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="group"/></TD><TD><xsl:value-of select="runtime"/></TD><TD><xsl:value-of select="filename"/></TD><TD><xsl:value-of select="runcount"/></TD><TD><xsl:value-of select="mode"/></TD></TR>
</xsl:for-each></TABLE>
<H4>recloser objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>retry_time</TH><TH>max_number_of_tries</TH><TH>number_of_tries</TH><TH>phase_A_state</TH><TH>phase_B_state</TH><TH>phase_C_state</TH><TH>operating_mode</TH><TH>status</TH><TH>from</TH><TH>to</TH><TH>power_in</TH><TH>power_out</TH><TH>power_out_real</TH><TH>power_losses</TH><TH>power_in_A</TH><TH>power_in_B</TH><TH>power_in_C</TH><TH>power_out_A</TH><TH>power_out_B</TH><TH>power_out_C</TH><TH>power_losses_A</TH><TH>power_losses_B</TH><TH>power_losses_C</TH><TH>current_out_A</TH><TH>current_out_B</TH><TH>current_out_C</TH><TH>current_in_A</TH><TH>current_in_B</TH><TH>current_in_C</TH><TH>fault_current_in_A</TH><TH>fault_current_in_B</TH><TH>fault_current_in_C</TH><TH>fault_current_out_A</TH><TH>fault_current_out_B</TH><TH>fault_current_out_C</TH><TH>flow_direction</TH><TH>mean_repair_time</TH><TH>continuous_rating</TH><TH>emergency_rating</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/recloser_list/recloser"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="retry_time"/></TD><TD><xsl:value-of select="max_number_of_tries"/></TD><TD><xsl:value-of select="number_of_tries"/></TD><TD><xsl:value-of select="phase_A_state"/></TD><TD><xsl:value-of select="phase_B_state"/></TD><TD><xsl:value-of select="phase_C_state"/></TD><TD><xsl:value-of select="operating_mode"/></TD><TD><xsl:value-of select="status"/></TD><TD><a href="#{from}"><xsl:value-of select="from"/></a></TD><TD><a href="#{to}"><xsl:value-of select="to"/></a></TD><TD><xsl:value-of select="power_in"/></TD><TD><xsl:value-of select="power_out"/></TD><TD><xsl:value-of select="power_out_real"/></TD><TD><xsl:value-of select="power_losses"/></TD><TD><xsl:value-of select="power_in_A"/></TD><TD><xsl:value-of select="power_in_B"/></TD><TD><xsl:value-of select="power_in_C"/></TD><TD><xsl:value-of select="power_out_A"/></TD><TD><xsl:value-of select="power_out_B"/></TD><TD><xsl:value-of select="power_out_C"/></TD><TD><xsl:value-of select="power_losses_A"/></TD><TD><xsl:value-of select="power_losses_B"/></TD><TD><xsl:value-of select="power_losses_C"/></TD><TD><xsl:value-of select="current_out_A"/></TD><TD><xsl:value-of select="current_out_B"/></TD><TD><xsl:value-of select="current_out_C"/></TD><TD><xsl:value-of select="current_in_A"/></TD><TD><xsl:value-of select="current_in_B"/></TD><TD><xsl:value-of select="current_in_C"/></TD><TD><xsl:value-of select="fault_current_in_A"/></TD><TD><xsl:value-of select="fault_current_in_B"/></TD><TD><xsl:value-of select="fault_current_in_C"/></TD><TD><xsl:value-of select="fault_current_out_A"/></TD><TD><xsl:value-of select="fault_current_out_B"/></TD><TD><xsl:value-of select="fault_current_out_C"/></TD><TD><xsl:value-of select="flow_direction"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="continuous_rating"/></TD><TD><xsl:value-of select="emergency_rating"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>sectionalizer objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>phase_A_state</TH><TH>phase_B_state</TH><TH>phase_C_state</TH><TH>operating_mode</TH><TH>status</TH><TH>from</TH><TH>to</TH><TH>power_in</TH><TH>power_out</TH><TH>power_out_real</TH><TH>power_losses</TH><TH>power_in_A</TH><TH>power_in_B</TH><TH>power_in_C</TH><TH>power_out_A</TH><TH>power_out_B</TH><TH>power_out_C</TH><TH>power_losses_A</TH><TH>power_losses_B</TH><TH>power_losses_C</TH><TH>current_out_A</TH><TH>current_out_B</TH><TH>current_out_C</TH><TH>current_in_A</TH><TH>current_in_B</TH><TH>current_in_C</TH><TH>fault_current_in_A</TH><TH>fault_current_in_B</TH><TH>fault_current_in_C</TH><TH>fault_current_out_A</TH><TH>fault_current_out_B</TH><TH>fault_current_out_C</TH><TH>flow_direction</TH><TH>mean_repair_time</TH><TH>continuous_rating</TH><TH>emergency_rating</TH><TH>phases</TH><TH>nominal_voltage</TH></TR>
<xsl:for-each select="powerflow/sectionalizer_list/sectionalizer"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="phase_A_state"/></TD><TD><xsl:value-of select="phase_B_state"/></TD><TD><xsl:value-of select="phase_C_state"/></TD><TD><xsl:value-of select="operating_mode"/></TD><TD><xsl:value-of select="status"/></TD><TD><a href="#{from}"><xsl:value-of select="from"/></a></TD><TD><a href="#{to}"><xsl:value-of select="to"/></a></TD><TD><xsl:value-of select="power_in"/></TD><TD><xsl:value-of select="power_out"/></TD><TD><xsl:value-of select="power_out_real"/></TD><TD><xsl:value-of select="power_losses"/></TD><TD><xsl:value-of select="power_in_A"/></TD><TD><xsl:value-of select="power_in_B"/></TD><TD><xsl:value-of select="power_in_C"/></TD><TD><xsl:value-of select="power_out_A"/></TD><TD><xsl:value-of select="power_out_B"/></TD><TD><xsl:value-of select="power_out_C"/></TD><TD><xsl:value-of select="power_losses_A"/></TD><TD><xsl:value-of select="power_losses_B"/></TD><TD><xsl:value-of select="power_losses_C"/></TD><TD><xsl:value-of select="current_out_A"/></TD><TD><xsl:value-of select="current_out_B"/></TD><TD><xsl:value-of select="current_out_C"/></TD><TD><xsl:value-of select="current_in_A"/></TD><TD><xsl:value-of select="current_in_B"/></TD><TD><xsl:value-of select="current_in_C"/></TD><TD><xsl:value-of select="fault_current_in_A"/></TD><TD><xsl:value-of select="fault_current_in_B"/></TD><TD><xsl:value-of select="fault_current_in_C"/></TD><TD><xsl:value-of select="fault_current_out_A"/></TD><TD><xsl:value-of select="fault_current_out_B"/></TD><TD><xsl:value-of select="fault_current_out_C"/></TD><TD><xsl:value-of select="flow_direction"/></TD><TD><xsl:value-of select="mean_repair_time"/></TD><TD><xsl:value-of select="continuous_rating"/></TD><TD><xsl:value-of select="emergency_rating"/></TD><TD><xsl:value-of select="phases"/></TD><TD><xsl:value-of select="nominal_voltage"/></TD></TR>
</xsl:for-each></TABLE>
<H4>emissions objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>Nuclear_Order</TH><TH>Hydroelectric_Order</TH><TH>Solarthermal_Order</TH><TH>Biomass_Order</TH><TH>Wind_Order</TH><TH>Coal_Order</TH><TH>Naturalgas_Order</TH><TH>Geothermal_Order</TH><TH>Petroleum_Order</TH><TH>Naturalgas_Max_Out</TH><TH>Coal_Max_Out</TH><TH>Biomass_Max_Out</TH><TH>Geothermal_Max_Out</TH><TH>Hydroelectric_Max_Out</TH><TH>Nuclear_Max_Out</TH><TH>Wind_Max_Out</TH><TH>Petroleum_Max_Out</TH><TH>Solarthermal_Max_Out</TH><TH>Naturalgas_Out</TH><TH>Coal_Out</TH><TH>Biomass_Out</TH><TH>Geothermal_Out</TH><TH>Hydroelectric_Out</TH><TH>Nuclear_Out</TH><TH>Wind_Out</TH><TH>Petroleum_Out</TH><TH>Solarthermal_Out</TH><TH>Naturalgas_Conv_Eff</TH><TH>Coal_Conv_Eff</TH><TH>Biomass_Conv_Eff</TH><TH>Geothermal_Conv_Eff</TH><TH>Hydroelectric_Conv_Eff</TH><TH>Nuclear_Conv_Eff</TH><TH>Wind_Conv_Eff</TH><TH>Petroleum_Conv_Eff</TH><TH>Solarthermal_Conv_Eff</TH><TH>Naturalgas_CO2</TH><TH>Coal_CO2</TH><TH>Biomass_CO2</TH><TH>Geothermal_CO2</TH><TH>Hydroelectric_CO2</TH><TH>Nuclear_CO2</TH><TH>Wind_CO2</TH><TH>Petroleum_CO2</TH><TH>Solarthermal_CO2</TH><TH>Naturalgas_SO2</TH><TH>Coal_SO2</TH><TH>Biomass_SO2</TH><TH>Geothermal_SO2</TH><TH>Hydroelectric_SO2</TH><TH>Nuclear_SO2</TH><TH>Wind_SO2</TH><TH>Petroleum_SO2</TH><TH>Solarthermal_SO2</TH><TH>Naturalgas_NOx</TH><TH>Coal_NOx</TH><TH>Biomass_NOx</TH><TH>Geothermal_NOx</TH><TH>Hydroelectric_NOx</TH><TH>Nuclear_NOx</TH><TH>Wind_NOx</TH><TH>Petroleum_NOx</TH><TH>Solarthermal_NOx</TH><TH>Naturalgas_emissions_CO2</TH><TH>Naturalgas_emissions_SO2</TH><TH>Naturalgas_emissions_NOx</TH><TH>Coal_emissions_CO2</TH><TH>Coal_emissions_SO2</TH><TH>Coal_emissions_NOx</TH><TH>Biomass_emissions_CO2</TH><TH>Biomass_emissions_SO2</TH><TH>Biomass_emissions_NOx</TH><TH>Geothermal_emissions_CO2</TH><TH>Geothermal_emissions_SO2</TH><TH>Geothermal_emissions_NOx</TH><TH>Hydroelectric_emissions_CO2</TH><TH>Hydroelectric_emissions_SO2</TH><TH>Hydroelectric_emissions_NOx</TH><TH>Nuclear_emissions_CO2</TH><TH>Nuclear_emissions_SO2</TH><TH>Nuclear_emissions_NOx</TH><TH>Wind_emissions_CO2</TH><TH>Wind_emissions_SO2</TH><TH>Wind_emissions_NOx</TH><TH>Petroleum_emissions_CO2</TH><TH>Petroleum_emissions_SO2</TH><TH>Petroleum_emissions_NOx</TH><TH>Solarthermal_emissions_CO2</TH><TH>Solarthermal_emissions_SO2</TH><TH>Solarthermal_emissions_NOx</TH><TH>Total_emissions_CO2</TH><TH>Total_emissions_SO2</TH><TH>Total_emissions_NOx</TH><TH>Total_energy_out</TH><TH>Region</TH><TH>cycle_interval</TH></TR>
<xsl:for-each select="powerflow/emissions_list/emissions"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="Nuclear_Order"/></TD><TD><xsl:value-of select="Hydroelectric_Order"/></TD><TD><xsl:value-of select="Solarthermal_Order"/></TD><TD><xsl:value-of select="Biomass_Order"/></TD><TD><xsl:value-of select="Wind_Order"/></TD><TD><xsl:value-of select="Coal_Order"/></TD><TD><xsl:value-of select="Naturalgas_Order"/></TD><TD><xsl:value-of select="Geothermal_Order"/></TD><TD><xsl:value-of select="Petroleum_Order"/></TD><TD><xsl:value-of select="Naturalgas_Max_Out"/></TD><TD><xsl:value-of select="Coal_Max_Out"/></TD><TD><xsl:value-of select="Biomass_Max_Out"/></TD><TD><xsl:value-of select="Geothermal_Max_Out"/></TD><TD><xsl:value-of select="Hydroelectric_Max_Out"/></TD><TD><xsl:value-of select="Nuclear_Max_Out"/></TD><TD><xsl:value-of select="Wind_Max_Out"/></TD><TD><xsl:value-of select="Petroleum_Max_Out"/></TD><TD><xsl:value-of select="Solarthermal_Max_Out"/></TD><TD><xsl:value-of select="Naturalgas_Out"/></TD><TD><xsl:value-of select="Coal_Out"/></TD><TD><xsl:value-of select="Biomass_Out"/></TD><TD><xsl:value-of select="Geothermal_Out"/></TD><TD><xsl:value-of select="Hydroelectric_Out"/></TD><TD><xsl:value-of select="Nuclear_Out"/></TD><TD><xsl:value-of select="Wind_Out"/></TD><TD><xsl:value-of select="Petroleum_Out"/></TD><TD><xsl:value-of select="Solarthermal_Out"/></TD><TD><xsl:value-of select="Naturalgas_Conv_Eff"/></TD><TD><xsl:value-of select="Coal_Conv_Eff"/></TD><TD><xsl:value-of select="Biomass_Conv_Eff"/></TD><TD><xsl:value-of select="Geothermal_Conv_Eff"/></TD><TD><xsl:value-of select="Hydroelectric_Conv_Eff"/></TD><TD><xsl:value-of select="Nuclear_Conv_Eff"/></TD><TD><xsl:value-of select="Wind_Conv_Eff"/></TD><TD><xsl:value-of select="Petroleum_Conv_Eff"/></TD><TD><xsl:value-of select="Solarthermal_Conv_Eff"/></TD><TD><xsl:value-of select="Naturalgas_CO2"/></TD><TD><xsl:value-of select="Coal_CO2"/></TD><TD><xsl:value-of select="Biomass_CO2"/></TD><TD><xsl:value-of select="Geothermal_CO2"/></TD><TD><xsl:value-of select="Hydroelectric_CO2"/></TD><TD><xsl:value-of select="Nuclear_CO2"/></TD><TD><xsl:value-of select="Wind_CO2"/></TD><TD><xsl:value-of select="Petroleum_CO2"/></TD><TD><xsl:value-of select="Solarthermal_CO2"/></TD><TD><xsl:value-of select="Naturalgas_SO2"/></TD><TD><xsl:value-of select="Coal_SO2"/></TD><TD><xsl:value-of select="Biomass_SO2"/></TD><TD><xsl:value-of select="Geothermal_SO2"/></TD><TD><xsl:value-of select="Hydroelectric_SO2"/></TD><TD><xsl:value-of select="Nuclear_SO2"/></TD><TD><xsl:value-of select="Wind_SO2"/></TD><TD><xsl:value-of select="Petroleum_SO2"/></TD><TD><xsl:value-of select="Solarthermal_SO2"/></TD><TD><xsl:value-of select="Naturalgas_NOx"/></TD><TD><xsl:value-of select="Coal_NOx"/></TD><TD><xsl:value-of select="Biomass_NOx"/></TD><TD><xsl:value-of select="Geothermal_NOx"/></TD><TD><xsl:value-of select="Hydroelectric_NOx"/></TD><TD><xsl:value-of select="Nuclear_NOx"/></TD><TD><xsl:value-of select="Wind_NOx"/></TD><TD><xsl:value-of select="Petroleum_NOx"/></TD><TD><xsl:value-of select="Solarthermal_NOx"/></TD><TD><xsl:value-of select="Naturalgas_emissions_CO2"/></TD><TD><xsl:value-of select="Naturalgas_emissions_SO2"/></TD><TD><xsl:value-of select="Naturalgas_emissions_NOx"/></TD><TD><xsl:value-of select="Coal_emissions_CO2"/></TD><TD><xsl:value-of select="Coal_emissions_SO2"/></TD><TD><xsl:value-of select="Coal_emissions_NOx"/></TD><TD><xsl:value-of select="Biomass_emissions_CO2"/></TD><TD><xsl:value-of select="Biomass_emissions_SO2"/></TD><TD><xsl:value-of select="Biomass_emissions_NOx"/></TD><TD><xsl:value-of select="Geothermal_emissions_CO2"/></TD><TD><xsl:value-of select="Geothermal_emissions_SO2"/></TD><TD><xsl:value-of select="Geothermal_emissions_NOx"/></TD><TD><xsl:value-of select="Hydroelectric_emissions_CO2"/></TD><TD><xsl:value-of select="Hydroelectric_emissions_SO2"/></TD><TD><xsl:value-of select="Hydroelectric_emissions_NOx"/></TD><TD><xsl:value-of select="Nuclear_emissions_CO2"/></TD><TD><xsl:value-of select="Nuclear_emissions_SO2"/></TD><TD><xsl:value-of select="Nuclear_emissions_NOx"/></TD><TD><xsl:value-of select="Wind_emissions_CO2"/></TD><TD><xsl:value-of select="Wind_emissions_SO2"/></TD><TD><xsl:value-of select="Wind_emissions_NOx"/></TD><TD><xsl:value-of select="Petroleum_emissions_CO2"/></TD><TD><xsl:value-of select="Petroleum_emissions_SO2"/></TD><TD><xsl:value-of select="Petroleum_emissions_NOx"/></TD><TD><xsl:value-of select="Solarthermal_emissions_CO2"/></TD><TD><xsl:value-of select="Solarthermal_emissions_SO2"/></TD><TD><xsl:value-of select="Solarthermal_emissions_NOx"/></TD><TD><xsl:value-of select="Total_emissions_CO2"/></TD><TD><xsl:value-of select="Total_emissions_SO2"/></TD><TD><xsl:value-of select="Total_emissions_NOx"/></TD><TD><xsl:value-of select="Total_energy_out"/></TD><TD><xsl:value-of select="Region"/></TD><TD><xsl:value-of select="cycle_interval"/></TD></TR>
</xsl:for-each></TABLE>
<H3><A NAME="modules_residential">residential</A></H3><TABLE BORDER="1">
<TR><TH>version.major</TH><TD><xsl:value-of select="residential/version.major"/></TD></TR><TR><TH>version.minor</TH><TD><xsl:value-of select="residential/version.minor"/></TD></TR><TR><TH>default_line_voltage</TH><TD><xsl:value-of select="residential/default_line_voltage"/></TD></TR><TR><TH>default_line_current</TH><TD><xsl:value-of select="residential/default_line_current"/></TD></TR><TR><TH>default_outdoor_temperature</TH><TD><xsl:value-of select="residential/default_outdoor_temperature"/></TD></TR><TR><TH>default_humidity</TH><TD><xsl:value-of select="residential/default_humidity"/></TD></TR><TR><TH>default_solar</TH><TD><xsl:value-of select="residential/default_solar"/></TD></TR><TR><TH>default_etp_iterations</TH><TD><xsl:value-of select="residential/default_etp_iterations"/></TD></TR><TR><TH>ANSI_voltage_check</TH><TD><xsl:value-of select="residential/ANSI_voltage_check"/></TD></TR><TR><TH>implicit_enduses</TH><TD><xsl:value-of select="residential/implicit_enduses"/></TD></TR><TR><TH>house_low_temperature_warning[degF]</TH><TD><xsl:value-of select="residential/house_low_temperature_warning[degF]"/></TD></TR><TR><TH>house_high_temperature_warning[degF]</TH><TD><xsl:value-of select="residential/house_high_temperature_warning[degF]"/></TD></TR><TR><TH>thermostat_control_warning</TH><TD><xsl:value-of select="residential/thermostat_control_warning"/></TD></TR><TR><TH>system_dwell_time[s]</TH><TD><xsl:value-of select="residential/system_dwell_time[s]"/></TD></TR><TR><TH>aux_cutin_temperature[degF]</TH><TD><xsl:value-of select="residential/aux_cutin_temperature[degF]"/></TD></TR></TABLE>
<H4>residential_enduse objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/residential_enduse_list/residential_enduse"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>house_a objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>floor_area</TH><TH>gross_wall_area</TH><TH>ceiling_height</TH><TH>aspect_ratio</TH><TH>envelope_UA</TH><TH>window_wall_ratio</TH><TH>glazing_shgc</TH><TH>airchange_per_hour</TH><TH>solar_gain</TH><TH>heat_cool_gain</TH><TH>thermostat_deadband</TH><TH>heating_setpoint</TH><TH>cooling_setpoint</TH><TH>design_heating_capacity</TH><TH>design_cooling_capacity</TH><TH>heating_COP</TH><TH>cooling_COP</TH><TH>COP_coeff</TH><TH>air_temperature</TH><TH>outside_temp</TH><TH>mass_temperature</TH><TH>mass_heat_coeff</TH><TH>outdoor_temperature</TH><TH>house_thermal_mass</TH><TH>heat_mode</TH><TH>hc_mode</TH><TH>houseload</TH><TH>houseload.energy</TH><TH>houseload.power</TH><TH>houseload.peak_demand</TH><TH>houseload.heatgain</TH><TH>houseload.cumulative_heatgain</TH><TH>houseload.heatgain_fraction</TH><TH>houseload.current_fraction</TH><TH>houseload.impedance_fraction</TH><TH>houseload.power_fraction</TH><TH>houseload.power_factor</TH><TH>houseload.constant_power</TH><TH>houseload.constant_current</TH><TH>houseload.constant_admittance</TH><TH>houseload.voltage_factor</TH><TH>houseload.breaker_amps</TH><TH>houseload.configuration</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/house_a_list/house_a"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="floor_area"/></TD><TD><xsl:value-of select="gross_wall_area"/></TD><TD><xsl:value-of select="ceiling_height"/></TD><TD><xsl:value-of select="aspect_ratio"/></TD><TD><xsl:value-of select="envelope_UA"/></TD><TD><xsl:value-of select="window_wall_ratio"/></TD><TD><xsl:value-of select="glazing_shgc"/></TD><TD><xsl:value-of select="airchange_per_hour"/></TD><TD><xsl:value-of select="solar_gain"/></TD><TD><xsl:value-of select="heat_cool_gain"/></TD><TD><xsl:value-of select="thermostat_deadband"/></TD><TD><xsl:value-of select="heating_setpoint"/></TD><TD><xsl:value-of select="cooling_setpoint"/></TD><TD><xsl:value-of select="design_heating_capacity"/></TD><TD><xsl:value-of select="design_cooling_capacity"/></TD><TD><xsl:value-of select="heating_COP"/></TD><TD><xsl:value-of select="cooling_COP"/></TD><TD><xsl:value-of select="COP_coeff"/></TD><TD><xsl:value-of select="air_temperature"/></TD><TD><xsl:value-of select="outside_temp"/></TD><TD><xsl:value-of select="mass_temperature"/></TD><TD><xsl:value-of select="mass_heat_coeff"/></TD><TD><xsl:value-of select="outdoor_temperature"/></TD><TD><xsl:value-of select="house_thermal_mass"/></TD><TD><xsl:value-of select="heat_mode"/></TD><TD><xsl:value-of select="hc_mode"/></TD><TD><xsl:value-of select="houseload"/></TD><TD><xsl:value-of select="houseload.energy"/></TD><TD><xsl:value-of select="houseload.power"/></TD><TD><xsl:value-of select="houseload.peak_demand"/></TD><TD><xsl:value-of select="houseload.heatgain"/></TD><TD><xsl:value-of select="houseload.cumulative_heatgain"/></TD><TD><xsl:value-of select="houseload.heatgain_fraction"/></TD><TD><xsl:value-of select="houseload.current_fraction"/></TD><TD><xsl:value-of select="houseload.impedance_fraction"/></TD><TD><xsl:value-of select="houseload.power_fraction"/></TD><TD><xsl:value-of select="houseload.power_factor"/></TD><TD><xsl:value-of select="houseload.constant_power"/></TD><TD><xsl:value-of select="houseload.constant_current"/></TD><TD><xsl:value-of select="houseload.constant_admittance"/></TD><TD><xsl:value-of select="houseload.voltage_factor"/></TD><TD><xsl:value-of select="houseload.breaker_amps"/></TD><TD><xsl:value-of select="houseload.configuration"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>house objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>weather</TH><TH>floor_area</TH><TH>gross_wall_area</TH><TH>ceiling_height</TH><TH>aspect_ratio</TH><TH>envelope_UA</TH><TH>window_wall_ratio</TH><TH>number_of_doors</TH><TH>exterior_wall_fraction</TH><TH>interior_exterior_wall_ratio</TH><TH>exterior_ceiling_fraction</TH><TH>exterior_floor_fraction</TH><TH>window_shading</TH><TH>window_exterior_transmission_coefficient</TH><TH>solar_heatgain_factor</TH><TH>airchange_per_hour</TH><TH>airchange_UA</TH><TH>UA</TH><TH>internal_gain</TH><TH>solar_gain</TH><TH>incident_solar_radiation</TH><TH>heat_cool_gain</TH><TH>include_solar_quadrant</TH><TH>heating_cop_curve</TH><TH>heating_cap_curve</TH><TH>cooling_cop_curve</TH><TH>cooling_cap_curve</TH><TH>use_latent_heat</TH><TH>include_fan_heatgain</TH><TH>thermostat_deadband</TH><TH>dlc_offset</TH><TH>thermostat_cycle_time</TH><TH>thermostat_last_cycle_time</TH><TH>heating_setpoint</TH><TH>cooling_setpoint</TH><TH>design_heating_setpoint</TH><TH>design_cooling_setpoint</TH><TH>over_sizing_factor</TH><TH>design_heating_capacity</TH><TH>design_cooling_capacity</TH><TH>cooling_design_temperature</TH><TH>heating_design_temperature</TH><TH>design_peak_solar</TH><TH>design_internal_gains</TH><TH>air_heat_fraction</TH><TH>mass_solar_gain_fraction</TH><TH>mass_internal_gain_fraction</TH><TH>auxiliary_heat_capacity</TH><TH>aux_heat_deadband</TH><TH>aux_heat_temperature_lockout</TH><TH>aux_heat_time_delay</TH><TH>cooling_supply_air_temp</TH><TH>heating_supply_air_temp</TH><TH>duct_pressure_drop</TH><TH>fan_design_power</TH><TH>fan_low_power_fraction</TH><TH>fan_power</TH><TH>fan_design_airflow</TH><TH>fan_impedance_fraction</TH><TH>fan_power_fraction</TH><TH>fan_current_fraction</TH><TH>fan_power_factor</TH><TH>heating_demand</TH><TH>cooling_demand</TH><TH>heating_COP</TH><TH>cooling_COP</TH><TH>air_temperature</TH><TH>outdoor_temperature</TH><TH>outdoor_rh</TH><TH>mass_heat_capacity</TH><TH>mass_heat_coeff</TH><TH>mass_temperature</TH><TH>air_volume</TH><TH>air_mass</TH><TH>air_heat_capacity</TH><TH>latent_load_fraction</TH><TH>total_thermal_mass_per_floor_area</TH><TH>interior_surface_heat_transfer_coeff</TH><TH>number_of_stories</TH><TH>is_AUX_on</TH><TH>is_HEAT_on</TH><TH>is_COOL_on</TH><TH>thermal_storage_present</TH><TH>thermal_storage_in_use</TH><TH>system_type</TH><TH>auxiliary_strategy</TH><TH>system_mode</TH><TH>last_system_mode</TH><TH>heating_system_type</TH><TH>cooling_system_type</TH><TH>auxiliary_system_type</TH><TH>fan_type</TH><TH>thermal_integrity_level</TH><TH>glass_type</TH><TH>window_frame</TH><TH>glazing_treatment</TH><TH>glazing_layers</TH><TH>motor_model</TH><TH>motor_efficiency</TH><TH>last_mode_timer</TH><TH>hvac_motor_efficiency</TH><TH>hvac_motor_loss_power_factor</TH><TH>Rroof</TH><TH>Rwall</TH><TH>Rfloor</TH><TH>Rwindows</TH><TH>Rdoors</TH><TH>hvac_breaker_rating</TH><TH>hvac_power_factor</TH><TH>hvac_load</TH><TH>last_heating_load</TH><TH>last_cooling_load</TH><TH>hvac_power</TH><TH>total_load</TH><TH>panel</TH><TH>panel.energy</TH><TH>panel.power</TH><TH>panel.peak_demand</TH><TH>panel.heatgain</TH><TH>panel.cumulative_heatgain</TH><TH>panel.heatgain_fraction</TH><TH>panel.current_fraction</TH><TH>panel.impedance_fraction</TH><TH>panel.power_fraction</TH><TH>panel.power_factor</TH><TH>panel.constant_power</TH><TH>panel.constant_current</TH><TH>panel.constant_admittance</TH><TH>panel.voltage_factor</TH><TH>panel.breaker_amps</TH><TH>panel.configuration</TH><TH>design_internal_gain_density</TH><TH>compressor_on</TH><TH>compressor_count</TH><TH>hvac_last_on</TH><TH>hvac_last_off</TH><TH>hvac_period_length</TH><TH>hvac_duty_cycle</TH><TH>a</TH><TH>b</TH><TH>c</TH><TH>d</TH><TH>c1</TH><TH>c2</TH><TH>A3</TH><TH>A4</TH><TH>k1</TH><TH>k2</TH><TH>r1</TH><TH>r2</TH><TH>Teq</TH><TH>Tevent</TH><TH>Qi</TH><TH>Qa</TH><TH>Qm</TH><TH>Qh</TH><TH>Qlatent</TH><TH>dTair</TH><TH>adj_cooling_cap</TH><TH>adj_heating_cap</TH><TH>adj_cooling_cop</TH><TH>adj_heating_cop</TH><TH>thermostat_control</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/house_list/house"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><a href="#{weather}"><xsl:value-of select="weather"/></a></TD><TD><xsl:value-of select="floor_area"/></TD><TD><xsl:value-of select="gross_wall_area"/></TD><TD><xsl:value-of select="ceiling_height"/></TD><TD><xsl:value-of select="aspect_ratio"/></TD><TD><xsl:value-of select="envelope_UA"/></TD><TD><xsl:value-of select="window_wall_ratio"/></TD><TD><xsl:value-of select="number_of_doors"/></TD><TD><xsl:value-of select="exterior_wall_fraction"/></TD><TD><xsl:value-of select="interior_exterior_wall_ratio"/></TD><TD><xsl:value-of select="exterior_ceiling_fraction"/></TD><TD><xsl:value-of select="exterior_floor_fraction"/></TD><TD><xsl:value-of select="window_shading"/></TD><TD><xsl:value-of select="window_exterior_transmission_coefficient"/></TD><TD><xsl:value-of select="solar_heatgain_factor"/></TD><TD><xsl:value-of select="airchange_per_hour"/></TD><TD><xsl:value-of select="airchange_UA"/></TD><TD><xsl:value-of select="UA"/></TD><TD><xsl:value-of select="internal_gain"/></TD><TD><xsl:value-of select="solar_gain"/></TD><TD><xsl:value-of select="incident_solar_radiation"/></TD><TD><xsl:value-of select="heat_cool_gain"/></TD><TD><xsl:value-of select="include_solar_quadrant"/></TD><TD><xsl:value-of select="heating_cop_curve"/></TD><TD><xsl:value-of select="heating_cap_curve"/></TD><TD><xsl:value-of select="cooling_cop_curve"/></TD><TD><xsl:value-of select="cooling_cap_curve"/></TD><TD><xsl:value-of select="use_latent_heat"/></TD><TD><xsl:value-of select="include_fan_heatgain"/></TD><TD><xsl:value-of select="thermostat_deadband"/></TD><TD><xsl:value-of select="dlc_offset"/></TD><TD><xsl:value-of select="thermostat_cycle_time"/></TD><TD><xsl:value-of select="thermostat_last_cycle_time"/></TD><TD><xsl:value-of select="heating_setpoint"/></TD><TD><xsl:value-of select="cooling_setpoint"/></TD><TD><xsl:value-of select="design_heating_setpoint"/></TD><TD><xsl:value-of select="design_cooling_setpoint"/></TD><TD><xsl:value-of select="over_sizing_factor"/></TD><TD><xsl:value-of select="design_heating_capacity"/></TD><TD><xsl:value-of select="design_cooling_capacity"/></TD><TD><xsl:value-of select="cooling_design_temperature"/></TD><TD><xsl:value-of select="heating_design_temperature"/></TD><TD><xsl:value-of select="design_peak_solar"/></TD><TD><xsl:value-of select="design_internal_gains"/></TD><TD><xsl:value-of select="air_heat_fraction"/></TD><TD><xsl:value-of select="mass_solar_gain_fraction"/></TD><TD><xsl:value-of select="mass_internal_gain_fraction"/></TD><TD><xsl:value-of select="auxiliary_heat_capacity"/></TD><TD><xsl:value-of select="aux_heat_deadband"/></TD><TD><xsl:value-of select="aux_heat_temperature_lockout"/></TD><TD><xsl:value-of select="aux_heat_time_delay"/></TD><TD><xsl:value-of select="cooling_supply_air_temp"/></TD><TD><xsl:value-of select="heating_supply_air_temp"/></TD><TD><xsl:value-of select="duct_pressure_drop"/></TD><TD><xsl:value-of select="fan_design_power"/></TD><TD><xsl:value-of select="fan_low_power_fraction"/></TD><TD><xsl:value-of select="fan_power"/></TD><TD><xsl:value-of select="fan_design_airflow"/></TD><TD><xsl:value-of select="fan_impedance_fraction"/></TD><TD><xsl:value-of select="fan_power_fraction"/></TD><TD><xsl:value-of select="fan_current_fraction"/></TD><TD><xsl:value-of select="fan_power_factor"/></TD><TD><xsl:value-of select="heating_demand"/></TD><TD><xsl:value-of select="cooling_demand"/></TD><TD><xsl:value-of select="heating_COP"/></TD><TD><xsl:value-of select="cooling_COP"/></TD><TD><xsl:value-of select="air_temperature"/></TD><TD><xsl:value-of select="outdoor_temperature"/></TD><TD><xsl:value-of select="outdoor_rh"/></TD><TD><xsl:value-of select="mass_heat_capacity"/></TD><TD><xsl:value-of select="mass_heat_coeff"/></TD><TD><xsl:value-of select="mass_temperature"/></TD><TD><xsl:value-of select="air_volume"/></TD><TD><xsl:value-of select="air_mass"/></TD><TD><xsl:value-of select="air_heat_capacity"/></TD><TD><xsl:value-of select="latent_load_fraction"/></TD><TD><xsl:value-of select="total_thermal_mass_per_floor_area"/></TD><TD><xsl:value-of select="interior_surface_heat_transfer_coeff"/></TD><TD><xsl:value-of select="number_of_stories"/></TD><TD><xsl:value-of select="is_AUX_on"/></TD><TD><xsl:value-of select="is_HEAT_on"/></TD><TD><xsl:value-of select="is_COOL_on"/></TD><TD><xsl:value-of select="thermal_storage_present"/></TD><TD><xsl:value-of select="thermal_storage_in_use"/></TD><TD><xsl:value-of select="system_type"/></TD><TD><xsl:value-of select="auxiliary_strategy"/></TD><TD><xsl:value-of select="system_mode"/></TD><TD><xsl:value-of select="last_system_mode"/></TD><TD><xsl:value-of select="heating_system_type"/></TD><TD><xsl:value-of select="cooling_system_type"/></TD><TD><xsl:value-of select="auxiliary_system_type"/></TD><TD><xsl:value-of select="fan_type"/></TD><TD><xsl:value-of select="thermal_integrity_level"/></TD><TD><xsl:value-of select="glass_type"/></TD><TD><xsl:value-of select="window_frame"/></TD><TD><xsl:value-of select="glazing_treatment"/></TD><TD><xsl:value-of select="glazing_layers"/></TD><TD><xsl:value-of select="motor_model"/></TD><TD><xsl:value-of select="motor_efficiency"/></TD><TD><xsl:value-of select="last_mode_timer"/></TD><TD><xsl:value-of select="hvac_motor_efficiency"/></TD><TD><xsl:value-of select="hvac_motor_loss_power_factor"/></TD><TD><xsl:value-of select="Rroof"/></TD><TD><xsl:value-of select="Rwall"/></TD><TD><xsl:value-of select="Rfloor"/></TD><TD><xsl:value-of select="Rwindows"/></TD><TD><xsl:value-of select="Rdoors"/></TD><TD><xsl:value-of select="hvac_breaker_rating"/></TD><TD><xsl:value-of select="hvac_power_factor"/></TD><TD><xsl:value-of select="hvac_load"/></TD><TD><xsl:value-of select="last_heating_load"/></TD><TD><xsl:value-of select="last_cooling_load"/></TD><TD><xsl:value-of select="hvac_power"/></TD><TD><xsl:value-of select="total_load"/></TD><TD><xsl:value-of select="panel"/></TD><TD><xsl:value-of select="panel.energy"/></TD><TD><xsl:value-of select="panel.power"/></TD><TD><xsl:value-of select="panel.peak_demand"/></TD><TD><xsl:value-of select="panel.heatgain"/></TD><TD><xsl:value-of select="panel.cumulative_heatgain"/></TD><TD><xsl:value-of select="panel.heatgain_fraction"/></TD><TD><xsl:value-of select="panel.current_fraction"/></TD><TD><xsl:value-of select="panel.impedance_fraction"/></TD><TD><xsl:value-of select="panel.power_fraction"/></TD><TD><xsl:value-of select="panel.power_factor"/></TD><TD><xsl:value-of select="panel.constant_power"/></TD><TD><xsl:value-of select="panel.constant_current"/></TD><TD><xsl:value-of select="panel.constant_admittance"/></TD><TD><xsl:value-of select="panel.voltage_factor"/></TD><TD><xsl:value-of select="panel.breaker_amps"/></TD><TD><xsl:value-of select="panel.configuration"/></TD><TD><xsl:value-of select="design_internal_gain_density"/></TD><TD><xsl:value-of select="compressor_on"/></TD><TD><xsl:value-of select="compressor_count"/></TD><TD><xsl:value-of select="hvac_last_on"/></TD><TD><xsl:value-of select="hvac_last_off"/></TD><TD><xsl:value-of select="hvac_period_length"/></TD><TD><xsl:value-of select="hvac_duty_cycle"/></TD><TD><xsl:value-of select="a"/></TD><TD><xsl:value-of select="b"/></TD><TD><xsl:value-of select="c"/></TD><TD><xsl:value-of select="d"/></TD><TD><xsl:value-of select="c1"/></TD><TD><xsl:value-of select="c2"/></TD><TD><xsl:value-of select="A3"/></TD><TD><xsl:value-of select="A4"/></TD><TD><xsl:value-of select="k1"/></TD><TD><xsl:value-of select="k2"/></TD><TD><xsl:value-of select="r1"/></TD><TD><xsl:value-of select="r2"/></TD><TD><xsl:value-of select="Teq"/></TD><TD><xsl:value-of select="Tevent"/></TD><TD><xsl:value-of select="Qi"/></TD><TD><xsl:value-of select="Qa"/></TD><TD><xsl:value-of select="Qm"/></TD><TD><xsl:value-of select="Qh"/></TD><TD><xsl:value-of select="Qlatent"/></TD><TD><xsl:value-of select="dTair"/></TD><TD><xsl:value-of select="adj_cooling_cap"/></TD><TD><xsl:value-of select="adj_heating_cap"/></TD><TD><xsl:value-of select="adj_cooling_cop"/></TD><TD><xsl:value-of select="adj_heating_cop"/></TD><TD><xsl:value-of select="thermostat_control"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>waterheater objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>tank_volume</TH><TH>tank_UA</TH><TH>tank_diameter</TH><TH>water_demand</TH><TH>heating_element_capacity</TH><TH>inlet_water_temperature</TH><TH>heat_mode</TH><TH>location</TH><TH>tank_setpoint</TH><TH>thermostat_deadband</TH><TH>temperature</TH><TH>height</TH><TH>demand</TH><TH>actual_load</TH><TH>previous_load</TH><TH>actual_power</TH><TH>is_waterheater_on</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/waterheater_list/waterheater"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="tank_volume"/></TD><TD><xsl:value-of select="tank_UA"/></TD><TD><xsl:value-of select="tank_diameter"/></TD><TD><xsl:value-of select="water_demand"/></TD><TD><xsl:value-of select="heating_element_capacity"/></TD><TD><xsl:value-of select="inlet_water_temperature"/></TD><TD><xsl:value-of select="heat_mode"/></TD><TD><xsl:value-of select="location"/></TD><TD><xsl:value-of select="tank_setpoint"/></TD><TD><xsl:value-of select="thermostat_deadband"/></TD><TD><xsl:value-of select="temperature"/></TD><TD><xsl:value-of select="height"/></TD><TD><xsl:value-of select="demand"/></TD><TD><xsl:value-of select="actual_load"/></TD><TD><xsl:value-of select="previous_load"/></TD><TD><xsl:value-of select="actual_power"/></TD><TD><xsl:value-of select="is_waterheater_on"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>lights objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>type</TH><TH>placement</TH><TH>installed_power</TH><TH>power_density</TH><TH>curtailment</TH><TH>demand</TH><TH>actual_power</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/lights_list/lights"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="type"/></TD><TD><xsl:value-of select="placement"/></TD><TD><xsl:value-of select="installed_power"/></TD><TD><xsl:value-of select="power_density"/></TD><TD><xsl:value-of select="curtailment"/></TD><TD><xsl:value-of select="demand"/></TD><TD><xsl:value-of select="actual_power"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>refrigerator objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>size</TH><TH>rated_capacity</TH><TH>temperature</TH><TH>setpoint</TH><TH>deadband</TH><TH>cycle_time</TH><TH>output</TH><TH>event_temp</TH><TH>UA</TH><TH>compressor_off_normal_energy</TH><TH>compressor_off_normal_power</TH><TH>compressor_on_normal_energy</TH><TH>compressor_on_normal_power</TH><TH>defrost_energy</TH><TH>defrost_power</TH><TH>icemaking_energy</TH><TH>icemaking_power</TH><TH>ice_making_probability</TH><TH>FF_Door_Openings</TH><TH>door_opening_energy</TH><TH>door_opening_power</TH><TH>DO_Thershold</TH><TH>dr_mode_double</TH><TH>energy_needed</TH><TH>energy_used</TH><TH>refrigerator_power</TH><TH>icemaker_running</TH><TH>check_DO</TH><TH>is_240</TH><TH>defrostDelayed</TH><TH>long_compressor_cycle_due</TH><TH>long_compressor_cycle_time</TH><TH>long_compressor_cycle_power</TH><TH>long_compressor_cycle_energy</TH><TH>long_compressor_cycle_threshold</TH><TH>defrost_criterion</TH><TH>run_defrost</TH><TH>door_opening_criterion</TH><TH>compressor_defrost_time</TH><TH>delay_defrost_time</TH><TH>daily_door_opening</TH><TH>state</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/refrigerator_list/refrigerator"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="size"/></TD><TD><xsl:value-of select="rated_capacity"/></TD><TD><xsl:value-of select="temperature"/></TD><TD><xsl:value-of select="setpoint"/></TD><TD><xsl:value-of select="deadband"/></TD><TD><xsl:value-of select="cycle_time"/></TD><TD><xsl:value-of select="output"/></TD><TD><xsl:value-of select="event_temp"/></TD><TD><xsl:value-of select="UA"/></TD><TD><xsl:value-of select="compressor_off_normal_energy"/></TD><TD><xsl:value-of select="compressor_off_normal_power"/></TD><TD><xsl:value-of select="compressor_on_normal_energy"/></TD><TD><xsl:value-of select="compressor_on_normal_power"/></TD><TD><xsl:value-of select="defrost_energy"/></TD><TD><xsl:value-of select="defrost_power"/></TD><TD><xsl:value-of select="icemaking_energy"/></TD><TD><xsl:value-of select="icemaking_power"/></TD><TD><xsl:value-of select="ice_making_probability"/></TD><TD><xsl:value-of select="FF_Door_Openings"/></TD><TD><xsl:value-of select="door_opening_energy"/></TD><TD><xsl:value-of select="door_opening_power"/></TD><TD><xsl:value-of select="DO_Thershold"/></TD><TD><xsl:value-of select="dr_mode_double"/></TD><TD><xsl:value-of select="energy_needed"/></TD><TD><xsl:value-of select="energy_used"/></TD><TD><xsl:value-of select="refrigerator_power"/></TD><TD><xsl:value-of select="icemaker_running"/></TD><TD><xsl:value-of select="check_DO"/></TD><TD><xsl:value-of select="is_240"/></TD><TD><xsl:value-of select="defrostDelayed"/></TD><TD><xsl:value-of select="long_compressor_cycle_due"/></TD><TD><xsl:value-of select="long_compressor_cycle_time"/></TD><TD><xsl:value-of select="long_compressor_cycle_power"/></TD><TD><xsl:value-of select="long_compressor_cycle_energy"/></TD><TD><xsl:value-of select="long_compressor_cycle_threshold"/></TD><TD><xsl:value-of select="defrost_criterion"/></TD><TD><xsl:value-of select="run_defrost"/></TD><TD><xsl:value-of select="door_opening_criterion"/></TD><TD><xsl:value-of select="compressor_defrost_time"/></TD><TD><xsl:value-of select="delay_defrost_time"/></TD><TD><xsl:value-of select="daily_door_opening"/></TD><TD><xsl:value-of select="state"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>clotheswasher objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>motor_power</TH><TH>circuit_split</TH><TH>queue</TH><TH>demand</TH><TH>energy_meter</TH><TH>stall_voltage</TH><TH>start_voltage</TH><TH>stall_impedance</TH><TH>trip_delay</TH><TH>reset_delay</TH><TH>state</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/clotheswasher_list/clotheswasher"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="motor_power"/></TD><TD><xsl:value-of select="circuit_split"/></TD><TD><xsl:value-of select="queue"/></TD><TD><xsl:value-of select="demand"/></TD><TD><xsl:value-of select="energy_meter"/></TD><TD><xsl:value-of select="stall_voltage"/></TD><TD><xsl:value-of select="start_voltage"/></TD><TD><xsl:value-of select="stall_impedance"/></TD><TD><xsl:value-of select="trip_delay"/></TD><TD><xsl:value-of select="reset_delay"/></TD><TD><xsl:value-of select="state"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>dishwasher objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>installed_power</TH><TH>demand</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/dishwasher_list/dishwasher"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="installed_power"/></TD><TD><xsl:value-of select="demand"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>occupantload objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>number_of_occupants</TH><TH>occupancy_fraction</TH><TH>heatgain_per_person</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/occupantload_list/occupantload"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="number_of_occupants"/></TD><TD><xsl:value-of select="occupancy_fraction"/></TD><TD><xsl:value-of select="heatgain_per_person"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>plugload objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>circuit_split</TH><TH>demand</TH><TH>installed_power</TH><TH>actual_power</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/plugload_list/plugload"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="circuit_split"/></TD><TD><xsl:value-of select="demand"/></TD><TD><xsl:value-of select="installed_power"/></TD><TD><xsl:value-of select="actual_power"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>microwave objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>installed_power</TH><TH>standby_power</TH><TH>circuit_split</TH><TH>state</TH><TH>cycle_length</TH><TH>runtime</TH><TH>state_time</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/microwave_list/microwave"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="installed_power"/></TD><TD><xsl:value-of select="standby_power"/></TD><TD><xsl:value-of select="circuit_split"/></TD><TD><xsl:value-of select="state"/></TD><TD><xsl:value-of select="cycle_length"/></TD><TD><xsl:value-of select="runtime"/></TD><TD><xsl:value-of select="state_time"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>range objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>installed_power</TH><TH>circuit_split</TH><TH>demand</TH><TH>energy_meter</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/range_list/range"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="installed_power"/></TD><TD><xsl:value-of select="circuit_split"/></TD><TD><xsl:value-of select="demand"/></TD><TD><xsl:value-of select="energy_meter"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>freezer objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>size</TH><TH>rated_capacity</TH><TH>temperature</TH><TH>setpoint</TH><TH>deadband</TH><TH>next_time</TH><TH>output</TH><TH>event_temp</TH><TH>UA</TH><TH>state</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/freezer_list/freezer"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="size"/></TD><TD><xsl:value-of select="rated_capacity"/></TD><TD><xsl:value-of select="temperature"/></TD><TD><xsl:value-of select="setpoint"/></TD><TD><xsl:value-of select="deadband"/></TD><TD><xsl:value-of select="next_time"/></TD><TD><xsl:value-of select="output"/></TD><TD><xsl:value-of select="event_temp"/></TD><TD><xsl:value-of select="UA"/></TD><TD><xsl:value-of select="state"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>dryer objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>motor_power</TH><TH>coil_power</TH><TH>circuit_split</TH><TH>demand</TH><TH>queue</TH><TH>stall_voltage</TH><TH>start_voltage</TH><TH>stall_impedance</TH><TH>trip_delay</TH><TH>reset_delay</TH><TH>actual_power</TH><TH>state</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/dryer_list/dryer"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="motor_power"/></TD><TD><xsl:value-of select="coil_power"/></TD><TD><xsl:value-of select="circuit_split"/></TD><TD><xsl:value-of select="demand"/></TD><TD><xsl:value-of select="queue"/></TD><TD><xsl:value-of select="stall_voltage"/></TD><TD><xsl:value-of select="start_voltage"/></TD><TD><xsl:value-of select="stall_impedance"/></TD><TD><xsl:value-of select="trip_delay"/></TD><TD><xsl:value-of select="reset_delay"/></TD><TD><xsl:value-of select="actual_power"/></TD><TD><xsl:value-of select="state"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>evcharger objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>charger_type</TH><TH>vehicle_type</TH><TH>state</TH><TH>p_go_home</TH><TH>p_go_work</TH><TH>work_dist</TH><TH>capacity</TH><TH>charge</TH><TH>charge_at_work</TH><TH>charge_throttle</TH><TH>charger_efficiency</TH><TH>power_train_efficiency</TH><TH>mileage_classification</TH><TH>demand_profile</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/evcharger_list/evcharger"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="charger_type"/></TD><TD><xsl:value-of select="vehicle_type"/></TD><TD><xsl:value-of select="state"/></TD><TD><xsl:value-of select="p_go_home"/></TD><TD><xsl:value-of select="p_go_work"/></TD><TD><xsl:value-of select="work_dist"/></TD><TD><xsl:value-of select="capacity"/></TD><TD><xsl:value-of select="charge"/></TD><TD><xsl:value-of select="charge_at_work"/></TD><TD><xsl:value-of select="charge_throttle"/></TD><TD><xsl:value-of select="charger_efficiency"/></TD><TD><xsl:value-of select="power_train_efficiency"/></TD><TD><xsl:value-of select="mileage_classification"/></TD><TD><xsl:value-of select="demand_profile"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>ZIPload objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>heat_fraction</TH><TH>base_power</TH><TH>power_pf</TH><TH>current_pf</TH><TH>impedance_pf</TH><TH>is_240</TH><TH>breaker_val</TH><TH>actual_power</TH><TH>multiplier</TH><TH>heatgain_only</TH><TH>demand_response_mode</TH><TH>number_of_devices</TH><TH>thermostatic_control_range</TH><TH>number_of_devices_off</TH><TH>number_of_devices_on</TH><TH>rate_of_cooling</TH><TH>rate_of_heating</TH><TH>temperature</TH><TH>phi</TH><TH>demand_rate</TH><TH>nominal_power</TH><TH>duty_cycle</TH><TH>recovery_duty_cycle</TH><TH>period</TH><TH>phase</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/ZIPload_list/ZIPload"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="heat_fraction"/></TD><TD><xsl:value-of select="base_power"/></TD><TD><xsl:value-of select="power_pf"/></TD><TD><xsl:value-of select="current_pf"/></TD><TD><xsl:value-of select="impedance_pf"/></TD><TD><xsl:value-of select="is_240"/></TD><TD><xsl:value-of select="breaker_val"/></TD><TD><xsl:value-of select="actual_power"/></TD><TD><xsl:value-of select="multiplier"/></TD><TD><xsl:value-of select="heatgain_only"/></TD><TD><xsl:value-of select="demand_response_mode"/></TD><TD><xsl:value-of select="number_of_devices"/></TD><TD><xsl:value-of select="thermostatic_control_range"/></TD><TD><xsl:value-of select="number_of_devices_off"/></TD><TD><xsl:value-of select="number_of_devices_on"/></TD><TD><xsl:value-of select="rate_of_cooling"/></TD><TD><xsl:value-of select="rate_of_heating"/></TD><TD><xsl:value-of select="temperature"/></TD><TD><xsl:value-of select="phi"/></TD><TD><xsl:value-of select="demand_rate"/></TD><TD><xsl:value-of select="nominal_power"/></TD><TD><xsl:value-of select="duty_cycle"/></TD><TD><xsl:value-of select="recovery_duty_cycle"/></TD><TD><xsl:value-of select="period"/></TD><TD><xsl:value-of select="phase"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>thermal_storage objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>total_capacity</TH><TH>stored_capacity</TH><TH>recharge_power</TH><TH>discharge_power</TH><TH>recharge_pf</TH><TH>discharge_pf</TH><TH>discharge_schedule_type</TH><TH>recharge_schedule_type</TH><TH>recharge_time</TH><TH>discharge_time</TH><TH>discharge_rate</TH><TH>SOC</TH><TH>k</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/thermal_storage_list/thermal_storage"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="total_capacity"/></TD><TD><xsl:value-of select="stored_capacity"/></TD><TD><xsl:value-of select="recharge_power"/></TD><TD><xsl:value-of select="discharge_power"/></TD><TD><xsl:value-of select="recharge_pf"/></TD><TD><xsl:value-of select="discharge_pf"/></TD><TD><xsl:value-of select="discharge_schedule_type"/></TD><TD><xsl:value-of select="recharge_schedule_type"/></TD><TD><xsl:value-of select="recharge_time"/></TD><TD><xsl:value-of select="discharge_time"/></TD><TD><xsl:value-of select="discharge_rate"/></TD><TD><xsl:value-of select="SOC"/></TD><TD><xsl:value-of select="k"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H4>evcharger_det objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>charge_rate</TH><TH>variation_mean</TH><TH>variation_std_dev</TH><TH>variation_trip_mean</TH><TH>variation_trip_std_dev</TH><TH>mileage_classification</TH><TH>work_charging_available</TH><TH>data_file</TH><TH>vehicle_index</TH><TH>vehicle_location</TH><TH>travel_distance</TH><TH>arrival_at_work</TH><TH>duration_at_work</TH><TH>arrival_at_home</TH><TH>duration_at_home</TH><TH>battery_capacity</TH><TH>battery_SOC</TH><TH>battery_size</TH><TH>mileage_efficiency</TH><TH>maximum_charge_rate</TH><TH>charging_efficiency</TH><TH>shape</TH><TH>load</TH><TH>energy</TH><TH>power</TH><TH>peak_demand</TH><TH>heatgain</TH><TH>cumulative_heatgain</TH><TH>heatgain_fraction</TH><TH>current_fraction</TH><TH>impedance_fraction</TH><TH>power_fraction</TH><TH>power_factor</TH><TH>constant_power</TH><TH>constant_current</TH><TH>constant_admittance</TH><TH>voltage_factor</TH><TH>breaker_amps</TH><TH>configuration</TH><TH>override</TH><TH>power_state</TH></TR>
<xsl:for-each select="residential/evcharger_det_list/evcharger_det"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="charge_rate"/></TD><TD><xsl:value-of select="variation_mean"/></TD><TD><xsl:value-of select="variation_std_dev"/></TD><TD><xsl:value-of select="variation_trip_mean"/></TD><TD><xsl:value-of select="variation_trip_std_dev"/></TD><TD><xsl:value-of select="mileage_classification"/></TD><TD><xsl:value-of select="work_charging_available"/></TD><TD><xsl:value-of select="data_file"/></TD><TD><xsl:value-of select="vehicle_index"/></TD><TD><xsl:value-of select="vehicle_location"/></TD><TD><xsl:value-of select="travel_distance"/></TD><TD><xsl:value-of select="arrival_at_work"/></TD><TD><xsl:value-of select="duration_at_work"/></TD><TD><xsl:value-of select="arrival_at_home"/></TD><TD><xsl:value-of select="duration_at_home"/></TD><TD><xsl:value-of select="battery_capacity"/></TD><TD><xsl:value-of select="battery_SOC"/></TD><TD><xsl:value-of select="battery_size"/></TD><TD><xsl:value-of select="mileage_efficiency"/></TD><TD><xsl:value-of select="maximum_charge_rate"/></TD><TD><xsl:value-of select="charging_efficiency"/></TD><TD><xsl:value-of select="shape"/></TD><TD><xsl:value-of select="load"/></TD><TD><xsl:value-of select="energy"/></TD><TD><xsl:value-of select="power"/></TD><TD><xsl:value-of select="peak_demand"/></TD><TD><xsl:value-of select="heatgain"/></TD><TD><xsl:value-of select="cumulative_heatgain"/></TD><TD><xsl:value-of select="heatgain_fraction"/></TD><TD><xsl:value-of select="current_fraction"/></TD><TD><xsl:value-of select="impedance_fraction"/></TD><TD><xsl:value-of select="power_fraction"/></TD><TD><xsl:value-of select="power_factor"/></TD><TD><xsl:value-of select="constant_power"/></TD><TD><xsl:value-of select="constant_current"/></TD><TD><xsl:value-of select="constant_admittance"/></TD><TD><xsl:value-of select="voltage_factor"/></TD><TD><xsl:value-of select="breaker_amps"/></TD><TD><xsl:value-of select="configuration"/></TD><TD><xsl:value-of select="override"/></TD><TD><xsl:value-of select="power_state"/></TD></TR>
</xsl:for-each></TABLE>
<H3><A NAME="modules_tape">tape</A></H3><TABLE BORDER="1">
<TR><TH>version.major</TH><TD><xsl:value-of select="tape/version.major"/></TD></TR><TR><TH>version.minor</TH><TD><xsl:value-of select="tape/version.minor"/></TD></TR><TR><TH>gnuplot_path</TH><TD><xsl:value-of select="tape/gnuplot_path"/></TD></TR><TR><TH>flush_interval</TH><TD><xsl:value-of select="tape/flush_interval"/></TD></TR><TR><TH>delta_mode_needed</TH><TD><xsl:value-of select="tape/delta_mode_needed"/></TD></TR></TABLE>
<H4>player objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>property</TH><TH>file</TH><TH>filetype</TH><TH>loop</TH></TR>
<xsl:for-each select="tape/player_list/player"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="property"/></TD><TD><xsl:value-of select="file"/></TD><TD><xsl:value-of select="filetype"/></TD><TD><xsl:value-of select="loop"/></TD></TR>
</xsl:for-each></TABLE>
<H4>shaper objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>file</TH><TH>filetype</TH><TH>group</TH><TH>property</TH><TH>magnitude</TH><TH>events</TH></TR>
<xsl:for-each select="tape/shaper_list/shaper"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="file"/></TD><TD><xsl:value-of select="filetype"/></TD><TD><xsl:value-of select="group"/></TD><TD><xsl:value-of select="property"/></TD><TD><xsl:value-of select="magnitude"/></TD><TD><xsl:value-of select="events"/></TD></TR>
</xsl:for-each></TABLE>
<H4>recorder objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>property</TH><TH>trigger</TH><TH>file</TH><TH>multifile</TH><TH>limit</TH><TH>plotcommands</TH><TH>xdata</TH><TH>columns</TH><TH>interval</TH><TH>output</TH><TH>header_units</TH><TH>line_units</TH></TR>
<xsl:for-each select="tape/recorder_list/recorder"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="property"/></TD><TD><xsl:value-of select="trigger"/></TD><TD><xsl:value-of select="file"/></TD><TD><xsl:value-of select="multifile"/></TD><TD><xsl:value-of select="limit"/></TD><TD><xsl:value-of select="plotcommands"/></TD><TD><xsl:value-of select="xdata"/></TD><TD><xsl:value-of select="columns"/></TD><TD><xsl:value-of select="interval"/></TD><TD><xsl:value-of select="output"/></TD><TD><xsl:value-of select="header_units"/></TD><TD><xsl:value-of select="line_units"/></TD></TR>
</xsl:for-each></TABLE>
<H4>multi_recorder objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>interval</TH><TH>property</TH><TH>trigger</TH><TH>file</TH><TH>multifile</TH><TH>limit</TH><TH>plotcommands</TH><TH>xdata</TH><TH>columns</TH><TH>output</TH><TH>header_units</TH><TH>line_units</TH></TR>
<xsl:for-each select="tape/multi_recorder_list/multi_recorder"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="interval"/></TD><TD><xsl:value-of select="property"/></TD><TD><xsl:value-of select="trigger"/></TD><TD><xsl:value-of select="file"/></TD><TD><xsl:value-of select="multifile"/></TD><TD><xsl:value-of select="limit"/></TD><TD><xsl:value-of select="plotcommands"/></TD><TD><xsl:value-of select="xdata"/></TD><TD><xsl:value-of select="columns"/></TD><TD><xsl:value-of select="output"/></TD><TD><xsl:value-of select="header_units"/></TD><TD><xsl:value-of select="line_units"/></TD></TR>
</xsl:for-each></TABLE>
<H4>collector objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>property</TH><TH>trigger</TH><TH>file</TH><TH>limit</TH><TH>group</TH><TH>interval</TH></TR>
<xsl:for-each select="tape/collector_list/collector"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="property"/></TD><TD><xsl:value-of select="trigger"/></TD><TD><xsl:value-of select="file"/></TD><TD><xsl:value-of select="limit"/></TD><TD><xsl:value-of select="group"/></TD><TD><xsl:value-of select="interval"/></TD></TR>
</xsl:for-each></TABLE>
<H4>histogram objects</H4><TABLE BORDER="1">
<TR><TH>Name</TH><TH>filename</TH><TH>group</TH><TH>bins</TH><TH>property</TH><TH>min</TH><TH>max</TH><TH>samplerate</TH><TH>countrate</TH><TH>bin_count</TH><TH>limit</TH></TR>
<xsl:for-each select="tape/histogram_list/histogram"><TR><TD><a name="#{name}"/><xsl:value-of select="name"/> (#<xsl:value-of select="id"/>)</TD><TD><xsl:value-of select="filename"/></TD><TD><xsl:value-of select="group"/></TD><TD><xsl:value-of select="bins"/></TD><TD><xsl:value-of select="property"/></TD><TD><xsl:value-of select="min"/></TD><TD><xsl:value-of select="max"/></TD><TD><xsl:value-of select="samplerate"/></TD><TD><xsl:value-of select="countrate"/></TD><TD><xsl:value-of select="bin_count"/></TD><TD><xsl:value-of select="limit"/></TD></TR>
</xsl:for-each></TABLE>
<H2><A NAME="output">GLM Output</A></H2>
<table border="1" width="100%"><tr><td><pre>
# Generated by GridLAB-D <xsl:value-of select="version.major"/>.<xsl:value-of select="version.minor"/>
# Command line..... <xsl:value-of select="command_line"/>
# Model name....... <xsl:value-of select="modelname"/>
# Start at......... <xsl:value-of select="starttime"/>
#
clock {
	timestamp '<xsl:value-of select="clock"/>';
	timezone <xsl:value-of select="timezone"/>;
}
<xsl:for-each select="climate">
##############################################
# climate module
module climate {
	version.major <xsl:value-of select="version.major"/>;
	version.minor <xsl:value-of select="version.minor"/>;
}

# climate::climate objects
<xsl:for-each select="climate_list/climate"><a name="#GLM.{name}"/>object climate:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="solar_elevation">	solar_elevation <xsl:value-of select="solar_elevation"/>;
</xsl:if><xsl:if test="solar_azimuth">	solar_azimuth <xsl:value-of select="solar_azimuth"/>;
</xsl:if><xsl:if test="city">	city <xsl:value-of select="city"/>;
</xsl:if><xsl:if test="tmyfile">	tmyfile <xsl:value-of select="tmyfile"/>;
</xsl:if><xsl:if test="temperature">	temperature <xsl:value-of select="temperature"/>;
</xsl:if><xsl:if test="humidity">	humidity <xsl:value-of select="humidity"/>;
</xsl:if><xsl:if test="solar_flux">	solar_flux <xsl:value-of select="solar_flux"/>;
</xsl:if><xsl:if test="solar_direct">	solar_direct <xsl:value-of select="solar_direct"/>;
</xsl:if><xsl:if test="solar_diffuse">	solar_diffuse <xsl:value-of select="solar_diffuse"/>;
</xsl:if><xsl:if test="solar_global">	solar_global <xsl:value-of select="solar_global"/>;
</xsl:if><xsl:if test="extraterrestrial_direct_normal">	extraterrestrial_direct_normal <xsl:value-of select="extraterrestrial_direct_normal"/>;
</xsl:if><xsl:if test="pressure">	pressure <xsl:value-of select="pressure"/>;
</xsl:if><xsl:if test="wind_speed">	wind_speed <xsl:value-of select="wind_speed"/>;
</xsl:if><xsl:if test="wind_dir">	wind_dir <xsl:value-of select="wind_dir"/>;
</xsl:if><xsl:if test="wind_gust">	wind_gust <xsl:value-of select="wind_gust"/>;
</xsl:if><xsl:if test="record.low">	record.low <xsl:value-of select="record.low"/>;
</xsl:if><xsl:if test="record.low_day">	record.low_day <xsl:value-of select="record.low_day"/>;
</xsl:if><xsl:if test="record.high">	record.high <xsl:value-of select="record.high"/>;
</xsl:if><xsl:if test="record.high_day">	record.high_day <xsl:value-of select="record.high_day"/>;
</xsl:if><xsl:if test="record.solar">	record.solar <xsl:value-of select="record.solar"/>;
</xsl:if><xsl:if test="rainfall">	rainfall <xsl:value-of select="rainfall"/>;
</xsl:if><xsl:if test="snowdepth">	snowdepth <xsl:value-of select="snowdepth"/>;
</xsl:if><xsl:if test="interpolate">	interpolate <xsl:value-of select="interpolate"/>;
</xsl:if><xsl:if test="solar_horiz">	solar_horiz <xsl:value-of select="solar_horiz"/>;
</xsl:if><xsl:if test="solar_north">	solar_north <xsl:value-of select="solar_north"/>;
</xsl:if><xsl:if test="solar_northeast">	solar_northeast <xsl:value-of select="solar_northeast"/>;
</xsl:if><xsl:if test="solar_east">	solar_east <xsl:value-of select="solar_east"/>;
</xsl:if><xsl:if test="solar_southeast">	solar_southeast <xsl:value-of select="solar_southeast"/>;
</xsl:if><xsl:if test="solar_south">	solar_south <xsl:value-of select="solar_south"/>;
</xsl:if><xsl:if test="solar_southwest">	solar_southwest <xsl:value-of select="solar_southwest"/>;
</xsl:if><xsl:if test="solar_west">	solar_west <xsl:value-of select="solar_west"/>;
</xsl:if><xsl:if test="solar_northwest">	solar_northwest <xsl:value-of select="solar_northwest"/>;
</xsl:if><xsl:if test="solar_raw">	solar_raw <xsl:value-of select="solar_raw"/>;
</xsl:if><xsl:if test="ground_reflectivity">	ground_reflectivity <xsl:value-of select="ground_reflectivity"/>;
</xsl:if><xsl:if test="reader">	reader <a href="#GLM.{reader}"><xsl:value-of select="reader"/></a>;
</xsl:if><xsl:if test="forecast">	forecast <xsl:value-of select="forecast"/>;
</xsl:if>}
</xsl:for-each>
# climate::weather objects
<xsl:for-each select="weather_list/weather"><a name="#GLM.{name}"/>object weather:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="temperature">	temperature <xsl:value-of select="temperature"/>;
</xsl:if><xsl:if test="humidity">	humidity <xsl:value-of select="humidity"/>;
</xsl:if><xsl:if test="solar_dir">	solar_dir <xsl:value-of select="solar_dir"/>;
</xsl:if><xsl:if test="solar_diff">	solar_diff <xsl:value-of select="solar_diff"/>;
</xsl:if><xsl:if test="solar_global">	solar_global <xsl:value-of select="solar_global"/>;
</xsl:if><xsl:if test="wind_speed">	wind_speed <xsl:value-of select="wind_speed"/>;
</xsl:if><xsl:if test="rainfall">	rainfall <xsl:value-of select="rainfall"/>;
</xsl:if><xsl:if test="snowdepth">	snowdepth <xsl:value-of select="snowdepth"/>;
</xsl:if><xsl:if test="month">	month <xsl:value-of select="month"/>;
</xsl:if><xsl:if test="day">	day <xsl:value-of select="day"/>;
</xsl:if><xsl:if test="hour">	hour <xsl:value-of select="hour"/>;
</xsl:if><xsl:if test="minute">	minute <xsl:value-of select="minute"/>;
</xsl:if><xsl:if test="second">	second <xsl:value-of select="second"/>;
</xsl:if>}
</xsl:for-each>
# climate::csv_reader objects
<xsl:for-each select="csv_reader_list/csv_reader"><a name="#GLM.{name}"/>object csv_reader:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="index">	index <xsl:value-of select="index"/>;
</xsl:if><xsl:if test="city_name">	city_name <xsl:value-of select="city_name"/>;
</xsl:if><xsl:if test="state_name">	state_name <xsl:value-of select="state_name"/>;
</xsl:if><xsl:if test="lat_deg">	lat_deg <xsl:value-of select="lat_deg"/>;
</xsl:if><xsl:if test="lat_min">	lat_min <xsl:value-of select="lat_min"/>;
</xsl:if><xsl:if test="long_deg">	long_deg <xsl:value-of select="long_deg"/>;
</xsl:if><xsl:if test="long_min">	long_min <xsl:value-of select="long_min"/>;
</xsl:if><xsl:if test="low_temp">	low_temp <xsl:value-of select="low_temp"/>;
</xsl:if><xsl:if test="high_temp">	high_temp <xsl:value-of select="high_temp"/>;
</xsl:if><xsl:if test="peak_solar">	peak_solar <xsl:value-of select="peak_solar"/>;
</xsl:if><xsl:if test="status">	status <xsl:value-of select="status"/>;
</xsl:if><xsl:if test="timefmt">	timefmt <xsl:value-of select="timefmt"/>;
</xsl:if><xsl:if test="timezone">	timezone <xsl:value-of select="timezone"/>;
</xsl:if><xsl:if test="timezone_offset">	timezone_offset <xsl:value-of select="timezone_offset"/>;
</xsl:if><xsl:if test="columns">	columns <xsl:value-of select="columns"/>;
</xsl:if><xsl:if test="filename">	filename <xsl:value-of select="filename"/>;
</xsl:if>}
</xsl:for-each></xsl:for-each><xsl:for-each select="commercial">
##############################################
# commercial module
module commercial {
	version.major <xsl:value-of select="version.major"/>;
	version.minor <xsl:value-of select="version.minor"/>;
	warn_control <xsl:value-of select="warn_control"/>;
	warn_low_temp <xsl:value-of select="warn_low_temp"/>;
	warn_high_temp <xsl:value-of select="warn_high_temp"/>;
}

# commercial::office objects
<xsl:for-each select="office_list/office"><a name="#GLM.{name}"/>object office:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="floor_area">	floor_area <xsl:value-of select="floor_area"/>;
</xsl:if><xsl:if test="floor_height">	floor_height <xsl:value-of select="floor_height"/>;
</xsl:if><xsl:if test="exterior_ua">	exterior_ua <xsl:value-of select="exterior_ua"/>;
</xsl:if><xsl:if test="interior_ua">	interior_ua <xsl:value-of select="interior_ua"/>;
</xsl:if><xsl:if test="interior_mass">	interior_mass <xsl:value-of select="interior_mass"/>;
</xsl:if><xsl:if test="glazing">	glazing <xsl:value-of select="glazing"/>;
</xsl:if><xsl:if test="glazing.north">	glazing.north <xsl:value-of select="glazing.north"/>;
</xsl:if><xsl:if test="glazing.northeast">	glazing.northeast <xsl:value-of select="glazing.northeast"/>;
</xsl:if><xsl:if test="glazing.east">	glazing.east <xsl:value-of select="glazing.east"/>;
</xsl:if><xsl:if test="glazing.southeast">	glazing.southeast <xsl:value-of select="glazing.southeast"/>;
</xsl:if><xsl:if test="glazing.south">	glazing.south <xsl:value-of select="glazing.south"/>;
</xsl:if><xsl:if test="glazing.southwest">	glazing.southwest <xsl:value-of select="glazing.southwest"/>;
</xsl:if><xsl:if test="glazing.west">	glazing.west <xsl:value-of select="glazing.west"/>;
</xsl:if><xsl:if test="glazing.northwest">	glazing.northwest <xsl:value-of select="glazing.northwest"/>;
</xsl:if><xsl:if test="glazing.horizontal">	glazing.horizontal <xsl:value-of select="glazing.horizontal"/>;
</xsl:if><xsl:if test="glazing.coefficient">	glazing.coefficient <xsl:value-of select="glazing.coefficient"/>;
</xsl:if><xsl:if test="occupancy">	occupancy <xsl:value-of select="occupancy"/>;
</xsl:if><xsl:if test="occupants">	occupants <xsl:value-of select="occupants"/>;
</xsl:if><xsl:if test="schedule">	schedule <xsl:value-of select="schedule"/>;
</xsl:if><xsl:if test="air_temperature">	air_temperature <xsl:value-of select="air_temperature"/>;
</xsl:if><xsl:if test="mass_temperature">	mass_temperature <xsl:value-of select="mass_temperature"/>;
</xsl:if><xsl:if test="temperature_change">	temperature_change <xsl:value-of select="temperature_change"/>;
</xsl:if><xsl:if test="outdoor_temperature">	outdoor_temperature <xsl:value-of select="outdoor_temperature"/>;
</xsl:if><xsl:if test="Qh">	Qh <xsl:value-of select="Qh"/>;
</xsl:if><xsl:if test="Qs">	Qs <xsl:value-of select="Qs"/>;
</xsl:if><xsl:if test="Qi">	Qi <xsl:value-of select="Qi"/>;
</xsl:if><xsl:if test="Qz">	Qz <xsl:value-of select="Qz"/>;
</xsl:if><xsl:if test="hvac_mode">	hvac_mode <xsl:value-of select="hvac_mode"/>;
</xsl:if><xsl:if test="hvac.cooling.balance_temperature">	hvac.cooling.balance_temperature <xsl:value-of select="hvac.cooling.balance_temperature"/>;
</xsl:if><xsl:if test="hvac.cooling.capacity">	hvac.cooling.capacity <xsl:value-of select="hvac.cooling.capacity"/>;
</xsl:if><xsl:if test="hvac.cooling.capacity_perF">	hvac.cooling.capacity_perF <xsl:value-of select="hvac.cooling.capacity_perF"/>;
</xsl:if><xsl:if test="hvac.cooling.design_temperature">	hvac.cooling.design_temperature <xsl:value-of select="hvac.cooling.design_temperature"/>;
</xsl:if><xsl:if test="hvac.cooling.efficiency">	hvac.cooling.efficiency <xsl:value-of select="hvac.cooling.efficiency"/>;
</xsl:if><xsl:if test="hvac.cooling.cop">	hvac.cooling.cop <xsl:value-of select="hvac.cooling.cop"/>;
</xsl:if><xsl:if test="hvac.heating.balance_temperature">	hvac.heating.balance_temperature <xsl:value-of select="hvac.heating.balance_temperature"/>;
</xsl:if><xsl:if test="hvac.heating.capacity">	hvac.heating.capacity <xsl:value-of select="hvac.heating.capacity"/>;
</xsl:if><xsl:if test="hvac.heating.capacity_perF">	hvac.heating.capacity_perF <xsl:value-of select="hvac.heating.capacity_perF"/>;
</xsl:if><xsl:if test="hvac.heating.design_temperature">	hvac.heating.design_temperature <xsl:value-of select="hvac.heating.design_temperature"/>;
</xsl:if><xsl:if test="hvac.heating.efficiency">	hvac.heating.efficiency <xsl:value-of select="hvac.heating.efficiency"/>;
</xsl:if><xsl:if test="hvac.heating.cop">	hvac.heating.cop <xsl:value-of select="hvac.heating.cop"/>;
</xsl:if><xsl:if test="lights.capacity">	lights.capacity <xsl:value-of select="lights.capacity"/>;
</xsl:if><xsl:if test="lights.fraction">	lights.fraction <xsl:value-of select="lights.fraction"/>;
</xsl:if><xsl:if test="plugs.capacity">	plugs.capacity <xsl:value-of select="plugs.capacity"/>;
</xsl:if><xsl:if test="plugs.fraction">	plugs.fraction <xsl:value-of select="plugs.fraction"/>;
</xsl:if><xsl:if test="demand">	demand <xsl:value-of select="demand"/>;
</xsl:if><xsl:if test="total_load">	total_load <xsl:value-of select="total_load"/>;
</xsl:if><xsl:if test="energy">	energy <xsl:value-of select="energy"/>;
</xsl:if><xsl:if test="power_factor">	power_factor <xsl:value-of select="power_factor"/>;
</xsl:if><xsl:if test="power">	power <xsl:value-of select="power"/>;
</xsl:if><xsl:if test="current">	current <xsl:value-of select="current"/>;
</xsl:if><xsl:if test="admittance">	admittance <xsl:value-of select="admittance"/>;
</xsl:if><xsl:if test="hvac.demand">	hvac.demand <xsl:value-of select="hvac.demand"/>;
</xsl:if><xsl:if test="hvac.load">	hvac.load <xsl:value-of select="hvac.load"/>;
</xsl:if><xsl:if test="hvac.energy">	hvac.energy <xsl:value-of select="hvac.energy"/>;
</xsl:if><xsl:if test="hvac.power_factor">	hvac.power_factor <xsl:value-of select="hvac.power_factor"/>;
</xsl:if><xsl:if test="lights.demand">	lights.demand <xsl:value-of select="lights.demand"/>;
</xsl:if><xsl:if test="lights.load">	lights.load <xsl:value-of select="lights.load"/>;
</xsl:if><xsl:if test="lights.energy">	lights.energy <xsl:value-of select="lights.energy"/>;
</xsl:if><xsl:if test="lights.power_factor">	lights.power_factor <xsl:value-of select="lights.power_factor"/>;
</xsl:if><xsl:if test="lights.heatgain_fraction">	lights.heatgain_fraction <xsl:value-of select="lights.heatgain_fraction"/>;
</xsl:if><xsl:if test="lights.heatgain">	lights.heatgain <xsl:value-of select="lights.heatgain"/>;
</xsl:if><xsl:if test="plugs.demand">	plugs.demand <xsl:value-of select="plugs.demand"/>;
</xsl:if><xsl:if test="plugs.load">	plugs.load <xsl:value-of select="plugs.load"/>;
</xsl:if><xsl:if test="plugs.energy">	plugs.energy <xsl:value-of select="plugs.energy"/>;
</xsl:if><xsl:if test="plugs.power_factor">	plugs.power_factor <xsl:value-of select="plugs.power_factor"/>;
</xsl:if><xsl:if test="plugs.heatgain_fraction">	plugs.heatgain_fraction <xsl:value-of select="plugs.heatgain_fraction"/>;
</xsl:if><xsl:if test="plugs.heatgain">	plugs.heatgain <xsl:value-of select="plugs.heatgain"/>;
</xsl:if><xsl:if test="cooling_setpoint">	cooling_setpoint <xsl:value-of select="cooling_setpoint"/>;
</xsl:if><xsl:if test="heating_setpoint">	heating_setpoint <xsl:value-of select="heating_setpoint"/>;
</xsl:if><xsl:if test="thermostat_deadband">	thermostat_deadband <xsl:value-of select="thermostat_deadband"/>;
</xsl:if><xsl:if test="control.ventilation_fraction">	control.ventilation_fraction <xsl:value-of select="control.ventilation_fraction"/>;
</xsl:if><xsl:if test="control.lighting_fraction">	control.lighting_fraction <xsl:value-of select="control.lighting_fraction"/>;
</xsl:if><xsl:if test="ACH">	ACH <xsl:value-of select="ACH"/>;
</xsl:if>}
</xsl:for-each>
# commercial::multizone objects
<xsl:for-each select="multizone_list/multizone"><a name="#GLM.{name}"/>object multizone:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="from">	from <a href="#GLM.{from}"><xsl:value-of select="from"/></a>;
</xsl:if><xsl:if test="to">	to <a href="#GLM.{to}"><xsl:value-of select="to"/></a>;
</xsl:if><xsl:if test="ua">	ua <xsl:value-of select="ua"/>;
</xsl:if>}
</xsl:for-each></xsl:for-each><xsl:for-each select="generators">
##############################################
# generators module
module generators {
	version.major <xsl:value-of select="version.major"/>;
	version.minor <xsl:value-of select="version.minor"/>;
	enable_subsecond_models <xsl:value-of select="enable_subsecond_models"/>;
	deltamode_timestep <xsl:value-of select="deltamode_timestep"/>;
	maximum_deltamode_time <xsl:value-of select="maximum_deltamode_time"/>;
	deltamode_update_count <xsl:value-of select="deltamode_update_count"/>;
}

# generators::diesel_dg objects
<xsl:for-each select="diesel_dg_list/diesel_dg"><a name="#GLM.{name}"/>object diesel_dg:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="Gen_mode">	Gen_mode <xsl:value-of select="Gen_mode"/>;
</xsl:if><xsl:if test="Gen_status">	Gen_status <xsl:value-of select="Gen_status"/>;
</xsl:if><xsl:if test="Gen_type">	Gen_type <xsl:value-of select="Gen_type"/>;
</xsl:if><xsl:if test="pf">	pf <xsl:value-of select="pf"/>;
</xsl:if><xsl:if test="GenElecEff">	GenElecEff <xsl:value-of select="GenElecEff"/>;
</xsl:if><xsl:if test="TotalRealPow">	TotalRealPow <xsl:value-of select="TotalRealPow"/>;
</xsl:if><xsl:if test="TotalReacPow">	TotalReacPow <xsl:value-of select="TotalReacPow"/>;
</xsl:if><xsl:if test="speed">	speed <xsl:value-of select="speed"/>;
</xsl:if><xsl:if test="cylinders">	cylinders <xsl:value-of select="cylinders"/>;
</xsl:if><xsl:if test="stroke">	stroke <xsl:value-of select="stroke"/>;
</xsl:if><xsl:if test="torque">	torque <xsl:value-of select="torque"/>;
</xsl:if><xsl:if test="pressure">	pressure <xsl:value-of select="pressure"/>;
</xsl:if><xsl:if test="time_operation">	time_operation <xsl:value-of select="time_operation"/>;
</xsl:if><xsl:if test="fuel">	fuel <xsl:value-of select="fuel"/>;
</xsl:if><xsl:if test="w_coolingwater">	w_coolingwater <xsl:value-of select="w_coolingwater"/>;
</xsl:if><xsl:if test="inlet_temperature">	inlet_temperature <xsl:value-of select="inlet_temperature"/>;
</xsl:if><xsl:if test="outlet_temperature">	outlet_temperature <xsl:value-of select="outlet_temperature"/>;
</xsl:if><xsl:if test="air_fuel">	air_fuel <xsl:value-of select="air_fuel"/>;
</xsl:if><xsl:if test="room_temperature">	room_temperature <xsl:value-of select="room_temperature"/>;
</xsl:if><xsl:if test="exhaust_temperature">	exhaust_temperature <xsl:value-of select="exhaust_temperature"/>;
</xsl:if><xsl:if test="cylinder_length">	cylinder_length <xsl:value-of select="cylinder_length"/>;
</xsl:if><xsl:if test="cylinder_radius">	cylinder_radius <xsl:value-of select="cylinder_radius"/>;
</xsl:if><xsl:if test="brake_diameter">	brake_diameter <xsl:value-of select="brake_diameter"/>;
</xsl:if><xsl:if test="calotific_fuel">	calotific_fuel <xsl:value-of select="calotific_fuel"/>;
</xsl:if><xsl:if test="steam_exhaust">	steam_exhaust <xsl:value-of select="steam_exhaust"/>;
</xsl:if><xsl:if test="specific_heat_steam">	specific_heat_steam <xsl:value-of select="specific_heat_steam"/>;
</xsl:if><xsl:if test="specific_heat_dry">	specific_heat_dry <xsl:value-of select="specific_heat_dry"/>;
</xsl:if><xsl:if test="indicated_hp">	indicated_hp <xsl:value-of select="indicated_hp"/>;
</xsl:if><xsl:if test="brake_hp">	brake_hp <xsl:value-of select="brake_hp"/>;
</xsl:if><xsl:if test="thermal_efficiency">	thermal_efficiency <xsl:value-of select="thermal_efficiency"/>;
</xsl:if><xsl:if test="energy_supplied">	energy_supplied <xsl:value-of select="energy_supplied"/>;
</xsl:if><xsl:if test="heat_equivalent_ip">	heat_equivalent_ip <xsl:value-of select="heat_equivalent_ip"/>;
</xsl:if><xsl:if test="energy_coolingwater">	energy_coolingwater <xsl:value-of select="energy_coolingwater"/>;
</xsl:if><xsl:if test="mass_exhaustgas">	mass_exhaustgas <xsl:value-of select="mass_exhaustgas"/>;
</xsl:if><xsl:if test="energy_exhaustgas">	energy_exhaustgas <xsl:value-of select="energy_exhaustgas"/>;
</xsl:if><xsl:if test="energy_steam">	energy_steam <xsl:value-of select="energy_steam"/>;
</xsl:if><xsl:if test="total_energy_exhaustgas">	total_energy_exhaustgas <xsl:value-of select="total_energy_exhaustgas"/>;
</xsl:if><xsl:if test="unaccounted_energyloss">	unaccounted_energyloss <xsl:value-of select="unaccounted_energyloss"/>;
</xsl:if><xsl:if test="f">	f <xsl:value-of select="f"/>;
</xsl:if><xsl:if test="poles">	poles <xsl:value-of select="poles"/>;
</xsl:if><xsl:if test="wm">	wm <xsl:value-of select="wm"/>;
</xsl:if><xsl:if test="Pconv">	Pconv <xsl:value-of select="Pconv"/>;
</xsl:if><xsl:if test="Tind">	Tind <xsl:value-of select="Tind"/>;
</xsl:if><xsl:if test="EA">	EA <xsl:value-of select="EA"/>;
</xsl:if><xsl:if test="Vo">	Vo <xsl:value-of select="Vo"/>;
</xsl:if><xsl:if test="Rs1">	Rs1 <xsl:value-of select="Rs1"/>;
</xsl:if><xsl:if test="Xs1">	Xs1 <xsl:value-of select="Xs1"/>;
</xsl:if><xsl:if test="delta">	delta <xsl:value-of select="delta"/>;
</xsl:if><xsl:if test="IA">	IA <xsl:value-of select="IA"/>;
</xsl:if><xsl:if test="Ploss">	Ploss <xsl:value-of select="Ploss"/>;
</xsl:if><xsl:if test="Pout">	Pout <xsl:value-of select="Pout"/>;
</xsl:if><xsl:if test="effe">	effe <xsl:value-of select="effe"/>;
</xsl:if><xsl:if test="effo">	effo <xsl:value-of select="effo"/>;
</xsl:if><xsl:if test="Rated_V">	Rated_V <xsl:value-of select="Rated_V"/>;
</xsl:if><xsl:if test="Rated_VA">	Rated_VA <xsl:value-of select="Rated_VA"/>;
</xsl:if><xsl:if test="Rs">	Rs <xsl:value-of select="Rs"/>;
</xsl:if><xsl:if test="Xs">	Xs <xsl:value-of select="Xs"/>;
</xsl:if><xsl:if test="Rg">	Rg <xsl:value-of select="Rg"/>;
</xsl:if><xsl:if test="Xg">	Xg <xsl:value-of select="Xg"/>;
</xsl:if><xsl:if test="voltage_A">	voltage_A <xsl:value-of select="voltage_A"/>;
</xsl:if><xsl:if test="voltage_B">	voltage_B <xsl:value-of select="voltage_B"/>;
</xsl:if><xsl:if test="voltage_C">	voltage_C <xsl:value-of select="voltage_C"/>;
</xsl:if><xsl:if test="current_A">	current_A <xsl:value-of select="current_A"/>;
</xsl:if><xsl:if test="current_B">	current_B <xsl:value-of select="current_B"/>;
</xsl:if><xsl:if test="current_C">	current_C <xsl:value-of select="current_C"/>;
</xsl:if><xsl:if test="EfA">	EfA <xsl:value-of select="EfA"/>;
</xsl:if><xsl:if test="EfB">	EfB <xsl:value-of select="EfB"/>;
</xsl:if><xsl:if test="EfC">	EfC <xsl:value-of select="EfC"/>;
</xsl:if><xsl:if test="power_A">	power_A <xsl:value-of select="power_A"/>;
</xsl:if><xsl:if test="power_B">	power_B <xsl:value-of select="power_B"/>;
</xsl:if><xsl:if test="power_C">	power_C <xsl:value-of select="power_C"/>;
</xsl:if><xsl:if test="power_A_sch">	power_A_sch <xsl:value-of select="power_A_sch"/>;
</xsl:if><xsl:if test="power_B_sch">	power_B_sch <xsl:value-of select="power_B_sch"/>;
</xsl:if><xsl:if test="power_C_sch">	power_C_sch <xsl:value-of select="power_C_sch"/>;
</xsl:if><xsl:if test="EfA_sch">	EfA_sch <xsl:value-of select="EfA_sch"/>;
</xsl:if><xsl:if test="EfB_sch">	EfB_sch <xsl:value-of select="EfB_sch"/>;
</xsl:if><xsl:if test="EfC_sch">	EfC_sch <xsl:value-of select="EfC_sch"/>;
</xsl:if><xsl:if test="SlackBus">	SlackBus <xsl:value-of select="SlackBus"/>;
</xsl:if><xsl:if test="phases">	phases <xsl:value-of select="phases"/>;
</xsl:if>}
</xsl:for-each>
# generators::windturb_dg objects
<xsl:for-each select="windturb_dg_list/windturb_dg"><a name="#GLM.{name}"/>object windturb_dg:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="Gen_status">	Gen_status <xsl:value-of select="Gen_status"/>;
</xsl:if><xsl:if test="Gen_type">	Gen_type <xsl:value-of select="Gen_type"/>;
</xsl:if><xsl:if test="Gen_mode">	Gen_mode <xsl:value-of select="Gen_mode"/>;
</xsl:if><xsl:if test="Turbine_Model">	Turbine_Model <xsl:value-of select="Turbine_Model"/>;
</xsl:if><xsl:if test="turbine_height">	turbine_height <xsl:value-of select="turbine_height"/>;
</xsl:if><xsl:if test="blade_diam">	blade_diam <xsl:value-of select="blade_diam"/>;
</xsl:if><xsl:if test="cut_in_ws">	cut_in_ws <xsl:value-of select="cut_in_ws"/>;
</xsl:if><xsl:if test="cut_out_ws">	cut_out_ws <xsl:value-of select="cut_out_ws"/>;
</xsl:if><xsl:if test="ws_rated">	ws_rated <xsl:value-of select="ws_rated"/>;
</xsl:if><xsl:if test="ws_maxcp">	ws_maxcp <xsl:value-of select="ws_maxcp"/>;
</xsl:if><xsl:if test="Cp_max">	Cp_max <xsl:value-of select="Cp_max"/>;
</xsl:if><xsl:if test="Cp_rated">	Cp_rated <xsl:value-of select="Cp_rated"/>;
</xsl:if><xsl:if test="Rated_VA">	Rated_VA <xsl:value-of select="Rated_VA"/>;
</xsl:if><xsl:if test="Rated_V">	Rated_V <xsl:value-of select="Rated_V"/>;
</xsl:if><xsl:if test="Pconv">	Pconv <xsl:value-of select="Pconv"/>;
</xsl:if><xsl:if test="WSadj">	WSadj <xsl:value-of select="WSadj"/>;
</xsl:if><xsl:if test="Wind_Speed">	Wind_Speed <xsl:value-of select="Wind_Speed"/>;
</xsl:if><xsl:if test="R_stator">	R_stator <xsl:value-of select="R_stator"/>;
</xsl:if><xsl:if test="X_stator">	X_stator <xsl:value-of select="X_stator"/>;
</xsl:if><xsl:if test="R_rotor">	R_rotor <xsl:value-of select="R_rotor"/>;
</xsl:if><xsl:if test="X_rotor">	X_rotor <xsl:value-of select="X_rotor"/>;
</xsl:if><xsl:if test="R_core">	R_core <xsl:value-of select="R_core"/>;
</xsl:if><xsl:if test="X_magnetic">	X_magnetic <xsl:value-of select="X_magnetic"/>;
</xsl:if><xsl:if test="Max_Vrotor">	Max_Vrotor <xsl:value-of select="Max_Vrotor"/>;
</xsl:if><xsl:if test="Min_Vrotor">	Min_Vrotor <xsl:value-of select="Min_Vrotor"/>;
</xsl:if><xsl:if test="Max_Ef">	Max_Ef <xsl:value-of select="Max_Ef"/>;
</xsl:if><xsl:if test="Min_Ef">	Min_Ef <xsl:value-of select="Min_Ef"/>;
</xsl:if><xsl:if test="Rs">	Rs <xsl:value-of select="Rs"/>;
</xsl:if><xsl:if test="Xs">	Xs <xsl:value-of select="Xs"/>;
</xsl:if><xsl:if test="Rg">	Rg <xsl:value-of select="Rg"/>;
</xsl:if><xsl:if test="Xg">	Xg <xsl:value-of select="Xg"/>;
</xsl:if><xsl:if test="pf">	pf <xsl:value-of select="pf"/>;
</xsl:if><xsl:if test="GenElecEff">	GenElecEff <xsl:value-of select="GenElecEff"/>;
</xsl:if><xsl:if test="TotalRealPow">	TotalRealPow <xsl:value-of select="TotalRealPow"/>;
</xsl:if><xsl:if test="TotalReacPow">	TotalReacPow <xsl:value-of select="TotalReacPow"/>;
</xsl:if><xsl:if test="voltage_A">	voltage_A <xsl:value-of select="voltage_A"/>;
</xsl:if><xsl:if test="voltage_B">	voltage_B <xsl:value-of select="voltage_B"/>;
</xsl:if><xsl:if test="voltage_C">	voltage_C <xsl:value-of select="voltage_C"/>;
</xsl:if><xsl:if test="current_A">	current_A <xsl:value-of select="current_A"/>;
</xsl:if><xsl:if test="current_B">	current_B <xsl:value-of select="current_B"/>;
</xsl:if><xsl:if test="current_C">	current_C <xsl:value-of select="current_C"/>;
</xsl:if><xsl:if test="EfA">	EfA <xsl:value-of select="EfA"/>;
</xsl:if><xsl:if test="EfB">	EfB <xsl:value-of select="EfB"/>;
</xsl:if><xsl:if test="EfC">	EfC <xsl:value-of select="EfC"/>;
</xsl:if><xsl:if test="Vrotor_A">	Vrotor_A <xsl:value-of select="Vrotor_A"/>;
</xsl:if><xsl:if test="Vrotor_B">	Vrotor_B <xsl:value-of select="Vrotor_B"/>;
</xsl:if><xsl:if test="Vrotor_C">	Vrotor_C <xsl:value-of select="Vrotor_C"/>;
</xsl:if><xsl:if test="Irotor_A">	Irotor_A <xsl:value-of select="Irotor_A"/>;
</xsl:if><xsl:if test="Irotor_B">	Irotor_B <xsl:value-of select="Irotor_B"/>;
</xsl:if><xsl:if test="Irotor_C">	Irotor_C <xsl:value-of select="Irotor_C"/>;
</xsl:if><xsl:if test="phases">	phases <xsl:value-of select="phases"/>;
</xsl:if>}
</xsl:for-each>
# generators::power_electronics objects
<xsl:for-each select="power_electronics_list/power_electronics"><a name="#GLM.{name}"/>object power_electronics:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="generator_mode">	generator_mode <xsl:value-of select="generator_mode"/>;
</xsl:if><xsl:if test="generator_status">	generator_status <xsl:value-of select="generator_status"/>;
</xsl:if><xsl:if test="converter_type">	converter_type <xsl:value-of select="converter_type"/>;
</xsl:if><xsl:if test="switch_type">	switch_type <xsl:value-of select="switch_type"/>;
</xsl:if><xsl:if test="filter_type">	filter_type <xsl:value-of select="filter_type"/>;
</xsl:if><xsl:if test="filter_implementation">	filter_implementation <xsl:value-of select="filter_implementation"/>;
</xsl:if><xsl:if test="filter_frequency">	filter_frequency <xsl:value-of select="filter_frequency"/>;
</xsl:if><xsl:if test="power_type">	power_type <xsl:value-of select="power_type"/>;
</xsl:if><xsl:if test="Rated_kW">	Rated_kW <xsl:value-of select="Rated_kW"/>;
</xsl:if><xsl:if test="Max_P">	Max_P <xsl:value-of select="Max_P"/>;
</xsl:if><xsl:if test="Min_P">	Min_P <xsl:value-of select="Min_P"/>;
</xsl:if><xsl:if test="Rated_kVA">	Rated_kVA <xsl:value-of select="Rated_kVA"/>;
</xsl:if><xsl:if test="Rated_kV">	Rated_kV <xsl:value-of select="Rated_kV"/>;
</xsl:if><xsl:if test="phases">	phases <xsl:value-of select="phases"/>;
</xsl:if>}
</xsl:for-each>
# generators::energy_storage objects
<xsl:for-each select="energy_storage_list/energy_storage"><a name="#GLM.{name}"/>object energy_storage:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="generator_mode">	generator_mode <xsl:value-of select="generator_mode"/>;
</xsl:if><xsl:if test="generator_status">	generator_status <xsl:value-of select="generator_status"/>;
</xsl:if><xsl:if test="power_type">	power_type <xsl:value-of select="power_type"/>;
</xsl:if><xsl:if test="Rinternal">	Rinternal <xsl:value-of select="Rinternal"/>;
</xsl:if><xsl:if test="V_Max">	V_Max <xsl:value-of select="V_Max"/>;
</xsl:if><xsl:if test="I_Max">	I_Max <xsl:value-of select="I_Max"/>;
</xsl:if><xsl:if test="E_Max">	E_Max <xsl:value-of select="E_Max"/>;
</xsl:if><xsl:if test="Energy">	Energy <xsl:value-of select="Energy"/>;
</xsl:if><xsl:if test="efficiency">	efficiency <xsl:value-of select="efficiency"/>;
</xsl:if><xsl:if test="Rated_kVA">	Rated_kVA <xsl:value-of select="Rated_kVA"/>;
</xsl:if><xsl:if test="V_Out">	V_Out <xsl:value-of select="V_Out"/>;
</xsl:if><xsl:if test="I_Out">	I_Out <xsl:value-of select="I_Out"/>;
</xsl:if><xsl:if test="VA_Out">	VA_Out <xsl:value-of select="VA_Out"/>;
</xsl:if><xsl:if test="V_In">	V_In <xsl:value-of select="V_In"/>;
</xsl:if><xsl:if test="I_In">	I_In <xsl:value-of select="I_In"/>;
</xsl:if><xsl:if test="V_Internal">	V_Internal <xsl:value-of select="V_Internal"/>;
</xsl:if><xsl:if test="I_Internal">	I_Internal <xsl:value-of select="I_Internal"/>;
</xsl:if><xsl:if test="I_Prev">	I_Prev <xsl:value-of select="I_Prev"/>;
</xsl:if><xsl:if test="phases">	phases <xsl:value-of select="phases"/>;
</xsl:if>}
</xsl:for-each>
# generators::inverter objects
<xsl:for-each select="inverter_list/inverter"><a name="#GLM.{name}"/>object inverter:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="inverter_type">	inverter_type <xsl:value-of select="inverter_type"/>;
</xsl:if><xsl:if test="four_quadrant_control_mode">	four_quadrant_control_mode <xsl:value-of select="four_quadrant_control_mode"/>;
</xsl:if><xsl:if test="generator_status">	generator_status <xsl:value-of select="generator_status"/>;
</xsl:if><xsl:if test="generator_mode">	generator_mode <xsl:value-of select="generator_mode"/>;
</xsl:if><xsl:if test="V_In">	V_In <xsl:value-of select="V_In"/>;
</xsl:if><xsl:if test="I_In">	I_In <xsl:value-of select="I_In"/>;
</xsl:if><xsl:if test="VA_In">	VA_In <xsl:value-of select="VA_In"/>;
</xsl:if><xsl:if test="VA_Out">	VA_Out <xsl:value-of select="VA_Out"/>;
</xsl:if><xsl:if test="Vdc">	Vdc <xsl:value-of select="Vdc"/>;
</xsl:if><xsl:if test="phaseA_V_Out">	phaseA_V_Out <xsl:value-of select="phaseA_V_Out"/>;
</xsl:if><xsl:if test="phaseB_V_Out">	phaseB_V_Out <xsl:value-of select="phaseB_V_Out"/>;
</xsl:if><xsl:if test="phaseC_V_Out">	phaseC_V_Out <xsl:value-of select="phaseC_V_Out"/>;
</xsl:if><xsl:if test="phaseA_I_Out">	phaseA_I_Out <xsl:value-of select="phaseA_I_Out"/>;
</xsl:if><xsl:if test="phaseB_I_Out">	phaseB_I_Out <xsl:value-of select="phaseB_I_Out"/>;
</xsl:if><xsl:if test="phaseC_I_Out">	phaseC_I_Out <xsl:value-of select="phaseC_I_Out"/>;
</xsl:if><xsl:if test="power_A">	power_A <xsl:value-of select="power_A"/>;
</xsl:if><xsl:if test="power_B">	power_B <xsl:value-of select="power_B"/>;
</xsl:if><xsl:if test="power_C">	power_C <xsl:value-of select="power_C"/>;
</xsl:if><xsl:if test="P_Out">	P_Out <xsl:value-of select="P_Out"/>;
</xsl:if><xsl:if test="Q_Out">	Q_Out <xsl:value-of select="Q_Out"/>;
</xsl:if><xsl:if test="power_in">	power_in <xsl:value-of select="power_in"/>;
</xsl:if><xsl:if test="rated_power">	rated_power <xsl:value-of select="rated_power"/>;
</xsl:if><xsl:if test="rated_battery_power">	rated_battery_power <xsl:value-of select="rated_battery_power"/>;
</xsl:if><xsl:if test="inverter_efficiency">	inverter_efficiency <xsl:value-of select="inverter_efficiency"/>;
</xsl:if><xsl:if test="battery_soc">	battery_soc <xsl:value-of select="battery_soc"/>;
</xsl:if><xsl:if test="soc_reserve">	soc_reserve <xsl:value-of select="soc_reserve"/>;
</xsl:if><xsl:if test="power_factor">	power_factor <xsl:value-of select="power_factor"/>;
</xsl:if><xsl:if test="phases">	phases <xsl:value-of select="phases"/>;
</xsl:if><xsl:if test="use_multipoint_efficiency">	use_multipoint_efficiency <xsl:value-of select="use_multipoint_efficiency"/>;
</xsl:if><xsl:if test="inverter_manufacturer">	inverter_manufacturer <xsl:value-of select="inverter_manufacturer"/>;
</xsl:if><xsl:if test="maximum_dc_power">	maximum_dc_power <xsl:value-of select="maximum_dc_power"/>;
</xsl:if><xsl:if test="maximum_dc_voltage">	maximum_dc_voltage <xsl:value-of select="maximum_dc_voltage"/>;
</xsl:if><xsl:if test="minimum_dc_power">	minimum_dc_power <xsl:value-of select="minimum_dc_power"/>;
</xsl:if><xsl:if test="c_0">	c_0 <xsl:value-of select="c_0"/>;
</xsl:if><xsl:if test="c_1">	c_1 <xsl:value-of select="c_1"/>;
</xsl:if><xsl:if test="c_2">	c_2 <xsl:value-of select="c_2"/>;
</xsl:if><xsl:if test="c_3">	c_3 <xsl:value-of select="c_3"/>;
</xsl:if>}
</xsl:for-each>
# generators::dc_dc_converter objects
<xsl:for-each select="dc_dc_converter_list/dc_dc_converter"><a name="#GLM.{name}"/>object dc_dc_converter:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="dc_dc_converter_type">	dc_dc_converter_type <xsl:value-of select="dc_dc_converter_type"/>;
</xsl:if><xsl:if test="generator_mode">	generator_mode <xsl:value-of select="generator_mode"/>;
</xsl:if><xsl:if test="V_Out">	V_Out <xsl:value-of select="V_Out"/>;
</xsl:if><xsl:if test="I_Out">	I_Out <xsl:value-of select="I_Out"/>;
</xsl:if><xsl:if test="Vdc">	Vdc <xsl:value-of select="Vdc"/>;
</xsl:if><xsl:if test="VA_Out">	VA_Out <xsl:value-of select="VA_Out"/>;
</xsl:if><xsl:if test="P_Out">	P_Out <xsl:value-of select="P_Out"/>;
</xsl:if><xsl:if test="Q_Out">	Q_Out <xsl:value-of select="Q_Out"/>;
</xsl:if><xsl:if test="service_ratio">	service_ratio <xsl:value-of select="service_ratio"/>;
</xsl:if><xsl:if test="V_In">	V_In <xsl:value-of select="V_In"/>;
</xsl:if><xsl:if test="I_In">	I_In <xsl:value-of select="I_In"/>;
</xsl:if><xsl:if test="VA_In">	VA_In <xsl:value-of select="VA_In"/>;
</xsl:if><xsl:if test="phases">	phases <xsl:value-of select="phases"/>;
</xsl:if>}
</xsl:for-each>
# generators::rectifier objects
<xsl:for-each select="rectifier_list/rectifier"><a name="#GLM.{name}"/>object rectifier:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="rectifier_type">	rectifier_type <xsl:value-of select="rectifier_type"/>;
</xsl:if><xsl:if test="generator_mode">	generator_mode <xsl:value-of select="generator_mode"/>;
</xsl:if><xsl:if test="V_Out">	V_Out <xsl:value-of select="V_Out"/>;
</xsl:if><xsl:if test="I_Out">	I_Out <xsl:value-of select="I_Out"/>;
</xsl:if><xsl:if test="VA_Out">	VA_Out <xsl:value-of select="VA_Out"/>;
</xsl:if><xsl:if test="P_Out">	P_Out <xsl:value-of select="P_Out"/>;
</xsl:if><xsl:if test="Q_Out">	Q_Out <xsl:value-of select="Q_Out"/>;
</xsl:if><xsl:if test="Vdc">	Vdc <xsl:value-of select="Vdc"/>;
</xsl:if><xsl:if test="phaseA_V_In">	phaseA_V_In <xsl:value-of select="phaseA_V_In"/>;
</xsl:if><xsl:if test="phaseB_V_In">	phaseB_V_In <xsl:value-of select="phaseB_V_In"/>;
</xsl:if><xsl:if test="phaseC_V_In">	phaseC_V_In <xsl:value-of select="phaseC_V_In"/>;
</xsl:if><xsl:if test="phaseA_I_In">	phaseA_I_In <xsl:value-of select="phaseA_I_In"/>;
</xsl:if><xsl:if test="phaseB_I_In">	phaseB_I_In <xsl:value-of select="phaseB_I_In"/>;
</xsl:if><xsl:if test="phaseC_I_In">	phaseC_I_In <xsl:value-of select="phaseC_I_In"/>;
</xsl:if><xsl:if test="power_A_In">	power_A_In <xsl:value-of select="power_A_In"/>;
</xsl:if><xsl:if test="power_B_In">	power_B_In <xsl:value-of select="power_B_In"/>;
</xsl:if><xsl:if test="power_C_In">	power_C_In <xsl:value-of select="power_C_In"/>;
</xsl:if><xsl:if test="phases">	phases <xsl:value-of select="phases"/>;
</xsl:if>}
</xsl:for-each>
# generators::microturbine objects
<xsl:for-each select="microturbine_list/microturbine"><a name="#GLM.{name}"/>object microturbine:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="generator_mode">	generator_mode <xsl:value-of select="generator_mode"/>;
</xsl:if><xsl:if test="generator_status">	generator_status <xsl:value-of select="generator_status"/>;
</xsl:if><xsl:if test="power_type">	power_type <xsl:value-of select="power_type"/>;
</xsl:if><xsl:if test="Rinternal">	Rinternal <xsl:value-of select="Rinternal"/>;
</xsl:if><xsl:if test="Rload">	Rload <xsl:value-of select="Rload"/>;
</xsl:if><xsl:if test="V_Max">	V_Max <xsl:value-of select="V_Max"/>;
</xsl:if><xsl:if test="I_Max">	I_Max <xsl:value-of select="I_Max"/>;
</xsl:if><xsl:if test="frequency">	frequency <xsl:value-of select="frequency"/>;
</xsl:if><xsl:if test="Max_Frequency">	Max_Frequency <xsl:value-of select="Max_Frequency"/>;
</xsl:if><xsl:if test="Min_Frequency">	Min_Frequency <xsl:value-of select="Min_Frequency"/>;
</xsl:if><xsl:if test="Fuel_Used">	Fuel_Used <xsl:value-of select="Fuel_Used"/>;
</xsl:if><xsl:if test="Heat_Out">	Heat_Out <xsl:value-of select="Heat_Out"/>;
</xsl:if><xsl:if test="KV">	KV <xsl:value-of select="KV"/>;
</xsl:if><xsl:if test="Power_Angle">	Power_Angle <xsl:value-of select="Power_Angle"/>;
</xsl:if><xsl:if test="Max_P">	Max_P <xsl:value-of select="Max_P"/>;
</xsl:if><xsl:if test="Min_P">	Min_P <xsl:value-of select="Min_P"/>;
</xsl:if><xsl:if test="phaseA_V_Out">	phaseA_V_Out <xsl:value-of select="phaseA_V_Out"/>;
</xsl:if><xsl:if test="phaseB_V_Out">	phaseB_V_Out <xsl:value-of select="phaseB_V_Out"/>;
</xsl:if><xsl:if test="phaseC_V_Out">	phaseC_V_Out <xsl:value-of select="phaseC_V_Out"/>;
</xsl:if><xsl:if test="phaseA_I_Out">	phaseA_I_Out <xsl:value-of select="phaseA_I_Out"/>;
</xsl:if><xsl:if test="phaseB_I_Out">	phaseB_I_Out <xsl:value-of select="phaseB_I_Out"/>;
</xsl:if><xsl:if test="phaseC_I_Out">	phaseC_I_Out <xsl:value-of select="phaseC_I_Out"/>;
</xsl:if><xsl:if test="power_A_Out">	power_A_Out <xsl:value-of select="power_A_Out"/>;
</xsl:if><xsl:if test="power_B_Out">	power_B_Out <xsl:value-of select="power_B_Out"/>;
</xsl:if><xsl:if test="power_C_Out">	power_C_Out <xsl:value-of select="power_C_Out"/>;
</xsl:if><xsl:if test="VA_Out">	VA_Out <xsl:value-of select="VA_Out"/>;
</xsl:if><xsl:if test="pf_Out">	pf_Out <xsl:value-of select="pf_Out"/>;
</xsl:if><xsl:if test="E_A_Internal">	E_A_Internal <xsl:value-of select="E_A_Internal"/>;
</xsl:if><xsl:if test="E_B_Internal">	E_B_Internal <xsl:value-of select="E_B_Internal"/>;
</xsl:if><xsl:if test="E_C_Internal">	E_C_Internal <xsl:value-of select="E_C_Internal"/>;
</xsl:if><xsl:if test="efficiency">	efficiency <xsl:value-of select="efficiency"/>;
</xsl:if><xsl:if test="Rated_kVA">	Rated_kVA <xsl:value-of select="Rated_kVA"/>;
</xsl:if><xsl:if test="phases">	phases <xsl:value-of select="phases"/>;
</xsl:if>}
</xsl:for-each>
# generators::battery objects
<xsl:for-each select="battery_list/battery"><a name="#GLM.{name}"/>object battery:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="generator_mode">	generator_mode <xsl:value-of select="generator_mode"/>;
</xsl:if><xsl:if test="additional_controls">	additional_controls <xsl:value-of select="additional_controls"/>;
</xsl:if><xsl:if test="generator_status">	generator_status <xsl:value-of select="generator_status"/>;
</xsl:if><xsl:if test="rfb_size">	rfb_size <xsl:value-of select="rfb_size"/>;
</xsl:if><xsl:if test="power_type">	power_type <xsl:value-of select="power_type"/>;
</xsl:if><xsl:if test="battery_state">	battery_state <xsl:value-of select="battery_state"/>;
</xsl:if><xsl:if test="number_battery_state_changes">	number_battery_state_changes <xsl:value-of select="number_battery_state_changes"/>;
</xsl:if><xsl:if test="monitored_power">	monitored_power <xsl:value-of select="monitored_power"/>;
</xsl:if><xsl:if test="power_set_high">	power_set_high <xsl:value-of select="power_set_high"/>;
</xsl:if><xsl:if test="power_set_low">	power_set_low <xsl:value-of select="power_set_low"/>;
</xsl:if><xsl:if test="power_set_high_highT">	power_set_high_highT <xsl:value-of select="power_set_high_highT"/>;
</xsl:if><xsl:if test="power_set_low_highT">	power_set_low_highT <xsl:value-of select="power_set_low_highT"/>;
</xsl:if><xsl:if test="check_power_low">	check_power_low <xsl:value-of select="check_power_low"/>;
</xsl:if><xsl:if test="check_power_high">	check_power_high <xsl:value-of select="check_power_high"/>;
</xsl:if><xsl:if test="voltage_set_high">	voltage_set_high <xsl:value-of select="voltage_set_high"/>;
</xsl:if><xsl:if test="voltage_set_low">	voltage_set_low <xsl:value-of select="voltage_set_low"/>;
</xsl:if><xsl:if test="deadband">	deadband <xsl:value-of select="deadband"/>;
</xsl:if><xsl:if test="sensitivity">	sensitivity <xsl:value-of select="sensitivity"/>;
</xsl:if><xsl:if test="high_temperature">	high_temperature <xsl:value-of select="high_temperature"/>;
</xsl:if><xsl:if test="midpoint_temperature">	midpoint_temperature <xsl:value-of select="midpoint_temperature"/>;
</xsl:if><xsl:if test="low_temperature">	low_temperature <xsl:value-of select="low_temperature"/>;
</xsl:if><xsl:if test="scheduled_power">	scheduled_power <xsl:value-of select="scheduled_power"/>;
</xsl:if><xsl:if test="Rinternal">	Rinternal <xsl:value-of select="Rinternal"/>;
</xsl:if><xsl:if test="V_Max">	V_Max <xsl:value-of select="V_Max"/>;
</xsl:if><xsl:if test="I_Max">	I_Max <xsl:value-of select="I_Max"/>;
</xsl:if><xsl:if test="E_Max">	E_Max <xsl:value-of select="E_Max"/>;
</xsl:if><xsl:if test="P_Max">	P_Max <xsl:value-of select="P_Max"/>;
</xsl:if><xsl:if test="power_factor">	power_factor <xsl:value-of select="power_factor"/>;
</xsl:if><xsl:if test="Energy">	Energy <xsl:value-of select="Energy"/>;
</xsl:if><xsl:if test="efficiency">	efficiency <xsl:value-of select="efficiency"/>;
</xsl:if><xsl:if test="base_efficiency">	base_efficiency <xsl:value-of select="base_efficiency"/>;
</xsl:if><xsl:if test="parasitic_power_draw">	parasitic_power_draw <xsl:value-of select="parasitic_power_draw"/>;
</xsl:if><xsl:if test="Rated_kVA">	Rated_kVA <xsl:value-of select="Rated_kVA"/>;
</xsl:if><xsl:if test="V_Out">	V_Out <xsl:value-of select="V_Out"/>;
</xsl:if><xsl:if test="I_Out">	I_Out <xsl:value-of select="I_Out"/>;
</xsl:if><xsl:if test="VA_Out">	VA_Out <xsl:value-of select="VA_Out"/>;
</xsl:if><xsl:if test="V_In">	V_In <xsl:value-of select="V_In"/>;
</xsl:if><xsl:if test="I_In">	I_In <xsl:value-of select="I_In"/>;
</xsl:if><xsl:if test="V_Internal">	V_Internal <xsl:value-of select="V_Internal"/>;
</xsl:if><xsl:if test="I_Internal">	I_Internal <xsl:value-of select="I_Internal"/>;
</xsl:if><xsl:if test="I_Prev">	I_Prev <xsl:value-of select="I_Prev"/>;
</xsl:if><xsl:if test="power_transferred">	power_transferred <xsl:value-of select="power_transferred"/>;
</xsl:if><xsl:if test="phases">	phases <xsl:value-of select="phases"/>;
</xsl:if><xsl:if test="use_internal_battery_model">	use_internal_battery_model <xsl:value-of select="use_internal_battery_model"/>;
</xsl:if><xsl:if test="battery_type">	battery_type <xsl:value-of select="battery_type"/>;
</xsl:if><xsl:if test="nominal_voltage">	nominal_voltage <xsl:value-of select="nominal_voltage"/>;
</xsl:if><xsl:if test="rated_power">	rated_power <xsl:value-of select="rated_power"/>;
</xsl:if><xsl:if test="battery_capacity">	battery_capacity <xsl:value-of select="battery_capacity"/>;
</xsl:if><xsl:if test="round_trip_efficiency">	round_trip_efficiency <xsl:value-of select="round_trip_efficiency"/>;
</xsl:if><xsl:if test="state_of_charge">	state_of_charge <xsl:value-of select="state_of_charge"/>;
</xsl:if><xsl:if test="battery_load">	battery_load <xsl:value-of select="battery_load"/>;
</xsl:if><xsl:if test="reserve_state_of_charge">	reserve_state_of_charge <xsl:value-of select="reserve_state_of_charge"/>;
</xsl:if>}
</xsl:for-each>
# generators::solar objects
<xsl:for-each select="solar_list/solar"><a name="#GLM.{name}"/>object solar:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="generator_mode">	generator_mode <xsl:value-of select="generator_mode"/>;
</xsl:if><xsl:if test="generator_status">	generator_status <xsl:value-of select="generator_status"/>;
</xsl:if><xsl:if test="panel_type">	panel_type <xsl:value-of select="panel_type"/>;
</xsl:if><xsl:if test="power_type">	power_type <xsl:value-of select="power_type"/>;
</xsl:if><xsl:if test="INSTALLATION_TYPE">	INSTALLATION_TYPE <xsl:value-of select="INSTALLATION_TYPE"/>;
</xsl:if><xsl:if test="SOLAR_TILT_MODEL">	SOLAR_TILT_MODEL <xsl:value-of select="SOLAR_TILT_MODEL"/>;
</xsl:if><xsl:if test="SOLAR_POWER_MODEL">	SOLAR_POWER_MODEL <xsl:value-of select="SOLAR_POWER_MODEL"/>;
</xsl:if><xsl:if test="a_coeff">	a_coeff <xsl:value-of select="a_coeff"/>;
</xsl:if><xsl:if test="b_coeff">	b_coeff <xsl:value-of select="b_coeff"/>;
</xsl:if><xsl:if test="dT_coeff">	dT_coeff <xsl:value-of select="dT_coeff"/>;
</xsl:if><xsl:if test="T_coeff">	T_coeff <xsl:value-of select="T_coeff"/>;
</xsl:if><xsl:if test="NOCT">	NOCT <xsl:value-of select="NOCT"/>;
</xsl:if><xsl:if test="Tmodule">	Tmodule <xsl:value-of select="Tmodule"/>;
</xsl:if><xsl:if test="Tambient">	Tambient <xsl:value-of select="Tambient"/>;
</xsl:if><xsl:if test="wind_speed">	wind_speed <xsl:value-of select="wind_speed"/>;
</xsl:if><xsl:if test="ambient_temperature">	ambient_temperature <xsl:value-of select="ambient_temperature"/>;
</xsl:if><xsl:if test="Insolation">	Insolation <xsl:value-of select="Insolation"/>;
</xsl:if><xsl:if test="Rinternal">	Rinternal <xsl:value-of select="Rinternal"/>;
</xsl:if><xsl:if test="Rated_Insolation">	Rated_Insolation <xsl:value-of select="Rated_Insolation"/>;
</xsl:if><xsl:if test="Pmax_temp_coeff">	Pmax_temp_coeff <xsl:value-of select="Pmax_temp_coeff"/>;
</xsl:if><xsl:if test="Voc_temp_coeff">	Voc_temp_coeff <xsl:value-of select="Voc_temp_coeff"/>;
</xsl:if><xsl:if test="V_Max">	V_Max <xsl:value-of select="V_Max"/>;
</xsl:if><xsl:if test="Voc_Max">	Voc_Max <xsl:value-of select="Voc_Max"/>;
</xsl:if><xsl:if test="Voc">	Voc <xsl:value-of select="Voc"/>;
</xsl:if><xsl:if test="efficiency">	efficiency <xsl:value-of select="efficiency"/>;
</xsl:if><xsl:if test="area">	area <xsl:value-of select="area"/>;
</xsl:if><xsl:if test="soiling">	soiling <xsl:value-of select="soiling"/>;
</xsl:if><xsl:if test="derating">	derating <xsl:value-of select="derating"/>;
</xsl:if><xsl:if test="Tcell">	Tcell <xsl:value-of select="Tcell"/>;
</xsl:if><xsl:if test="Rated_kVA">	Rated_kVA <xsl:value-of select="Rated_kVA"/>;
</xsl:if><xsl:if test="P_Out">	P_Out <xsl:value-of select="P_Out"/>;
</xsl:if><xsl:if test="V_Out">	V_Out <xsl:value-of select="V_Out"/>;
</xsl:if><xsl:if test="I_Out">	I_Out <xsl:value-of select="I_Out"/>;
</xsl:if><xsl:if test="VA_Out">	VA_Out <xsl:value-of select="VA_Out"/>;
</xsl:if><xsl:if test="weather">	weather <a href="#GLM.{weather}"><xsl:value-of select="weather"/></a>;
</xsl:if><xsl:if test="shading_factor">	shading_factor <xsl:value-of select="shading_factor"/>;
</xsl:if><xsl:if test="tilt_angle">	tilt_angle <xsl:value-of select="tilt_angle"/>;
</xsl:if><xsl:if test="orientation_azimuth">	orientation_azimuth <xsl:value-of select="orientation_azimuth"/>;
</xsl:if><xsl:if test="latitude_angle_fix">	latitude_angle_fix <xsl:value-of select="latitude_angle_fix"/>;
</xsl:if><xsl:if test="orientation">	orientation <xsl:value-of select="orientation"/>;
</xsl:if><xsl:if test="phases">	phases <xsl:value-of select="phases"/>;
</xsl:if>}
</xsl:for-each></xsl:for-each><xsl:for-each select="market">
##############################################
# market module
module market {
	version.major <xsl:value-of select="version.major"/>;
	version.minor <xsl:value-of select="version.minor"/>;
	bid_offset <xsl:value-of select="bid_offset"/>;
}

# market::auction objects
<xsl:for-each select="auction_list/auction"><a name="#GLM.{name}"/>object auction:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="unit">	unit <xsl:value-of select="unit"/>;
</xsl:if><xsl:if test="period">	period <xsl:value-of select="period"/>;
</xsl:if><xsl:if test="latency">	latency <xsl:value-of select="latency"/>;
</xsl:if><xsl:if test="market_id">	market_id <xsl:value-of select="market_id"/>;
</xsl:if><xsl:if test="network">	network <a href="#GLM.{network}"><xsl:value-of select="network"/></a>;
</xsl:if><xsl:if test="verbose">	verbose <xsl:value-of select="verbose"/>;
</xsl:if><xsl:if test="linkref">	linkref <a href="#GLM.{linkref}"><xsl:value-of select="linkref"/></a>;
</xsl:if><xsl:if test="pricecap">	pricecap <xsl:value-of select="pricecap"/>;
</xsl:if><xsl:if test="price_cap">	price_cap <xsl:value-of select="price_cap"/>;
</xsl:if><xsl:if test="special_mode">	special_mode <xsl:value-of select="special_mode"/>;
</xsl:if><xsl:if test="statistic_mode">	statistic_mode <xsl:value-of select="statistic_mode"/>;
</xsl:if><xsl:if test="fixed_price">	fixed_price <xsl:value-of select="fixed_price"/>;
</xsl:if><xsl:if test="fixed_quantity">	fixed_quantity <xsl:value-of select="fixed_quantity"/>;
</xsl:if><xsl:if test="capacity_reference_object">	capacity_reference_object <a href="#GLM.{capacity_reference_object}"><xsl:value-of select="capacity_reference_object"/></a>;
</xsl:if><xsl:if test="capacity_reference_property">	capacity_reference_property <xsl:value-of select="capacity_reference_property"/>;
</xsl:if><xsl:if test="capacity_reference_bid_price">	capacity_reference_bid_price <xsl:value-of select="capacity_reference_bid_price"/>;
</xsl:if><xsl:if test="max_capacity_reference_bid_quantity">	max_capacity_reference_bid_quantity <xsl:value-of select="max_capacity_reference_bid_quantity"/>;
</xsl:if><xsl:if test="capacity_reference_bid_quantity">	capacity_reference_bid_quantity <xsl:value-of select="capacity_reference_bid_quantity"/>;
</xsl:if><xsl:if test="init_price">	init_price <xsl:value-of select="init_price"/>;
</xsl:if><xsl:if test="init_stdev">	init_stdev <xsl:value-of select="init_stdev"/>;
</xsl:if><xsl:if test="future_mean_price">	future_mean_price <xsl:value-of select="future_mean_price"/>;
</xsl:if><xsl:if test="use_future_mean_price">	use_future_mean_price <xsl:value-of select="use_future_mean_price"/>;
</xsl:if><xsl:if test="current_market.start_time">	current_market.start_time <xsl:value-of select="current_market.start_time"/>;
</xsl:if><xsl:if test="current_market.end_time">	current_market.end_time <xsl:value-of select="current_market.end_time"/>;
</xsl:if><xsl:if test="current_market.clearing_price">	current_market.clearing_price <xsl:value-of select="current_market.clearing_price"/>;
</xsl:if><xsl:if test="current_market.clearing_quantity">	current_market.clearing_quantity <xsl:value-of select="current_market.clearing_quantity"/>;
</xsl:if><xsl:if test="current_market.clearing_type">	current_market.clearing_type <xsl:value-of select="current_market.clearing_type"/>;
</xsl:if><xsl:if test="current_market.marginal_quantity_load">	current_market.marginal_quantity_load <xsl:value-of select="current_market.marginal_quantity_load"/>;
</xsl:if><xsl:if test="current_market.marginal_quantity">	current_market.marginal_quantity <xsl:value-of select="current_market.marginal_quantity"/>;
</xsl:if><xsl:if test="current_market.marginal_quantity_bid">	current_market.marginal_quantity_bid <xsl:value-of select="current_market.marginal_quantity_bid"/>;
</xsl:if><xsl:if test="current_market.marginal_quantity_frac">	current_market.marginal_quantity_frac <xsl:value-of select="current_market.marginal_quantity_frac"/>;
</xsl:if><xsl:if test="current_market.seller_total_quantity">	current_market.seller_total_quantity <xsl:value-of select="current_market.seller_total_quantity"/>;
</xsl:if><xsl:if test="current_market.buyer_total_quantity">	current_market.buyer_total_quantity <xsl:value-of select="current_market.buyer_total_quantity"/>;
</xsl:if><xsl:if test="current_market.seller_min_price">	current_market.seller_min_price <xsl:value-of select="current_market.seller_min_price"/>;
</xsl:if><xsl:if test="current_market.buyer_total_unrep">	current_market.buyer_total_unrep <xsl:value-of select="current_market.buyer_total_unrep"/>;
</xsl:if><xsl:if test="current_market.cap_ref_unrep">	current_market.cap_ref_unrep <xsl:value-of select="current_market.cap_ref_unrep"/>;
</xsl:if><xsl:if test="next_market.start_time">	next_market.start_time <xsl:value-of select="next_market.start_time"/>;
</xsl:if><xsl:if test="next_market.end_time">	next_market.end_time <xsl:value-of select="next_market.end_time"/>;
</xsl:if><xsl:if test="next_market.clearing_price">	next_market.clearing_price <xsl:value-of select="next_market.clearing_price"/>;
</xsl:if><xsl:if test="next_market.clearing_quantity">	next_market.clearing_quantity <xsl:value-of select="next_market.clearing_quantity"/>;
</xsl:if><xsl:if test="next_market.clearing_type">	next_market.clearing_type <xsl:value-of select="next_market.clearing_type"/>;
</xsl:if><xsl:if test="next_market.marginal_quantity_load">	next_market.marginal_quantity_load <xsl:value-of select="next_market.marginal_quantity_load"/>;
</xsl:if><xsl:if test="next_market.marginal_quantity_bid">	next_market.marginal_quantity_bid <xsl:value-of select="next_market.marginal_quantity_bid"/>;
</xsl:if><xsl:if test="next_market.marginal_quantity_frac">	next_market.marginal_quantity_frac <xsl:value-of select="next_market.marginal_quantity_frac"/>;
</xsl:if><xsl:if test="next_market.seller_total_quantity">	next_market.seller_total_quantity <xsl:value-of select="next_market.seller_total_quantity"/>;
</xsl:if><xsl:if test="next_market.buyer_total_quantity">	next_market.buyer_total_quantity <xsl:value-of select="next_market.buyer_total_quantity"/>;
</xsl:if><xsl:if test="next_market.seller_min_price">	next_market.seller_min_price <xsl:value-of select="next_market.seller_min_price"/>;
</xsl:if><xsl:if test="next_market.cap_ref_unrep">	next_market.cap_ref_unrep <xsl:value-of select="next_market.cap_ref_unrep"/>;
</xsl:if><xsl:if test="past_market.start_time">	past_market.start_time <xsl:value-of select="past_market.start_time"/>;
</xsl:if><xsl:if test="past_market.end_time">	past_market.end_time <xsl:value-of select="past_market.end_time"/>;
</xsl:if><xsl:if test="past_market.clearing_price">	past_market.clearing_price <xsl:value-of select="past_market.clearing_price"/>;
</xsl:if><xsl:if test="past_market.clearing_quantity">	past_market.clearing_quantity <xsl:value-of select="past_market.clearing_quantity"/>;
</xsl:if><xsl:if test="past_market.clearing_type">	past_market.clearing_type <xsl:value-of select="past_market.clearing_type"/>;
</xsl:if><xsl:if test="past_market.marginal_quantity_load">	past_market.marginal_quantity_load <xsl:value-of select="past_market.marginal_quantity_load"/>;
</xsl:if><xsl:if test="past_market.marginal_quantity_bid">	past_market.marginal_quantity_bid <xsl:value-of select="past_market.marginal_quantity_bid"/>;
</xsl:if><xsl:if test="past_market.marginal_quantity_frac">	past_market.marginal_quantity_frac <xsl:value-of select="past_market.marginal_quantity_frac"/>;
</xsl:if><xsl:if test="past_market.seller_total_quantity">	past_market.seller_total_quantity <xsl:value-of select="past_market.seller_total_quantity"/>;
</xsl:if><xsl:if test="past_market.buyer_total_quantity">	past_market.buyer_total_quantity <xsl:value-of select="past_market.buyer_total_quantity"/>;
</xsl:if><xsl:if test="past_market.seller_min_price">	past_market.seller_min_price <xsl:value-of select="past_market.seller_min_price"/>;
</xsl:if><xsl:if test="past_market.cap_ref_unrep">	past_market.cap_ref_unrep <xsl:value-of select="past_market.cap_ref_unrep"/>;
</xsl:if><xsl:if test="margin_mode">	margin_mode <xsl:value-of select="margin_mode"/>;
</xsl:if><xsl:if test="warmup">	warmup <xsl:value-of select="warmup"/>;
</xsl:if><xsl:if test="ignore_pricecap">	ignore_pricecap <xsl:value-of select="ignore_pricecap"/>;
</xsl:if><xsl:if test="transaction_log_file">	transaction_log_file <xsl:value-of select="transaction_log_file"/>;
</xsl:if><xsl:if test="transaction_log_limit">	transaction_log_limit <xsl:value-of select="transaction_log_limit"/>;
</xsl:if><xsl:if test="curve_log_file">	curve_log_file <xsl:value-of select="curve_log_file"/>;
</xsl:if><xsl:if test="curve_log_limit">	curve_log_limit <xsl:value-of select="curve_log_limit"/>;
</xsl:if><xsl:if test="curve_log_info">	curve_log_info <xsl:value-of select="curve_log_info"/>;
</xsl:if>}
</xsl:for-each>
# market::controller objects
<xsl:for-each select="controller_list/controller"><a name="#GLM.{name}"/>object controller:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="simple_mode">	simple_mode <xsl:value-of select="simple_mode"/>;
</xsl:if><xsl:if test="bid_mode">	bid_mode <xsl:value-of select="bid_mode"/>;
</xsl:if><xsl:if test="use_override">	use_override <xsl:value-of select="use_override"/>;
</xsl:if><xsl:if test="ramp_low">	ramp_low <xsl:value-of select="ramp_low"/>;
</xsl:if><xsl:if test="ramp_high">	ramp_high <xsl:value-of select="ramp_high"/>;
</xsl:if><xsl:if test="range_low">	range_low <xsl:value-of select="range_low"/>;
</xsl:if><xsl:if test="range_high">	range_high <xsl:value-of select="range_high"/>;
</xsl:if><xsl:if test="target">	target <xsl:value-of select="target"/>;
</xsl:if><xsl:if test="setpoint">	setpoint <xsl:value-of select="setpoint"/>;
</xsl:if><xsl:if test="demand">	demand <xsl:value-of select="demand"/>;
</xsl:if><xsl:if test="load">	load <xsl:value-of select="load"/>;
</xsl:if><xsl:if test="total">	total <xsl:value-of select="total"/>;
</xsl:if><xsl:if test="market">	market <a href="#GLM.{market}"><xsl:value-of select="market"/></a>;
</xsl:if><xsl:if test="state">	state <xsl:value-of select="state"/>;
</xsl:if><xsl:if test="avg_target">	avg_target <xsl:value-of select="avg_target"/>;
</xsl:if><xsl:if test="std_target">	std_target <xsl:value-of select="std_target"/>;
</xsl:if><xsl:if test="bid_price">	bid_price <xsl:value-of select="bid_price"/>;
</xsl:if><xsl:if test="bid_quantity">	bid_quantity <xsl:value-of select="bid_quantity"/>;
</xsl:if><xsl:if test="set_temp">	set_temp <xsl:value-of select="set_temp"/>;
</xsl:if><xsl:if test="base_setpoint">	base_setpoint <xsl:value-of select="base_setpoint"/>;
</xsl:if><xsl:if test="period">	period <xsl:value-of select="period"/>;
</xsl:if><xsl:if test="control_mode">	control_mode <xsl:value-of select="control_mode"/>;
</xsl:if><xsl:if test="resolve_mode">	resolve_mode <xsl:value-of select="resolve_mode"/>;
</xsl:if><xsl:if test="slider_setting">	slider_setting <xsl:value-of select="slider_setting"/>;
</xsl:if><xsl:if test="slider_setting_heat">	slider_setting_heat <xsl:value-of select="slider_setting_heat"/>;
</xsl:if><xsl:if test="slider_setting_cool">	slider_setting_cool <xsl:value-of select="slider_setting_cool"/>;
</xsl:if><xsl:if test="override">	override <xsl:value-of select="override"/>;
</xsl:if><xsl:if test="heating_range_high">	heating_range_high <xsl:value-of select="heating_range_high"/>;
</xsl:if><xsl:if test="heating_range_low">	heating_range_low <xsl:value-of select="heating_range_low"/>;
</xsl:if><xsl:if test="heating_ramp_high">	heating_ramp_high <xsl:value-of select="heating_ramp_high"/>;
</xsl:if><xsl:if test="heating_ramp_low">	heating_ramp_low <xsl:value-of select="heating_ramp_low"/>;
</xsl:if><xsl:if test="cooling_range_high">	cooling_range_high <xsl:value-of select="cooling_range_high"/>;
</xsl:if><xsl:if test="cooling_range_low">	cooling_range_low <xsl:value-of select="cooling_range_low"/>;
</xsl:if><xsl:if test="cooling_ramp_high">	cooling_ramp_high <xsl:value-of select="cooling_ramp_high"/>;
</xsl:if><xsl:if test="cooling_ramp_low">	cooling_ramp_low <xsl:value-of select="cooling_ramp_low"/>;
</xsl:if><xsl:if test="heating_base_setpoint">	heating_base_setpoint <xsl:value-of select="heating_base_setpoint"/>;
</xsl:if><xsl:if test="cooling_base_setpoint">	cooling_base_setpoint <xsl:value-of select="cooling_base_setpoint"/>;
</xsl:if><xsl:if test="deadband">	deadband <xsl:value-of select="deadband"/>;
</xsl:if><xsl:if test="heating_setpoint">	heating_setpoint <xsl:value-of select="heating_setpoint"/>;
</xsl:if><xsl:if test="heating_demand">	heating_demand <xsl:value-of select="heating_demand"/>;
</xsl:if><xsl:if test="cooling_setpoint">	cooling_setpoint <xsl:value-of select="cooling_setpoint"/>;
</xsl:if><xsl:if test="cooling_demand">	cooling_demand <xsl:value-of select="cooling_demand"/>;
</xsl:if><xsl:if test="sliding_time_delay">	sliding_time_delay <xsl:value-of select="sliding_time_delay"/>;
</xsl:if><xsl:if test="average_target">	average_target <xsl:value-of select="average_target"/>;
</xsl:if><xsl:if test="standard_deviation_target">	standard_deviation_target <xsl:value-of select="standard_deviation_target"/>;
</xsl:if><xsl:if test="current_mode">	current_mode <xsl:value-of select="current_mode"/>;
</xsl:if><xsl:if test="dominant_mode">	dominant_mode <xsl:value-of select="dominant_mode"/>;
</xsl:if><xsl:if test="previous_mode">	previous_mode <xsl:value-of select="previous_mode"/>;
</xsl:if><xsl:if test="heat_max">	heat_max <xsl:value-of select="heat_max"/>;
</xsl:if><xsl:if test="cool_min">	cool_min <xsl:value-of select="cool_min"/>;
</xsl:if><xsl:if test="bid_delay">	bid_delay <xsl:value-of select="bid_delay"/>;
</xsl:if>}
</xsl:for-each>
# market::stubauction objects
<xsl:for-each select="stubauction_list/stubauction"><a name="#GLM.{name}"/>object stubauction:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="unit">	unit <xsl:value-of select="unit"/>;
</xsl:if><xsl:if test="period">	period <xsl:value-of select="period"/>;
</xsl:if><xsl:if test="last.P">	last.P <xsl:value-of select="last.P"/>;
</xsl:if><xsl:if test="current_market.clearing_price">	current_market.clearing_price <xsl:value-of select="current_market.clearing_price"/>;
</xsl:if><xsl:if test="past_market.clearing_price">	past_market.clearing_price <xsl:value-of select="past_market.clearing_price"/>;
</xsl:if><xsl:if test="next.P">	next.P <xsl:value-of select="next.P"/>;
</xsl:if><xsl:if test="avg24">	avg24 <xsl:value-of select="avg24"/>;
</xsl:if><xsl:if test="std24">	std24 <xsl:value-of select="std24"/>;
</xsl:if><xsl:if test="avg72">	avg72 <xsl:value-of select="avg72"/>;
</xsl:if><xsl:if test="std72">	std72 <xsl:value-of select="std72"/>;
</xsl:if><xsl:if test="avg168">	avg168 <xsl:value-of select="avg168"/>;
</xsl:if><xsl:if test="std168">	std168 <xsl:value-of select="std168"/>;
</xsl:if><xsl:if test="market_id">	market_id <xsl:value-of select="market_id"/>;
</xsl:if><xsl:if test="verbose">	verbose <xsl:value-of select="verbose"/>;
</xsl:if><xsl:if test="control_mode">	control_mode <xsl:value-of select="control_mode"/>;
</xsl:if>}
</xsl:for-each>
# market::passive_controller objects
<xsl:for-each select="passive_controller_list/passive_controller"><a name="#GLM.{name}"/>object passive_controller:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="input_state">	input_state <xsl:value-of select="input_state"/>;
</xsl:if><xsl:if test="input_setpoint">	input_setpoint <xsl:value-of select="input_setpoint"/>;
</xsl:if><xsl:if test="input_chained">	input_chained <xsl:value-of select="input_chained"/>;
</xsl:if><xsl:if test="observation">	observation <xsl:value-of select="observation"/>;
</xsl:if><xsl:if test="mean_observation">	mean_observation <xsl:value-of select="mean_observation"/>;
</xsl:if><xsl:if test="stdev_observation">	stdev_observation <xsl:value-of select="stdev_observation"/>;
</xsl:if><xsl:if test="expectation">	expectation <xsl:value-of select="expectation"/>;
</xsl:if><xsl:if test="sensitivity">	sensitivity <xsl:value-of select="sensitivity"/>;
</xsl:if><xsl:if test="period">	period <xsl:value-of select="period"/>;
</xsl:if><xsl:if test="expectation_prop">	expectation_prop <xsl:value-of select="expectation_prop"/>;
</xsl:if><xsl:if test="expectation_obj">	expectation_obj <a href="#GLM.{expectation_obj}"><xsl:value-of select="expectation_obj"/></a>;
</xsl:if><xsl:if test="expectation_property">	expectation_property <xsl:value-of select="expectation_property"/>;
</xsl:if><xsl:if test="expectation_object">	expectation_object <a href="#GLM.{expectation_object}"><xsl:value-of select="expectation_object"/></a>;
</xsl:if><xsl:if test="setpoint_prop">	setpoint_prop <xsl:value-of select="setpoint_prop"/>;
</xsl:if><xsl:if test="setpoint">	setpoint <xsl:value-of select="setpoint"/>;
</xsl:if><xsl:if test="state_prop">	state_prop <xsl:value-of select="state_prop"/>;
</xsl:if><xsl:if test="state_property">	state_property <xsl:value-of select="state_property"/>;
</xsl:if><xsl:if test="observation_obj">	observation_obj <a href="#GLM.{observation_obj}"><xsl:value-of select="observation_obj"/></a>;
</xsl:if><xsl:if test="observation_prop">	observation_prop <xsl:value-of select="observation_prop"/>;
</xsl:if><xsl:if test="observation_object">	observation_object <a href="#GLM.{observation_object}"><xsl:value-of select="observation_object"/></a>;
</xsl:if><xsl:if test="observation_property">	observation_property <xsl:value-of select="observation_property"/>;
</xsl:if><xsl:if test="mean_observation_prop">	mean_observation_prop <xsl:value-of select="mean_observation_prop"/>;
</xsl:if><xsl:if test="stdev_observation_prop">	stdev_observation_prop <xsl:value-of select="stdev_observation_prop"/>;
</xsl:if><xsl:if test="stdev_observation_property">	stdev_observation_property <xsl:value-of select="stdev_observation_property"/>;
</xsl:if><xsl:if test="cycle_length">	cycle_length <xsl:value-of select="cycle_length"/>;
</xsl:if><xsl:if test="base_setpoint">	base_setpoint <xsl:value-of select="base_setpoint"/>;
</xsl:if><xsl:if test="critical_day">	critical_day <xsl:value-of select="critical_day"/>;
</xsl:if><xsl:if test="two_tier_cpp">	two_tier_cpp <xsl:value-of select="two_tier_cpp"/>;
</xsl:if><xsl:if test="daily_elasticity">	daily_elasticity <xsl:value-of select="daily_elasticity"/>;
</xsl:if><xsl:if test="sub_elasticity_first_second">	sub_elasticity_first_second <xsl:value-of select="sub_elasticity_first_second"/>;
</xsl:if><xsl:if test="sub_elasticity_first_third">	sub_elasticity_first_third <xsl:value-of select="sub_elasticity_first_third"/>;
</xsl:if><xsl:if test="second_tier_hours">	second_tier_hours <xsl:value-of select="second_tier_hours"/>;
</xsl:if><xsl:if test="third_tier_hours">	third_tier_hours <xsl:value-of select="third_tier_hours"/>;
</xsl:if><xsl:if test="first_tier_hours">	first_tier_hours <xsl:value-of select="first_tier_hours"/>;
</xsl:if><xsl:if test="first_tier_price">	first_tier_price <xsl:value-of select="first_tier_price"/>;
</xsl:if><xsl:if test="second_tier_price">	second_tier_price <xsl:value-of select="second_tier_price"/>;
</xsl:if><xsl:if test="third_tier_price">	third_tier_price <xsl:value-of select="third_tier_price"/>;
</xsl:if><xsl:if test="old_first_tier_price">	old_first_tier_price <xsl:value-of select="old_first_tier_price"/>;
</xsl:if><xsl:if test="old_second_tier_price">	old_second_tier_price <xsl:value-of select="old_second_tier_price"/>;
</xsl:if><xsl:if test="old_third_tier_price">	old_third_tier_price <xsl:value-of select="old_third_tier_price"/>;
</xsl:if><xsl:if test="Percent_change_in_price">	Percent_change_in_price <xsl:value-of select="Percent_change_in_price"/>;
</xsl:if><xsl:if test="Percent_change_in_peakoffpeak_ratio">	Percent_change_in_peakoffpeak_ratio <xsl:value-of select="Percent_change_in_peakoffpeak_ratio"/>;
</xsl:if><xsl:if test="Percent_change_in_Criticalpeakoffpeak_ratio">	Percent_change_in_Criticalpeakoffpeak_ratio <xsl:value-of select="Percent_change_in_Criticalpeakoffpeak_ratio"/>;
</xsl:if><xsl:if test="linearize_elasticity">	linearize_elasticity <xsl:value-of select="linearize_elasticity"/>;
</xsl:if><xsl:if test="price_offset">	price_offset <xsl:value-of select="price_offset"/>;
</xsl:if><xsl:if test="pool_pump_model">	pool_pump_model <xsl:value-of select="pool_pump_model"/>;
</xsl:if><xsl:if test="base_duty_cycle">	base_duty_cycle <xsl:value-of select="base_duty_cycle"/>;
</xsl:if><xsl:if test="distribution_type">	distribution_type <xsl:value-of select="distribution_type"/>;
</xsl:if><xsl:if test="comfort_level">	comfort_level <xsl:value-of select="comfort_level"/>;
</xsl:if><xsl:if test="range_high">	range_high <xsl:value-of select="range_high"/>;
</xsl:if><xsl:if test="range_low">	range_low <xsl:value-of select="range_low"/>;
</xsl:if><xsl:if test="ramp_high">	ramp_high <xsl:value-of select="ramp_high"/>;
</xsl:if><xsl:if test="ramp_low">	ramp_low <xsl:value-of select="ramp_low"/>;
</xsl:if><xsl:if test="prob_off">	prob_off <xsl:value-of select="prob_off"/>;
</xsl:if><xsl:if test="output_state">	output_state <xsl:value-of select="output_state"/>;
</xsl:if><xsl:if test="output_setpoint">	output_setpoint <xsl:value-of select="output_setpoint"/>;
</xsl:if><xsl:if test="control_mode">	control_mode <xsl:value-of select="control_mode"/>;
</xsl:if><xsl:if test="dlc_mode">	dlc_mode <xsl:value-of select="dlc_mode"/>;
</xsl:if><xsl:if test="cycle_length_off">	cycle_length_off <xsl:value-of select="cycle_length_off"/>;
</xsl:if><xsl:if test="cycle_length_on">	cycle_length_on <xsl:value-of select="cycle_length_on"/>;
</xsl:if>}
</xsl:for-each>
# market::double_controller objects
<xsl:for-each select="double_controller_list/double_controller"><a name="#GLM.{name}"/>object double_controller:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="thermostat_mode">	thermostat_mode <xsl:value-of select="thermostat_mode"/>;
</xsl:if><xsl:if test="last_mode">	last_mode <xsl:value-of select="last_mode"/>;
</xsl:if><xsl:if test="resolve_mode">	resolve_mode <xsl:value-of select="resolve_mode"/>;
</xsl:if><xsl:if test="setup_mode">	setup_mode <xsl:value-of select="setup_mode"/>;
</xsl:if><xsl:if test="bid_mode">	bid_mode <xsl:value-of select="bid_mode"/>;
</xsl:if><xsl:if test="last_mode_timer">	last_mode_timer <xsl:value-of select="last_mode_timer"/>;
</xsl:if><xsl:if test="cool_ramp_low">	cool_ramp_low <xsl:value-of select="cool_ramp_low"/>;
</xsl:if><xsl:if test="cool_ramp_high">	cool_ramp_high <xsl:value-of select="cool_ramp_high"/>;
</xsl:if><xsl:if test="cool_range_low">	cool_range_low <xsl:value-of select="cool_range_low"/>;
</xsl:if><xsl:if test="cool_range_high">	cool_range_high <xsl:value-of select="cool_range_high"/>;
</xsl:if><xsl:if test="cool_set_base">	cool_set_base <xsl:value-of select="cool_set_base"/>;
</xsl:if><xsl:if test="cool_setpoint">	cool_setpoint <xsl:value-of select="cool_setpoint"/>;
</xsl:if><xsl:if test="heat_ramp_low">	heat_ramp_low <xsl:value-of select="heat_ramp_low"/>;
</xsl:if><xsl:if test="heat_ramp_high">	heat_ramp_high <xsl:value-of select="heat_ramp_high"/>;
</xsl:if><xsl:if test="heat_range_low">	heat_range_low <xsl:value-of select="heat_range_low"/>;
</xsl:if><xsl:if test="heat_range_high">	heat_range_high <xsl:value-of select="heat_range_high"/>;
</xsl:if><xsl:if test="heat_set_base">	heat_set_base <xsl:value-of select="heat_set_base"/>;
</xsl:if><xsl:if test="heat_setpoint">	heat_setpoint <xsl:value-of select="heat_setpoint"/>;
</xsl:if><xsl:if test="temperature_name">	temperature_name <xsl:value-of select="temperature_name"/>;
</xsl:if><xsl:if test="cool_setpoint_name">	cool_setpoint_name <xsl:value-of select="cool_setpoint_name"/>;
</xsl:if><xsl:if test="cool_demand_name">	cool_demand_name <xsl:value-of select="cool_demand_name"/>;
</xsl:if><xsl:if test="heat_setpoint_name">	heat_setpoint_name <xsl:value-of select="heat_setpoint_name"/>;
</xsl:if><xsl:if test="heat_demand_name">	heat_demand_name <xsl:value-of select="heat_demand_name"/>;
</xsl:if><xsl:if test="load_name">	load_name <xsl:value-of select="load_name"/>;
</xsl:if><xsl:if test="total_load_name">	total_load_name <xsl:value-of select="total_load_name"/>;
</xsl:if><xsl:if test="deadband_name">	deadband_name <xsl:value-of select="deadband_name"/>;
</xsl:if><xsl:if test="state_name">	state_name <xsl:value-of select="state_name"/>;
</xsl:if><xsl:if test="market">	market <a href="#GLM.{market}"><xsl:value-of select="market"/></a>;
</xsl:if><xsl:if test="market_period">	market_period <xsl:value-of select="market_period"/>;
</xsl:if><xsl:if test="bid_price">	bid_price <xsl:value-of select="bid_price"/>;
</xsl:if><xsl:if test="bid_quant">	bid_quant <xsl:value-of select="bid_quant"/>;
</xsl:if><xsl:if test="load">	load <xsl:value-of select="load"/>;
</xsl:if><xsl:if test="total">	total <xsl:value-of select="total"/>;
</xsl:if><xsl:if test="last_price">	last_price <xsl:value-of select="last_price"/>;
</xsl:if><xsl:if test="temperature">	temperature <xsl:value-of select="temperature"/>;
</xsl:if><xsl:if test="cool_bid">	cool_bid <xsl:value-of select="cool_bid"/>;
</xsl:if><xsl:if test="heat_bid">	heat_bid <xsl:value-of select="heat_bid"/>;
</xsl:if><xsl:if test="cool_demand">	cool_demand <xsl:value-of select="cool_demand"/>;
</xsl:if><xsl:if test="heat_demand">	heat_demand <xsl:value-of select="heat_demand"/>;
</xsl:if><xsl:if test="price">	price <xsl:value-of select="price"/>;
</xsl:if><xsl:if test="avg_price">	avg_price <xsl:value-of select="avg_price"/>;
</xsl:if><xsl:if test="stdev_price">	stdev_price <xsl:value-of select="stdev_price"/>;
</xsl:if>}
</xsl:for-each>
# market::stub_bidder objects
<xsl:for-each select="stub_bidder_list/stub_bidder"><a name="#GLM.{name}"/>object stub_bidder:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="bid_period">	bid_period <xsl:value-of select="bid_period"/>;
</xsl:if><xsl:if test="count">	count <xsl:value-of select="count"/>;
</xsl:if><xsl:if test="market">	market <a href="#GLM.{market}"><xsl:value-of select="market"/></a>;
</xsl:if><xsl:if test="role">	role <xsl:value-of select="role"/>;
</xsl:if><xsl:if test="price">	price <xsl:value-of select="price"/>;
</xsl:if><xsl:if test="quantity">	quantity <xsl:value-of select="quantity"/>;
</xsl:if>}
</xsl:for-each>
# market::generator_controller objects
<xsl:for-each select="generator_controller_list/generator_controller"><a name="#GLM.{name}"/>object generator_controller:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="generator_state">	generator_state <xsl:value-of select="generator_state"/>;
</xsl:if><xsl:if test="amortization_type">	amortization_type <xsl:value-of select="amortization_type"/>;
</xsl:if><xsl:if test="generator_state_number">	generator_state_number <xsl:value-of select="generator_state_number"/>;
</xsl:if><xsl:if test="market">	market <a href="#GLM.{market}"><xsl:value-of select="market"/></a>;
</xsl:if><xsl:if test="bid_curve">	bid_curve <xsl:value-of select="bid_curve"/>;
</xsl:if><xsl:if test="bid_curve_file">	bid_curve_file <xsl:value-of select="bid_curve_file"/>;
</xsl:if><xsl:if test="bid_generator_rating">	bid_generator_rating <xsl:value-of select="bid_generator_rating"/>;
</xsl:if><xsl:if test="update_curve">	update_curve <xsl:value-of select="update_curve"/>;
</xsl:if><xsl:if test="is_marginal_gen">	is_marginal_gen <xsl:value-of select="is_marginal_gen"/>;
</xsl:if><xsl:if test="generator_rating">	generator_rating <xsl:value-of select="generator_rating"/>;
</xsl:if><xsl:if test="generator_output">	generator_output <xsl:value-of select="generator_output"/>;
</xsl:if><xsl:if test="input_unit_base">	input_unit_base <xsl:value-of select="input_unit_base"/>;
</xsl:if><xsl:if test="startup_cost">	startup_cost <xsl:value-of select="startup_cost"/>;
</xsl:if><xsl:if test="shutdown_cost">	shutdown_cost <xsl:value-of select="shutdown_cost"/>;
</xsl:if><xsl:if test="minimum_runtime">	minimum_runtime <xsl:value-of select="minimum_runtime"/>;
</xsl:if><xsl:if test="minimum_downtime">	minimum_downtime <xsl:value-of select="minimum_downtime"/>;
</xsl:if><xsl:if test="capacity_factor">	capacity_factor <xsl:value-of select="capacity_factor"/>;
</xsl:if><xsl:if test="amortization_factor">	amortization_factor <xsl:value-of select="amortization_factor"/>;
</xsl:if><xsl:if test="bid_delay">	bid_delay <xsl:value-of select="bid_delay"/>;
</xsl:if><xsl:if test="generator_attachment">	generator_attachment <xsl:value-of select="generator_attachment"/>;
</xsl:if><xsl:if test="building_load_curr">	building_load_curr <xsl:value-of select="building_load_curr"/>;
</xsl:if><xsl:if test="building_load_bid">	building_load_bid <xsl:value-of select="building_load_bid"/>;
</xsl:if><xsl:if test="year_runtime_limit">	year_runtime_limit <xsl:value-of select="year_runtime_limit"/>;
</xsl:if><xsl:if test="current_year_runtime">	current_year_runtime <xsl:value-of select="current_year_runtime"/>;
</xsl:if><xsl:if test="runtime_year_end">	runtime_year_end <xsl:value-of select="runtime_year_end"/>;
</xsl:if><xsl:if test="scaling_factor">	scaling_factor <xsl:value-of select="scaling_factor"/>;
</xsl:if><xsl:if test="license_premium">	license_premium <xsl:value-of select="license_premium"/>;
</xsl:if><xsl:if test="hours_in_year">	hours_in_year <xsl:value-of select="hours_in_year"/>;
</xsl:if><xsl:if test="op_and_maint_cost">	op_and_maint_cost <xsl:value-of select="op_and_maint_cost"/>;
</xsl:if>}
</xsl:for-each></xsl:for-each><xsl:for-each select="plc">
##############################################
# plc module
module plc {
	version.major <xsl:value-of select="version.major"/>;
	version.minor <xsl:value-of select="version.minor"/>;
	libpath <xsl:value-of select="libpath"/>;
	incpath <xsl:value-of select="incpath"/>;
}

# plc::plc objects
<xsl:for-each select="plc_list/plc"><a name="#GLM.{name}"/>object plc:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="source">	source <xsl:value-of select="source"/>;
</xsl:if><xsl:if test="network">	network <a href="#GLM.{network}"><xsl:value-of select="network"/></a>;
</xsl:if>}
</xsl:for-each>
# plc::comm objects
<xsl:for-each select="comm_list/comm"><a name="#GLM.{name}"/>object comm:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="latency">	latency <xsl:value-of select="latency"/>;
</xsl:if><xsl:if test="reliability">	reliability <xsl:value-of select="reliability"/>;
</xsl:if><xsl:if test="bitrate">	bitrate <xsl:value-of select="bitrate"/>;
</xsl:if><xsl:if test="timeout">	timeout <xsl:value-of select="timeout"/>;
</xsl:if>}
</xsl:for-each></xsl:for-each><xsl:for-each select="powerflow">
##############################################
# powerflow module
module powerflow {
	version.major <xsl:value-of select="version.major"/>;
	version.minor <xsl:value-of select="version.minor"/>;
	show_matrix_values <xsl:value-of select="show_matrix_values"/>;
	primary_voltage_ratio <xsl:value-of select="primary_voltage_ratio"/>;
	nominal_frequency <xsl:value-of select="nominal_frequency"/>;
	require_voltage_control <xsl:value-of select="require_voltage_control"/>;
	geographic_degree <xsl:value-of select="geographic_degree"/>;
	fault_impedance <xsl:value-of select="fault_impedance"/>;
	warning_underfrequency <xsl:value-of select="warning_underfrequency"/>;
	warning_overfrequency <xsl:value-of select="warning_overfrequency"/>;
	warning_undervoltage <xsl:value-of select="warning_undervoltage"/>;
	warning_overvoltage <xsl:value-of select="warning_overvoltage"/>;
	warning_voltageangle <xsl:value-of select="warning_voltageangle"/>;
	maximum_voltage_error <xsl:value-of select="maximum_voltage_error"/>;
	solver_method <xsl:value-of select="solver_method"/>;
	line_capacitance <xsl:value-of select="line_capacitance"/>;
	line_limits <xsl:value-of select="line_limits"/>;
	lu_solver <xsl:value-of select="lu_solver"/>;
	acceleration_factor <xsl:value-of select="acceleration_factor"/>;
	NR_iteration_limit <xsl:value-of select="NR_iteration_limit"/>;
	NR_superLU_procs <xsl:value-of select="NR_superLU_procs"/>;
	default_maximum_voltage_error <xsl:value-of select="default_maximum_voltage_error"/>;
}

# powerflow::node objects
<xsl:for-each select="node_list/node"><a name="#GLM.{name}"/>object node:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="bustype">	bustype <xsl:value-of select="bustype"/>;
</xsl:if><xsl:if test="busflags">	busflags <xsl:value-of select="busflags"/>;
</xsl:if><xsl:if test="reference_bus">	reference_bus <a href="#GLM.{reference_bus}"><xsl:value-of select="reference_bus"/></a>;
</xsl:if><xsl:if test="maximum_voltage_error">	maximum_voltage_error <xsl:value-of select="maximum_voltage_error"/>;
</xsl:if><xsl:if test="voltage_A">	voltage_A <xsl:value-of select="voltage_A"/>;
</xsl:if><xsl:if test="voltage_B">	voltage_B <xsl:value-of select="voltage_B"/>;
</xsl:if><xsl:if test="voltage_C">	voltage_C <xsl:value-of select="voltage_C"/>;
</xsl:if><xsl:if test="voltage_AB">	voltage_AB <xsl:value-of select="voltage_AB"/>;
</xsl:if><xsl:if test="voltage_BC">	voltage_BC <xsl:value-of select="voltage_BC"/>;
</xsl:if><xsl:if test="voltage_CA">	voltage_CA <xsl:value-of select="voltage_CA"/>;
</xsl:if><xsl:if test="current_A">	current_A <xsl:value-of select="current_A"/>;
</xsl:if><xsl:if test="current_B">	current_B <xsl:value-of select="current_B"/>;
</xsl:if><xsl:if test="current_C">	current_C <xsl:value-of select="current_C"/>;
</xsl:if><xsl:if test="power_A">	power_A <xsl:value-of select="power_A"/>;
</xsl:if><xsl:if test="power_B">	power_B <xsl:value-of select="power_B"/>;
</xsl:if><xsl:if test="power_C">	power_C <xsl:value-of select="power_C"/>;
</xsl:if><xsl:if test="shunt_A">	shunt_A <xsl:value-of select="shunt_A"/>;
</xsl:if><xsl:if test="shunt_B">	shunt_B <xsl:value-of select="shunt_B"/>;
</xsl:if><xsl:if test="shunt_C">	shunt_C <xsl:value-of select="shunt_C"/>;
</xsl:if><xsl:if test="NR_mode">	NR_mode <xsl:value-of select="NR_mode"/>;
</xsl:if><xsl:if test="mean_repair_time">	mean_repair_time <xsl:value-of select="mean_repair_time"/>;
</xsl:if><xsl:if test="service_status">	service_status <xsl:value-of select="service_status"/>;
</xsl:if><xsl:if test="service_status_double">	service_status_double <xsl:value-of select="service_status_double"/>;
</xsl:if><xsl:if test="previous_uptime">	previous_uptime <xsl:value-of select="previous_uptime"/>;
</xsl:if><xsl:if test="current_uptime">	current_uptime <xsl:value-of select="current_uptime"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::link objects
<xsl:for-each select="link_list/link"><a name="#GLM.{name}"/>object link:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="status">	status <xsl:value-of select="status"/>;
</xsl:if><xsl:if test="from">	from <a href="#GLM.{from}"><xsl:value-of select="from"/></a>;
</xsl:if><xsl:if test="to">	to <a href="#GLM.{to}"><xsl:value-of select="to"/></a>;
</xsl:if><xsl:if test="power_in">	power_in <xsl:value-of select="power_in"/>;
</xsl:if><xsl:if test="power_out">	power_out <xsl:value-of select="power_out"/>;
</xsl:if><xsl:if test="power_out_real">	power_out_real <xsl:value-of select="power_out_real"/>;
</xsl:if><xsl:if test="power_losses">	power_losses <xsl:value-of select="power_losses"/>;
</xsl:if><xsl:if test="power_in_A">	power_in_A <xsl:value-of select="power_in_A"/>;
</xsl:if><xsl:if test="power_in_B">	power_in_B <xsl:value-of select="power_in_B"/>;
</xsl:if><xsl:if test="power_in_C">	power_in_C <xsl:value-of select="power_in_C"/>;
</xsl:if><xsl:if test="power_out_A">	power_out_A <xsl:value-of select="power_out_A"/>;
</xsl:if><xsl:if test="power_out_B">	power_out_B <xsl:value-of select="power_out_B"/>;
</xsl:if><xsl:if test="power_out_C">	power_out_C <xsl:value-of select="power_out_C"/>;
</xsl:if><xsl:if test="power_losses_A">	power_losses_A <xsl:value-of select="power_losses_A"/>;
</xsl:if><xsl:if test="power_losses_B">	power_losses_B <xsl:value-of select="power_losses_B"/>;
</xsl:if><xsl:if test="power_losses_C">	power_losses_C <xsl:value-of select="power_losses_C"/>;
</xsl:if><xsl:if test="current_out_A">	current_out_A <xsl:value-of select="current_out_A"/>;
</xsl:if><xsl:if test="current_out_B">	current_out_B <xsl:value-of select="current_out_B"/>;
</xsl:if><xsl:if test="current_out_C">	current_out_C <xsl:value-of select="current_out_C"/>;
</xsl:if><xsl:if test="current_in_A">	current_in_A <xsl:value-of select="current_in_A"/>;
</xsl:if><xsl:if test="current_in_B">	current_in_B <xsl:value-of select="current_in_B"/>;
</xsl:if><xsl:if test="current_in_C">	current_in_C <xsl:value-of select="current_in_C"/>;
</xsl:if><xsl:if test="fault_current_in_A">	fault_current_in_A <xsl:value-of select="fault_current_in_A"/>;
</xsl:if><xsl:if test="fault_current_in_B">	fault_current_in_B <xsl:value-of select="fault_current_in_B"/>;
</xsl:if><xsl:if test="fault_current_in_C">	fault_current_in_C <xsl:value-of select="fault_current_in_C"/>;
</xsl:if><xsl:if test="fault_current_out_A">	fault_current_out_A <xsl:value-of select="fault_current_out_A"/>;
</xsl:if><xsl:if test="fault_current_out_B">	fault_current_out_B <xsl:value-of select="fault_current_out_B"/>;
</xsl:if><xsl:if test="fault_current_out_C">	fault_current_out_C <xsl:value-of select="fault_current_out_C"/>;
</xsl:if><xsl:if test="flow_direction">	flow_direction <xsl:value-of select="flow_direction"/>;
</xsl:if><xsl:if test="mean_repair_time">	mean_repair_time <xsl:value-of select="mean_repair_time"/>;
</xsl:if><xsl:if test="continuous_rating">	continuous_rating <xsl:value-of select="continuous_rating"/>;
</xsl:if><xsl:if test="emergency_rating">	emergency_rating <xsl:value-of select="emergency_rating"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::capacitor objects
<xsl:for-each select="capacitor_list/capacitor"><a name="#GLM.{name}"/>object capacitor:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="pt_phase">	pt_phase <xsl:value-of select="pt_phase"/>;
</xsl:if><xsl:if test="phases_connected">	phases_connected <xsl:value-of select="phases_connected"/>;
</xsl:if><xsl:if test="switchA">	switchA <xsl:value-of select="switchA"/>;
</xsl:if><xsl:if test="switchB">	switchB <xsl:value-of select="switchB"/>;
</xsl:if><xsl:if test="switchC">	switchC <xsl:value-of select="switchC"/>;
</xsl:if><xsl:if test="control">	control <xsl:value-of select="control"/>;
</xsl:if><xsl:if test="cap_A_switch_count">	cap_A_switch_count <xsl:value-of select="cap_A_switch_count"/>;
</xsl:if><xsl:if test="cap_B_switch_count">	cap_B_switch_count <xsl:value-of select="cap_B_switch_count"/>;
</xsl:if><xsl:if test="cap_C_switch_count">	cap_C_switch_count <xsl:value-of select="cap_C_switch_count"/>;
</xsl:if><xsl:if test="voltage_set_high">	voltage_set_high <xsl:value-of select="voltage_set_high"/>;
</xsl:if><xsl:if test="voltage_set_low">	voltage_set_low <xsl:value-of select="voltage_set_low"/>;
</xsl:if><xsl:if test="VAr_set_high">	VAr_set_high <xsl:value-of select="VAr_set_high"/>;
</xsl:if><xsl:if test="VAr_set_low">	VAr_set_low <xsl:value-of select="VAr_set_low"/>;
</xsl:if><xsl:if test="current_set_low">	current_set_low <xsl:value-of select="current_set_low"/>;
</xsl:if><xsl:if test="current_set_high">	current_set_high <xsl:value-of select="current_set_high"/>;
</xsl:if><xsl:if test="capacitor_A">	capacitor_A <xsl:value-of select="capacitor_A"/>;
</xsl:if><xsl:if test="capacitor_B">	capacitor_B <xsl:value-of select="capacitor_B"/>;
</xsl:if><xsl:if test="capacitor_C">	capacitor_C <xsl:value-of select="capacitor_C"/>;
</xsl:if><xsl:if test="cap_nominal_voltage">	cap_nominal_voltage <xsl:value-of select="cap_nominal_voltage"/>;
</xsl:if><xsl:if test="time_delay">	time_delay <xsl:value-of select="time_delay"/>;
</xsl:if><xsl:if test="dwell_time">	dwell_time <xsl:value-of select="dwell_time"/>;
</xsl:if><xsl:if test="lockout_time">	lockout_time <xsl:value-of select="lockout_time"/>;
</xsl:if><xsl:if test="remote_sense">	remote_sense <a href="#GLM.{remote_sense}"><xsl:value-of select="remote_sense"/></a>;
</xsl:if><xsl:if test="remote_sense_B">	remote_sense_B <a href="#GLM.{remote_sense_B}"><xsl:value-of select="remote_sense_B"/></a>;
</xsl:if><xsl:if test="control_level">	control_level <xsl:value-of select="control_level"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::fuse objects
<xsl:for-each select="fuse_list/fuse"><a name="#GLM.{name}"/>object fuse:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="phase_A_status">	phase_A_status <xsl:value-of select="phase_A_status"/>;
</xsl:if><xsl:if test="phase_B_status">	phase_B_status <xsl:value-of select="phase_B_status"/>;
</xsl:if><xsl:if test="phase_C_status">	phase_C_status <xsl:value-of select="phase_C_status"/>;
</xsl:if><xsl:if test="repair_dist_type">	repair_dist_type <xsl:value-of select="repair_dist_type"/>;
</xsl:if><xsl:if test="current_limit">	current_limit <xsl:value-of select="current_limit"/>;
</xsl:if><xsl:if test="mean_replacement_time">	mean_replacement_time <xsl:value-of select="mean_replacement_time"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::meter objects
<xsl:for-each select="meter_list/meter"><a name="#GLM.{name}"/>object meter:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="measured_real_energy">	measured_real_energy <xsl:value-of select="measured_real_energy"/>;
</xsl:if><xsl:if test="measured_reactive_energy">	measured_reactive_energy <xsl:value-of select="measured_reactive_energy"/>;
</xsl:if><xsl:if test="measured_power">	measured_power <xsl:value-of select="measured_power"/>;
</xsl:if><xsl:if test="measured_power_A">	measured_power_A <xsl:value-of select="measured_power_A"/>;
</xsl:if><xsl:if test="measured_power_B">	measured_power_B <xsl:value-of select="measured_power_B"/>;
</xsl:if><xsl:if test="measured_power_C">	measured_power_C <xsl:value-of select="measured_power_C"/>;
</xsl:if><xsl:if test="measured_demand">	measured_demand <xsl:value-of select="measured_demand"/>;
</xsl:if><xsl:if test="measured_real_power">	measured_real_power <xsl:value-of select="measured_real_power"/>;
</xsl:if><xsl:if test="measured_reactive_power">	measured_reactive_power <xsl:value-of select="measured_reactive_power"/>;
</xsl:if><xsl:if test="meter_power_consumption">	meter_power_consumption <xsl:value-of select="meter_power_consumption"/>;
</xsl:if><xsl:if test="measured_voltage_A">	measured_voltage_A <xsl:value-of select="measured_voltage_A"/>;
</xsl:if><xsl:if test="measured_voltage_B">	measured_voltage_B <xsl:value-of select="measured_voltage_B"/>;
</xsl:if><xsl:if test="measured_voltage_C">	measured_voltage_C <xsl:value-of select="measured_voltage_C"/>;
</xsl:if><xsl:if test="measured_voltage_AB">	measured_voltage_AB <xsl:value-of select="measured_voltage_AB"/>;
</xsl:if><xsl:if test="measured_voltage_BC">	measured_voltage_BC <xsl:value-of select="measured_voltage_BC"/>;
</xsl:if><xsl:if test="measured_voltage_CA">	measured_voltage_CA <xsl:value-of select="measured_voltage_CA"/>;
</xsl:if><xsl:if test="measured_current_A">	measured_current_A <xsl:value-of select="measured_current_A"/>;
</xsl:if><xsl:if test="measured_current_B">	measured_current_B <xsl:value-of select="measured_current_B"/>;
</xsl:if><xsl:if test="measured_current_C">	measured_current_C <xsl:value-of select="measured_current_C"/>;
</xsl:if><xsl:if test="customer_interrupted">	customer_interrupted <xsl:value-of select="customer_interrupted"/>;
</xsl:if><xsl:if test="customer_interrupted_secondary">	customer_interrupted_secondary <xsl:value-of select="customer_interrupted_secondary"/>;
</xsl:if><xsl:if test="monthly_bill">	monthly_bill <xsl:value-of select="monthly_bill"/>;
</xsl:if><xsl:if test="previous_monthly_bill">	previous_monthly_bill <xsl:value-of select="previous_monthly_bill"/>;
</xsl:if><xsl:if test="previous_monthly_energy">	previous_monthly_energy <xsl:value-of select="previous_monthly_energy"/>;
</xsl:if><xsl:if test="monthly_fee">	monthly_fee <xsl:value-of select="monthly_fee"/>;
</xsl:if><xsl:if test="monthly_energy">	monthly_energy <xsl:value-of select="monthly_energy"/>;
</xsl:if><xsl:if test="bill_mode">	bill_mode <xsl:value-of select="bill_mode"/>;
</xsl:if><xsl:if test="power_market">	power_market <a href="#GLM.{power_market}"><xsl:value-of select="power_market"/></a>;
</xsl:if><xsl:if test="bill_day">	bill_day <xsl:value-of select="bill_day"/>;
</xsl:if><xsl:if test="price">	price <xsl:value-of select="price"/>;
</xsl:if><xsl:if test="price_base">	price_base <xsl:value-of select="price_base"/>;
</xsl:if><xsl:if test="first_tier_price">	first_tier_price <xsl:value-of select="first_tier_price"/>;
</xsl:if><xsl:if test="first_tier_energy">	first_tier_energy <xsl:value-of select="first_tier_energy"/>;
</xsl:if><xsl:if test="second_tier_price">	second_tier_price <xsl:value-of select="second_tier_price"/>;
</xsl:if><xsl:if test="second_tier_energy">	second_tier_energy <xsl:value-of select="second_tier_energy"/>;
</xsl:if><xsl:if test="third_tier_price">	third_tier_price <xsl:value-of select="third_tier_price"/>;
</xsl:if><xsl:if test="third_tier_energy">	third_tier_energy <xsl:value-of select="third_tier_energy"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::line objects
<xsl:for-each select="line_list/line"><a name="#GLM.{name}"/>object line:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="configuration">	configuration <a href="#GLM.{configuration}"><xsl:value-of select="configuration"/></a>;
</xsl:if><xsl:if test="length">	length <xsl:value-of select="length"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::line_spacing objects
<xsl:for-each select="line_spacing_list/line_spacing"><a name="#GLM.{name}"/>object line_spacing:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="distance_AB">	distance_AB <xsl:value-of select="distance_AB"/>;
</xsl:if><xsl:if test="distance_BC">	distance_BC <xsl:value-of select="distance_BC"/>;
</xsl:if><xsl:if test="distance_AC">	distance_AC <xsl:value-of select="distance_AC"/>;
</xsl:if><xsl:if test="distance_AN">	distance_AN <xsl:value-of select="distance_AN"/>;
</xsl:if><xsl:if test="distance_BN">	distance_BN <xsl:value-of select="distance_BN"/>;
</xsl:if><xsl:if test="distance_CN">	distance_CN <xsl:value-of select="distance_CN"/>;
</xsl:if><xsl:if test="distance_AE">	distance_AE <xsl:value-of select="distance_AE"/>;
</xsl:if><xsl:if test="distance_BE">	distance_BE <xsl:value-of select="distance_BE"/>;
</xsl:if><xsl:if test="distance_CE">	distance_CE <xsl:value-of select="distance_CE"/>;
</xsl:if><xsl:if test="distance_NE">	distance_NE <xsl:value-of select="distance_NE"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::overhead_line objects
<xsl:for-each select="overhead_line_list/overhead_line"><a name="#GLM.{name}"/>object overhead_line:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::underground_line objects
<xsl:for-each select="underground_line_list/underground_line"><a name="#GLM.{name}"/>object underground_line:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::overhead_line_conductor objects
<xsl:for-each select="overhead_line_conductor_list/overhead_line_conductor"><a name="#GLM.{name}"/>object overhead_line_conductor:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="geometric_mean_radius">	geometric_mean_radius <xsl:value-of select="geometric_mean_radius"/>;
</xsl:if><xsl:if test="resistance">	resistance <xsl:value-of select="resistance"/>;
</xsl:if><xsl:if test="diameter">	diameter <xsl:value-of select="diameter"/>;
</xsl:if><xsl:if test="rating.summer.continuous">	rating.summer.continuous <xsl:value-of select="rating.summer.continuous"/>;
</xsl:if><xsl:if test="rating.summer.emergency">	rating.summer.emergency <xsl:value-of select="rating.summer.emergency"/>;
</xsl:if><xsl:if test="rating.winter.continuous">	rating.winter.continuous <xsl:value-of select="rating.winter.continuous"/>;
</xsl:if><xsl:if test="rating.winter.emergency">	rating.winter.emergency <xsl:value-of select="rating.winter.emergency"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::underground_line_conductor objects
<xsl:for-each select="underground_line_conductor_list/underground_line_conductor"><a name="#GLM.{name}"/>object underground_line_conductor:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="outer_diameter">	outer_diameter <xsl:value-of select="outer_diameter"/>;
</xsl:if><xsl:if test="conductor_gmr">	conductor_gmr <xsl:value-of select="conductor_gmr"/>;
</xsl:if><xsl:if test="conductor_diameter">	conductor_diameter <xsl:value-of select="conductor_diameter"/>;
</xsl:if><xsl:if test="conductor_resistance">	conductor_resistance <xsl:value-of select="conductor_resistance"/>;
</xsl:if><xsl:if test="neutral_gmr">	neutral_gmr <xsl:value-of select="neutral_gmr"/>;
</xsl:if><xsl:if test="neutral_diameter">	neutral_diameter <xsl:value-of select="neutral_diameter"/>;
</xsl:if><xsl:if test="neutral_resistance">	neutral_resistance <xsl:value-of select="neutral_resistance"/>;
</xsl:if><xsl:if test="neutral_strands">	neutral_strands <xsl:value-of select="neutral_strands"/>;
</xsl:if><xsl:if test="insulation_relative_permitivitty">	insulation_relative_permitivitty <xsl:value-of select="insulation_relative_permitivitty"/>;
</xsl:if><xsl:if test="shield_gmr">	shield_gmr <xsl:value-of select="shield_gmr"/>;
</xsl:if><xsl:if test="shield_resistance">	shield_resistance <xsl:value-of select="shield_resistance"/>;
</xsl:if><xsl:if test="rating.summer.continuous">	rating.summer.continuous <xsl:value-of select="rating.summer.continuous"/>;
</xsl:if><xsl:if test="rating.summer.emergency">	rating.summer.emergency <xsl:value-of select="rating.summer.emergency"/>;
</xsl:if><xsl:if test="rating.winter.continuous">	rating.winter.continuous <xsl:value-of select="rating.winter.continuous"/>;
</xsl:if><xsl:if test="rating.winter.emergency">	rating.winter.emergency <xsl:value-of select="rating.winter.emergency"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::line_configuration objects
<xsl:for-each select="line_configuration_list/line_configuration"><a name="#GLM.{name}"/>object line_configuration:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="conductor_A">	conductor_A <a href="#GLM.{conductor_A}"><xsl:value-of select="conductor_A"/></a>;
</xsl:if><xsl:if test="conductor_B">	conductor_B <a href="#GLM.{conductor_B}"><xsl:value-of select="conductor_B"/></a>;
</xsl:if><xsl:if test="conductor_C">	conductor_C <a href="#GLM.{conductor_C}"><xsl:value-of select="conductor_C"/></a>;
</xsl:if><xsl:if test="conductor_N">	conductor_N <a href="#GLM.{conductor_N}"><xsl:value-of select="conductor_N"/></a>;
</xsl:if><xsl:if test="spacing">	spacing <a href="#GLM.{spacing}"><xsl:value-of select="spacing"/></a>;
</xsl:if><xsl:if test="z11">	z11 <xsl:value-of select="z11"/>;
</xsl:if><xsl:if test="z12">	z12 <xsl:value-of select="z12"/>;
</xsl:if><xsl:if test="z13">	z13 <xsl:value-of select="z13"/>;
</xsl:if><xsl:if test="z21">	z21 <xsl:value-of select="z21"/>;
</xsl:if><xsl:if test="z22">	z22 <xsl:value-of select="z22"/>;
</xsl:if><xsl:if test="z23">	z23 <xsl:value-of select="z23"/>;
</xsl:if><xsl:if test="z31">	z31 <xsl:value-of select="z31"/>;
</xsl:if><xsl:if test="z32">	z32 <xsl:value-of select="z32"/>;
</xsl:if><xsl:if test="z33">	z33 <xsl:value-of select="z33"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::relay objects
<xsl:for-each select="relay_list/relay"><a name="#GLM.{name}"/>object relay:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="time_to_change">	time_to_change <xsl:value-of select="time_to_change"/>;
</xsl:if><xsl:if test="recloser_delay">	recloser_delay <xsl:value-of select="recloser_delay"/>;
</xsl:if><xsl:if test="recloser_tries">	recloser_tries <xsl:value-of select="recloser_tries"/>;
</xsl:if><xsl:if test="recloser_limit">	recloser_limit <xsl:value-of select="recloser_limit"/>;
</xsl:if><xsl:if test="recloser_event">	recloser_event <xsl:value-of select="recloser_event"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::transformer_configuration objects
<xsl:for-each select="transformer_configuration_list/transformer_configuration"><a name="#GLM.{name}"/>object transformer_configuration:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="connect_type">	connect_type <xsl:value-of select="connect_type"/>;
</xsl:if><xsl:if test="install_type">	install_type <xsl:value-of select="install_type"/>;
</xsl:if><xsl:if test="coolant_type">	coolant_type <xsl:value-of select="coolant_type"/>;
</xsl:if><xsl:if test="cooling_type">	cooling_type <xsl:value-of select="cooling_type"/>;
</xsl:if><xsl:if test="primary_voltage">	primary_voltage <xsl:value-of select="primary_voltage"/>;
</xsl:if><xsl:if test="secondary_voltage">	secondary_voltage <xsl:value-of select="secondary_voltage"/>;
</xsl:if><xsl:if test="power_rating">	power_rating <xsl:value-of select="power_rating"/>;
</xsl:if><xsl:if test="powerA_rating">	powerA_rating <xsl:value-of select="powerA_rating"/>;
</xsl:if><xsl:if test="powerB_rating">	powerB_rating <xsl:value-of select="powerB_rating"/>;
</xsl:if><xsl:if test="powerC_rating">	powerC_rating <xsl:value-of select="powerC_rating"/>;
</xsl:if><xsl:if test="resistance">	resistance <xsl:value-of select="resistance"/>;
</xsl:if><xsl:if test="reactance">	reactance <xsl:value-of select="reactance"/>;
</xsl:if><xsl:if test="impedance">	impedance <xsl:value-of select="impedance"/>;
</xsl:if><xsl:if test="resistance1">	resistance1 <xsl:value-of select="resistance1"/>;
</xsl:if><xsl:if test="reactance1">	reactance1 <xsl:value-of select="reactance1"/>;
</xsl:if><xsl:if test="impedance1">	impedance1 <xsl:value-of select="impedance1"/>;
</xsl:if><xsl:if test="resistance2">	resistance2 <xsl:value-of select="resistance2"/>;
</xsl:if><xsl:if test="reactance2">	reactance2 <xsl:value-of select="reactance2"/>;
</xsl:if><xsl:if test="impedance2">	impedance2 <xsl:value-of select="impedance2"/>;
</xsl:if><xsl:if test="shunt_resistance">	shunt_resistance <xsl:value-of select="shunt_resistance"/>;
</xsl:if><xsl:if test="shunt_reactance">	shunt_reactance <xsl:value-of select="shunt_reactance"/>;
</xsl:if><xsl:if test="shunt_impedance">	shunt_impedance <xsl:value-of select="shunt_impedance"/>;
</xsl:if><xsl:if test="core_coil_weight">	core_coil_weight <xsl:value-of select="core_coil_weight"/>;
</xsl:if><xsl:if test="tank_fittings_weight">	tank_fittings_weight <xsl:value-of select="tank_fittings_weight"/>;
</xsl:if><xsl:if test="oil_volume">	oil_volume <xsl:value-of select="oil_volume"/>;
</xsl:if><xsl:if test="rated_winding_time_constant">	rated_winding_time_constant <xsl:value-of select="rated_winding_time_constant"/>;
</xsl:if><xsl:if test="rated_winding_hot_spot_rise">	rated_winding_hot_spot_rise <xsl:value-of select="rated_winding_hot_spot_rise"/>;
</xsl:if><xsl:if test="rated_top_oil_rise">	rated_top_oil_rise <xsl:value-of select="rated_top_oil_rise"/>;
</xsl:if><xsl:if test="no_load_loss">	no_load_loss <xsl:value-of select="no_load_loss"/>;
</xsl:if><xsl:if test="full_load_loss">	full_load_loss <xsl:value-of select="full_load_loss"/>;
</xsl:if><xsl:if test="reactance_resistance_ratio">	reactance_resistance_ratio <xsl:value-of select="reactance_resistance_ratio"/>;
</xsl:if><xsl:if test="installed_insulation_life">	installed_insulation_life <xsl:value-of select="installed_insulation_life"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::transformer objects
<xsl:for-each select="transformer_list/transformer"><a name="#GLM.{name}"/>object transformer:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="configuration">	configuration <a href="#GLM.{configuration}"><xsl:value-of select="configuration"/></a>;
</xsl:if><xsl:if test="climate">	climate <a href="#GLM.{climate}"><xsl:value-of select="climate"/></a>;
</xsl:if><xsl:if test="ambient_temperature">	ambient_temperature <xsl:value-of select="ambient_temperature"/>;
</xsl:if><xsl:if test="top_oil_hot_spot_temperature">	top_oil_hot_spot_temperature <xsl:value-of select="top_oil_hot_spot_temperature"/>;
</xsl:if><xsl:if test="winding_hot_spot_temperature">	winding_hot_spot_temperature <xsl:value-of select="winding_hot_spot_temperature"/>;
</xsl:if><xsl:if test="percent_loss_of_life">	percent_loss_of_life <xsl:value-of select="percent_loss_of_life"/>;
</xsl:if><xsl:if test="aging_constant">	aging_constant <xsl:value-of select="aging_constant"/>;
</xsl:if><xsl:if test="use_thermal_model">	use_thermal_model <xsl:value-of select="use_thermal_model"/>;
</xsl:if><xsl:if test="transformer_replacement_count">	transformer_replacement_count <xsl:value-of select="transformer_replacement_count"/>;
</xsl:if><xsl:if test="aging_granularity">	aging_granularity <xsl:value-of select="aging_granularity"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::load objects
<xsl:for-each select="load_list/load"><a name="#GLM.{name}"/>object load:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="load_class">	load_class <xsl:value-of select="load_class"/>;
</xsl:if><xsl:if test="constant_power_A">	constant_power_A <xsl:value-of select="constant_power_A"/>;
</xsl:if><xsl:if test="constant_power_B">	constant_power_B <xsl:value-of select="constant_power_B"/>;
</xsl:if><xsl:if test="constant_power_C">	constant_power_C <xsl:value-of select="constant_power_C"/>;
</xsl:if><xsl:if test="constant_power_A_real">	constant_power_A_real <xsl:value-of select="constant_power_A_real"/>;
</xsl:if><xsl:if test="constant_power_B_real">	constant_power_B_real <xsl:value-of select="constant_power_B_real"/>;
</xsl:if><xsl:if test="constant_power_C_real">	constant_power_C_real <xsl:value-of select="constant_power_C_real"/>;
</xsl:if><xsl:if test="constant_power_A_reac">	constant_power_A_reac <xsl:value-of select="constant_power_A_reac"/>;
</xsl:if><xsl:if test="constant_power_B_reac">	constant_power_B_reac <xsl:value-of select="constant_power_B_reac"/>;
</xsl:if><xsl:if test="constant_power_C_reac">	constant_power_C_reac <xsl:value-of select="constant_power_C_reac"/>;
</xsl:if><xsl:if test="constant_current_A">	constant_current_A <xsl:value-of select="constant_current_A"/>;
</xsl:if><xsl:if test="constant_current_B">	constant_current_B <xsl:value-of select="constant_current_B"/>;
</xsl:if><xsl:if test="constant_current_C">	constant_current_C <xsl:value-of select="constant_current_C"/>;
</xsl:if><xsl:if test="constant_current_A_real">	constant_current_A_real <xsl:value-of select="constant_current_A_real"/>;
</xsl:if><xsl:if test="constant_current_B_real">	constant_current_B_real <xsl:value-of select="constant_current_B_real"/>;
</xsl:if><xsl:if test="constant_current_C_real">	constant_current_C_real <xsl:value-of select="constant_current_C_real"/>;
</xsl:if><xsl:if test="constant_current_A_reac">	constant_current_A_reac <xsl:value-of select="constant_current_A_reac"/>;
</xsl:if><xsl:if test="constant_current_B_reac">	constant_current_B_reac <xsl:value-of select="constant_current_B_reac"/>;
</xsl:if><xsl:if test="constant_current_C_reac">	constant_current_C_reac <xsl:value-of select="constant_current_C_reac"/>;
</xsl:if><xsl:if test="constant_impedance_A">	constant_impedance_A <xsl:value-of select="constant_impedance_A"/>;
</xsl:if><xsl:if test="constant_impedance_B">	constant_impedance_B <xsl:value-of select="constant_impedance_B"/>;
</xsl:if><xsl:if test="constant_impedance_C">	constant_impedance_C <xsl:value-of select="constant_impedance_C"/>;
</xsl:if><xsl:if test="constant_impedance_A_real">	constant_impedance_A_real <xsl:value-of select="constant_impedance_A_real"/>;
</xsl:if><xsl:if test="constant_impedance_B_real">	constant_impedance_B_real <xsl:value-of select="constant_impedance_B_real"/>;
</xsl:if><xsl:if test="constant_impedance_C_real">	constant_impedance_C_real <xsl:value-of select="constant_impedance_C_real"/>;
</xsl:if><xsl:if test="constant_impedance_A_reac">	constant_impedance_A_reac <xsl:value-of select="constant_impedance_A_reac"/>;
</xsl:if><xsl:if test="constant_impedance_B_reac">	constant_impedance_B_reac <xsl:value-of select="constant_impedance_B_reac"/>;
</xsl:if><xsl:if test="constant_impedance_C_reac">	constant_impedance_C_reac <xsl:value-of select="constant_impedance_C_reac"/>;
</xsl:if><xsl:if test="measured_voltage_A">	measured_voltage_A <xsl:value-of select="measured_voltage_A"/>;
</xsl:if><xsl:if test="measured_voltage_B">	measured_voltage_B <xsl:value-of select="measured_voltage_B"/>;
</xsl:if><xsl:if test="measured_voltage_C">	measured_voltage_C <xsl:value-of select="measured_voltage_C"/>;
</xsl:if><xsl:if test="measured_voltage_AB">	measured_voltage_AB <xsl:value-of select="measured_voltage_AB"/>;
</xsl:if><xsl:if test="measured_voltage_BC">	measured_voltage_BC <xsl:value-of select="measured_voltage_BC"/>;
</xsl:if><xsl:if test="measured_voltage_CA">	measured_voltage_CA <xsl:value-of select="measured_voltage_CA"/>;
</xsl:if><xsl:if test="phase_loss_protection">	phase_loss_protection <xsl:value-of select="phase_loss_protection"/>;
</xsl:if><xsl:if test="base_power_A">	base_power_A <xsl:value-of select="base_power_A"/>;
</xsl:if><xsl:if test="base_power_B">	base_power_B <xsl:value-of select="base_power_B"/>;
</xsl:if><xsl:if test="base_power_C">	base_power_C <xsl:value-of select="base_power_C"/>;
</xsl:if><xsl:if test="power_pf_A">	power_pf_A <xsl:value-of select="power_pf_A"/>;
</xsl:if><xsl:if test="current_pf_A">	current_pf_A <xsl:value-of select="current_pf_A"/>;
</xsl:if><xsl:if test="impedance_pf_A">	impedance_pf_A <xsl:value-of select="impedance_pf_A"/>;
</xsl:if><xsl:if test="power_pf_B">	power_pf_B <xsl:value-of select="power_pf_B"/>;
</xsl:if><xsl:if test="current_pf_B">	current_pf_B <xsl:value-of select="current_pf_B"/>;
</xsl:if><xsl:if test="impedance_pf_B">	impedance_pf_B <xsl:value-of select="impedance_pf_B"/>;
</xsl:if><xsl:if test="power_pf_C">	power_pf_C <xsl:value-of select="power_pf_C"/>;
</xsl:if><xsl:if test="current_pf_C">	current_pf_C <xsl:value-of select="current_pf_C"/>;
</xsl:if><xsl:if test="impedance_pf_C">	impedance_pf_C <xsl:value-of select="impedance_pf_C"/>;
</xsl:if><xsl:if test="power_fraction_A">	power_fraction_A <xsl:value-of select="power_fraction_A"/>;
</xsl:if><xsl:if test="current_fraction_A">	current_fraction_A <xsl:value-of select="current_fraction_A"/>;
</xsl:if><xsl:if test="impedance_fraction_A">	impedance_fraction_A <xsl:value-of select="impedance_fraction_A"/>;
</xsl:if><xsl:if test="power_fraction_B">	power_fraction_B <xsl:value-of select="power_fraction_B"/>;
</xsl:if><xsl:if test="current_fraction_B">	current_fraction_B <xsl:value-of select="current_fraction_B"/>;
</xsl:if><xsl:if test="impedance_fraction_B">	impedance_fraction_B <xsl:value-of select="impedance_fraction_B"/>;
</xsl:if><xsl:if test="power_fraction_C">	power_fraction_C <xsl:value-of select="power_fraction_C"/>;
</xsl:if><xsl:if test="current_fraction_C">	current_fraction_C <xsl:value-of select="current_fraction_C"/>;
</xsl:if><xsl:if test="impedance_fraction_C">	impedance_fraction_C <xsl:value-of select="impedance_fraction_C"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::regulator_configuration objects
<xsl:for-each select="regulator_configuration_list/regulator_configuration"><a name="#GLM.{name}"/>object regulator_configuration:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="connect_type">	connect_type <xsl:value-of select="connect_type"/>;
</xsl:if><xsl:if test="band_center">	band_center <xsl:value-of select="band_center"/>;
</xsl:if><xsl:if test="band_width">	band_width <xsl:value-of select="band_width"/>;
</xsl:if><xsl:if test="time_delay">	time_delay <xsl:value-of select="time_delay"/>;
</xsl:if><xsl:if test="dwell_time">	dwell_time <xsl:value-of select="dwell_time"/>;
</xsl:if><xsl:if test="raise_taps">	raise_taps <xsl:value-of select="raise_taps"/>;
</xsl:if><xsl:if test="lower_taps">	lower_taps <xsl:value-of select="lower_taps"/>;
</xsl:if><xsl:if test="current_transducer_ratio">	current_transducer_ratio <xsl:value-of select="current_transducer_ratio"/>;
</xsl:if><xsl:if test="power_transducer_ratio">	power_transducer_ratio <xsl:value-of select="power_transducer_ratio"/>;
</xsl:if><xsl:if test="compensator_r_setting_A">	compensator_r_setting_A <xsl:value-of select="compensator_r_setting_A"/>;
</xsl:if><xsl:if test="compensator_r_setting_B">	compensator_r_setting_B <xsl:value-of select="compensator_r_setting_B"/>;
</xsl:if><xsl:if test="compensator_r_setting_C">	compensator_r_setting_C <xsl:value-of select="compensator_r_setting_C"/>;
</xsl:if><xsl:if test="compensator_x_setting_A">	compensator_x_setting_A <xsl:value-of select="compensator_x_setting_A"/>;
</xsl:if><xsl:if test="compensator_x_setting_B">	compensator_x_setting_B <xsl:value-of select="compensator_x_setting_B"/>;
</xsl:if><xsl:if test="compensator_x_setting_C">	compensator_x_setting_C <xsl:value-of select="compensator_x_setting_C"/>;
</xsl:if><xsl:if test="CT_phase">	CT_phase <xsl:value-of select="CT_phase"/>;
</xsl:if><xsl:if test="PT_phase">	PT_phase <xsl:value-of select="PT_phase"/>;
</xsl:if><xsl:if test="regulation">	regulation <xsl:value-of select="regulation"/>;
</xsl:if><xsl:if test="control_level">	control_level <xsl:value-of select="control_level"/>;
</xsl:if><xsl:if test="Control">	Control <xsl:value-of select="Control"/>;
</xsl:if><xsl:if test="Type">	Type <xsl:value-of select="Type"/>;
</xsl:if><xsl:if test="tap_pos_A">	tap_pos_A <xsl:value-of select="tap_pos_A"/>;
</xsl:if><xsl:if test="tap_pos_B">	tap_pos_B <xsl:value-of select="tap_pos_B"/>;
</xsl:if><xsl:if test="tap_pos_C">	tap_pos_C <xsl:value-of select="tap_pos_C"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::regulator objects
<xsl:for-each select="regulator_list/regulator"><a name="#GLM.{name}"/>object regulator:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="configuration">	configuration <a href="#GLM.{configuration}"><xsl:value-of select="configuration"/></a>;
</xsl:if><xsl:if test="tap_A">	tap_A <xsl:value-of select="tap_A"/>;
</xsl:if><xsl:if test="tap_B">	tap_B <xsl:value-of select="tap_B"/>;
</xsl:if><xsl:if test="tap_C">	tap_C <xsl:value-of select="tap_C"/>;
</xsl:if><xsl:if test="tap_A_change_count">	tap_A_change_count <xsl:value-of select="tap_A_change_count"/>;
</xsl:if><xsl:if test="tap_B_change_count">	tap_B_change_count <xsl:value-of select="tap_B_change_count"/>;
</xsl:if><xsl:if test="tap_C_change_count">	tap_C_change_count <xsl:value-of select="tap_C_change_count"/>;
</xsl:if><xsl:if test="sense_node">	sense_node <a href="#GLM.{sense_node}"><xsl:value-of select="sense_node"/></a>;
</xsl:if>}
</xsl:for-each>
# powerflow::triplex_node objects
<xsl:for-each select="triplex_node_list/triplex_node"><a name="#GLM.{name}"/>object triplex_node:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="bustype">	bustype <xsl:value-of select="bustype"/>;
</xsl:if><xsl:if test="busflags">	busflags <xsl:value-of select="busflags"/>;
</xsl:if><xsl:if test="reference_bus">	reference_bus <a href="#GLM.{reference_bus}"><xsl:value-of select="reference_bus"/></a>;
</xsl:if><xsl:if test="maximum_voltage_error">	maximum_voltage_error <xsl:value-of select="maximum_voltage_error"/>;
</xsl:if><xsl:if test="voltage_1">	voltage_1 <xsl:value-of select="voltage_1"/>;
</xsl:if><xsl:if test="voltage_2">	voltage_2 <xsl:value-of select="voltage_2"/>;
</xsl:if><xsl:if test="voltage_N">	voltage_N <xsl:value-of select="voltage_N"/>;
</xsl:if><xsl:if test="voltage_12">	voltage_12 <xsl:value-of select="voltage_12"/>;
</xsl:if><xsl:if test="voltage_1N">	voltage_1N <xsl:value-of select="voltage_1N"/>;
</xsl:if><xsl:if test="voltage_2N">	voltage_2N <xsl:value-of select="voltage_2N"/>;
</xsl:if><xsl:if test="current_1">	current_1 <xsl:value-of select="current_1"/>;
</xsl:if><xsl:if test="current_2">	current_2 <xsl:value-of select="current_2"/>;
</xsl:if><xsl:if test="current_N">	current_N <xsl:value-of select="current_N"/>;
</xsl:if><xsl:if test="current_1_real">	current_1_real <xsl:value-of select="current_1_real"/>;
</xsl:if><xsl:if test="current_2_real">	current_2_real <xsl:value-of select="current_2_real"/>;
</xsl:if><xsl:if test="current_N_real">	current_N_real <xsl:value-of select="current_N_real"/>;
</xsl:if><xsl:if test="current_1_reac">	current_1_reac <xsl:value-of select="current_1_reac"/>;
</xsl:if><xsl:if test="current_2_reac">	current_2_reac <xsl:value-of select="current_2_reac"/>;
</xsl:if><xsl:if test="current_N_reac">	current_N_reac <xsl:value-of select="current_N_reac"/>;
</xsl:if><xsl:if test="current_12">	current_12 <xsl:value-of select="current_12"/>;
</xsl:if><xsl:if test="current_12_real">	current_12_real <xsl:value-of select="current_12_real"/>;
</xsl:if><xsl:if test="current_12_reac">	current_12_reac <xsl:value-of select="current_12_reac"/>;
</xsl:if><xsl:if test="residential_nominal_current_1">	residential_nominal_current_1 <xsl:value-of select="residential_nominal_current_1"/>;
</xsl:if><xsl:if test="residential_nominal_current_2">	residential_nominal_current_2 <xsl:value-of select="residential_nominal_current_2"/>;
</xsl:if><xsl:if test="residential_nominal_current_12">	residential_nominal_current_12 <xsl:value-of select="residential_nominal_current_12"/>;
</xsl:if><xsl:if test="residential_nominal_current_1_real">	residential_nominal_current_1_real <xsl:value-of select="residential_nominal_current_1_real"/>;
</xsl:if><xsl:if test="residential_nominal_current_1_imag">	residential_nominal_current_1_imag <xsl:value-of select="residential_nominal_current_1_imag"/>;
</xsl:if><xsl:if test="residential_nominal_current_2_real">	residential_nominal_current_2_real <xsl:value-of select="residential_nominal_current_2_real"/>;
</xsl:if><xsl:if test="residential_nominal_current_2_imag">	residential_nominal_current_2_imag <xsl:value-of select="residential_nominal_current_2_imag"/>;
</xsl:if><xsl:if test="residential_nominal_current_12_real">	residential_nominal_current_12_real <xsl:value-of select="residential_nominal_current_12_real"/>;
</xsl:if><xsl:if test="residential_nominal_current_12_imag">	residential_nominal_current_12_imag <xsl:value-of select="residential_nominal_current_12_imag"/>;
</xsl:if><xsl:if test="power_1">	power_1 <xsl:value-of select="power_1"/>;
</xsl:if><xsl:if test="power_2">	power_2 <xsl:value-of select="power_2"/>;
</xsl:if><xsl:if test="power_12">	power_12 <xsl:value-of select="power_12"/>;
</xsl:if><xsl:if test="power_1_real">	power_1_real <xsl:value-of select="power_1_real"/>;
</xsl:if><xsl:if test="power_2_real">	power_2_real <xsl:value-of select="power_2_real"/>;
</xsl:if><xsl:if test="power_12_real">	power_12_real <xsl:value-of select="power_12_real"/>;
</xsl:if><xsl:if test="power_1_reac">	power_1_reac <xsl:value-of select="power_1_reac"/>;
</xsl:if><xsl:if test="power_2_reac">	power_2_reac <xsl:value-of select="power_2_reac"/>;
</xsl:if><xsl:if test="power_12_reac">	power_12_reac <xsl:value-of select="power_12_reac"/>;
</xsl:if><xsl:if test="shunt_1">	shunt_1 <xsl:value-of select="shunt_1"/>;
</xsl:if><xsl:if test="shunt_2">	shunt_2 <xsl:value-of select="shunt_2"/>;
</xsl:if><xsl:if test="shunt_12">	shunt_12 <xsl:value-of select="shunt_12"/>;
</xsl:if><xsl:if test="impedance_1">	impedance_1 <xsl:value-of select="impedance_1"/>;
</xsl:if><xsl:if test="impedance_2">	impedance_2 <xsl:value-of select="impedance_2"/>;
</xsl:if><xsl:if test="impedance_12">	impedance_12 <xsl:value-of select="impedance_12"/>;
</xsl:if><xsl:if test="impedance_1_real">	impedance_1_real <xsl:value-of select="impedance_1_real"/>;
</xsl:if><xsl:if test="impedance_2_real">	impedance_2_real <xsl:value-of select="impedance_2_real"/>;
</xsl:if><xsl:if test="impedance_12_real">	impedance_12_real <xsl:value-of select="impedance_12_real"/>;
</xsl:if><xsl:if test="impedance_1_reac">	impedance_1_reac <xsl:value-of select="impedance_1_reac"/>;
</xsl:if><xsl:if test="impedance_2_reac">	impedance_2_reac <xsl:value-of select="impedance_2_reac"/>;
</xsl:if><xsl:if test="impedance_12_reac">	impedance_12_reac <xsl:value-of select="impedance_12_reac"/>;
</xsl:if><xsl:if test="house_present">	house_present <xsl:value-of select="house_present"/>;
</xsl:if><xsl:if test="NR_mode">	NR_mode <xsl:value-of select="NR_mode"/>;
</xsl:if><xsl:if test="service_status">	service_status <xsl:value-of select="service_status"/>;
</xsl:if><xsl:if test="service_status_double">	service_status_double <xsl:value-of select="service_status_double"/>;
</xsl:if><xsl:if test="previous_uptime">	previous_uptime <xsl:value-of select="previous_uptime"/>;
</xsl:if><xsl:if test="current_uptime">	current_uptime <xsl:value-of select="current_uptime"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::triplex_meter objects
<xsl:for-each select="triplex_meter_list/triplex_meter"><a name="#GLM.{name}"/>object triplex_meter:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="measured_real_energy">	measured_real_energy <xsl:value-of select="measured_real_energy"/>;
</xsl:if><xsl:if test="measured_reactive_energy">	measured_reactive_energy <xsl:value-of select="measured_reactive_energy"/>;
</xsl:if><xsl:if test="measured_power">	measured_power <xsl:value-of select="measured_power"/>;
</xsl:if><xsl:if test="indiv_measured_power_1">	indiv_measured_power_1 <xsl:value-of select="indiv_measured_power_1"/>;
</xsl:if><xsl:if test="indiv_measured_power_2">	indiv_measured_power_2 <xsl:value-of select="indiv_measured_power_2"/>;
</xsl:if><xsl:if test="indiv_measured_power_N">	indiv_measured_power_N <xsl:value-of select="indiv_measured_power_N"/>;
</xsl:if><xsl:if test="measured_demand">	measured_demand <xsl:value-of select="measured_demand"/>;
</xsl:if><xsl:if test="measured_real_power">	measured_real_power <xsl:value-of select="measured_real_power"/>;
</xsl:if><xsl:if test="measured_reactive_power">	measured_reactive_power <xsl:value-of select="measured_reactive_power"/>;
</xsl:if><xsl:if test="meter_power_consumption">	meter_power_consumption <xsl:value-of select="meter_power_consumption"/>;
</xsl:if><xsl:if test="measured_voltage_1">	measured_voltage_1 <xsl:value-of select="measured_voltage_1"/>;
</xsl:if><xsl:if test="measured_voltage_2">	measured_voltage_2 <xsl:value-of select="measured_voltage_2"/>;
</xsl:if><xsl:if test="measured_voltage_N">	measured_voltage_N <xsl:value-of select="measured_voltage_N"/>;
</xsl:if><xsl:if test="measured_current_1">	measured_current_1 <xsl:value-of select="measured_current_1"/>;
</xsl:if><xsl:if test="measured_current_2">	measured_current_2 <xsl:value-of select="measured_current_2"/>;
</xsl:if><xsl:if test="measured_current_N">	measured_current_N <xsl:value-of select="measured_current_N"/>;
</xsl:if><xsl:if test="customer_interrupted">	customer_interrupted <xsl:value-of select="customer_interrupted"/>;
</xsl:if><xsl:if test="customer_interrupted_secondary">	customer_interrupted_secondary <xsl:value-of select="customer_interrupted_secondary"/>;
</xsl:if><xsl:if test="monthly_bill">	monthly_bill <xsl:value-of select="monthly_bill"/>;
</xsl:if><xsl:if test="previous_monthly_bill">	previous_monthly_bill <xsl:value-of select="previous_monthly_bill"/>;
</xsl:if><xsl:if test="previous_monthly_energy">	previous_monthly_energy <xsl:value-of select="previous_monthly_energy"/>;
</xsl:if><xsl:if test="monthly_fee">	monthly_fee <xsl:value-of select="monthly_fee"/>;
</xsl:if><xsl:if test="monthly_energy">	monthly_energy <xsl:value-of select="monthly_energy"/>;
</xsl:if><xsl:if test="bill_mode">	bill_mode <xsl:value-of select="bill_mode"/>;
</xsl:if><xsl:if test="power_market">	power_market <a href="#GLM.{power_market}"><xsl:value-of select="power_market"/></a>;
</xsl:if><xsl:if test="bill_day">	bill_day <xsl:value-of select="bill_day"/>;
</xsl:if><xsl:if test="price">	price <xsl:value-of select="price"/>;
</xsl:if><xsl:if test="price_base">	price_base <xsl:value-of select="price_base"/>;
</xsl:if><xsl:if test="first_tier_price">	first_tier_price <xsl:value-of select="first_tier_price"/>;
</xsl:if><xsl:if test="first_tier_energy">	first_tier_energy <xsl:value-of select="first_tier_energy"/>;
</xsl:if><xsl:if test="second_tier_price">	second_tier_price <xsl:value-of select="second_tier_price"/>;
</xsl:if><xsl:if test="second_tier_energy">	second_tier_energy <xsl:value-of select="second_tier_energy"/>;
</xsl:if><xsl:if test="third_tier_price">	third_tier_price <xsl:value-of select="third_tier_price"/>;
</xsl:if><xsl:if test="third_tier_energy">	third_tier_energy <xsl:value-of select="third_tier_energy"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::triplex_line objects
<xsl:for-each select="triplex_line_list/triplex_line"><a name="#GLM.{name}"/>object triplex_line:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::triplex_line_configuration objects
<xsl:for-each select="triplex_line_configuration_list/triplex_line_configuration"><a name="#GLM.{name}"/>object triplex_line_configuration:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="conductor_1">	conductor_1 <a href="#GLM.{conductor_1}"><xsl:value-of select="conductor_1"/></a>;
</xsl:if><xsl:if test="conductor_2">	conductor_2 <a href="#GLM.{conductor_2}"><xsl:value-of select="conductor_2"/></a>;
</xsl:if><xsl:if test="conductor_N">	conductor_N <a href="#GLM.{conductor_N}"><xsl:value-of select="conductor_N"/></a>;
</xsl:if><xsl:if test="insulation_thickness">	insulation_thickness <xsl:value-of select="insulation_thickness"/>;
</xsl:if><xsl:if test="diameter">	diameter <xsl:value-of select="diameter"/>;
</xsl:if><xsl:if test="spacing">	spacing <a href="#GLM.{spacing}"><xsl:value-of select="spacing"/></a>;
</xsl:if><xsl:if test="z11">	z11 <xsl:value-of select="z11"/>;
</xsl:if><xsl:if test="z12">	z12 <xsl:value-of select="z12"/>;
</xsl:if><xsl:if test="z21">	z21 <xsl:value-of select="z21"/>;
</xsl:if><xsl:if test="z22">	z22 <xsl:value-of select="z22"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::triplex_line_conductor objects
<xsl:for-each select="triplex_line_conductor_list/triplex_line_conductor"><a name="#GLM.{name}"/>object triplex_line_conductor:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="resistance">	resistance <xsl:value-of select="resistance"/>;
</xsl:if><xsl:if test="geometric_mean_radius">	geometric_mean_radius <xsl:value-of select="geometric_mean_radius"/>;
</xsl:if><xsl:if test="rating.summer.continuous">	rating.summer.continuous <xsl:value-of select="rating.summer.continuous"/>;
</xsl:if><xsl:if test="rating.summer.emergency">	rating.summer.emergency <xsl:value-of select="rating.summer.emergency"/>;
</xsl:if><xsl:if test="rating.winter.continuous">	rating.winter.continuous <xsl:value-of select="rating.winter.continuous"/>;
</xsl:if><xsl:if test="rating.winter.emergency">	rating.winter.emergency <xsl:value-of select="rating.winter.emergency"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::switch objects
<xsl:for-each select="switch_list/switch"><a name="#GLM.{name}"/>object switch:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="phase_A_state">	phase_A_state <xsl:value-of select="phase_A_state"/>;
</xsl:if><xsl:if test="phase_B_state">	phase_B_state <xsl:value-of select="phase_B_state"/>;
</xsl:if><xsl:if test="phase_C_state">	phase_C_state <xsl:value-of select="phase_C_state"/>;
</xsl:if><xsl:if test="operating_mode">	operating_mode <xsl:value-of select="operating_mode"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::substation objects
<xsl:for-each select="substation_list/substation"><a name="#GLM.{name}"/>object substation:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="distribution_energy">	distribution_energy <xsl:value-of select="distribution_energy"/>;
</xsl:if><xsl:if test="distribution_power">	distribution_power <xsl:value-of select="distribution_power"/>;
</xsl:if><xsl:if test="distribution_demand">	distribution_demand <xsl:value-of select="distribution_demand"/>;
</xsl:if><xsl:if test="distribution_voltage_A">	distribution_voltage_A <xsl:value-of select="distribution_voltage_A"/>;
</xsl:if><xsl:if test="distribution_voltage_B">	distribution_voltage_B <xsl:value-of select="distribution_voltage_B"/>;
</xsl:if><xsl:if test="distribution_voltage_C">	distribution_voltage_C <xsl:value-of select="distribution_voltage_C"/>;
</xsl:if><xsl:if test="distribution_current_A">	distribution_current_A <xsl:value-of select="distribution_current_A"/>;
</xsl:if><xsl:if test="distribution_current_B">	distribution_current_B <xsl:value-of select="distribution_current_B"/>;
</xsl:if><xsl:if test="distribution_current_C">	distribution_current_C <xsl:value-of select="distribution_current_C"/>;
</xsl:if><xsl:if test="Network_Node_Base_Power">	Network_Node_Base_Power <xsl:value-of select="Network_Node_Base_Power"/>;
</xsl:if><xsl:if test="Network_Node_Base_Voltage">	Network_Node_Base_Voltage <xsl:value-of select="Network_Node_Base_Voltage"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::pqload objects
<xsl:for-each select="pqload_list/pqload"><a name="#GLM.{name}"/>object pqload:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="weather">	weather <a href="#GLM.{weather}"><xsl:value-of select="weather"/></a>;
</xsl:if><xsl:if test="T_nominal">	T_nominal <xsl:value-of select="T_nominal"/>;
</xsl:if><xsl:if test="Zp_T">	Zp_T <xsl:value-of select="Zp_T"/>;
</xsl:if><xsl:if test="Zp_H">	Zp_H <xsl:value-of select="Zp_H"/>;
</xsl:if><xsl:if test="Zp_S">	Zp_S <xsl:value-of select="Zp_S"/>;
</xsl:if><xsl:if test="Zp_W">	Zp_W <xsl:value-of select="Zp_W"/>;
</xsl:if><xsl:if test="Zp_R">	Zp_R <xsl:value-of select="Zp_R"/>;
</xsl:if><xsl:if test="Zp">	Zp <xsl:value-of select="Zp"/>;
</xsl:if><xsl:if test="Zq_T">	Zq_T <xsl:value-of select="Zq_T"/>;
</xsl:if><xsl:if test="Zq_H">	Zq_H <xsl:value-of select="Zq_H"/>;
</xsl:if><xsl:if test="Zq_S">	Zq_S <xsl:value-of select="Zq_S"/>;
</xsl:if><xsl:if test="Zq_W">	Zq_W <xsl:value-of select="Zq_W"/>;
</xsl:if><xsl:if test="Zq_R">	Zq_R <xsl:value-of select="Zq_R"/>;
</xsl:if><xsl:if test="Zq">	Zq <xsl:value-of select="Zq"/>;
</xsl:if><xsl:if test="Im_T">	Im_T <xsl:value-of select="Im_T"/>;
</xsl:if><xsl:if test="Im_H">	Im_H <xsl:value-of select="Im_H"/>;
</xsl:if><xsl:if test="Im_S">	Im_S <xsl:value-of select="Im_S"/>;
</xsl:if><xsl:if test="Im_W">	Im_W <xsl:value-of select="Im_W"/>;
</xsl:if><xsl:if test="Im_R">	Im_R <xsl:value-of select="Im_R"/>;
</xsl:if><xsl:if test="Im">	Im <xsl:value-of select="Im"/>;
</xsl:if><xsl:if test="Ia_T">	Ia_T <xsl:value-of select="Ia_T"/>;
</xsl:if><xsl:if test="Ia_H">	Ia_H <xsl:value-of select="Ia_H"/>;
</xsl:if><xsl:if test="Ia_S">	Ia_S <xsl:value-of select="Ia_S"/>;
</xsl:if><xsl:if test="Ia_W">	Ia_W <xsl:value-of select="Ia_W"/>;
</xsl:if><xsl:if test="Ia_R">	Ia_R <xsl:value-of select="Ia_R"/>;
</xsl:if><xsl:if test="Ia">	Ia <xsl:value-of select="Ia"/>;
</xsl:if><xsl:if test="Pp_T">	Pp_T <xsl:value-of select="Pp_T"/>;
</xsl:if><xsl:if test="Pp_H">	Pp_H <xsl:value-of select="Pp_H"/>;
</xsl:if><xsl:if test="Pp_S">	Pp_S <xsl:value-of select="Pp_S"/>;
</xsl:if><xsl:if test="Pp_W">	Pp_W <xsl:value-of select="Pp_W"/>;
</xsl:if><xsl:if test="Pp_R">	Pp_R <xsl:value-of select="Pp_R"/>;
</xsl:if><xsl:if test="Pp">	Pp <xsl:value-of select="Pp"/>;
</xsl:if><xsl:if test="Pq_T">	Pq_T <xsl:value-of select="Pq_T"/>;
</xsl:if><xsl:if test="Pq_H">	Pq_H <xsl:value-of select="Pq_H"/>;
</xsl:if><xsl:if test="Pq_S">	Pq_S <xsl:value-of select="Pq_S"/>;
</xsl:if><xsl:if test="Pq_W">	Pq_W <xsl:value-of select="Pq_W"/>;
</xsl:if><xsl:if test="Pq_R">	Pq_R <xsl:value-of select="Pq_R"/>;
</xsl:if><xsl:if test="Pq">	Pq <xsl:value-of select="Pq"/>;
</xsl:if><xsl:if test="input_temp">	input_temp <xsl:value-of select="input_temp"/>;
</xsl:if><xsl:if test="input_humid">	input_humid <xsl:value-of select="input_humid"/>;
</xsl:if><xsl:if test="input_solar">	input_solar <xsl:value-of select="input_solar"/>;
</xsl:if><xsl:if test="input_wind">	input_wind <xsl:value-of select="input_wind"/>;
</xsl:if><xsl:if test="input_rain">	input_rain <xsl:value-of select="input_rain"/>;
</xsl:if><xsl:if test="output_imped_p">	output_imped_p <xsl:value-of select="output_imped_p"/>;
</xsl:if><xsl:if test="output_imped_q">	output_imped_q <xsl:value-of select="output_imped_q"/>;
</xsl:if><xsl:if test="output_current_m">	output_current_m <xsl:value-of select="output_current_m"/>;
</xsl:if><xsl:if test="output_current_a">	output_current_a <xsl:value-of select="output_current_a"/>;
</xsl:if><xsl:if test="output_power_p">	output_power_p <xsl:value-of select="output_power_p"/>;
</xsl:if><xsl:if test="output_power_q">	output_power_q <xsl:value-of select="output_power_q"/>;
</xsl:if><xsl:if test="output_impedance">	output_impedance <xsl:value-of select="output_impedance"/>;
</xsl:if><xsl:if test="output_current">	output_current <xsl:value-of select="output_current"/>;
</xsl:if><xsl:if test="output_power">	output_power <xsl:value-of select="output_power"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::voltdump objects
<xsl:for-each select="voltdump_list/voltdump"><a name="#GLM.{name}"/>object voltdump:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="group">	group <xsl:value-of select="group"/>;
</xsl:if><xsl:if test="runtime">	runtime <xsl:value-of select="runtime"/>;
</xsl:if><xsl:if test="filename">	filename <xsl:value-of select="filename"/>;
</xsl:if><xsl:if test="runcount">	runcount <xsl:value-of select="runcount"/>;
</xsl:if><xsl:if test="mode">	mode <xsl:value-of select="mode"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::series_reactor objects
<xsl:for-each select="series_reactor_list/series_reactor"><a name="#GLM.{name}"/>object series_reactor:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="phase_A_impedance">	phase_A_impedance <xsl:value-of select="phase_A_impedance"/>;
</xsl:if><xsl:if test="phase_A_resistance">	phase_A_resistance <xsl:value-of select="phase_A_resistance"/>;
</xsl:if><xsl:if test="phase_A_reactance">	phase_A_reactance <xsl:value-of select="phase_A_reactance"/>;
</xsl:if><xsl:if test="phase_B_impedance">	phase_B_impedance <xsl:value-of select="phase_B_impedance"/>;
</xsl:if><xsl:if test="phase_B_resistance">	phase_B_resistance <xsl:value-of select="phase_B_resistance"/>;
</xsl:if><xsl:if test="phase_B_reactance">	phase_B_reactance <xsl:value-of select="phase_B_reactance"/>;
</xsl:if><xsl:if test="phase_C_impedance">	phase_C_impedance <xsl:value-of select="phase_C_impedance"/>;
</xsl:if><xsl:if test="phase_C_resistance">	phase_C_resistance <xsl:value-of select="phase_C_resistance"/>;
</xsl:if><xsl:if test="phase_C_reactance">	phase_C_reactance <xsl:value-of select="phase_C_reactance"/>;
</xsl:if><xsl:if test="rated_current_limit">	rated_current_limit <xsl:value-of select="rated_current_limit"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::restoration objects
<xsl:for-each select="restoration_list/restoration"><a name="#GLM.{name}"/>object restoration:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="reconfig_attempts">	reconfig_attempts <xsl:value-of select="reconfig_attempts"/>;
</xsl:if><xsl:if test="reconfig_iteration_limit">	reconfig_iteration_limit <xsl:value-of select="reconfig_iteration_limit"/>;
</xsl:if><xsl:if test="populate_tree">	populate_tree <xsl:value-of select="populate_tree"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::frequency_gen objects
<xsl:for-each select="frequency_gen_list/frequency_gen"><a name="#GLM.{name}"/>object frequency_gen:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="Frequency_Mode">	Frequency_Mode <xsl:value-of select="Frequency_Mode"/>;
</xsl:if><xsl:if test="Frequency">	Frequency <xsl:value-of select="Frequency"/>;
</xsl:if><xsl:if test="FreqChange">	FreqChange <xsl:value-of select="FreqChange"/>;
</xsl:if><xsl:if test="Deadband">	Deadband <xsl:value-of select="Deadband"/>;
</xsl:if><xsl:if test="Tolerance">	Tolerance <xsl:value-of select="Tolerance"/>;
</xsl:if><xsl:if test="M">	M <xsl:value-of select="M"/>;
</xsl:if><xsl:if test="D">	D <xsl:value-of select="D"/>;
</xsl:if><xsl:if test="Rated_power">	Rated_power <xsl:value-of select="Rated_power"/>;
</xsl:if><xsl:if test="Gen_power">	Gen_power <xsl:value-of select="Gen_power"/>;
</xsl:if><xsl:if test="Load_power">	Load_power <xsl:value-of select="Load_power"/>;
</xsl:if><xsl:if test="Gov_delay">	Gov_delay <xsl:value-of select="Gov_delay"/>;
</xsl:if><xsl:if test="Ramp_rate">	Ramp_rate <xsl:value-of select="Ramp_rate"/>;
</xsl:if><xsl:if test="Low_Freq_OI">	Low_Freq_OI <xsl:value-of select="Low_Freq_OI"/>;
</xsl:if><xsl:if test="High_Freq_OI">	High_Freq_OI <xsl:value-of select="High_Freq_OI"/>;
</xsl:if><xsl:if test="avg24">	avg24 <xsl:value-of select="avg24"/>;
</xsl:if><xsl:if test="std24">	std24 <xsl:value-of select="std24"/>;
</xsl:if><xsl:if test="avg168">	avg168 <xsl:value-of select="avg168"/>;
</xsl:if><xsl:if test="std168">	std168 <xsl:value-of select="std168"/>;
</xsl:if><xsl:if test="Num_Resp_Eqs">	Num_Resp_Eqs <xsl:value-of select="Num_Resp_Eqs"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::volt_var_control objects
<xsl:for-each select="volt_var_control_list/volt_var_control"><a name="#GLM.{name}"/>object volt_var_control:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="control_method">	control_method <xsl:value-of select="control_method"/>;
</xsl:if><xsl:if test="capacitor_delay">	capacitor_delay <xsl:value-of select="capacitor_delay"/>;
</xsl:if><xsl:if test="regulator_delay">	regulator_delay <xsl:value-of select="regulator_delay"/>;
</xsl:if><xsl:if test="desired_pf">	desired_pf <xsl:value-of select="desired_pf"/>;
</xsl:if><xsl:if test="d_max">	d_max <xsl:value-of select="d_max"/>;
</xsl:if><xsl:if test="d_min">	d_min <xsl:value-of select="d_min"/>;
</xsl:if><xsl:if test="substation_link">	substation_link <a href="#GLM.{substation_link}"><xsl:value-of select="substation_link"/></a>;
</xsl:if><xsl:if test="pf_phase">	pf_phase <xsl:value-of select="pf_phase"/>;
</xsl:if><xsl:if test="regulator_list">	regulator_list <xsl:value-of select="regulator_list"/>;
</xsl:if><xsl:if test="capacitor_list">	capacitor_list <xsl:value-of select="capacitor_list"/>;
</xsl:if><xsl:if test="voltage_measurements">	voltage_measurements <xsl:value-of select="voltage_measurements"/>;
</xsl:if><xsl:if test="minimum_voltages">	minimum_voltages <xsl:value-of select="minimum_voltages"/>;
</xsl:if><xsl:if test="maximum_voltages">	maximum_voltages <xsl:value-of select="maximum_voltages"/>;
</xsl:if><xsl:if test="desired_voltages">	desired_voltages <xsl:value-of select="desired_voltages"/>;
</xsl:if><xsl:if test="max_vdrop">	max_vdrop <xsl:value-of select="max_vdrop"/>;
</xsl:if><xsl:if test="high_load_deadband">	high_load_deadband <xsl:value-of select="high_load_deadband"/>;
</xsl:if><xsl:if test="low_load_deadband">	low_load_deadband <xsl:value-of select="low_load_deadband"/>;
</xsl:if><xsl:if test="pf_signed">	pf_signed <xsl:value-of select="pf_signed"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::fault_check objects
<xsl:for-each select="fault_check_list/fault_check"><a name="#GLM.{name}"/>object fault_check:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="check_mode">	check_mode <xsl:value-of select="check_mode"/>;
</xsl:if><xsl:if test="output_filename">	output_filename <xsl:value-of select="output_filename"/>;
</xsl:if><xsl:if test="reliability_mode">	reliability_mode <xsl:value-of select="reliability_mode"/>;
</xsl:if><xsl:if test="eventgen_object">	eventgen_object <a href="#GLM.{eventgen_object}"><xsl:value-of select="eventgen_object"/></a>;
</xsl:if>}
</xsl:for-each>
# powerflow::motor objects
<xsl:for-each select="motor_list/motor"><a name="#GLM.{name}"/>object motor:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::billdump objects
<xsl:for-each select="billdump_list/billdump"><a name="#GLM.{name}"/>object billdump:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="group">	group <xsl:value-of select="group"/>;
</xsl:if><xsl:if test="runtime">	runtime <xsl:value-of select="runtime"/>;
</xsl:if><xsl:if test="filename">	filename <xsl:value-of select="filename"/>;
</xsl:if><xsl:if test="runcount">	runcount <xsl:value-of select="runcount"/>;
</xsl:if><xsl:if test="meter_type">	meter_type <xsl:value-of select="meter_type"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::power_metrics objects
<xsl:for-each select="power_metrics_list/power_metrics"><a name="#GLM.{name}"/>object power_metrics:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="SAIFI">	SAIFI <xsl:value-of select="SAIFI"/>;
</xsl:if><xsl:if test="SAIFI_int">	SAIFI_int <xsl:value-of select="SAIFI_int"/>;
</xsl:if><xsl:if test="SAIDI">	SAIDI <xsl:value-of select="SAIDI"/>;
</xsl:if><xsl:if test="SAIDI_int">	SAIDI_int <xsl:value-of select="SAIDI_int"/>;
</xsl:if><xsl:if test="CAIDI">	CAIDI <xsl:value-of select="CAIDI"/>;
</xsl:if><xsl:if test="CAIDI_int">	CAIDI_int <xsl:value-of select="CAIDI_int"/>;
</xsl:if><xsl:if test="ASAI">	ASAI <xsl:value-of select="ASAI"/>;
</xsl:if><xsl:if test="ASAI_int">	ASAI_int <xsl:value-of select="ASAI_int"/>;
</xsl:if><xsl:if test="MAIFI">	MAIFI <xsl:value-of select="MAIFI"/>;
</xsl:if><xsl:if test="MAIFI_int">	MAIFI_int <xsl:value-of select="MAIFI_int"/>;
</xsl:if><xsl:if test="base_time_value">	base_time_value <xsl:value-of select="base_time_value"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::currdump objects
<xsl:for-each select="currdump_list/currdump"><a name="#GLM.{name}"/>object currdump:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="group">	group <xsl:value-of select="group"/>;
</xsl:if><xsl:if test="runtime">	runtime <xsl:value-of select="runtime"/>;
</xsl:if><xsl:if test="filename">	filename <xsl:value-of select="filename"/>;
</xsl:if><xsl:if test="runcount">	runcount <xsl:value-of select="runcount"/>;
</xsl:if><xsl:if test="mode">	mode <xsl:value-of select="mode"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::recloser objects
<xsl:for-each select="recloser_list/recloser"><a name="#GLM.{name}"/>object recloser:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="retry_time">	retry_time <xsl:value-of select="retry_time"/>;
</xsl:if><xsl:if test="max_number_of_tries">	max_number_of_tries <xsl:value-of select="max_number_of_tries"/>;
</xsl:if><xsl:if test="number_of_tries">	number_of_tries <xsl:value-of select="number_of_tries"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::sectionalizer objects
<xsl:for-each select="sectionalizer_list/sectionalizer"><a name="#GLM.{name}"/>object sectionalizer:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if>}
</xsl:for-each>
# powerflow::emissions objects
<xsl:for-each select="emissions_list/emissions"><a name="#GLM.{name}"/>object emissions:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="Nuclear_Order">	Nuclear_Order <xsl:value-of select="Nuclear_Order"/>;
</xsl:if><xsl:if test="Hydroelectric_Order">	Hydroelectric_Order <xsl:value-of select="Hydroelectric_Order"/>;
</xsl:if><xsl:if test="Solarthermal_Order">	Solarthermal_Order <xsl:value-of select="Solarthermal_Order"/>;
</xsl:if><xsl:if test="Biomass_Order">	Biomass_Order <xsl:value-of select="Biomass_Order"/>;
</xsl:if><xsl:if test="Wind_Order">	Wind_Order <xsl:value-of select="Wind_Order"/>;
</xsl:if><xsl:if test="Coal_Order">	Coal_Order <xsl:value-of select="Coal_Order"/>;
</xsl:if><xsl:if test="Naturalgas_Order">	Naturalgas_Order <xsl:value-of select="Naturalgas_Order"/>;
</xsl:if><xsl:if test="Geothermal_Order">	Geothermal_Order <xsl:value-of select="Geothermal_Order"/>;
</xsl:if><xsl:if test="Petroleum_Order">	Petroleum_Order <xsl:value-of select="Petroleum_Order"/>;
</xsl:if><xsl:if test="Naturalgas_Max_Out">	Naturalgas_Max_Out <xsl:value-of select="Naturalgas_Max_Out"/>;
</xsl:if><xsl:if test="Coal_Max_Out">	Coal_Max_Out <xsl:value-of select="Coal_Max_Out"/>;
</xsl:if><xsl:if test="Biomass_Max_Out">	Biomass_Max_Out <xsl:value-of select="Biomass_Max_Out"/>;
</xsl:if><xsl:if test="Geothermal_Max_Out">	Geothermal_Max_Out <xsl:value-of select="Geothermal_Max_Out"/>;
</xsl:if><xsl:if test="Hydroelectric_Max_Out">	Hydroelectric_Max_Out <xsl:value-of select="Hydroelectric_Max_Out"/>;
</xsl:if><xsl:if test="Nuclear_Max_Out">	Nuclear_Max_Out <xsl:value-of select="Nuclear_Max_Out"/>;
</xsl:if><xsl:if test="Wind_Max_Out">	Wind_Max_Out <xsl:value-of select="Wind_Max_Out"/>;
</xsl:if><xsl:if test="Petroleum_Max_Out">	Petroleum_Max_Out <xsl:value-of select="Petroleum_Max_Out"/>;
</xsl:if><xsl:if test="Solarthermal_Max_Out">	Solarthermal_Max_Out <xsl:value-of select="Solarthermal_Max_Out"/>;
</xsl:if><xsl:if test="Naturalgas_Out">	Naturalgas_Out <xsl:value-of select="Naturalgas_Out"/>;
</xsl:if><xsl:if test="Coal_Out">	Coal_Out <xsl:value-of select="Coal_Out"/>;
</xsl:if><xsl:if test="Biomass_Out">	Biomass_Out <xsl:value-of select="Biomass_Out"/>;
</xsl:if><xsl:if test="Geothermal_Out">	Geothermal_Out <xsl:value-of select="Geothermal_Out"/>;
</xsl:if><xsl:if test="Hydroelectric_Out">	Hydroelectric_Out <xsl:value-of select="Hydroelectric_Out"/>;
</xsl:if><xsl:if test="Nuclear_Out">	Nuclear_Out <xsl:value-of select="Nuclear_Out"/>;
</xsl:if><xsl:if test="Wind_Out">	Wind_Out <xsl:value-of select="Wind_Out"/>;
</xsl:if><xsl:if test="Petroleum_Out">	Petroleum_Out <xsl:value-of select="Petroleum_Out"/>;
</xsl:if><xsl:if test="Solarthermal_Out">	Solarthermal_Out <xsl:value-of select="Solarthermal_Out"/>;
</xsl:if><xsl:if test="Naturalgas_Conv_Eff">	Naturalgas_Conv_Eff <xsl:value-of select="Naturalgas_Conv_Eff"/>;
</xsl:if><xsl:if test="Coal_Conv_Eff">	Coal_Conv_Eff <xsl:value-of select="Coal_Conv_Eff"/>;
</xsl:if><xsl:if test="Biomass_Conv_Eff">	Biomass_Conv_Eff <xsl:value-of select="Biomass_Conv_Eff"/>;
</xsl:if><xsl:if test="Geothermal_Conv_Eff">	Geothermal_Conv_Eff <xsl:value-of select="Geothermal_Conv_Eff"/>;
</xsl:if><xsl:if test="Hydroelectric_Conv_Eff">	Hydroelectric_Conv_Eff <xsl:value-of select="Hydroelectric_Conv_Eff"/>;
</xsl:if><xsl:if test="Nuclear_Conv_Eff">	Nuclear_Conv_Eff <xsl:value-of select="Nuclear_Conv_Eff"/>;
</xsl:if><xsl:if test="Wind_Conv_Eff">	Wind_Conv_Eff <xsl:value-of select="Wind_Conv_Eff"/>;
</xsl:if><xsl:if test="Petroleum_Conv_Eff">	Petroleum_Conv_Eff <xsl:value-of select="Petroleum_Conv_Eff"/>;
</xsl:if><xsl:if test="Solarthermal_Conv_Eff">	Solarthermal_Conv_Eff <xsl:value-of select="Solarthermal_Conv_Eff"/>;
</xsl:if><xsl:if test="Naturalgas_CO2">	Naturalgas_CO2 <xsl:value-of select="Naturalgas_CO2"/>;
</xsl:if><xsl:if test="Coal_CO2">	Coal_CO2 <xsl:value-of select="Coal_CO2"/>;
</xsl:if><xsl:if test="Biomass_CO2">	Biomass_CO2 <xsl:value-of select="Biomass_CO2"/>;
</xsl:if><xsl:if test="Geothermal_CO2">	Geothermal_CO2 <xsl:value-of select="Geothermal_CO2"/>;
</xsl:if><xsl:if test="Hydroelectric_CO2">	Hydroelectric_CO2 <xsl:value-of select="Hydroelectric_CO2"/>;
</xsl:if><xsl:if test="Nuclear_CO2">	Nuclear_CO2 <xsl:value-of select="Nuclear_CO2"/>;
</xsl:if><xsl:if test="Wind_CO2">	Wind_CO2 <xsl:value-of select="Wind_CO2"/>;
</xsl:if><xsl:if test="Petroleum_CO2">	Petroleum_CO2 <xsl:value-of select="Petroleum_CO2"/>;
</xsl:if><xsl:if test="Solarthermal_CO2">	Solarthermal_CO2 <xsl:value-of select="Solarthermal_CO2"/>;
</xsl:if><xsl:if test="Naturalgas_SO2">	Naturalgas_SO2 <xsl:value-of select="Naturalgas_SO2"/>;
</xsl:if><xsl:if test="Coal_SO2">	Coal_SO2 <xsl:value-of select="Coal_SO2"/>;
</xsl:if><xsl:if test="Biomass_SO2">	Biomass_SO2 <xsl:value-of select="Biomass_SO2"/>;
</xsl:if><xsl:if test="Geothermal_SO2">	Geothermal_SO2 <xsl:value-of select="Geothermal_SO2"/>;
</xsl:if><xsl:if test="Hydroelectric_SO2">	Hydroelectric_SO2 <xsl:value-of select="Hydroelectric_SO2"/>;
</xsl:if><xsl:if test="Nuclear_SO2">	Nuclear_SO2 <xsl:value-of select="Nuclear_SO2"/>;
</xsl:if><xsl:if test="Wind_SO2">	Wind_SO2 <xsl:value-of select="Wind_SO2"/>;
</xsl:if><xsl:if test="Petroleum_SO2">	Petroleum_SO2 <xsl:value-of select="Petroleum_SO2"/>;
</xsl:if><xsl:if test="Solarthermal_SO2">	Solarthermal_SO2 <xsl:value-of select="Solarthermal_SO2"/>;
</xsl:if><xsl:if test="Naturalgas_NOx">	Naturalgas_NOx <xsl:value-of select="Naturalgas_NOx"/>;
</xsl:if><xsl:if test="Coal_NOx">	Coal_NOx <xsl:value-of select="Coal_NOx"/>;
</xsl:if><xsl:if test="Biomass_NOx">	Biomass_NOx <xsl:value-of select="Biomass_NOx"/>;
</xsl:if><xsl:if test="Geothermal_NOx">	Geothermal_NOx <xsl:value-of select="Geothermal_NOx"/>;
</xsl:if><xsl:if test="Hydroelectric_NOx">	Hydroelectric_NOx <xsl:value-of select="Hydroelectric_NOx"/>;
</xsl:if><xsl:if test="Nuclear_NOx">	Nuclear_NOx <xsl:value-of select="Nuclear_NOx"/>;
</xsl:if><xsl:if test="Wind_NOx">	Wind_NOx <xsl:value-of select="Wind_NOx"/>;
</xsl:if><xsl:if test="Petroleum_NOx">	Petroleum_NOx <xsl:value-of select="Petroleum_NOx"/>;
</xsl:if><xsl:if test="Solarthermal_NOx">	Solarthermal_NOx <xsl:value-of select="Solarthermal_NOx"/>;
</xsl:if><xsl:if test="Naturalgas_emissions_CO2">	Naturalgas_emissions_CO2 <xsl:value-of select="Naturalgas_emissions_CO2"/>;
</xsl:if><xsl:if test="Naturalgas_emissions_SO2">	Naturalgas_emissions_SO2 <xsl:value-of select="Naturalgas_emissions_SO2"/>;
</xsl:if><xsl:if test="Naturalgas_emissions_NOx">	Naturalgas_emissions_NOx <xsl:value-of select="Naturalgas_emissions_NOx"/>;
</xsl:if><xsl:if test="Coal_emissions_CO2">	Coal_emissions_CO2 <xsl:value-of select="Coal_emissions_CO2"/>;
</xsl:if><xsl:if test="Coal_emissions_SO2">	Coal_emissions_SO2 <xsl:value-of select="Coal_emissions_SO2"/>;
</xsl:if><xsl:if test="Coal_emissions_NOx">	Coal_emissions_NOx <xsl:value-of select="Coal_emissions_NOx"/>;
</xsl:if><xsl:if test="Biomass_emissions_CO2">	Biomass_emissions_CO2 <xsl:value-of select="Biomass_emissions_CO2"/>;
</xsl:if><xsl:if test="Biomass_emissions_SO2">	Biomass_emissions_SO2 <xsl:value-of select="Biomass_emissions_SO2"/>;
</xsl:if><xsl:if test="Biomass_emissions_NOx">	Biomass_emissions_NOx <xsl:value-of select="Biomass_emissions_NOx"/>;
</xsl:if><xsl:if test="Geothermal_emissions_CO2">	Geothermal_emissions_CO2 <xsl:value-of select="Geothermal_emissions_CO2"/>;
</xsl:if><xsl:if test="Geothermal_emissions_SO2">	Geothermal_emissions_SO2 <xsl:value-of select="Geothermal_emissions_SO2"/>;
</xsl:if><xsl:if test="Geothermal_emissions_NOx">	Geothermal_emissions_NOx <xsl:value-of select="Geothermal_emissions_NOx"/>;
</xsl:if><xsl:if test="Hydroelectric_emissions_CO2">	Hydroelectric_emissions_CO2 <xsl:value-of select="Hydroelectric_emissions_CO2"/>;
</xsl:if><xsl:if test="Hydroelectric_emissions_SO2">	Hydroelectric_emissions_SO2 <xsl:value-of select="Hydroelectric_emissions_SO2"/>;
</xsl:if><xsl:if test="Hydroelectric_emissions_NOx">	Hydroelectric_emissions_NOx <xsl:value-of select="Hydroelectric_emissions_NOx"/>;
</xsl:if><xsl:if test="Nuclear_emissions_CO2">	Nuclear_emissions_CO2 <xsl:value-of select="Nuclear_emissions_CO2"/>;
</xsl:if><xsl:if test="Nuclear_emissions_SO2">	Nuclear_emissions_SO2 <xsl:value-of select="Nuclear_emissions_SO2"/>;
</xsl:if><xsl:if test="Nuclear_emissions_NOx">	Nuclear_emissions_NOx <xsl:value-of select="Nuclear_emissions_NOx"/>;
</xsl:if><xsl:if test="Wind_emissions_CO2">	Wind_emissions_CO2 <xsl:value-of select="Wind_emissions_CO2"/>;
</xsl:if><xsl:if test="Wind_emissions_SO2">	Wind_emissions_SO2 <xsl:value-of select="Wind_emissions_SO2"/>;
</xsl:if><xsl:if test="Wind_emissions_NOx">	Wind_emissions_NOx <xsl:value-of select="Wind_emissions_NOx"/>;
</xsl:if><xsl:if test="Petroleum_emissions_CO2">	Petroleum_emissions_CO2 <xsl:value-of select="Petroleum_emissions_CO2"/>;
</xsl:if><xsl:if test="Petroleum_emissions_SO2">	Petroleum_emissions_SO2 <xsl:value-of select="Petroleum_emissions_SO2"/>;
</xsl:if><xsl:if test="Petroleum_emissions_NOx">	Petroleum_emissions_NOx <xsl:value-of select="Petroleum_emissions_NOx"/>;
</xsl:if><xsl:if test="Solarthermal_emissions_CO2">	Solarthermal_emissions_CO2 <xsl:value-of select="Solarthermal_emissions_CO2"/>;
</xsl:if><xsl:if test="Solarthermal_emissions_SO2">	Solarthermal_emissions_SO2 <xsl:value-of select="Solarthermal_emissions_SO2"/>;
</xsl:if><xsl:if test="Solarthermal_emissions_NOx">	Solarthermal_emissions_NOx <xsl:value-of select="Solarthermal_emissions_NOx"/>;
</xsl:if><xsl:if test="Total_emissions_CO2">	Total_emissions_CO2 <xsl:value-of select="Total_emissions_CO2"/>;
</xsl:if><xsl:if test="Total_emissions_SO2">	Total_emissions_SO2 <xsl:value-of select="Total_emissions_SO2"/>;
</xsl:if><xsl:if test="Total_emissions_NOx">	Total_emissions_NOx <xsl:value-of select="Total_emissions_NOx"/>;
</xsl:if><xsl:if test="Total_energy_out">	Total_energy_out <xsl:value-of select="Total_energy_out"/>;
</xsl:if><xsl:if test="Region">	Region <xsl:value-of select="Region"/>;
</xsl:if><xsl:if test="cycle_interval">	cycle_interval <xsl:value-of select="cycle_interval"/>;
</xsl:if>}
</xsl:for-each></xsl:for-each><xsl:for-each select="residential">
##############################################
# residential module
module residential {
	version.major <xsl:value-of select="version.major"/>;
	version.minor <xsl:value-of select="version.minor"/>;
	default_line_voltage <xsl:value-of select="default_line_voltage"/>;
	default_line_current <xsl:value-of select="default_line_current"/>;
	default_outdoor_temperature <xsl:value-of select="default_outdoor_temperature"/>;
	default_humidity <xsl:value-of select="default_humidity"/>;
	default_solar <xsl:value-of select="default_solar"/>;
	default_etp_iterations <xsl:value-of select="default_etp_iterations"/>;
	ANSI_voltage_check <xsl:value-of select="ANSI_voltage_check"/>;
	implicit_enduses <xsl:value-of select="implicit_enduses"/>;
	house_low_temperature_warning[degF] <xsl:value-of select="house_low_temperature_warning[degF]"/>;
	house_high_temperature_warning[degF] <xsl:value-of select="house_high_temperature_warning[degF]"/>;
	thermostat_control_warning <xsl:value-of select="thermostat_control_warning"/>;
	system_dwell_time[s] <xsl:value-of select="system_dwell_time[s]"/>;
	aux_cutin_temperature[degF] <xsl:value-of select="aux_cutin_temperature[degF]"/>;
}

# residential::residential_enduse objects
<xsl:for-each select="residential_enduse_list/residential_enduse"><a name="#GLM.{name}"/>object residential_enduse:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="shape">	shape <xsl:value-of select="shape"/>;
</xsl:if><xsl:if test="load">	load <xsl:value-of select="load"/>;
</xsl:if><xsl:if test="energy">	energy <xsl:value-of select="energy"/>;
</xsl:if><xsl:if test="power">	power <xsl:value-of select="power"/>;
</xsl:if><xsl:if test="peak_demand">	peak_demand <xsl:value-of select="peak_demand"/>;
</xsl:if><xsl:if test="heatgain">	heatgain <xsl:value-of select="heatgain"/>;
</xsl:if><xsl:if test="cumulative_heatgain">	cumulative_heatgain <xsl:value-of select="cumulative_heatgain"/>;
</xsl:if><xsl:if test="heatgain_fraction">	heatgain_fraction <xsl:value-of select="heatgain_fraction"/>;
</xsl:if><xsl:if test="current_fraction">	current_fraction <xsl:value-of select="current_fraction"/>;
</xsl:if><xsl:if test="impedance_fraction">	impedance_fraction <xsl:value-of select="impedance_fraction"/>;
</xsl:if><xsl:if test="power_fraction">	power_fraction <xsl:value-of select="power_fraction"/>;
</xsl:if><xsl:if test="power_factor">	power_factor <xsl:value-of select="power_factor"/>;
</xsl:if><xsl:if test="constant_power">	constant_power <xsl:value-of select="constant_power"/>;
</xsl:if><xsl:if test="constant_current">	constant_current <xsl:value-of select="constant_current"/>;
</xsl:if><xsl:if test="constant_admittance">	constant_admittance <xsl:value-of select="constant_admittance"/>;
</xsl:if><xsl:if test="voltage_factor">	voltage_factor <xsl:value-of select="voltage_factor"/>;
</xsl:if><xsl:if test="breaker_amps">	breaker_amps <xsl:value-of select="breaker_amps"/>;
</xsl:if><xsl:if test="configuration">	configuration <xsl:value-of select="configuration"/>;
</xsl:if><xsl:if test="override">	override <xsl:value-of select="override"/>;
</xsl:if><xsl:if test="power_state">	power_state <xsl:value-of select="power_state"/>;
</xsl:if>}
</xsl:for-each>
# residential::house_a objects
<xsl:for-each select="house_a_list/house_a"><a name="#GLM.{name}"/>object house_a:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="floor_area">	floor_area <xsl:value-of select="floor_area"/>;
</xsl:if><xsl:if test="gross_wall_area">	gross_wall_area <xsl:value-of select="gross_wall_area"/>;
</xsl:if><xsl:if test="ceiling_height">	ceiling_height <xsl:value-of select="ceiling_height"/>;
</xsl:if><xsl:if test="aspect_ratio">	aspect_ratio <xsl:value-of select="aspect_ratio"/>;
</xsl:if><xsl:if test="envelope_UA">	envelope_UA <xsl:value-of select="envelope_UA"/>;
</xsl:if><xsl:if test="window_wall_ratio">	window_wall_ratio <xsl:value-of select="window_wall_ratio"/>;
</xsl:if><xsl:if test="glazing_shgc">	glazing_shgc <xsl:value-of select="glazing_shgc"/>;
</xsl:if><xsl:if test="airchange_per_hour">	airchange_per_hour <xsl:value-of select="airchange_per_hour"/>;
</xsl:if><xsl:if test="solar_gain">	solar_gain <xsl:value-of select="solar_gain"/>;
</xsl:if><xsl:if test="heat_cool_gain">	heat_cool_gain <xsl:value-of select="heat_cool_gain"/>;
</xsl:if><xsl:if test="thermostat_deadband">	thermostat_deadband <xsl:value-of select="thermostat_deadband"/>;
</xsl:if><xsl:if test="heating_setpoint">	heating_setpoint <xsl:value-of select="heating_setpoint"/>;
</xsl:if><xsl:if test="cooling_setpoint">	cooling_setpoint <xsl:value-of select="cooling_setpoint"/>;
</xsl:if><xsl:if test="design_heating_capacity">	design_heating_capacity <xsl:value-of select="design_heating_capacity"/>;
</xsl:if><xsl:if test="design_cooling_capacity">	design_cooling_capacity <xsl:value-of select="design_cooling_capacity"/>;
</xsl:if><xsl:if test="heating_COP">	heating_COP <xsl:value-of select="heating_COP"/>;
</xsl:if><xsl:if test="cooling_COP">	cooling_COP <xsl:value-of select="cooling_COP"/>;
</xsl:if><xsl:if test="COP_coeff">	COP_coeff <xsl:value-of select="COP_coeff"/>;
</xsl:if><xsl:if test="air_temperature">	air_temperature <xsl:value-of select="air_temperature"/>;
</xsl:if><xsl:if test="outside_temp">	outside_temp <xsl:value-of select="outside_temp"/>;
</xsl:if><xsl:if test="mass_temperature">	mass_temperature <xsl:value-of select="mass_temperature"/>;
</xsl:if><xsl:if test="mass_heat_coeff">	mass_heat_coeff <xsl:value-of select="mass_heat_coeff"/>;
</xsl:if><xsl:if test="outdoor_temperature">	outdoor_temperature <xsl:value-of select="outdoor_temperature"/>;
</xsl:if><xsl:if test="house_thermal_mass">	house_thermal_mass <xsl:value-of select="house_thermal_mass"/>;
</xsl:if><xsl:if test="heat_mode">	heat_mode <xsl:value-of select="heat_mode"/>;
</xsl:if><xsl:if test="hc_mode">	hc_mode <xsl:value-of select="hc_mode"/>;
</xsl:if><xsl:if test="houseload">	houseload <xsl:value-of select="houseload"/>;
</xsl:if><xsl:if test="houseload.energy">	houseload.energy <xsl:value-of select="houseload.energy"/>;
</xsl:if><xsl:if test="houseload.power">	houseload.power <xsl:value-of select="houseload.power"/>;
</xsl:if><xsl:if test="houseload.peak_demand">	houseload.peak_demand <xsl:value-of select="houseload.peak_demand"/>;
</xsl:if><xsl:if test="houseload.heatgain">	houseload.heatgain <xsl:value-of select="houseload.heatgain"/>;
</xsl:if><xsl:if test="houseload.cumulative_heatgain">	houseload.cumulative_heatgain <xsl:value-of select="houseload.cumulative_heatgain"/>;
</xsl:if><xsl:if test="houseload.heatgain_fraction">	houseload.heatgain_fraction <xsl:value-of select="houseload.heatgain_fraction"/>;
</xsl:if><xsl:if test="houseload.current_fraction">	houseload.current_fraction <xsl:value-of select="houseload.current_fraction"/>;
</xsl:if><xsl:if test="houseload.impedance_fraction">	houseload.impedance_fraction <xsl:value-of select="houseload.impedance_fraction"/>;
</xsl:if><xsl:if test="houseload.power_fraction">	houseload.power_fraction <xsl:value-of select="houseload.power_fraction"/>;
</xsl:if><xsl:if test="houseload.power_factor">	houseload.power_factor <xsl:value-of select="houseload.power_factor"/>;
</xsl:if><xsl:if test="houseload.constant_power">	houseload.constant_power <xsl:value-of select="houseload.constant_power"/>;
</xsl:if><xsl:if test="houseload.constant_current">	houseload.constant_current <xsl:value-of select="houseload.constant_current"/>;
</xsl:if><xsl:if test="houseload.constant_admittance">	houseload.constant_admittance <xsl:value-of select="houseload.constant_admittance"/>;
</xsl:if><xsl:if test="houseload.voltage_factor">	houseload.voltage_factor <xsl:value-of select="houseload.voltage_factor"/>;
</xsl:if><xsl:if test="houseload.breaker_amps">	houseload.breaker_amps <xsl:value-of select="houseload.breaker_amps"/>;
</xsl:if><xsl:if test="houseload.configuration">	houseload.configuration <xsl:value-of select="houseload.configuration"/>;
</xsl:if>}
</xsl:for-each>
# residential::house objects
<xsl:for-each select="house_list/house"><a name="#GLM.{name}"/>object house:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="weather">	weather <a href="#GLM.{weather}"><xsl:value-of select="weather"/></a>;
</xsl:if><xsl:if test="floor_area">	floor_area <xsl:value-of select="floor_area"/>;
</xsl:if><xsl:if test="gross_wall_area">	gross_wall_area <xsl:value-of select="gross_wall_area"/>;
</xsl:if><xsl:if test="ceiling_height">	ceiling_height <xsl:value-of select="ceiling_height"/>;
</xsl:if><xsl:if test="aspect_ratio">	aspect_ratio <xsl:value-of select="aspect_ratio"/>;
</xsl:if><xsl:if test="envelope_UA">	envelope_UA <xsl:value-of select="envelope_UA"/>;
</xsl:if><xsl:if test="window_wall_ratio">	window_wall_ratio <xsl:value-of select="window_wall_ratio"/>;
</xsl:if><xsl:if test="number_of_doors">	number_of_doors <xsl:value-of select="number_of_doors"/>;
</xsl:if><xsl:if test="exterior_wall_fraction">	exterior_wall_fraction <xsl:value-of select="exterior_wall_fraction"/>;
</xsl:if><xsl:if test="interior_exterior_wall_ratio">	interior_exterior_wall_ratio <xsl:value-of select="interior_exterior_wall_ratio"/>;
</xsl:if><xsl:if test="exterior_ceiling_fraction">	exterior_ceiling_fraction <xsl:value-of select="exterior_ceiling_fraction"/>;
</xsl:if><xsl:if test="exterior_floor_fraction">	exterior_floor_fraction <xsl:value-of select="exterior_floor_fraction"/>;
</xsl:if><xsl:if test="window_shading">	window_shading <xsl:value-of select="window_shading"/>;
</xsl:if><xsl:if test="window_exterior_transmission_coefficient">	window_exterior_transmission_coefficient <xsl:value-of select="window_exterior_transmission_coefficient"/>;
</xsl:if><xsl:if test="solar_heatgain_factor">	solar_heatgain_factor <xsl:value-of select="solar_heatgain_factor"/>;
</xsl:if><xsl:if test="airchange_per_hour">	airchange_per_hour <xsl:value-of select="airchange_per_hour"/>;
</xsl:if><xsl:if test="airchange_UA">	airchange_UA <xsl:value-of select="airchange_UA"/>;
</xsl:if><xsl:if test="UA">	UA <xsl:value-of select="UA"/>;
</xsl:if><xsl:if test="internal_gain">	internal_gain <xsl:value-of select="internal_gain"/>;
</xsl:if><xsl:if test="solar_gain">	solar_gain <xsl:value-of select="solar_gain"/>;
</xsl:if><xsl:if test="incident_solar_radiation">	incident_solar_radiation <xsl:value-of select="incident_solar_radiation"/>;
</xsl:if><xsl:if test="heat_cool_gain">	heat_cool_gain <xsl:value-of select="heat_cool_gain"/>;
</xsl:if><xsl:if test="include_solar_quadrant">	include_solar_quadrant <xsl:value-of select="include_solar_quadrant"/>;
</xsl:if><xsl:if test="heating_cop_curve">	heating_cop_curve <xsl:value-of select="heating_cop_curve"/>;
</xsl:if><xsl:if test="heating_cap_curve">	heating_cap_curve <xsl:value-of select="heating_cap_curve"/>;
</xsl:if><xsl:if test="cooling_cop_curve">	cooling_cop_curve <xsl:value-of select="cooling_cop_curve"/>;
</xsl:if><xsl:if test="cooling_cap_curve">	cooling_cap_curve <xsl:value-of select="cooling_cap_curve"/>;
</xsl:if><xsl:if test="use_latent_heat">	use_latent_heat <xsl:value-of select="use_latent_heat"/>;
</xsl:if><xsl:if test="include_fan_heatgain">	include_fan_heatgain <xsl:value-of select="include_fan_heatgain"/>;
</xsl:if><xsl:if test="thermostat_deadband">	thermostat_deadband <xsl:value-of select="thermostat_deadband"/>;
</xsl:if><xsl:if test="dlc_offset">	dlc_offset <xsl:value-of select="dlc_offset"/>;
</xsl:if><xsl:if test="thermostat_cycle_time">	thermostat_cycle_time <xsl:value-of select="thermostat_cycle_time"/>;
</xsl:if><xsl:if test="thermostat_last_cycle_time">	thermostat_last_cycle_time <xsl:value-of select="thermostat_last_cycle_time"/>;
</xsl:if><xsl:if test="heating_setpoint">	heating_setpoint <xsl:value-of select="heating_setpoint"/>;
</xsl:if><xsl:if test="cooling_setpoint">	cooling_setpoint <xsl:value-of select="cooling_setpoint"/>;
</xsl:if><xsl:if test="design_heating_setpoint">	design_heating_setpoint <xsl:value-of select="design_heating_setpoint"/>;
</xsl:if><xsl:if test="design_cooling_setpoint">	design_cooling_setpoint <xsl:value-of select="design_cooling_setpoint"/>;
</xsl:if><xsl:if test="over_sizing_factor">	over_sizing_factor <xsl:value-of select="over_sizing_factor"/>;
</xsl:if><xsl:if test="design_heating_capacity">	design_heating_capacity <xsl:value-of select="design_heating_capacity"/>;
</xsl:if><xsl:if test="design_cooling_capacity">	design_cooling_capacity <xsl:value-of select="design_cooling_capacity"/>;
</xsl:if><xsl:if test="cooling_design_temperature">	cooling_design_temperature <xsl:value-of select="cooling_design_temperature"/>;
</xsl:if><xsl:if test="heating_design_temperature">	heating_design_temperature <xsl:value-of select="heating_design_temperature"/>;
</xsl:if><xsl:if test="design_peak_solar">	design_peak_solar <xsl:value-of select="design_peak_solar"/>;
</xsl:if><xsl:if test="design_internal_gains">	design_internal_gains <xsl:value-of select="design_internal_gains"/>;
</xsl:if><xsl:if test="air_heat_fraction">	air_heat_fraction <xsl:value-of select="air_heat_fraction"/>;
</xsl:if><xsl:if test="mass_solar_gain_fraction">	mass_solar_gain_fraction <xsl:value-of select="mass_solar_gain_fraction"/>;
</xsl:if><xsl:if test="mass_internal_gain_fraction">	mass_internal_gain_fraction <xsl:value-of select="mass_internal_gain_fraction"/>;
</xsl:if><xsl:if test="auxiliary_heat_capacity">	auxiliary_heat_capacity <xsl:value-of select="auxiliary_heat_capacity"/>;
</xsl:if><xsl:if test="aux_heat_deadband">	aux_heat_deadband <xsl:value-of select="aux_heat_deadband"/>;
</xsl:if><xsl:if test="aux_heat_temperature_lockout">	aux_heat_temperature_lockout <xsl:value-of select="aux_heat_temperature_lockout"/>;
</xsl:if><xsl:if test="aux_heat_time_delay">	aux_heat_time_delay <xsl:value-of select="aux_heat_time_delay"/>;
</xsl:if><xsl:if test="cooling_supply_air_temp">	cooling_supply_air_temp <xsl:value-of select="cooling_supply_air_temp"/>;
</xsl:if><xsl:if test="heating_supply_air_temp">	heating_supply_air_temp <xsl:value-of select="heating_supply_air_temp"/>;
</xsl:if><xsl:if test="duct_pressure_drop">	duct_pressure_drop <xsl:value-of select="duct_pressure_drop"/>;
</xsl:if><xsl:if test="fan_design_power">	fan_design_power <xsl:value-of select="fan_design_power"/>;
</xsl:if><xsl:if test="fan_low_power_fraction">	fan_low_power_fraction <xsl:value-of select="fan_low_power_fraction"/>;
</xsl:if><xsl:if test="fan_power">	fan_power <xsl:value-of select="fan_power"/>;
</xsl:if><xsl:if test="fan_design_airflow">	fan_design_airflow <xsl:value-of select="fan_design_airflow"/>;
</xsl:if><xsl:if test="fan_impedance_fraction">	fan_impedance_fraction <xsl:value-of select="fan_impedance_fraction"/>;
</xsl:if><xsl:if test="fan_power_fraction">	fan_power_fraction <xsl:value-of select="fan_power_fraction"/>;
</xsl:if><xsl:if test="fan_current_fraction">	fan_current_fraction <xsl:value-of select="fan_current_fraction"/>;
</xsl:if><xsl:if test="fan_power_factor">	fan_power_factor <xsl:value-of select="fan_power_factor"/>;
</xsl:if><xsl:if test="heating_demand">	heating_demand <xsl:value-of select="heating_demand"/>;
</xsl:if><xsl:if test="cooling_demand">	cooling_demand <xsl:value-of select="cooling_demand"/>;
</xsl:if><xsl:if test="heating_COP">	heating_COP <xsl:value-of select="heating_COP"/>;
</xsl:if><xsl:if test="cooling_COP">	cooling_COP <xsl:value-of select="cooling_COP"/>;
</xsl:if><xsl:if test="air_temperature">	air_temperature <xsl:value-of select="air_temperature"/>;
</xsl:if><xsl:if test="outdoor_temperature">	outdoor_temperature <xsl:value-of select="outdoor_temperature"/>;
</xsl:if><xsl:if test="outdoor_rh">	outdoor_rh <xsl:value-of select="outdoor_rh"/>;
</xsl:if><xsl:if test="mass_heat_capacity">	mass_heat_capacity <xsl:value-of select="mass_heat_capacity"/>;
</xsl:if><xsl:if test="mass_heat_coeff">	mass_heat_coeff <xsl:value-of select="mass_heat_coeff"/>;
</xsl:if><xsl:if test="mass_temperature">	mass_temperature <xsl:value-of select="mass_temperature"/>;
</xsl:if><xsl:if test="air_volume">	air_volume <xsl:value-of select="air_volume"/>;
</xsl:if><xsl:if test="air_mass">	air_mass <xsl:value-of select="air_mass"/>;
</xsl:if><xsl:if test="air_heat_capacity">	air_heat_capacity <xsl:value-of select="air_heat_capacity"/>;
</xsl:if><xsl:if test="latent_load_fraction">	latent_load_fraction <xsl:value-of select="latent_load_fraction"/>;
</xsl:if><xsl:if test="total_thermal_mass_per_floor_area">	total_thermal_mass_per_floor_area <xsl:value-of select="total_thermal_mass_per_floor_area"/>;
</xsl:if><xsl:if test="interior_surface_heat_transfer_coeff">	interior_surface_heat_transfer_coeff <xsl:value-of select="interior_surface_heat_transfer_coeff"/>;
</xsl:if><xsl:if test="number_of_stories">	number_of_stories <xsl:value-of select="number_of_stories"/>;
</xsl:if><xsl:if test="is_AUX_on">	is_AUX_on <xsl:value-of select="is_AUX_on"/>;
</xsl:if><xsl:if test="is_HEAT_on">	is_HEAT_on <xsl:value-of select="is_HEAT_on"/>;
</xsl:if><xsl:if test="is_COOL_on">	is_COOL_on <xsl:value-of select="is_COOL_on"/>;
</xsl:if><xsl:if test="thermal_storage_present">	thermal_storage_present <xsl:value-of select="thermal_storage_present"/>;
</xsl:if><xsl:if test="thermal_storage_in_use">	thermal_storage_in_use <xsl:value-of select="thermal_storage_in_use"/>;
</xsl:if><xsl:if test="system_type">	system_type <xsl:value-of select="system_type"/>;
</xsl:if><xsl:if test="auxiliary_strategy">	auxiliary_strategy <xsl:value-of select="auxiliary_strategy"/>;
</xsl:if><xsl:if test="system_mode">	system_mode <xsl:value-of select="system_mode"/>;
</xsl:if><xsl:if test="last_system_mode">	last_system_mode <xsl:value-of select="last_system_mode"/>;
</xsl:if><xsl:if test="heating_system_type">	heating_system_type <xsl:value-of select="heating_system_type"/>;
</xsl:if><xsl:if test="cooling_system_type">	cooling_system_type <xsl:value-of select="cooling_system_type"/>;
</xsl:if><xsl:if test="auxiliary_system_type">	auxiliary_system_type <xsl:value-of select="auxiliary_system_type"/>;
</xsl:if><xsl:if test="fan_type">	fan_type <xsl:value-of select="fan_type"/>;
</xsl:if><xsl:if test="thermal_integrity_level">	thermal_integrity_level <xsl:value-of select="thermal_integrity_level"/>;
</xsl:if><xsl:if test="glass_type">	glass_type <xsl:value-of select="glass_type"/>;
</xsl:if><xsl:if test="window_frame">	window_frame <xsl:value-of select="window_frame"/>;
</xsl:if><xsl:if test="glazing_treatment">	glazing_treatment <xsl:value-of select="glazing_treatment"/>;
</xsl:if><xsl:if test="glazing_layers">	glazing_layers <xsl:value-of select="glazing_layers"/>;
</xsl:if><xsl:if test="motor_model">	motor_model <xsl:value-of select="motor_model"/>;
</xsl:if><xsl:if test="motor_efficiency">	motor_efficiency <xsl:value-of select="motor_efficiency"/>;
</xsl:if><xsl:if test="last_mode_timer">	last_mode_timer <xsl:value-of select="last_mode_timer"/>;
</xsl:if><xsl:if test="hvac_motor_efficiency">	hvac_motor_efficiency <xsl:value-of select="hvac_motor_efficiency"/>;
</xsl:if><xsl:if test="hvac_motor_loss_power_factor">	hvac_motor_loss_power_factor <xsl:value-of select="hvac_motor_loss_power_factor"/>;
</xsl:if><xsl:if test="Rroof">	Rroof <xsl:value-of select="Rroof"/>;
</xsl:if><xsl:if test="Rwall">	Rwall <xsl:value-of select="Rwall"/>;
</xsl:if><xsl:if test="Rfloor">	Rfloor <xsl:value-of select="Rfloor"/>;
</xsl:if><xsl:if test="Rwindows">	Rwindows <xsl:value-of select="Rwindows"/>;
</xsl:if><xsl:if test="Rdoors">	Rdoors <xsl:value-of select="Rdoors"/>;
</xsl:if><xsl:if test="hvac_breaker_rating">	hvac_breaker_rating <xsl:value-of select="hvac_breaker_rating"/>;
</xsl:if><xsl:if test="hvac_power_factor">	hvac_power_factor <xsl:value-of select="hvac_power_factor"/>;
</xsl:if><xsl:if test="hvac_load">	hvac_load <xsl:value-of select="hvac_load"/>;
</xsl:if><xsl:if test="last_heating_load">	last_heating_load <xsl:value-of select="last_heating_load"/>;
</xsl:if><xsl:if test="last_cooling_load">	last_cooling_load <xsl:value-of select="last_cooling_load"/>;
</xsl:if><xsl:if test="hvac_power">	hvac_power <xsl:value-of select="hvac_power"/>;
</xsl:if><xsl:if test="total_load">	total_load <xsl:value-of select="total_load"/>;
</xsl:if><xsl:if test="panel">	panel <xsl:value-of select="panel"/>;
</xsl:if><xsl:if test="panel.energy">	panel.energy <xsl:value-of select="panel.energy"/>;
</xsl:if><xsl:if test="panel.power">	panel.power <xsl:value-of select="panel.power"/>;
</xsl:if><xsl:if test="panel.peak_demand">	panel.peak_demand <xsl:value-of select="panel.peak_demand"/>;
</xsl:if><xsl:if test="panel.heatgain">	panel.heatgain <xsl:value-of select="panel.heatgain"/>;
</xsl:if><xsl:if test="panel.cumulative_heatgain">	panel.cumulative_heatgain <xsl:value-of select="panel.cumulative_heatgain"/>;
</xsl:if><xsl:if test="panel.heatgain_fraction">	panel.heatgain_fraction <xsl:value-of select="panel.heatgain_fraction"/>;
</xsl:if><xsl:if test="panel.current_fraction">	panel.current_fraction <xsl:value-of select="panel.current_fraction"/>;
</xsl:if><xsl:if test="panel.impedance_fraction">	panel.impedance_fraction <xsl:value-of select="panel.impedance_fraction"/>;
</xsl:if><xsl:if test="panel.power_fraction">	panel.power_fraction <xsl:value-of select="panel.power_fraction"/>;
</xsl:if><xsl:if test="panel.power_factor">	panel.power_factor <xsl:value-of select="panel.power_factor"/>;
</xsl:if><xsl:if test="panel.constant_power">	panel.constant_power <xsl:value-of select="panel.constant_power"/>;
</xsl:if><xsl:if test="panel.constant_current">	panel.constant_current <xsl:value-of select="panel.constant_current"/>;
</xsl:if><xsl:if test="panel.constant_admittance">	panel.constant_admittance <xsl:value-of select="panel.constant_admittance"/>;
</xsl:if><xsl:if test="panel.voltage_factor">	panel.voltage_factor <xsl:value-of select="panel.voltage_factor"/>;
</xsl:if><xsl:if test="panel.breaker_amps">	panel.breaker_amps <xsl:value-of select="panel.breaker_amps"/>;
</xsl:if><xsl:if test="panel.configuration">	panel.configuration <xsl:value-of select="panel.configuration"/>;
</xsl:if><xsl:if test="design_internal_gain_density">	design_internal_gain_density <xsl:value-of select="design_internal_gain_density"/>;
</xsl:if><xsl:if test="compressor_on">	compressor_on <xsl:value-of select="compressor_on"/>;
</xsl:if><xsl:if test="compressor_count">	compressor_count <xsl:value-of select="compressor_count"/>;
</xsl:if><xsl:if test="hvac_last_on">	hvac_last_on <xsl:value-of select="hvac_last_on"/>;
</xsl:if><xsl:if test="hvac_last_off">	hvac_last_off <xsl:value-of select="hvac_last_off"/>;
</xsl:if><xsl:if test="hvac_period_length">	hvac_period_length <xsl:value-of select="hvac_period_length"/>;
</xsl:if><xsl:if test="hvac_duty_cycle">	hvac_duty_cycle <xsl:value-of select="hvac_duty_cycle"/>;
</xsl:if><xsl:if test="a">	a <xsl:value-of select="a"/>;
</xsl:if><xsl:if test="b">	b <xsl:value-of select="b"/>;
</xsl:if><xsl:if test="c">	c <xsl:value-of select="c"/>;
</xsl:if><xsl:if test="d">	d <xsl:value-of select="d"/>;
</xsl:if><xsl:if test="c1">	c1 <xsl:value-of select="c1"/>;
</xsl:if><xsl:if test="c2">	c2 <xsl:value-of select="c2"/>;
</xsl:if><xsl:if test="A3">	A3 <xsl:value-of select="A3"/>;
</xsl:if><xsl:if test="A4">	A4 <xsl:value-of select="A4"/>;
</xsl:if><xsl:if test="k1">	k1 <xsl:value-of select="k1"/>;
</xsl:if><xsl:if test="k2">	k2 <xsl:value-of select="k2"/>;
</xsl:if><xsl:if test="r1">	r1 <xsl:value-of select="r1"/>;
</xsl:if><xsl:if test="r2">	r2 <xsl:value-of select="r2"/>;
</xsl:if><xsl:if test="Teq">	Teq <xsl:value-of select="Teq"/>;
</xsl:if><xsl:if test="Tevent">	Tevent <xsl:value-of select="Tevent"/>;
</xsl:if><xsl:if test="Qi">	Qi <xsl:value-of select="Qi"/>;
</xsl:if><xsl:if test="Qa">	Qa <xsl:value-of select="Qa"/>;
</xsl:if><xsl:if test="Qm">	Qm <xsl:value-of select="Qm"/>;
</xsl:if><xsl:if test="Qh">	Qh <xsl:value-of select="Qh"/>;
</xsl:if><xsl:if test="Qlatent">	Qlatent <xsl:value-of select="Qlatent"/>;
</xsl:if><xsl:if test="dTair">	dTair <xsl:value-of select="dTair"/>;
</xsl:if><xsl:if test="adj_cooling_cap">	adj_cooling_cap <xsl:value-of select="adj_cooling_cap"/>;
</xsl:if><xsl:if test="adj_heating_cap">	adj_heating_cap <xsl:value-of select="adj_heating_cap"/>;
</xsl:if><xsl:if test="adj_cooling_cop">	adj_cooling_cop <xsl:value-of select="adj_cooling_cop"/>;
</xsl:if><xsl:if test="adj_heating_cop">	adj_heating_cop <xsl:value-of select="adj_heating_cop"/>;
</xsl:if><xsl:if test="thermostat_control">	thermostat_control <xsl:value-of select="thermostat_control"/>;
</xsl:if>}
</xsl:for-each>
# residential::waterheater objects
<xsl:for-each select="waterheater_list/waterheater"><a name="#GLM.{name}"/>object waterheater:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="tank_volume">	tank_volume <xsl:value-of select="tank_volume"/>;
</xsl:if><xsl:if test="tank_UA">	tank_UA <xsl:value-of select="tank_UA"/>;
</xsl:if><xsl:if test="tank_diameter">	tank_diameter <xsl:value-of select="tank_diameter"/>;
</xsl:if><xsl:if test="water_demand">	water_demand <xsl:value-of select="water_demand"/>;
</xsl:if><xsl:if test="heating_element_capacity">	heating_element_capacity <xsl:value-of select="heating_element_capacity"/>;
</xsl:if><xsl:if test="inlet_water_temperature">	inlet_water_temperature <xsl:value-of select="inlet_water_temperature"/>;
</xsl:if><xsl:if test="heat_mode">	heat_mode <xsl:value-of select="heat_mode"/>;
</xsl:if><xsl:if test="location">	location <xsl:value-of select="location"/>;
</xsl:if><xsl:if test="tank_setpoint">	tank_setpoint <xsl:value-of select="tank_setpoint"/>;
</xsl:if><xsl:if test="thermostat_deadband">	thermostat_deadband <xsl:value-of select="thermostat_deadband"/>;
</xsl:if><xsl:if test="temperature">	temperature <xsl:value-of select="temperature"/>;
</xsl:if><xsl:if test="height">	height <xsl:value-of select="height"/>;
</xsl:if><xsl:if test="demand">	demand <xsl:value-of select="demand"/>;
</xsl:if><xsl:if test="actual_load">	actual_load <xsl:value-of select="actual_load"/>;
</xsl:if><xsl:if test="previous_load">	previous_load <xsl:value-of select="previous_load"/>;
</xsl:if><xsl:if test="actual_power">	actual_power <xsl:value-of select="actual_power"/>;
</xsl:if><xsl:if test="is_waterheater_on">	is_waterheater_on <xsl:value-of select="is_waterheater_on"/>;
</xsl:if>}
</xsl:for-each>
# residential::lights objects
<xsl:for-each select="lights_list/lights"><a name="#GLM.{name}"/>object lights:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="type">	type <xsl:value-of select="type"/>;
</xsl:if><xsl:if test="placement">	placement <xsl:value-of select="placement"/>;
</xsl:if><xsl:if test="installed_power">	installed_power <xsl:value-of select="installed_power"/>;
</xsl:if><xsl:if test="power_density">	power_density <xsl:value-of select="power_density"/>;
</xsl:if><xsl:if test="curtailment">	curtailment <xsl:value-of select="curtailment"/>;
</xsl:if><xsl:if test="demand">	demand <xsl:value-of select="demand"/>;
</xsl:if><xsl:if test="actual_power">	actual_power <xsl:value-of select="actual_power"/>;
</xsl:if>}
</xsl:for-each>
# residential::refrigerator objects
<xsl:for-each select="refrigerator_list/refrigerator"><a name="#GLM.{name}"/>object refrigerator:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="size">	size <xsl:value-of select="size"/>;
</xsl:if><xsl:if test="rated_capacity">	rated_capacity <xsl:value-of select="rated_capacity"/>;
</xsl:if><xsl:if test="temperature">	temperature <xsl:value-of select="temperature"/>;
</xsl:if><xsl:if test="setpoint">	setpoint <xsl:value-of select="setpoint"/>;
</xsl:if><xsl:if test="deadband">	deadband <xsl:value-of select="deadband"/>;
</xsl:if><xsl:if test="cycle_time">	cycle_time <xsl:value-of select="cycle_time"/>;
</xsl:if><xsl:if test="output">	output <xsl:value-of select="output"/>;
</xsl:if><xsl:if test="event_temp">	event_temp <xsl:value-of select="event_temp"/>;
</xsl:if><xsl:if test="UA">	UA <xsl:value-of select="UA"/>;
</xsl:if><xsl:if test="compressor_off_normal_energy">	compressor_off_normal_energy <xsl:value-of select="compressor_off_normal_energy"/>;
</xsl:if><xsl:if test="compressor_off_normal_power">	compressor_off_normal_power <xsl:value-of select="compressor_off_normal_power"/>;
</xsl:if><xsl:if test="compressor_on_normal_energy">	compressor_on_normal_energy <xsl:value-of select="compressor_on_normal_energy"/>;
</xsl:if><xsl:if test="compressor_on_normal_power">	compressor_on_normal_power <xsl:value-of select="compressor_on_normal_power"/>;
</xsl:if><xsl:if test="defrost_energy">	defrost_energy <xsl:value-of select="defrost_energy"/>;
</xsl:if><xsl:if test="defrost_power">	defrost_power <xsl:value-of select="defrost_power"/>;
</xsl:if><xsl:if test="icemaking_energy">	icemaking_energy <xsl:value-of select="icemaking_energy"/>;
</xsl:if><xsl:if test="icemaking_power">	icemaking_power <xsl:value-of select="icemaking_power"/>;
</xsl:if><xsl:if test="ice_making_probability">	ice_making_probability <xsl:value-of select="ice_making_probability"/>;
</xsl:if><xsl:if test="FF_Door_Openings">	FF_Door_Openings <xsl:value-of select="FF_Door_Openings"/>;
</xsl:if><xsl:if test="door_opening_energy">	door_opening_energy <xsl:value-of select="door_opening_energy"/>;
</xsl:if><xsl:if test="door_opening_power">	door_opening_power <xsl:value-of select="door_opening_power"/>;
</xsl:if><xsl:if test="DO_Thershold">	DO_Thershold <xsl:value-of select="DO_Thershold"/>;
</xsl:if><xsl:if test="dr_mode_double">	dr_mode_double <xsl:value-of select="dr_mode_double"/>;
</xsl:if><xsl:if test="energy_needed">	energy_needed <xsl:value-of select="energy_needed"/>;
</xsl:if><xsl:if test="energy_used">	energy_used <xsl:value-of select="energy_used"/>;
</xsl:if><xsl:if test="refrigerator_power">	refrigerator_power <xsl:value-of select="refrigerator_power"/>;
</xsl:if><xsl:if test="icemaker_running">	icemaker_running <xsl:value-of select="icemaker_running"/>;
</xsl:if><xsl:if test="check_DO">	check_DO <xsl:value-of select="check_DO"/>;
</xsl:if><xsl:if test="is_240">	is_240 <xsl:value-of select="is_240"/>;
</xsl:if><xsl:if test="defrostDelayed">	defrostDelayed <xsl:value-of select="defrostDelayed"/>;
</xsl:if><xsl:if test="long_compressor_cycle_due">	long_compressor_cycle_due <xsl:value-of select="long_compressor_cycle_due"/>;
</xsl:if><xsl:if test="long_compressor_cycle_time">	long_compressor_cycle_time <xsl:value-of select="long_compressor_cycle_time"/>;
</xsl:if><xsl:if test="long_compressor_cycle_power">	long_compressor_cycle_power <xsl:value-of select="long_compressor_cycle_power"/>;
</xsl:if><xsl:if test="long_compressor_cycle_energy">	long_compressor_cycle_energy <xsl:value-of select="long_compressor_cycle_energy"/>;
</xsl:if><xsl:if test="long_compressor_cycle_threshold">	long_compressor_cycle_threshold <xsl:value-of select="long_compressor_cycle_threshold"/>;
</xsl:if><xsl:if test="defrost_criterion">	defrost_criterion <xsl:value-of select="defrost_criterion"/>;
</xsl:if><xsl:if test="run_defrost">	run_defrost <xsl:value-of select="run_defrost"/>;
</xsl:if><xsl:if test="door_opening_criterion">	door_opening_criterion <xsl:value-of select="door_opening_criterion"/>;
</xsl:if><xsl:if test="compressor_defrost_time">	compressor_defrost_time <xsl:value-of select="compressor_defrost_time"/>;
</xsl:if><xsl:if test="delay_defrost_time">	delay_defrost_time <xsl:value-of select="delay_defrost_time"/>;
</xsl:if><xsl:if test="daily_door_opening">	daily_door_opening <xsl:value-of select="daily_door_opening"/>;
</xsl:if><xsl:if test="state">	state <xsl:value-of select="state"/>;
</xsl:if>}
</xsl:for-each>
# residential::clotheswasher objects
<xsl:for-each select="clotheswasher_list/clotheswasher"><a name="#GLM.{name}"/>object clotheswasher:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="motor_power">	motor_power <xsl:value-of select="motor_power"/>;
</xsl:if><xsl:if test="circuit_split">	circuit_split <xsl:value-of select="circuit_split"/>;
</xsl:if><xsl:if test="queue">	queue <xsl:value-of select="queue"/>;
</xsl:if><xsl:if test="demand">	demand <xsl:value-of select="demand"/>;
</xsl:if><xsl:if test="energy_meter">	energy_meter <xsl:value-of select="energy_meter"/>;
</xsl:if><xsl:if test="stall_voltage">	stall_voltage <xsl:value-of select="stall_voltage"/>;
</xsl:if><xsl:if test="start_voltage">	start_voltage <xsl:value-of select="start_voltage"/>;
</xsl:if><xsl:if test="stall_impedance">	stall_impedance <xsl:value-of select="stall_impedance"/>;
</xsl:if><xsl:if test="trip_delay">	trip_delay <xsl:value-of select="trip_delay"/>;
</xsl:if><xsl:if test="reset_delay">	reset_delay <xsl:value-of select="reset_delay"/>;
</xsl:if><xsl:if test="state">	state <xsl:value-of select="state"/>;
</xsl:if>}
</xsl:for-each>
# residential::dishwasher objects
<xsl:for-each select="dishwasher_list/dishwasher"><a name="#GLM.{name}"/>object dishwasher:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="installed_power">	installed_power <xsl:value-of select="installed_power"/>;
</xsl:if><xsl:if test="demand">	demand <xsl:value-of select="demand"/>;
</xsl:if>}
</xsl:for-each>
# residential::occupantload objects
<xsl:for-each select="occupantload_list/occupantload"><a name="#GLM.{name}"/>object occupantload:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="number_of_occupants">	number_of_occupants <xsl:value-of select="number_of_occupants"/>;
</xsl:if><xsl:if test="occupancy_fraction">	occupancy_fraction <xsl:value-of select="occupancy_fraction"/>;
</xsl:if><xsl:if test="heatgain_per_person">	heatgain_per_person <xsl:value-of select="heatgain_per_person"/>;
</xsl:if>}
</xsl:for-each>
# residential::plugload objects
<xsl:for-each select="plugload_list/plugload"><a name="#GLM.{name}"/>object plugload:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="circuit_split">	circuit_split <xsl:value-of select="circuit_split"/>;
</xsl:if><xsl:if test="demand">	demand <xsl:value-of select="demand"/>;
</xsl:if><xsl:if test="installed_power">	installed_power <xsl:value-of select="installed_power"/>;
</xsl:if><xsl:if test="actual_power">	actual_power <xsl:value-of select="actual_power"/>;
</xsl:if>}
</xsl:for-each>
# residential::microwave objects
<xsl:for-each select="microwave_list/microwave"><a name="#GLM.{name}"/>object microwave:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="installed_power">	installed_power <xsl:value-of select="installed_power"/>;
</xsl:if><xsl:if test="standby_power">	standby_power <xsl:value-of select="standby_power"/>;
</xsl:if><xsl:if test="circuit_split">	circuit_split <xsl:value-of select="circuit_split"/>;
</xsl:if><xsl:if test="state">	state <xsl:value-of select="state"/>;
</xsl:if><xsl:if test="cycle_length">	cycle_length <xsl:value-of select="cycle_length"/>;
</xsl:if><xsl:if test="runtime">	runtime <xsl:value-of select="runtime"/>;
</xsl:if><xsl:if test="state_time">	state_time <xsl:value-of select="state_time"/>;
</xsl:if>}
</xsl:for-each>
# residential::range objects
<xsl:for-each select="range_list/range"><a name="#GLM.{name}"/>object range:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="installed_power">	installed_power <xsl:value-of select="installed_power"/>;
</xsl:if><xsl:if test="circuit_split">	circuit_split <xsl:value-of select="circuit_split"/>;
</xsl:if><xsl:if test="demand">	demand <xsl:value-of select="demand"/>;
</xsl:if><xsl:if test="energy_meter">	energy_meter <xsl:value-of select="energy_meter"/>;
</xsl:if>}
</xsl:for-each>
# residential::freezer objects
<xsl:for-each select="freezer_list/freezer"><a name="#GLM.{name}"/>object freezer:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="size">	size <xsl:value-of select="size"/>;
</xsl:if><xsl:if test="rated_capacity">	rated_capacity <xsl:value-of select="rated_capacity"/>;
</xsl:if><xsl:if test="temperature">	temperature <xsl:value-of select="temperature"/>;
</xsl:if><xsl:if test="setpoint">	setpoint <xsl:value-of select="setpoint"/>;
</xsl:if><xsl:if test="deadband">	deadband <xsl:value-of select="deadband"/>;
</xsl:if><xsl:if test="next_time">	next_time <xsl:value-of select="next_time"/>;
</xsl:if><xsl:if test="output">	output <xsl:value-of select="output"/>;
</xsl:if><xsl:if test="event_temp">	event_temp <xsl:value-of select="event_temp"/>;
</xsl:if><xsl:if test="UA">	UA <xsl:value-of select="UA"/>;
</xsl:if><xsl:if test="state">	state <xsl:value-of select="state"/>;
</xsl:if>}
</xsl:for-each>
# residential::dryer objects
<xsl:for-each select="dryer_list/dryer"><a name="#GLM.{name}"/>object dryer:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="motor_power">	motor_power <xsl:value-of select="motor_power"/>;
</xsl:if><xsl:if test="coil_power">	coil_power <xsl:value-of select="coil_power"/>;
</xsl:if><xsl:if test="circuit_split">	circuit_split <xsl:value-of select="circuit_split"/>;
</xsl:if><xsl:if test="demand">	demand <xsl:value-of select="demand"/>;
</xsl:if><xsl:if test="queue">	queue <xsl:value-of select="queue"/>;
</xsl:if><xsl:if test="stall_voltage">	stall_voltage <xsl:value-of select="stall_voltage"/>;
</xsl:if><xsl:if test="start_voltage">	start_voltage <xsl:value-of select="start_voltage"/>;
</xsl:if><xsl:if test="stall_impedance">	stall_impedance <xsl:value-of select="stall_impedance"/>;
</xsl:if><xsl:if test="trip_delay">	trip_delay <xsl:value-of select="trip_delay"/>;
</xsl:if><xsl:if test="reset_delay">	reset_delay <xsl:value-of select="reset_delay"/>;
</xsl:if><xsl:if test="actual_power">	actual_power <xsl:value-of select="actual_power"/>;
</xsl:if><xsl:if test="state">	state <xsl:value-of select="state"/>;
</xsl:if>}
</xsl:for-each>
# residential::evcharger objects
<xsl:for-each select="evcharger_list/evcharger"><a name="#GLM.{name}"/>object evcharger:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="charger_type">	charger_type <xsl:value-of select="charger_type"/>;
</xsl:if><xsl:if test="vehicle_type">	vehicle_type <xsl:value-of select="vehicle_type"/>;
</xsl:if><xsl:if test="state">	state <xsl:value-of select="state"/>;
</xsl:if><xsl:if test="p_go_home">	p_go_home <xsl:value-of select="p_go_home"/>;
</xsl:if><xsl:if test="p_go_work">	p_go_work <xsl:value-of select="p_go_work"/>;
</xsl:if><xsl:if test="work_dist">	work_dist <xsl:value-of select="work_dist"/>;
</xsl:if><xsl:if test="capacity">	capacity <xsl:value-of select="capacity"/>;
</xsl:if><xsl:if test="charge">	charge <xsl:value-of select="charge"/>;
</xsl:if><xsl:if test="charge_at_work">	charge_at_work <xsl:value-of select="charge_at_work"/>;
</xsl:if><xsl:if test="charge_throttle">	charge_throttle <xsl:value-of select="charge_throttle"/>;
</xsl:if><xsl:if test="charger_efficiency">	charger_efficiency <xsl:value-of select="charger_efficiency"/>;
</xsl:if><xsl:if test="power_train_efficiency">	power_train_efficiency <xsl:value-of select="power_train_efficiency"/>;
</xsl:if><xsl:if test="mileage_classification">	mileage_classification <xsl:value-of select="mileage_classification"/>;
</xsl:if><xsl:if test="demand_profile">	demand_profile <xsl:value-of select="demand_profile"/>;
</xsl:if>}
</xsl:for-each>
# residential::ZIPload objects
<xsl:for-each select="ZIPload_list/ZIPload"><a name="#GLM.{name}"/>object ZIPload:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="heat_fraction">	heat_fraction <xsl:value-of select="heat_fraction"/>;
</xsl:if><xsl:if test="base_power">	base_power <xsl:value-of select="base_power"/>;
</xsl:if><xsl:if test="power_pf">	power_pf <xsl:value-of select="power_pf"/>;
</xsl:if><xsl:if test="current_pf">	current_pf <xsl:value-of select="current_pf"/>;
</xsl:if><xsl:if test="impedance_pf">	impedance_pf <xsl:value-of select="impedance_pf"/>;
</xsl:if><xsl:if test="is_240">	is_240 <xsl:value-of select="is_240"/>;
</xsl:if><xsl:if test="breaker_val">	breaker_val <xsl:value-of select="breaker_val"/>;
</xsl:if><xsl:if test="actual_power">	actual_power <xsl:value-of select="actual_power"/>;
</xsl:if><xsl:if test="multiplier">	multiplier <xsl:value-of select="multiplier"/>;
</xsl:if><xsl:if test="heatgain_only">	heatgain_only <xsl:value-of select="heatgain_only"/>;
</xsl:if><xsl:if test="demand_response_mode">	demand_response_mode <xsl:value-of select="demand_response_mode"/>;
</xsl:if><xsl:if test="number_of_devices">	number_of_devices <xsl:value-of select="number_of_devices"/>;
</xsl:if><xsl:if test="thermostatic_control_range">	thermostatic_control_range <xsl:value-of select="thermostatic_control_range"/>;
</xsl:if><xsl:if test="number_of_devices_off">	number_of_devices_off <xsl:value-of select="number_of_devices_off"/>;
</xsl:if><xsl:if test="number_of_devices_on">	number_of_devices_on <xsl:value-of select="number_of_devices_on"/>;
</xsl:if><xsl:if test="rate_of_cooling">	rate_of_cooling <xsl:value-of select="rate_of_cooling"/>;
</xsl:if><xsl:if test="rate_of_heating">	rate_of_heating <xsl:value-of select="rate_of_heating"/>;
</xsl:if><xsl:if test="temperature">	temperature <xsl:value-of select="temperature"/>;
</xsl:if><xsl:if test="phi">	phi <xsl:value-of select="phi"/>;
</xsl:if><xsl:if test="demand_rate">	demand_rate <xsl:value-of select="demand_rate"/>;
</xsl:if><xsl:if test="nominal_power">	nominal_power <xsl:value-of select="nominal_power"/>;
</xsl:if><xsl:if test="duty_cycle">	duty_cycle <xsl:value-of select="duty_cycle"/>;
</xsl:if><xsl:if test="recovery_duty_cycle">	recovery_duty_cycle <xsl:value-of select="recovery_duty_cycle"/>;
</xsl:if><xsl:if test="period">	period <xsl:value-of select="period"/>;
</xsl:if><xsl:if test="phase">	phase <xsl:value-of select="phase"/>;
</xsl:if>}
</xsl:for-each>
# residential::thermal_storage objects
<xsl:for-each select="thermal_storage_list/thermal_storage"><a name="#GLM.{name}"/>object thermal_storage:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="total_capacity">	total_capacity <xsl:value-of select="total_capacity"/>;
</xsl:if><xsl:if test="stored_capacity">	stored_capacity <xsl:value-of select="stored_capacity"/>;
</xsl:if><xsl:if test="recharge_power">	recharge_power <xsl:value-of select="recharge_power"/>;
</xsl:if><xsl:if test="discharge_power">	discharge_power <xsl:value-of select="discharge_power"/>;
</xsl:if><xsl:if test="recharge_pf">	recharge_pf <xsl:value-of select="recharge_pf"/>;
</xsl:if><xsl:if test="discharge_pf">	discharge_pf <xsl:value-of select="discharge_pf"/>;
</xsl:if><xsl:if test="discharge_schedule_type">	discharge_schedule_type <xsl:value-of select="discharge_schedule_type"/>;
</xsl:if><xsl:if test="recharge_schedule_type">	recharge_schedule_type <xsl:value-of select="recharge_schedule_type"/>;
</xsl:if><xsl:if test="recharge_time">	recharge_time <xsl:value-of select="recharge_time"/>;
</xsl:if><xsl:if test="discharge_time">	discharge_time <xsl:value-of select="discharge_time"/>;
</xsl:if><xsl:if test="discharge_rate">	discharge_rate <xsl:value-of select="discharge_rate"/>;
</xsl:if><xsl:if test="SOC">	SOC <xsl:value-of select="SOC"/>;
</xsl:if><xsl:if test="k">	k <xsl:value-of select="k"/>;
</xsl:if>}
</xsl:for-each>
# residential::evcharger_det objects
<xsl:for-each select="evcharger_det_list/evcharger_det"><a name="#GLM.{name}"/>object evcharger_det:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="charge_rate">	charge_rate <xsl:value-of select="charge_rate"/>;
</xsl:if><xsl:if test="variation_mean">	variation_mean <xsl:value-of select="variation_mean"/>;
</xsl:if><xsl:if test="variation_std_dev">	variation_std_dev <xsl:value-of select="variation_std_dev"/>;
</xsl:if><xsl:if test="variation_trip_mean">	variation_trip_mean <xsl:value-of select="variation_trip_mean"/>;
</xsl:if><xsl:if test="variation_trip_std_dev">	variation_trip_std_dev <xsl:value-of select="variation_trip_std_dev"/>;
</xsl:if><xsl:if test="mileage_classification">	mileage_classification <xsl:value-of select="mileage_classification"/>;
</xsl:if><xsl:if test="work_charging_available">	work_charging_available <xsl:value-of select="work_charging_available"/>;
</xsl:if><xsl:if test="data_file">	data_file <xsl:value-of select="data_file"/>;
</xsl:if><xsl:if test="vehicle_index">	vehicle_index <xsl:value-of select="vehicle_index"/>;
</xsl:if><xsl:if test="vehicle_location">	vehicle_location <xsl:value-of select="vehicle_location"/>;
</xsl:if><xsl:if test="travel_distance">	travel_distance <xsl:value-of select="travel_distance"/>;
</xsl:if><xsl:if test="arrival_at_work">	arrival_at_work <xsl:value-of select="arrival_at_work"/>;
</xsl:if><xsl:if test="duration_at_work">	duration_at_work <xsl:value-of select="duration_at_work"/>;
</xsl:if><xsl:if test="arrival_at_home">	arrival_at_home <xsl:value-of select="arrival_at_home"/>;
</xsl:if><xsl:if test="duration_at_home">	duration_at_home <xsl:value-of select="duration_at_home"/>;
</xsl:if><xsl:if test="battery_capacity">	battery_capacity <xsl:value-of select="battery_capacity"/>;
</xsl:if><xsl:if test="battery_SOC">	battery_SOC <xsl:value-of select="battery_SOC"/>;
</xsl:if><xsl:if test="battery_size">	battery_size <xsl:value-of select="battery_size"/>;
</xsl:if><xsl:if test="mileage_efficiency">	mileage_efficiency <xsl:value-of select="mileage_efficiency"/>;
</xsl:if><xsl:if test="maximum_charge_rate">	maximum_charge_rate <xsl:value-of select="maximum_charge_rate"/>;
</xsl:if><xsl:if test="charging_efficiency">	charging_efficiency <xsl:value-of select="charging_efficiency"/>;
</xsl:if>}
</xsl:for-each></xsl:for-each><xsl:for-each select="tape">
##############################################
# tape module
module tape {
	version.major <xsl:value-of select="version.major"/>;
	version.minor <xsl:value-of select="version.minor"/>;
	gnuplot_path <xsl:value-of select="gnuplot_path"/>;
	flush_interval <xsl:value-of select="flush_interval"/>;
	delta_mode_needed <xsl:value-of select="delta_mode_needed"/>;
}

# tape::player objects
<xsl:for-each select="player_list/player"><a name="#GLM.{name}"/>object player:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="property">	property <xsl:value-of select="property"/>;
</xsl:if><xsl:if test="file">	file <xsl:value-of select="file"/>;
</xsl:if><xsl:if test="filetype">	filetype <xsl:value-of select="filetype"/>;
</xsl:if><xsl:if test="loop">	loop <xsl:value-of select="loop"/>;
</xsl:if>}
</xsl:for-each>
# tape::shaper objects
<xsl:for-each select="shaper_list/shaper"><a name="#GLM.{name}"/>object shaper:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="file">	file <xsl:value-of select="file"/>;
</xsl:if><xsl:if test="filetype">	filetype <xsl:value-of select="filetype"/>;
</xsl:if><xsl:if test="group">	group <xsl:value-of select="group"/>;
</xsl:if><xsl:if test="property">	property <xsl:value-of select="property"/>;
</xsl:if><xsl:if test="magnitude">	magnitude <xsl:value-of select="magnitude"/>;
</xsl:if><xsl:if test="events">	events <xsl:value-of select="events"/>;
</xsl:if>}
</xsl:for-each>
# tape::recorder objects
<xsl:for-each select="recorder_list/recorder"><a name="#GLM.{name}"/>object recorder:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="property">	property <xsl:value-of select="property"/>;
</xsl:if><xsl:if test="trigger">	trigger <xsl:value-of select="trigger"/>;
</xsl:if><xsl:if test="file">	file <xsl:value-of select="file"/>;
</xsl:if><xsl:if test="multifile">	multifile <xsl:value-of select="multifile"/>;
</xsl:if><xsl:if test="limit">	limit <xsl:value-of select="limit"/>;
</xsl:if><xsl:if test="plotcommands">	plotcommands <xsl:value-of select="plotcommands"/>;
</xsl:if><xsl:if test="xdata">	xdata <xsl:value-of select="xdata"/>;
</xsl:if><xsl:if test="columns">	columns <xsl:value-of select="columns"/>;
</xsl:if><xsl:if test="interval">	interval <xsl:value-of select="interval"/>;
</xsl:if><xsl:if test="output">	output <xsl:value-of select="output"/>;
</xsl:if><xsl:if test="header_units">	header_units <xsl:value-of select="header_units"/>;
</xsl:if><xsl:if test="line_units">	line_units <xsl:value-of select="line_units"/>;
</xsl:if>}
</xsl:for-each>
# tape::multi_recorder objects
<xsl:for-each select="multi_recorder_list/multi_recorder"><a name="#GLM.{name}"/>object multi_recorder:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="interval">	interval <xsl:value-of select="interval"/>;
</xsl:if><xsl:if test="property">	property <xsl:value-of select="property"/>;
</xsl:if><xsl:if test="trigger">	trigger <xsl:value-of select="trigger"/>;
</xsl:if><xsl:if test="file">	file <xsl:value-of select="file"/>;
</xsl:if><xsl:if test="multifile">	multifile <xsl:value-of select="multifile"/>;
</xsl:if><xsl:if test="limit">	limit <xsl:value-of select="limit"/>;
</xsl:if><xsl:if test="plotcommands">	plotcommands <xsl:value-of select="plotcommands"/>;
</xsl:if><xsl:if test="xdata">	xdata <xsl:value-of select="xdata"/>;
</xsl:if><xsl:if test="columns">	columns <xsl:value-of select="columns"/>;
</xsl:if><xsl:if test="output">	output <xsl:value-of select="output"/>;
</xsl:if><xsl:if test="header_units">	header_units <xsl:value-of select="header_units"/>;
</xsl:if><xsl:if test="line_units">	line_units <xsl:value-of select="line_units"/>;
</xsl:if>}
</xsl:for-each>
# tape::collector objects
<xsl:for-each select="collector_list/collector"><a name="#GLM.{name}"/>object collector:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="property">	property <xsl:value-of select="property"/>;
</xsl:if><xsl:if test="trigger">	trigger <xsl:value-of select="trigger"/>;
</xsl:if><xsl:if test="file">	file <xsl:value-of select="file"/>;
</xsl:if><xsl:if test="limit">	limit <xsl:value-of select="limit"/>;
</xsl:if><xsl:if test="group">	group <xsl:value-of select="group"/>;
</xsl:if><xsl:if test="interval">	interval <xsl:value-of select="interval"/>;
</xsl:if>}
</xsl:for-each>
# tape::histogram objects
<xsl:for-each select="histogram_list/histogram"><a name="#GLM.{name}"/>object histogram:<xsl:value-of select="id"/> {
<xsl:if test="name!=''">	name "<xsl:value-of select="name"/>";
</xsl:if><xsl:if test="parent!=''">	parent "<a href="#GLM.{parent}"><xsl:value-of select="parent"/></a>";
</xsl:if><xsl:if test="clock!=''">	clock '<xsl:value-of select="clock"/>';
</xsl:if><xsl:if test="in_svc!=''">	in_svc '<xsl:value-of select="in_svc"/>';
</xsl:if><xsl:if test="out_svc!=''">	out_svc '<xsl:value-of select="out_svc"/>';
</xsl:if><xsl:if test="latitude!=''">	latitude <xsl:value-of select="latitude"/>;
</xsl:if><xsl:if test="longitude!=''">	longitude <xsl:value-of select="longitude"/>;
</xsl:if><xsl:if test="rank!=''">	rank <xsl:value-of select="rank"/>;
</xsl:if><xsl:if test="filename">	filename <xsl:value-of select="filename"/>;
</xsl:if><xsl:if test="group">	group <xsl:value-of select="group"/>;
</xsl:if><xsl:if test="bins">	bins <xsl:value-of select="bins"/>;
</xsl:if><xsl:if test="property">	property <xsl:value-of select="property"/>;
</xsl:if><xsl:if test="min">	min <xsl:value-of select="min"/>;
</xsl:if><xsl:if test="max">	max <xsl:value-of select="max"/>;
</xsl:if><xsl:if test="samplerate">	samplerate <xsl:value-of select="samplerate"/>;
</xsl:if><xsl:if test="countrate">	countrate <xsl:value-of select="countrate"/>;
</xsl:if><xsl:if test="bin_count">	bin_count <xsl:value-of select="bin_count"/>;
</xsl:if><xsl:if test="limit">	limit <xsl:value-of select="limit"/>;
</xsl:if>}
</xsl:for-each></xsl:for-each></pre></td></tr></table>
</body>
</xsl:for-each></html>
