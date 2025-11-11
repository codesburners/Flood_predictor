# tamilnadu_data.py - UPDATED WITH VELLORE
class TamilNaduData:
    def __init__(self):
        self.districts = {
            'Chennai': {
                'lat': 13.0827, 'lon': 80.2707, 'elevation': 6,
                'population_density': 26903, 'drainage_capacity': 0.4,
                'distance_to_river': 0.5, 'soil_type': 'sandy_loam',
                'urbanization_level': 0.95, 'infrastructure_quality': 0.8,
                'flood_history': 'severe'
            },
            'Trichy': {
                'lat': 10.7905, 'lon': 78.7047, 'elevation': 88,
                'population_density': 3456, 'drainage_capacity': 0.6,
                'distance_to_river': 0.8, 'soil_type': 'clay',
                'urbanization_level': 0.7, 'infrastructure_quality': 0.6,
                'flood_history': 'moderate'
            },
            'Cuddalore': {
                'lat': 11.7447, 'lon': 79.7680, 'elevation': 6,
                'population_density': 789, 'drainage_capacity': 0.3,
                'distance_to_river': 1.5, 'soil_type': 'sandy',
                'urbanization_level': 0.5, 'infrastructure_quality': 0.4,
                'flood_history': 'high'
            },
            'Madurai': {
                'lat': 9.9252, 'lon': 78.1198, 'elevation': 134,
                'population_density': 5432, 'drainage_capacity': 0.5,
                'distance_to_river': 2.1, 'soil_type': 'red_soil',
                'urbanization_level': 0.75, 'infrastructure_quality': 0.7,
                'flood_history': 'low'
            },
            'Coimbatore': {
                'lat': 11.0168, 'lon': 76.9558, 'elevation': 432,
                'population_density': 4567, 'drainage_capacity': 0.7,
                'distance_to_river': 3.2, 'soil_type': 'black_cotton',
                'urbanization_level': 0.8, 'infrastructure_quality': 0.75,
                'flood_history': 'very_low'
            },
            'Salem': {
                'lat': 11.6643, 'lon': 78.1460, 'elevation': 278,
                'population_density': 3210, 'drainage_capacity': 0.55,
                'distance_to_river': 2.8, 'soil_type': 'rocky',
                'urbanization_level': 0.65, 'infrastructure_quality': 0.6,
                'flood_history': 'moderate'
            },
            'Thanjavur': {
                'lat': 10.7870, 'lon': 79.1378, 'elevation': 59,
                'population_density': 2345, 'drainage_capacity': 0.4,
                'distance_to_river': 1.2, 'soil_type': 'clay',
                'urbanization_level': 0.6, 'infrastructure_quality': 0.5,
                'flood_history': 'high'
            },
            
            # NEW: VELLORE DISTRICT - Comprehensive Data
            'Vellore': {
                'lat': 12.9165,
                'lon': 79.1325,
                'elevation': 220,
                'population_density': 648,
                'drainage_capacity': 0.65,
                'distance_to_river': 1.2,  # Close to Palar River
                'soil_type': 'clay_loam',
                'urbanization_level': 0.7,
                'infrastructure_quality': 0.6,
                'flood_history': 'moderate',
                'river_basins': ['Palar', 'Ponnaiyar'],
                'topography': 'plain_with_hills',
                'critical_infrastructure': [
                    'Christian Medical College Hospital',
                    'VIT University', 
                    'Vellore Railway Junction',
                    'Adukkamparai Dam',
                    'Kaleri Hydroelectric Project'
                ],
                'vulnerable_areas': [
                    'Katpadi',
                    'Sathuvachari', 
                    'Bagayam',
                    'Gandhi Nagar',
                    'Kosapet'
                ],
                'emergency_shelters': 12,
                'healthcare_facilities': 25,
                'last_major_flood': 2015,
                'flood_prone_zones': [
                    'Palar River banks',
                    'Low-lying areas near Katpadi',
                    'Sathuvachari residential areas'
                ],
                'drainage_systems': [
                    'Municipal stormwater drains',
                    'Natural Palar river drainage',
                    'Agricultural drainage channels'
                ],
                'rescue_centers': [
                    'Collectorate Complex',
                    'CMC Hospital Campus',
                    'VIT University Hostels'
                ]
            },
            
            # You can add more districts as needed
            'Tirupur': {
                'lat': 11.1085, 'lon': 77.3411, 'elevation': 295,
                'population_density': 5123, 'drainage_capacity': 0.55,
                'distance_to_river': 2.5, 'soil_type': 'sandy_loam',
                'urbanization_level': 0.75, 'infrastructure_quality': 0.65,
                'flood_history': 'low'
            },
            'Erode': {
                'lat': 11.3410, 'lon': 77.7172, 'elevation': 183,
                'population_density': 2987, 'drainage_capacity': 0.6,
                'distance_to_river': 1.8, 'soil_type': 'clay',
                'urbanization_level': 0.6, 'infrastructure_quality': 0.55,
                'flood_history': 'moderate'
            }
        }