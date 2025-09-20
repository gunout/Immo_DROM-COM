import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DromcomImmobilierAnalyzer:
    def __init__(self, territoire_name):
        self.territoire = territoire_name
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A602', '#6A0572', 
                      '#AB83A1', '#5CAB7D', '#2A9D8F', '#E76F51', '#264653']
        
        self.start_year = 2002
        self.end_year = 2025
        
        # Configuration spécifique à chaque territoire
        self.config = self._get_territoire_config()
        
    def _get_territoire_config(self):
        """Retourne la configuration spécifique pour chaque DROM-COM"""
        configs = {
            "Guadeloupe": {
                "prix_m2_base": 1800,
                "loyer_m2_base": 10.5,
                "revenu_median": 18000,
                "specialites": ["tourisme", "résidentiel", "luxe"],
                "zones_cles": ["Pointe-à-Pitre", "Gosier", "Sainte-Anne", "Basse-Terre"]
            },
            "Martinique": {
                "prix_m2_base": 2200,
                "loyer_m2_base": 12.0,
                "revenu_median": 19500,
                "specialites": ["tourisme", "résidentiel", "vue mer"],
                "zones_cles": ["Fort-de-France", "Ducos", "Schoelcher", "Trois-Îlets"]
            },
            "Guyane": {
                "prix_m2_base": 1500,
                "loyer_m2_base": 9.0,
                "revenu_median": 16500,
                "specialites": ["spatial", "croissance", "défiscalisation"],
                "zones_cles": ["Cayenne", "Kourou", "Remire-Montjoly", "Matoury"]
            },
            "La Réunion": {
                "prix_m2_base": 2100,
                "loyer_m2_base": 11.5,
                "revenu_median": 19000,
                "specialites": ["tourisme", "résidentiel", "haute altitude"],
                "zones_cles": ["Saint-Denis", "Saint-Paul", "Saint-Pierre", "Le Tampon"]
            },
            "Mayotte": {
                "prix_m2_base": 1200,
                "loyer_m2_base": 7.5,
                "revenu_median": 9500,
                "specialites": ["croissance", "accession", "défavorisé"],
                "zones_cles": ["Mamoudzou", "Dzaoudzi", "Koungou", "Tsingoni"]
            },
            "Saint-Martin": {
                "prix_m2_base": 3500,
                "loyer_m2_base": 20.0,
                "revenu_median": 22000,
                "specialites": ["luxe", "tourisme", "international"],
                "zones_cles": ["Marigot", "Grand-Case", "Baie Orientale", "Terres Basses"]
            },
            "Saint-Barthélemy": {
                "prix_m2_base": 8500,
                "loyer_m2_base": 45.0,
                "revenu_median": 35000,
                "specialites": ["ultra-luxe", "jet-set", "international"],
                "zones_cles": ["Gustavia", "Saint-Jean", "Lorient", "Flamands"]
            },
            "Saint-Pierre-et-Miquelon": {
                "prix_m2_base": 1800,
                "loyer_m2_base": 9.5,
                "revenu_median": 21000,
                "specialites": ["pêche", "isolé", "climat froid"],
                "zones_cles": ["Saint-Pierre", "Miquelon", "Langlade"]
            },
            "Wallis-et-Futuna": {
                "prix_m2_base": 1300,
                "loyer_m2_base": 8.0,
                "revenu_median": 12000,
                "specialites": ["traditionnel", "communautaire", "isolé"],
                "zones_cles": ["Mata-Utu", "Leava", "Alo", "Sigave"]
            },
            "Polynésie française": {
                "prix_m2_base": 2800,
                "loyer_m2_base": 15.0,
                "revenu_median": 18500,
                "specialites": ["tourisme", "insulaire", "vue lagons"],
                "zones_cles": ["Papeete", "Punaauia", "Moorea", "Bora-Bora"]
            },
            "Nouvelle-Calédonie": {
                "prix_m2_base": 2500,
                "loyer_m2_base": 14.0,
                "revenu_median": 23000,
                "specialites": ["nickel", "austral", "vue mer"],
                "zones_cles": ["Nouméa", "Dumbéa", "Mont-Dore", "Païta"]
            },
            # Configuration par défaut
            "default": {
                "prix_m2_base": 2000,
                "loyer_m2_base": 11.0,
                "revenu_median": 18000,
                "specialites": ["résidentiel", "tourisme"],
                "zones_cles": ["Capitale", "Zone touristique", "Périurbain"]
            }
        }
        
        return configs.get(self.territoire, configs["default"])
    
    def generate_real_estate_data(self):
        """Génère des données immobilières pour le territoire"""
        print(f"🏠 Génération des données immobilières pour {self.territoire}...")
        
        # Créer une base de données annuelle
        dates = pd.date_range(start=f'{self.start_year}-01-01', 
                             end=f'{self.end_year}-12-31', freq='Y')
        
        data = {'Annee': [date.year for date in dates]}
        
        # Données immobilières de base
        data['Prix_m2_Maison'] = self._simulate_house_prices(dates)
        data['Prix_m2_Appartement'] = self._simulate_apartment_prices(dates)
        data['Loyer_m2_Maison'] = self._simulate_house_rents(dates)
        data['Loyer_m2_Appartement'] = self._simulate_apartment_rents(dates)
        
        # Indicateurs de marché
        data['Transactions_Total'] = self._simulate_transactions(dates)
        data['Duree_Vente_Moyenne'] = self._simulate_selling_time(dates)
        data['Taux_Vacance_Locatif'] = self._simulate_vacancy_rate(dates)
        
        # Indicateurs économiques liés
        data['Revenu_Median'] = self._simulate_median_income(dates)
        data['Taux_Interet_Hypothecaire'] = self._simulate_mortgage_rates(dates)
        data['Chomage'] = self._simulate_unemployment(dates)
        
        # Indicateurs d'accessibilité
        data['Annee_Salaire_Maison'] = self._simulate_years_of_income_house(dates)
        data['Annee_Salaire_Appartement'] = self._simulate_years_of_income_apartment(dates)
        data['Ratio_Loyer_Revenu'] = self._simulate_rent_income_ratio(dates)
        
        # Investissements et constructions
        data['Permis_Construire'] = self._simulate_building_permits(dates)
        data['Investissement_Etranger'] = self._simulate_foreign_investment(dates)
        data['Investissement_Locatif'] = self._simulate_rental_investment(dates)
        
        df = pd.DataFrame(data)
        
        # Ajouter des tendances spécifiques au territoire
        self._add_territory_trends(df)
        
        return df
    
    def _simulate_house_prices(self, dates):
        """Simule les prix au m² des maisons"""
        base_price = self.config["prix_m2_base"]
        
        prices = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Croissance différente selon les territoires
            if self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
                growth_rate = 0.045  # Croissance forte dans les îles luxueuses
            elif self.territoire in ["Guyane", "Mayotte"]:
                growth_rate = 0.038  # Croissance forte dans les territoires en développement
            elif self.territoire in ["Nouvelle-Calédonie", "Polynésie française"]:
                growth_rate = 0.032  # Croissance modérée
            else:
                growth_rate = 0.028  # Croissance standard
            
            # Appliquer la croissance
            growth = 1 + growth_rate * i
            
            # Variations cycliques
            if year in [2008, 2009, 2020, 2021]:  # Crises économiques
                multiplier = 0.92
            elif year in [2006, 2012, 2017, 2023]:  # Périodes fastes
                multiplier = 1.08
            else:
                multiplier = 1.0
            
            noise = np.random.normal(1, 0.06)
            prices.append(base_price * growth * multiplier * noise)
        
        return prices
    
    def _simulate_apartment_prices(self, dates):
        """Simule les prix au m² des appartements"""
        base_price = self.config["prix_m2_base"] * 1.15  # Généralement plus chers que les maisons
        
        prices = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Croissance différente selon les territoires
            if self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
                growth_rate = 0.048  # Croissance très forte
            elif self.territoire in ["Guyane", "Mayotte"]:
                growth_rate = 0.042  # Croissance forte
            elif self.territoire in ["La Réunion", "Martinique"]:
                growth_rate = 0.035  # Croissance modérée
            else:
                growth_rate = 0.030  # Croissance standard
            
            # Appliquer la croissance
            growth = 1 + growth_rate * i
            
            # Variations cycliques
            if year in [2008, 2009, 2020, 2021]:  # Crises économiques
                multiplier = 0.90
            elif year in [2006, 2012, 2017, 2023]:  # Périodes fastes
                multiplier = 1.10
            else:
                multiplier = 1.0
            
            noise = np.random.normal(1, 0.07)
            prices.append(base_price * growth * multiplier * noise)
        
        return prices
    
    def _simulate_house_rents(self, dates):
        """Simule les loyers au m² des maisons"""
        base_rent = self.config["loyer_m2_base"]
        
        rents = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Croissance différente selon les territoires
            if self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
                growth_rate = 0.032  # Croissance forte
            elif self.territoire in ["Guyane", "Mayotte"]:
                growth_rate = 0.028  # Croissance modérée
            else:
                growth_rate = 0.022  # Croissance standard
            
            # Appliquer la croissance
            growth = 1 + growth_rate * i
            
            # Variations cycliques (les loyers sont moins volatils que les prix)
            if year in [2008, 2009, 2020, 2021]:  # Crises économiques
                multiplier = 0.96
            elif year in [2006, 2012, 2017, 2023]:  # Périodes fastes
                multiplier = 1.04
            else:
                multiplier = 1.0
            
            noise = np.random.normal(1, 0.04)
            rents.append(base_rent * growth * multiplier * noise)
        
        return rents
    
    def _simulate_apartment_rents(self, dates):
        """Simule les loyers au m² des appartements"""
        base_rent = self.config["loyer_m2_base"] * 1.10  # Généralement plus chers que les maisons
        
        rents = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Croissance différente selon les territoires
            if self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
                growth_rate = 0.035  # Croissance forte
            elif self.territoire in ["Guyane", "Mayotte"]:
                growth_rate = 0.030  # Croissance modérée
            else:
                growth_rate = 0.025  # Croissance standard
            
            # Appliquer la croissance
            growth = 1 + growth_rate * i
            
            # Variations cycliques
            if year in [2008, 2009, 2020, 2021]:  # Crises économiques
                multiplier = 0.95
            elif year in [2006, 2012, 2017, 2023]:  # Périodes fastes
                multiplier = 1.05
            else:
                multiplier = 1.0
            
            noise = np.random.normal(1, 0.05)
            rents.append(base_rent * growth * multiplier * noise)
        
        return rents
    
    def _simulate_transactions(self, dates):
        """Simule le volume de transactions"""
        # Volume de base selon le territoire (en nombre de transactions)
        if self.territoire in ["La Réunion", "Martinique", "Guadeloupe"]:
            base_volume = 5000
        elif self.territoire in ["Guyane", "Nouvelle-Calédonie", "Polynésie française"]:
            base_volume = 2500
        elif self.territoire == "Mayotte":
            base_volume = 1500
        elif self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
            base_volume = 500
        else:
            base_volume = 1000
        
        transactions = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Croissance différente selon les territoires
            if self.territoire in ["Guyane", "Mayotte"]:
                growth_rate = 0.035  # Croissance forte
            elif self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
                growth_rate = 0.025  # Croissance modérée
            else:
                growth_rate = 0.015  # Croissance standard
            
            # Appliquer la croissance
            growth = 1 + growth_rate * i
            
            # Variations cycliques (forte sensibilité aux conditions économiques)
            if year in [2008, 2009, 2020, 2021]:  # Crises économiques
                multiplier = 0.65
            elif year in [2006, 2012, 2017, 2023]:  # Périodes fastes
                multiplier = 1.25
            else:
                multiplier = 1.0
            
            noise = np.random.normal(1, 0.12)
            transactions.append(base_volume * growth * multiplier * noise)
        
        return transactions
    
    def _simulate_selling_time(self, dates):
        """Simule la durée moyenne de vente (en jours)"""
        # Durée de base selon le territoire
        if self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
            base_duration = 60  # Marché dynamique
        elif self.territoire in ["Guyane", "Mayotte"]:
            base_duration = 90  # Marché en développement
        elif self.territoire in ["La Réunion", "Martinique"]:
            base_duration = 75  # Marché standard
        else:
            base_duration = 85  # Marché standard
        
        durations = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Évolution différente selon les territoires
            if self.territoire in ["Guyane", "Mayotte"]:
                trend = 1 - 0.01 * i  # Amélioration progressive
            else:
                trend = 1 - 0.005 * i  # Amélioration lente
            
            # Variations cycliques (plus long en période de crise)
            if year in [2008, 2009, 2020, 2021]:  # Crises économiques
                multiplier = 1.35
            elif year in [2006, 2012, 2017, 2023]:  # Périodes fastes
                multiplier = 0.80
            else:
                multiplier = 1.0
            
            noise = np.random.normal(1, 0.08)
            durations.append(base_duration * trend * multiplier * noise)
        
        return durations
    
    def _simulate_vacancy_rate(self, dates):
        """Simule le taux de vacance locative (en %)"""
        # Taux de base selon le territoire
        if self.territoire in ["Mayotte", "Guyane"]:
            base_rate = 4.5  # Faible vacance (demande forte)
        elif self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
            base_rate = 8.0  # Vacance modérée (saisonnalité)
        elif self.territoire in ["La Réunion", "Martinique"]:
            base_rate = 6.0  # Vacance standard
        else:
            base_rate = 5.5  # Vacance standard
        
        rates = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Évolution différente selon les territoires
            if self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
                trend = 1 + 0.01 * i  # Légère augmentation
            else:
                trend = 1 - 0.005 * i  # Légère diminution
            
            # Variations cycliques (plus élevé en période de crise)
            if year in [2008, 2009, 2020, 2021]:  # Crises économiques
                multiplier = 1.25
            elif year in [2006, 2012, 2017, 2023]:  # Périodes fastes
                multiplier = 0.85
            else:
                multiplier = 1.0
            
            noise = np.random.normal(1, 0.06)
            rates.append(base_rate * trend * multiplier * noise)
        
        return rates
    
    def _simulate_median_income(self, dates):
        """Simule le revenu médian (en euros)"""
        base_income = self.config["revenu_median"]
        
        incomes = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Croissance différente selon les territoires
            if self.territoire in ["Guyane", "Mayotte"]:
                growth_rate = 0.022  # Croissance forte
            elif self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
                growth_rate = 0.018  # Croissance modérée
            else:
                growth_rate = 0.015  # Croissance standard
            
            # Appliquer la croissance
            growth = 1 + growth_rate * i
            
            # Variations cycliques
            if year in [2008, 2009, 2020, 2021]:  # Crises économiques
                multiplier = 0.97
            elif year in [2006, 2012, 2017, 2023]:  # Périodes fastes
                multiplier = 1.04
            else:
                multiplier = 1.0
            
            noise = np.random.normal(1, 0.03)
            incomes.append(base_income * growth * multiplier * noise)
        
        return incomes
    
    def _simulate_mortgage_rates(self, dates):
        """Simule les taux d'intérêt hypothécaires (en %)"""
        # Taux de base selon la période
        rates = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Tendances historiques et prospectives des taux
            if year <= 2005:
                base_rate = 4.2
            elif year <= 2008:
                base_rate = 4.5
            elif year <= 2012:
                base_rate = 3.8
            elif year <= 2016:
                base_rate = 2.9
            elif year <= 2020:
                base_rate = 1.8
            elif year <= 2023:
                base_rate = 2.2
            else:
                base_rate = 2.8
            
            # Ajouter une prime spécifique aux DROM-COM
            if self.territoire in ["Mayotte", "Guyane", "Wallis-et-Futuna"]:
                territory_premium = 0.4
            elif self.territoire in ["Saint-Pierre-et-Miquelon", "Polynésie française"]:
                territory_premium = 0.3
            else:
                territory_premium = 0.2
            
            noise = np.random.normal(1, 0.05)
            rates.append((base_rate + territory_premium) * noise)
        
        return rates
    
    def _simulate_unemployment(self, dates):
        """Simule le taux de chômage (en %)"""
        # Taux de base selon le territoire
        if self.territoire in ["Mayotte", "Guyane"]:
            base_rate = 22.0
        elif self.territoire in ["Martinique", "Guadeloupe"]:
            base_rate = 18.0
        elif self.territoire == "La Réunion":
            base_rate = 16.0
        elif self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
            base_rate = 12.0
        else:
            base_rate = 14.0
        
        rates = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Évolution avec des variations cycliques
            if year in [2008, 2009, 2020, 2021]:  # Crises économiques
                multiplier = 1.15
            elif year in [2006, 2012, 2017, 2023]:  # Périodes plus favorables
                multiplier = 0.92
            else:
                multiplier = 1.0
            
            # Tendances à long terme
            if self.territoire in ["Mayotte", "Guyane"]:
                trend = 1 - 0.005 * i  # Légère amélioration
            elif self.territoire in ["Martinique", "Guadeloupe"]:
                trend = 1 - 0.004 * i  # Légère amélioration
            else:
                trend = 1 - 0.003 * i  # Très légère amélioration
            
            noise = np.random.normal(1, 0.05)
            rates.append(base_rate * trend * multiplier * noise)
        
        return rates
    
    def _simulate_years_of_income_house(self, dates):
        """Simule le nombre d'années de salaire nécessaire pour une maison"""
        years = []
        for i, date in enumerate(dates):
            # Prix moyen d'une maison (100m²)
            house_price = self._simulate_house_prices([date])[0] * 100
            
            # Revenu médian annuel
            median_income = self._simulate_median_income([date])[0]
            
            # Calcul du nombre d'années de salaire
            years_of_income = house_price / median_income
            
            years.append(years_of_income)
        
        return years
    
    def _simulate_years_of_income_apartment(self, dates):
        """Simule le nombre d'années de salaire nécessaire pour un appartement"""
        years = []
        for i, date in enumerate(dates):
            # Prix moyen d'un appartement (70m²)
            apartment_price = self._simulate_apartment_prices([date])[0] * 70
            
            # Revenu médian annuel
            median_income = self._simulate_median_income([date])[0]
            
            # Calcul du nombre d'années de salaire
            years_of_income = apartment_price / median_income
            
            years.append(years_of_income)
        
        return years
    
    def _simulate_rent_income_ratio(self, dates):
        """Simule le ratio loyer/revenu (en %)"""
        ratios = []
        for i, date in enumerate(dates):
            # Loyer mensuel moyen pour un appartement (70m²)
            monthly_rent = self._simulate_apartment_rents([date])[0] * 70
            
            # Revenu médian annuel
            median_income = self._simulate_median_income([date])[0]
            
            # Revenu médian mensuel
            monthly_income = median_income / 12
            
            # Calcul du ratio
            ratio = (monthly_rent / monthly_income) * 100
            
            ratios.append(ratio)
        
        return ratios
    
    def _simulate_building_permits(self, dates):
        """Simule le nombre de permis de construire"""
        # Volume de base selon le territoire
        if self.territoire in ["La Réunion", "Martinique", "Guadeloupe"]:
            base_volume = 2000
        elif self.territoire in ["Guyane", "Nouvelle-Calédonie"]:
            base_volume = 1200
        elif self.territoire == "Mayotte":
            base_volume = 800
        elif self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
            base_volume = 200
        else:
            base_volume = 600
        
        permits = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Croissance différente selon les territoires
            if self.territoire in ["Guyane", "Mayotte"]:
                growth_rate = 0.040  # Croissance forte
            elif self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
                growth_rate = 0.025  # Croissance modérée
            else:
                growth_rate = 0.015  # Croissance standard
            
            # Appliquer la croissance
            growth = 1 + growth_rate * i
            
            # Variations cycliques (forte sensibilité aux conditions économiques)
            if year in [2008, 2009, 2020, 2021]:  # Crises économiques
                multiplier = 0.60
            elif year in [2006, 2012, 2017, 2023]:  # Périodes fastes
                multiplier = 1.30
            else:
                multiplier = 1.0
            
            noise = np.random.normal(1, 0.10)
            permits.append(base_volume * growth * multiplier * noise)
        
        return permits
    
    def _simulate_foreign_investment(self, dates):
        """Simule l'investissement étranger (en millions d'euros)"""
        # Volume de base selon le territoire
        if self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
            base_volume = 120
        elif self.territoire in ["Polynésie française", "Nouvelle-Calédonie"]:
            base_volume = 80
        elif self.territoire in ["Martinique", "Guadeloupe"]:
            base_volume = 50
        elif self.territoire == "La Réunion":
            base_volume = 40
        else:
            base_volume = 20
        
        investments = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Croissance différente selon les territoires
            if self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
                growth_rate = 0.050  # Croissance forte
            elif self.territoire in ["Polynésie française", "Nouvelle-Calédonie"]:
                growth_rate = 0.035  # Croissance modérée
            else:
                growth_rate = 0.020  # Croissance standard
            
            # Appliquer la croissance
            growth = 1 + growth_rate * i
            
            # Variations cycliques
            if year in [2008, 2009, 2020, 2021]:  # Crises économiques
                multiplier = 0.70
            elif year in [2006, 2012, 2017, 2023]:  # Périodes fastes
                multiplier = 1.40
            else:
                multiplier = 1.0
            
            noise = np.random.normal(1, 0.15)
            investments.append(base_volume * growth * multiplier * noise)
        
        return investments
    
    def _simulate_rental_investment(self, dates):
        """Simule l'investissement locatif (en millions d'euros)"""
        # Volume de base selon le territoire
        if self.territoire in ["La Réunion", "Martinique", "Guadeloupe"]:
            base_volume = 150
        elif self.territoire in ["Guyane", "Nouvelle-Calédonie"]:
            base_volume = 80
        elif self.territoire == "Mayotte":
            base_volume = 50
        elif self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
            base_volume = 100
        else:
            base_volume = 60
        
        investments = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Croissance différente selon les territoires
            if self.territoire in ["Guyane", "Mayotte"]:
                growth_rate = 0.045  # Croissance forte
            elif self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
                growth_rate = 0.030  # Croissance modérée
            else:
                growth_rate = 0.020  # Croissance standard
            
            # Appliquer la croissance
            growth = 1 + growth_rate * i
            
            # Variations cycliques
            if year in [2008, 2009, 2020, 2021]:  # Crises économiques
                multiplier = 0.75
            elif year in [2006, 2012, 2017, 2023]:  # Périodes fastes
                multiplier = 1.25
            else:
                multiplier = 1.0
            
            noise = np.random.normal(1, 0.12)
            investments.append(base_volume * growth * multiplier * noise)
        
        return investments
    
    def _add_territory_trends(self, df):
        """Ajoute des tendances réalistes adaptées à chaque territoire"""
        for i, row in df.iterrows():
            year = row['Annee']
            
            # Événements communs à tous les territoires
            if 2008 <= year <= 2009:  # Crise financière mondiale
                df.loc[i, 'Prix_m2_Maison'] *= 0.88
                df.loc[i, 'Prix_m2_Appartement'] *= 0.85
                df.loc[i, 'Transactions_Total'] *= 0.65
            
            if 2020 <= year <= 2021:  # Pandémie COVID-19
                df.loc[i, 'Loyer_m2_Maison'] *= 0.95
                df.loc[i, 'Loyer_m2_Appartement'] *= 0.93
                df.loc[i, 'Transactions_Total'] *= 0.70
                df.loc[i, 'Taux_Vacance_Locatif'] *= 1.20
            
            # Événements spécifiques à certains territoires
            if self.territoire == "Mayotte":
                if year >= 2011:  # Départementalisation
                    df.loc[i, 'Investissement_Etranger'] *= 1.15
                    df.loc[i, 'Permis_Construire'] *= 1.20
            
            if self.territoire == "Guyane":
                if year in [2017, 2018]:  # Mouvements sociaux
                    df.loc[i, 'Transactions_Total'] *= 0.80
                    df.loc[i, 'Permis_Construire'] *= 0.85
            
            if self.territoire == "Nouvelle-Calédonie":
                if year in [2018, 2020, 2021]:  # Référendums et incertitudes politiques
                    df.loc[i, 'Investissement_Etranger'] *= 0.75
                    df.loc[i, 'Transactions_Total'] *= 0.85
            
            if self.territoire == "La Réunion":
                if year >= 2010:  # Développement du numérique et télétravail
                    df.loc[i, 'Loyer_m2_Appartement'] *= 1.03
                    df.loc[i, 'Prix_m2_Appartement'] *= 1.04
            
            # Tendances à long terme
            if year >= 2015:
                # Hausse générale des prix immobiliers
                df.loc[i, 'Prix_m2_Maison'] *= 1.02
                df.loc[i, 'Prix_m2_Appartement'] *= 1.03
            
            if year >= 2022:
                # Reprise post-COVID
                df.loc[i, 'Transactions_Total'] *= 1.15
                df.loc[i, 'Investissement_Locatif'] *= 1.10
    
    def create_real_estate_analysis(self, df):
        """Crée une analyse complète du marché immobilier"""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Évolution des prix au m²
        ax1 = plt.subplot(4, 2, 1)
        self._plot_price_evolution(df, ax1)
        
        # 2. Évolution des loyers au m²
        ax2 = plt.subplot(4, 2, 2)
        self._plot_rent_evolution(df, ax2)
        
        # 3. Accessibilité (années de salaire)
        ax3 = plt.subplot(4, 2, 3)
        self._plot_affordability(df, ax3)
        
        # 4. Transactions et durée de vente
        ax4 = plt.subplot(4, 2, 4)
        self._plot_transactions(df, ax4)
        
        # 5. Investissements
        ax5 = plt.subplot(4, 2, 5)
        self._plot_investments(df, ax5)
        
        # 6. Indicateurs économiques
        ax6 = plt.subplot(4, 2, 6)
        self._plot_economic_indicators(df, ax6)
        
        # 7. Ratios et indicateurs de marché
        ax7 = plt.subplot(4, 2, 7)
        self._plot_market_indicators(df, ax7)
        
        # 8. Comparaison prix/loyers
        ax8 = plt.subplot(4, 2, 8)
        self._plot_price_rent_comparison(df, ax8)
        
        plt.suptitle(f'Analyse du Marché Immobilier de {self.territoire} - DROM-COM ({self.start_year}-{self.end_year})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.territoire}_real_estate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Générer les insights
        self._generate_real_estate_insights(df)
    
    def _plot_price_evolution(self, df, ax):
        """Plot de l'évolution des prix au m²"""
        ax.plot(df['Annee'], df['Prix_m2_Maison'], label='Maison (€/m²)', 
               linewidth=2, color='#2A9D8F', alpha=0.8)
        ax.plot(df['Annee'], df['Prix_m2_Appartement'], label='Appartement (€/m²)', 
               linewidth=2, color='#E76F51', alpha=0.8)
        
        ax.set_title('Évolution des Prix Immobiliers (€/m²)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Prix (€/m²)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_rent_evolution(self, df, ax):
        """Plot de l'évolution des loyers au m²"""
        ax.plot(df['Annee'], df['Loyer_m2_Maison'], label='Maison (€/m²/mois)', 
               linewidth=2, color='#2A9D8F', alpha=0.8)
        ax.plot(df['Annee'], df['Loyer_m2_Appartement'], label='Appartement (€/m²/mois)', 
               linewidth=2, color='#E76F51', alpha=0.8)
        
        ax.set_title('Évolution des Loyers (€/m²/mois)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loyer (€/m²/mois)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_affordability(self, df, ax):
        """Plot de l'accessibilité (années de salaire)"""
        ax.plot(df['Annee'], df['Annee_Salaire_Maison'], label='Maison (années de salaire)', 
               linewidth=2, color='#2A9D8F', alpha=0.8)
        ax.plot(df['Annee'], df['Annee_Salaire_Appartement'], label='Appartement (années de salaire)', 
               linewidth=2, color='#E76F51', alpha=0.8)
        
        ax.set_title('Accessibilité: Années de Salaire Nécessaires', fontsize=12, fontweight='bold')
        ax.set_ylabel('Années de salaire')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_transactions(self, df, ax):
        """Plot des transactions et durée de vente"""
        # Transactions
        ax.bar(df['Annee'], df['Transactions_Total'], label='Transactions', 
              color='#2A9D8F', alpha=0.7)
        
        ax.set_title('Volume de Transactions et Durée de Vente', fontsize=12, fontweight='bold')
        ax.set_ylabel('Nombre de transactions', color='#2A9D8F')
        ax.tick_params(axis='y', labelcolor='#2A9D8F')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Durée de vente en second axe
        ax2 = ax.twinx()
        ax2.plot(df['Annee'], df['Duree_Vente_Moyenne'], label='Durée de vente (jours)', 
                linewidth=2, color='#E76F51', alpha=0.8)
        ax2.set_ylabel('Durée de vente (jours)', color='#E76F51')
        ax2.tick_params(axis='y', labelcolor='#E76F51')
        
        # Combiner les légendes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_investments(self, df, ax):
        """Plot des investissements"""
        ax.plot(df['Annee'], df['Investissement_Etranger'], label='Investissement étranger (M€)', 
               linewidth=2, color='#2A9D8F', alpha=0.8)
        ax.plot(df['Annee'], df['Investissement_Locatif'], label='Investissement locatif (M€)', 
               linewidth=2, color='#E76F51', alpha=0.8)
        
        ax.set_title('Investissements Immobiliers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Montant (M€)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_economic_indicators(self, df, ax):
        """Plot des indicateurs économiques"""
        # Taux de chômage
        ax.plot(df['Annee'], df['Chomage'], label='Taux de chômage (%)', 
               linewidth=2, color='#2A9D8F', alpha=0.8)
        
        ax.set_title('Indicateurs Économiques', fontsize=12, fontweight='bold')
        ax.set_ylabel('Taux de chômage (%)', color='#2A9D8F')
        ax.tick_params(axis='y', labelcolor='#2A9D8F')
        ax.grid(True, alpha=0.3)
        
        # Revenu médian en second axe
        ax2 = ax.twinx()
        ax2.plot(df['Annee'], df['Revenu_Median'], label='Revenu médian (€)', 
                linewidth=2, color='#E76F51', alpha=0.8)
        ax2.set_ylabel('Revenu médian (€)', color='#E76F51')
        ax2.tick_params(axis='y', labelcolor='#E76F51')
        
        # Combiner les légendes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_market_indicators(self, df, ax):
        """Plot des indicateurs de marché"""
        # Taux de vacance
        ax.plot(df['Annee'], df['Taux_Vacance_Locatif'], label='Taux de vacance locative (%)', 
               linewidth=2, color='#2A9D8F', alpha=0.8)
        
        ax.set_title('Indicateurs de Marché', fontsize=12, fontweight='bold')
        ax.set_ylabel('Taux de vacance (%)', color='#2A9D8F')
        ax.tick_params(axis='y', labelcolor='#2A9D8F')
        ax.grid(True, alpha=0.3)
        
        # Ratio loyer/revenu en second axe
        ax2 = ax.twinx()
        ax2.plot(df['Annee'], df['Ratio_Loyer_Revenu'], label='Ratio loyer/revenu (%)', 
                linewidth=2, color='#E76F51', alpha=0.8)
        ax2.set_ylabel('Ratio loyer/revenu (%)', color='#E76F51')
        ax2.tick_params(axis='y', labelcolor='#E76F51')
        
        # Combiner les légendes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_price_rent_comparison(self, df, ax):
        """Plot de la comparaison prix/loyers"""
        # Calcul du ratio prix/loyer (rendement brut)
        price_rent_ratio_house = []
        price_rent_ratio_apartment = []
        
        for i, row in df.iterrows():
            # Pour les maisons: prix annuel du loyer / prix d'achat
            rendement_maison = (row['Loyer_m2_Maison'] * 12 * 100) / row['Prix_m2_Maison']
            price_rent_ratio_house.append(rendement_maison)
            
            # Pour les appartements
            rendement_appartement = (row['Loyer_m2_Appartement'] * 12 * 100) / row['Prix_m2_Appartement']
            price_rent_ratio_apartment.append(rendement_appartement)
        
        ax.plot(df['Annee'], price_rent_ratio_house, label='Rendement brut maison (%)', 
               linewidth=2, color='#2A9D8F', alpha=0.8)
        ax.plot(df['Annee'], price_rent_ratio_apartment, label='Rendement brut appartement (%)', 
               linewidth=2, color='#E76F51', alpha=0.8)
        
        ax.set_title('Rendements Bruts Immobiliers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rendement brut (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _generate_real_estate_insights(self, df):
        """Génère des insights analytiques adaptés au territoire"""
        print(f"🏠 INSIGHTS IMMOBILIERS - {self.territoire} (DROM-COM)")
        print("=" * 60)
        
        # 1. Statistiques de base
        print("\n1. 📈 STATISTIQUES GÉNÉRALES:")
        avg_house_price = df['Prix_m2_Maison'].mean()
        avg_apartment_price = df['Prix_m2_Appartement'].mean()
        avg_house_rent = df['Loyer_m2_Maison'].mean()
        avg_apartment_rent = df['Loyer_m2_Appartement'].mean()
        
        print(f"Prix moyen au m² maison: {avg_house_price:.0f} €")
        print(f"Prix moyen au m² appartement: {avg_apartment_price:.0f} €")
        print(f"Loyer moyen au m² maison: {avg_house_rent:.1f} €/mois")
        print(f"Loyer moyen au m² appartement: {avg_apartment_rent:.1f} €/mois")
        
        # 2. Évolution des prix
        print("\n2. 📊 ÉVOLUTION DES PRIX:")
        house_price_growth = ((df['Prix_m2_Maison'].iloc[-1] / 
                              df['Prix_m2_Maison'].iloc[0]) - 1) * 100
        apartment_price_growth = ((df['Prix_m2_Appartement'].iloc[-1] / 
                                  df['Prix_m2_Appartement'].iloc[0]) - 1) * 100
        
        print(f"Croissance des prix maison ({self.start_year}-{self.end_year}): {house_price_growth:.1f}%")
        print(f"Croissance des prix appartement ({self.start_year}-{self.end_year}): {apartment_price_growth:.1f}%")
        
        # 3. Accessibilité
        print("\n3. 🏠 ACCESSIBILITÉ:")
        avg_years_house = df['Annee_Salaire_Maison'].mean()
        avg_years_apartment = df['Annee_Salaire_Appartement'].mean()
        avg_rent_ratio = df['Ratio_Loyer_Revenu'].mean()
        
        print(f"Années de salaire nécessaires pour une maison: {avg_years_house:.1f} ans")
        print(f"Années de salaire nécessaires pour un appartement: {avg_years_apartment:.1f} ans")
        print(f"Part du revenu consacrée au loyer: {avg_rent_ratio:.1f}%")
        
        # 4. Marché et investissements
        print("\n4. 📋 INDICATEURS DE MARCHÉ:")
        avg_transactions = df['Transactions_Total'].mean()
        avg_vacancy = df['Taux_Vacance_Locatif'].mean()
        avg_foreign_investment = df['Investissement_Etranger'].mean()
        
        print(f"Transactions annuelles moyennes: {avg_transactions:.0f}")
        print(f"Taux de vacance locative moyen: {avg_vacancy:.1f}%")
        print(f"Investissement étranger moyen: {avg_foreign_investment:.1f} M€/an")
        
        # 5. Spécificités du territoire
        print(f"\n5. 🌟 SPÉCIFICITÉS DE {self.territoire.upper()}:")
        print(f"Spécialités: {', '.join(self.config['specialites'])}")
        print(f"Zones clés: {', '.join(self.config['zones_cles'])}")
        
        # 6. Événements marquants
        print("\n6. 📅 ÉVÉNEMENTS MARQUANTS:")
        print("• 2008-2009: Crise financière mondiale (baisse des prix)")
        print("• 2011: Départementalisation de Mayotte (hausse des investissements)")
        print("• 2017: Mouvements sociaux en Guyane (ralentissement du marché)")
        print("• 2018-2021: Référendums en Nouvelle-Calédonie (incertitudes)")
        print("• 2020-2021: Pandémie de COVID-19 (baisse des transactions)")
        
        # 7. Recommandations
        print("\n7. 💡 RECOMMANDATIONS STRATÉGIQUES:")
        
        if avg_years_house > 10:  # Marché très inaccessible
            print("• Développer des programmes d'accession à la propriété")
            print("• Soutenir les dispositifs de prêts à taux zéro")
            print("• Encourager la construction de logements sociaux")
        
        if avg_vacancy > 7:  # Forte vacance locative
            print("• Diversifier l'offre locative (colocation, meublé, etc.)")
            print("• Améliorer la qualité du parc immobilier existant")
            print("• Développer le tourisme locatif")
        
        if "tourisme" in self.config["specialites"]:
            print("• Développer l'immobilier de tourisme et la location saisonnière")
            print("• Améliorer les infrastructures d'accueil touristique")
            print("• Former les professionnels de l'immobilier touristique")
        
        if "luxe" in self.config["specialites"]:
            print("• Positionner le territoire sur le marché international du luxe")
            print("• Développer des services haut de gamme pour résidents")
            print("• Promouvoir les atouts du territoire à l'international")
        
        if "croissance" in self.config["specialites"]:
            print("• Anticiper les besoins en logements de la population croissante")
            print("• Développer les infrastructures en parallèle de l'urbanisation")
            print("• Préserver les espaces naturels malgré la pression foncière")

def main():
    """Fonction principale pour les DROM-COM"""
    # Liste des DROM-COM
    territoires = [
        "Guadeloupe", "Martinique", "Guyane", "La Réunion", "Mayotte",
        "Saint-Martin", "Saint-Barthélemy", "Saint-Pierre-et-Miquelon",
        "Wallis-et-Futuna", "Polynésie française", "Nouvelle-Calédonie"
    ]
    
    print("🏠 ANALYSE DU MARCHÉ IMMOBILIER DES DROM-COM (2002-2025)")
    print("=" * 60)
    
    # Demander à l'utilisateur de choisir un territoire
    print("Liste des territoires disponibles:")
    for i, territoire in enumerate(territoires, 1):
        print(f"{i}. {territoire}")
    
    try:
        choix = int(input("\nChoisissez le numéro du territoire à analyser: "))
        if choix < 1 or choix > len(territoires):
            raise ValueError
        territoire_selectionne = territoires[choix-1]
    except (ValueError, IndexError):
        print("Choix invalide. Sélection de La Réunion par défaut.")
        territoire_selectionne = "La Réunion"
    
    # Initialiser l'analyseur
    analyzer = DromcomImmobilierAnalyzer(territoire_selectionne)
    
    # Générer les données
    real_estate_data = analyzer.generate_real_estate_data()
    
    # Sauvegarder les données
    output_file = f'{territoire_selectionne}_real_estate_data_2002_2025.csv'
    real_estate_data.to_csv(output_file, index=False)
    print(f"💾 Données sauvegardées: {output_file}")
    
    # Aperçu des données
    print("\n👀 Aperçu des données:")
    print(real_estate_data[['Annee', 'Prix_m2_Maison', 'Prix_m2_Appartement', 'Transactions_Total', 'Revenu_Median']].head())
    
    # Créer l'analyse
    print("\n📈 Création de l'analyse immobilière...")
    analyzer.create_real_estate_analysis(real_estate_data)
    
    print(f"\n✅ Analyse du marché immobilier de {territoire_selectionne} terminée!")
    print(f"📊 Période: {analyzer.start_year}-{analyzer.end_year}")
    print("📦 Données: Prix, loyers, transactions, investissements, accessibilité")

if __name__ == "__main__":
    main()