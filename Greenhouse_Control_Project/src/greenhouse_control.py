import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
from fpdf import FPDF
import tempfile
import shutil
import sys
from PIL import Image, ImageTk
import customtkinter as ctk
from datetime import datetime, timedelta
from tkcalendar import DateEntry
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from io import BytesIO
import xlsxwriter
import json

# Setări vizuale globale
plt.style.use('dark_background')
sns.set_theme(style="whitegrid", palette="muted")

# Culori și stiluri personalizate
COLORS = {
    'primary': '#1A1A1A',      # Negru foarte închis
    'primary_light': '#2D2D2D', # Negru mediu
    'accent': '#00B894',       # Verde turcoaz
    'background': '#121212',   # Negru foarte închis
    'surface': '#1E1E1E',      # Negru cu nuanță
    'text_primary': '#FFFFFF', # Alb
    'text_secondary': '#E0E0E0', # Alb deschis
    'success': '#00B894',      # Verde turcoaz
    'error': '#FF4D4D',        # Roșu
    'warning': '#FFD700',      # Galben
    'border': '#333333',       # Negru cu nuanță
    'hover': '#2A2A2A',        # Negru cu nuanță
    'shadow': '#000000',       # Negru pur
    'card_bg': '#1E1E1E',      # Fundal card
    'header_bg': '#121212',    # Fundal header
    'header_text': '#FFFFFF',  # Text header
    'menu_bg': '#2D2D2D',      # Fundal meniu
    'menu_text': '#FFFFFF',    # Text meniu
    'menu_hover': '#3D3D3D',   # Hover meniu
    'select_bg': '#2D2D2D',    # Fundal selectie
    'select_text': '#FFFFFF',  # Text selectie
    'select_hover': '#3D3D3D'  # Hover selectie
}

class ModernButton(ctk.CTkButton):
    def __init__(self, master, **kwargs):
        # Extract height from kwargs if provided, otherwise use default
        height = kwargs.pop('height', 32)
        super().__init__(
            master,
            fg_color=COLORS['accent'],
            hover_color=COLORS['success'],
            text_color=COLORS['text_primary'],
            corner_radius=6,
            height=height,
            font=ctk.CTkFont(family='Segoe UI', size=12, weight='bold'),
            **kwargs
        )

class ModernFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        # Extract fg_color from kwargs if provided, otherwise use default
        fg_color = kwargs.pop('fg_color', COLORS['surface'])
        super().__init__(
            master,
            fg_color=fg_color,
            corner_radius=6,
            border_width=1,
            border_color=COLORS['border'],
            **kwargs
        )

class ModernLabel(ctk.CTkLabel):
    def __init__(self, master, **kwargs):
        # Extrage text_color din kwargs dacă există, altfel folosește valoarea implicită
        text_color = kwargs.pop('text_color', COLORS['text_primary'])
        super().__init__(
            master,
            text_color=text_color,
            **kwargs
        )

class HeaderLabel(ctk.CTkLabel):
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            text_color=COLORS['header_text'],
            **kwargs
        )

class PredictiveSystem:
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.forecast_horizon = 24  # ore

    def train_models(self, data):
        """Antrenează modele pentru predicție"""
        for target in ['temperature', 'humidity']:
            if target in data.columns:
                X = data[['temperature', 'humidity']].shift(1).dropna()
                y = data[target].iloc[1:]
                
                model = LinearRegression()
                model.fit(X, y)
                
                self.models[target] = model

    def make_predictions(self, current_data):
        """Face predicții pentru următoarele ore"""
        predictions = {}
        for target, model in self.models.items():
            prediction = model.predict([current_data])[0]
            predictions[target] = prediction
        
        self.predictions = predictions
        return predictions

class OptimizationSystem:
    def __init__(self):
        self.optimal_ranges = {
            'temperature': {'min': 20, 'max': 25},
            'humidity': {'min': 60, 'max': 80},
            'light': {'min': 5000, 'max': 10000}
        }
        self.energy_consumption = {
            'ventilator': 100,  # W
            'umidificator': 150,
            'cortina': 50
        }
        self.optimization_history = []

    def optimize_parameters(self, current_values, predictions):
        """Optimizează parametrii pentru eficiență energetică"""
        recommendations = []
        energy_savings = 0
        
        for param, current in current_values.items():
            if param in self.optimal_ranges:
                optimal = self.optimal_ranges[param]
                predicted = predictions.get(param, current)
                
                if current < optimal['min']:
                    deviation = optimal['min'] - current
                    action = 'increase'
                elif current > optimal['max']:
                    deviation = current - optimal['max']
                    action = 'decrease'
                else:
                    deviation = 0
                    action = 'maintain'
                
                if deviation > 0:
                    actuator = self.get_actuator_for_param(param)
                    energy_cost = self.energy_consumption[actuator]
                    
                    recommendations.append({
                        'parameter': param,
                        'current_value': current,
                        'target_value': optimal['min'] if action == 'increase' else optimal['max'],
                        'action': action,
                        'actuator': actuator,
                        'energy_cost': energy_cost,
                        'priority': deviation * energy_cost
                    })
        
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        if recommendations:
            energy_savings = sum(rec['energy_cost'] for rec in recommendations)
        
        return recommendations, energy_savings

    def get_actuator_for_param(self, param):
        """Returnează actuatorul corespunzător parametrului"""
        actuator_map = {
            'temperature': 'ventilator',
            'humidity': 'umidificator',
            'light': 'cortina'
        }
        return actuator_map.get(param)

class ControlSystem:
    def __init__(self):
        # Inițializează sistemele avansate
        self.predictive_system = PredictiveSystem()
        self.optimization_system = OptimizationSystem()
        
        # Parametri optimi configurabili
        self.parametri_optimi = {
            'temperatura': {
                'min': 20,
                'max': 25,
                'toleranta': 1.0,
                'actuator': 'ventilator',
                'prioritate': 1
            },
            'umiditate': {
                'min': 60,
                'max': 80,
                'toleranta': 5.0,
                'actuator': 'umidificator',
                'prioritate': 2
            },
            'lumina': {
                'min': 5000,
                'max': 10000,
                'toleranta': 500,
                'actuator': 'cortina',
                'prioritate': 3
            }
        }
        
        # Starea curentă a sistemului
        self.stare_curenta = {
            'temperatura': None,
            'umiditate': None,
            'lumina': None,
            'actuatori': {
                'ventilator': False,
                'umidificator': False,
                'cortina': False
            }
        }
        
        # Istoric acțiuni
        self.istoric_actiuni = []
        
        # Logger pentru înregistrarea acțiunilor
        self.logger = []

    def actualizeaza_masuratori(self, temperatura, umiditate, lumina):
        """Actualizează măsurătorile curente și verifică necesitatea de acțiune"""
        # Actualizează starea curentă
        self.stare_curenta['temperatura'] = temperatura
        self.stare_curenta['umiditate'] = umiditate
        self.stare_curenta['lumina'] = lumina
        
        # Verifică siguranța
        sigur, mesaj = self.verifica_siguranta()
        if not sigur:
            self.log_actiune("EROARE", mesaj)
            return False, mesaj
        
        # Face predicții
        current_data = [temperatura, umiditate]
        predictions = self.predictive_system.make_predictions(current_data)
        
        # Optimizează parametrii
        current_values = {
            'temperature': temperatura,
            'humidity': umiditate,
            'light': lumina
        }
        recommendations, energy_savings = self.optimization_system.optimize_parameters(
            current_values, predictions
        )
        
        # Aplică recomandările
        for rec in recommendations:
            self.aplica_recomandare(rec)
        
        # Loggează optimizarea
        self.log_actiune(
            "OPTIMIZARE",
            f"Economii potențiale: {energy_savings}W"
        )
        
        return True, "Sistem actualizat cu succes"

    def aplica_recomandare(self, recomandare):
        """Aplică o recomandare de optimizare"""
        actuator = recomandare['actuator']
        action = recomandare['action']
        
        if action == 'increase':
            self.stare_curenta['actuatori'][actuator] = True
        elif action == 'decrease':
            self.stare_curenta['actuatori'][actuator] = False
        
        self.log_actiune(
            "ACTIUNE",
            f"{actuator.capitalize()} {'pornit' if self.stare_curenta['actuatori'][actuator] else 'oprit'} pentru {recomandare['parameter']}"
        )

    def train_predictive_models(self, data):
        """Antrenează modelele predictive"""
        self.predictive_system.train_models(data)
        self.log_actiune("CONFIG", "Modele predictive antrenate")

    def verifica_siguranta(self):
        """Verifică dacă valorile sunt în limite sigure"""
        if self.stare_curenta['temperatura'] > 35:
            return False, "Temperatura critică detectată"
        if self.stare_curenta['umiditate'] > 95:
            return False, "Umiditate critică detectată"
        if self.stare_curenta['lumina'] > 15000:
            return False, "Nivel de lumină critic detectat"
        return True, None

    def log_actiune(self, tip, mesaj):
        """Înregistrează o acțiune în log"""
        log = {
            'timestamp': datetime.now(),
            'tip': tip,
            'mesaj': mesaj,
            'stare': self.stare_curenta.copy()
        }
        self.logger.append(log)
        
        # Păstrează doar ultimele 1000 de înregistrări
        if len(self.logger) > 1000:
            self.logger = self.logger[-1000:]

    def get_raport_stare(self):
        """Generează un raport al stării curente"""
        return {
            'stare_curenta': self.stare_curenta,
            'ultimele_actiuni': self.logger[-5:] if self.logger else [],
            'parametri_optimi': self.parametri_optimi
        }

    def set_parametri_optimi(self, parametri):
        """Actualizează parametrii optimi ai sistemului"""
        for param, valori in parametri.items():
            if param in self.parametri_optimi:
                self.parametri_optimi[param].update(valori)
        self.log_actiune("CONFIG", "Parametri optimi actualizați")

class AdvancedAnalysis:
    def __init__(self):
        self.trend_analysis = {}
        self.correlation_matrix = None
        self.anomalies = []

    def analyze_trends(self, data):
        """Analizează tendințele din date"""
        for column in ['temperature', 'humidity', 'N', 'P', 'K']:
            if column in data.columns:
                # Calcul trend liniar
                x = np.arange(len(data))
                y = data[column].values
                slope, intercept = np.polyfit(x, y, 1)
                
                self.trend_analysis[column] = {
                    'slope': slope,
                    'intercept': intercept,
                    'trend': 'crescător' if slope > 0 else 'descrescător' if slope < 0 else 'constant'
                }

    def find_correlations(self, data):
        """Calculează matricea de corelație"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        self.correlation_matrix = data[numeric_columns].corr()

    def detect_anomalies(self, data):
        """Detectează anomalii în date"""
        for column in ['temperature', 'humidity', 'N', 'P', 'K']:
            if column in data.columns:
                # Folosește metoda IQR pentru detectarea outlier-ilor
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                anomalies = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
                if not anomalies.empty:
                    self.anomalies.extend([
                        {
                            'timestamp': row['date'],
                            'parameter': column,
                            'value': row[column],
                            'expected_range': f"{lower_bound:.2f} - {upper_bound:.2f}"
                        }
                        for _, row in anomalies.iterrows()
                    ])

class GreenhouseApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Inițializează sistemele avansate
        self.advanced_analysis = AdvancedAnalysis()
        self.control_system = ControlSystem()
        
        # Configurare fereastră principală
        self.title("🌿 Smart Greenhouse Controller")
        self.geometry("1200x700")
        self.configure(fg_color=COLORS['background'])
        
        # Stiluri globale
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        # Configurare stil pentru meniuri și combobox-uri
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurare notebook
        style.configure('TNotebook', background=COLORS['background'])
        style.configure('TNotebook.Tab', 
                       background=COLORS['menu_bg'], 
                       foreground=COLORS['menu_text'],
                       padding=[8, 4],
                       font=('Segoe UI', 10))
        style.map('TNotebook.Tab',
                 background=[('selected', COLORS['select_hover'])],
                 foreground=[('selected', COLORS['select_text'])])
        
        # Configurare combobox
        style.configure('TCombobox',
                      fieldbackground=COLORS['select_bg'],
                      background=COLORS['select_bg'],
                      foreground=COLORS['select_text'],
                      arrowcolor=COLORS['select_text'],
                      borderwidth=1,
                      relief='solid')
        
        style.map('TCombobox',
                 fieldbackground=[('readonly', COLORS['select_bg'])],
                 selectbackground=[('readonly', COLORS['select_hover'])],
                 selectforeground=[('readonly', COLORS['select_text'])])
        
        self.option_add('*TCombobox*Listbox*Background', COLORS['select_bg'])
        self.option_add('*TCombobox*Listbox*Foreground', COLORS['select_text'])
        self.option_add('*TCombobox*Listbox*selectBackground', COLORS['select_hover'])
        self.option_add('*TCombobox*Listbox*selectForeground', COLORS['select_text'])
        self.option_add('*TCombobox*Background', COLORS['select_bg'])
        self.option_add('*TCombobox*Foreground', COLORS['select_text'])
        self.option_add('*TCombobox*selectBackground', COLORS['select_hover'])
        self.option_add('*TCombobox*selectForeground', COLORS['select_text'])

        self.create_widgets()
        self.create_status_bar()
        self.create_side_panel()

    def create_widgets(self):
        # Main frame
        main_frame = ModernFrame(self)
        main_frame.pack(pady=6, padx=6, fill='both', expand=True)

        # Header
        header_frame = ctk.CTkFrame(main_frame, height=50, fg_color=COLORS['header_bg'], corner_radius=6)
        header_frame.pack(fill='x', padx=4, pady=4)
        
        title_frame = ModernFrame(header_frame, fg_color="transparent")
        title_frame.pack(side='left', padx=10, pady=2)
        
        title_label = HeaderLabel(
            title_frame,
                  text="🌡️ Greenhouse Monitoring System",
            font=ctk.CTkFont(family='Segoe UI', size=16, weight='bold')
        )
        title_label.pack(side='left')

        help_button = ModernButton(
            header_frame,
                   text="❔ Help",
                   command=self.show_help,
            width=80
        )
        help_button.pack(side='right', padx=10, pady=2)

        # Main panel with tabs
        self.main_notebook = ttk.Notebook(main_frame)
        self.main_notebook.pack(fill='both', expand=True, pady=4)

        # Data Tab
        data_tab = ModernFrame(self.main_notebook)
        self.main_notebook.add(data_tab, text='📁 Sensor Data')

        # Card for data operations
        operations_card = ModernFrame(data_tab)
        operations_card.pack(fill='x', padx=6, pady=6)
        
        HeaderLabel(
            operations_card,
            text="Data Operations",
            font=ctk.CTkFont(family='Segoe UI', size=13, weight='bold')
        ).pack(pady=3)
        
        button_frame = ModernFrame(operations_card)
        button_frame.pack(pady=3)
        
        ModernButton(
            button_frame,
            text="📤 Load CSV Data",
                   command=self.load_csv,
            width=150
        ).pack(side='left', padx=3)

        ModernButton(
            button_frame,
            text="🔍 View Data",
                   command=self.preview_data,
            width=150
        ).pack(side='left', padx=3)
        
        # Card for date range selection
        date_card = ModernFrame(data_tab)
        date_card.pack(fill='x', padx=4, pady=2)
        
        HeaderLabel(
            date_card,
            text="📅 Select Date Range",
            font=ctk.CTkFont(family='Segoe UI', size=11, weight='bold')
        ).pack(pady=1)
        
        date_frame = ModernFrame(date_card, fg_color="transparent")
        date_frame.pack(pady=1)
        
        # Start date frame
        start_frame = ModernFrame(date_frame, fg_color="transparent")
        start_frame.pack(fill='x', pady=0)
        
        ModernLabel(start_frame, text="Start Date:", font=ctk.CTkFont(family='Segoe UI', size=10)).pack(side='left', padx=2)
        
        # Start day selector
        self.start_day = ctk.CTkOptionMenu(
            start_frame,
            values=[str(i).zfill(2) for i in range(1, 32)],
            width=35,
            height=25,
            fg_color=COLORS['select_bg'],
            button_color=COLORS['accent'],
            button_hover_color=COLORS['success'],
            text_color=COLORS['select_text'],
            font=ctk.CTkFont(family='Segoe UI', size=10)
        )
        self.start_day.pack(side='left', padx=0)
        
        # Start month selector
        self.start_month = ctk.CTkOptionMenu(
            start_frame,
            values=[str(i).zfill(2) for i in range(1, 13)],
            width=35,
            height=25,
            fg_color=COLORS['select_bg'],
            button_color=COLORS['accent'],
            button_hover_color=COLORS['success'],
            text_color=COLORS['select_text'],
            font=ctk.CTkFont(family='Segoe UI', size=10)
        )
        self.start_month.pack(side='left', padx=0)
        
        # Start year selector
        current_year = datetime.now().year
        self.start_year = ctk.CTkOptionMenu(
            start_frame,
            values=[str(i) for i in range(current_year - 5, current_year + 1)],
            width=45,
            height=25,
            fg_color=COLORS['select_bg'],
            button_color=COLORS['accent'],
            button_hover_color=COLORS['success'],
            text_color=COLORS['select_text'],
            font=ctk.CTkFont(family='Segoe UI', size=10)
        )
        self.start_year.pack(side='left', padx=0)
        
        # End date frame
        end_frame = ModernFrame(date_frame, fg_color="transparent")
        end_frame.pack(fill='x', pady=0)
        
        ModernLabel(end_frame, text="End Date:", font=ctk.CTkFont(family='Segoe UI', size=10)).pack(side='left', padx=2)
        
        # End day selector
        self.end_day = ctk.CTkOptionMenu(
            end_frame,
            values=[str(i).zfill(2) for i in range(1, 32)],
            width=35,
            height=25,
            fg_color=COLORS['select_bg'],
            button_color=COLORS['accent'],
            button_hover_color=COLORS['success'],
            text_color=COLORS['select_text'],
            font=ctk.CTkFont(family='Segoe UI', size=10)
        )
        self.end_day.pack(side='left', padx=0)
        
        # End month selector
        self.end_month = ctk.CTkOptionMenu(
            end_frame,
            values=[str(i).zfill(2) for i in range(1, 13)],
            width=35,
            height=25,
            fg_color=COLORS['select_bg'],
            button_color=COLORS['accent'],
            button_hover_color=COLORS['success'],
            text_color=COLORS['select_text'],
            font=ctk.CTkFont(family='Segoe UI', size=10)
        )
        self.end_month.pack(side='left', padx=0)
        
        # End year selector
        self.end_year = ctk.CTkOptionMenu(
            end_frame,
            values=[str(i) for i in range(current_year - 5, current_year + 1)],
            width=45,
            height=25,
            fg_color=COLORS['select_bg'],
            button_color=COLORS['accent'],
            button_hover_color=COLORS['success'],
            text_color=COLORS['select_text'],
            font=ctk.CTkFont(family='Segoe UI', size=10)
        )
        self.end_year.pack(side='left', padx=0)
        
        # Set default values to current date
        now = datetime.now()
        self.start_day.set(str(now.day).zfill(2))
        self.start_month.set(str(now.month).zfill(2))
        self.start_year.set(str(now.year))
        self.end_day.set(str(now.day).zfill(2))
        self.end_month.set(str(now.month).zfill(2))
        self.end_year.set(str(now.year))
        
        # Card for actions
        actions_card = ModernFrame(data_tab)
        actions_card.pack(fill='x', padx=6, pady=4)
        
        HeaderLabel(
            actions_card,
            text="Actions",
            font=ctk.CTkFont(family='Segoe UI', size=12, weight='bold')
        ).pack(pady=2)
        
        action_frame = ModernFrame(actions_card, fg_color="transparent")
        action_frame.pack(pady=2)
        
        ModernButton(
            action_frame,
            text="📈 Generate Report",
                   command=self.generate_report,
            width=130
        ).grid(row=0, column=0, padx=2, pady=2)

        ModernButton(
            action_frame,
            text="🖼️ Graphs",
                   command=self.plot_data,
            width=130
        ).grid(row=0, column=1, padx=2, pady=2)

        ModernButton(
            action_frame,
            text="💾 Export",
                   command=self.save_file,
            width=130
        ).grid(row=0, column=2, padx=2, pady=2)

        # Add control panel
        control_card = ModernFrame(data_tab)
        control_card.pack(fill='x', padx=6, pady=4)
        
        HeaderLabel(
            control_card,
            text="🎛️ Automatic Control",
            font=ctk.CTkFont(family='Segoe UI', size=12, weight='bold')
        ).pack(pady=2)
        
        control_frame = ModernFrame(control_card, fg_color="transparent")
        control_frame.pack(pady=2)
        
        # Actuator status
        self.ventilator_status = ModernLabel(
            control_frame,
            text="Ventilator: Off",
            font=ctk.CTkFont(family='Segoe UI', size=11)
        )
        self.ventilator_status.pack(side='left', padx=5)
        
        self.umidificator_status = ModernLabel(
            control_frame,
            text="Humidifier: Off",
            font=ctk.CTkFont(family='Segoe UI', size=11)
        )
        self.umidificator_status.pack(side='left', padx=5)
        
        self.cortina_status = ModernLabel(
            control_frame,
            text="Curtain: Closed",
            font=ctk.CTkFont(family='Segoe UI', size=11)
        )
        self.cortina_status.pack(side='left', padx=5)
        
        # Manual update button
        ModernButton(
            control_frame,
            text="🔄 Update Control",
            command=self.actualizeaza_control,
            width=150
        ).pack(side='right', padx=5)

    def create_side_panel(self):
        # Create main side panel with modern look
        self.side_panel = ModernFrame(self, width=200, height=300)
        self.side_panel.pack(side='right', anchor='ne', padx=8, pady=8)
        
        # Modern title section with gradient
        title_frame = ModernFrame(self.side_panel, fg_color=COLORS['header_bg'])
        title_frame.pack(fill='x', padx=0, pady=0)
        
        title_label = HeaderLabel(
            title_frame,
            text="📊 Stats",
            font=ctk.CTkFont(family='Segoe UI', size=12, weight='bold')
        )
        title_label.pack(padx=8, pady=4)
        
        # Create modern scrollable area
        canvas = tk.Canvas(self.side_panel, bg=COLORS['background'], highlightthickness=0, height=250)
        scrollbar = ttk.Scrollbar(self.side_panel, orient="vertical", command=canvas.yview)
        
        # Create the scrollable frame
        self.stats_frame = ModernFrame(canvas, fg_color=COLORS['background'])
        self.stats_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Create window in canvas
        canvas.create_window((0, 0), window=self.stats_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel event
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # System Status Card with modern design
        status_card = ModernFrame(self.stats_frame, fg_color=COLORS['card_bg'])
        status_card.pack(fill='x', padx=2, pady=1)
        
        status_header = ModernFrame(status_card, fg_color="transparent")
        status_header.pack(fill='x', padx=2, pady=1)
        
        ModernLabel(
            status_header,
            text="⚙️ Status",
            font=ctk.CTkFont(family='Segoe UI', size=12, weight='bold')
        ).pack(side='left')
        
        self.status_indicator = ModernLabel(
            status_header,
            text="🔴 Offline",
            font=ctk.CTkFont(family='Segoe UI', size=12),
            text_color=COLORS['error']
        )
        self.status_indicator.pack(side='right')
        
        # Environment Stats Card with modern layout
        env_card = ModernFrame(self.stats_frame, fg_color=COLORS['card_bg'])
        env_card.pack(fill='x', padx=2, pady=1)
        
        env_header = ModernFrame(env_card, fg_color="transparent")
        env_header.pack(fill='x', padx=2, pady=1)
        
        ModernLabel(
            env_header,
            text="🌡️ Environment",
            font=ctk.CTkFont(family='Segoe UI', size=12, weight='bold')
        ).pack(side='left')
        
        # Temperature Section with modern grid
        temp_frame = ModernFrame(env_card, fg_color="transparent")
        temp_frame.pack(fill='x', padx=2, pady=0)
        
        temp_grid = ModernFrame(temp_frame, fg_color="transparent")
        temp_grid.pack(fill='x', expand=True)
        
        temp_left = ModernFrame(temp_grid, fg_color="transparent")
        temp_left.pack(side='left', fill='x', expand=True)
        
        temp_right = ModernFrame(temp_grid, fg_color="transparent")
        temp_right.pack(side='right', fill='x', expand=True)
        
        self.temp_current = ModernLabel(
            temp_left,
            text="Now: N/A",
            font=ctk.CTkFont(family='Segoe UI', size=14),
            anchor='center'
        )
        self.temp_current.pack(fill='x', padx=2, pady=0)
        
        self.temp_avg = ModernLabel(
            temp_right,
            text="Avg: N/A",
            font=ctk.CTkFont(family='Segoe UI', size=14),
            anchor='center'
        )
        self.temp_avg.pack(fill='x', padx=2, pady=0)
        
        # Humidity Section with modern grid
        humidity_frame = ModernFrame(env_card, fg_color="transparent")
        humidity_frame.pack(fill='x', padx=2, pady=0)
        
        humidity_grid = ModernFrame(humidity_frame, fg_color="transparent")
        humidity_grid.pack(fill='x', expand=True)
        
        humidity_left = ModernFrame(humidity_grid, fg_color="transparent")
        humidity_left.pack(side='left', fill='x', expand=True)
        
        humidity_right = ModernFrame(humidity_grid, fg_color="transparent")
        humidity_right.pack(side='right', fill='x', expand=True)
        
        self.humidity_current = ModernLabel(
            humidity_left,
            text="Now: N/A",
            font=ctk.CTkFont(family='Segoe UI', size=14),
            anchor='center'
        )
        self.humidity_current.pack(fill='x', padx=2, pady=0)
        
        self.humidity_avg = ModernLabel(
            humidity_right,
            text="Avg: N/A",
            font=ctk.CTkFont(family='Segoe UI', size=14),
            anchor='center'
        )
        self.humidity_avg.pack(fill='x', padx=2, pady=0)
        
        # NPK Levels Card with modern design
        npk_card = ModernFrame(self.stats_frame, fg_color=COLORS['card_bg'])
        npk_card.pack(fill='x', padx=2, pady=1)
        
        npk_header = ModernFrame(npk_card, fg_color="transparent")
        npk_header.pack(fill='x', padx=2, pady=1)
        
        ModernLabel(
            npk_header,
            text="🌱 NPK",
            font=ctk.CTkFont(family='Segoe UI', size=12, weight='bold')
        ).pack(side='left')
        
        npk_content = ModernFrame(npk_card, fg_color="transparent")
        npk_content.pack(fill='x', padx=2, pady=0)
        
        npk_grid = ModernFrame(npk_content, fg_color="transparent")
        npk_grid.pack(fill='x', expand=True)
        
        npk_left = ModernFrame(npk_grid, fg_color="transparent")
        npk_left.pack(side='left', fill='x', expand=True)
        
        npk_middle = ModernFrame(npk_grid, fg_color="transparent")
        npk_middle.pack(side='left', fill='x', expand=True)
        
        npk_right = ModernFrame(npk_grid, fg_color="transparent")
        npk_right.pack(side='right', fill='x', expand=True)
        
        self.n_level = ModernLabel(
            npk_left,
            text="N: N/A",
            font=ctk.CTkFont(family='Segoe UI', size=14),
            anchor='center'
        )
        self.n_level.pack(fill='x', padx=2, pady=0)
        
        self.p_level = ModernLabel(
            npk_middle,
            text="P: N/A",
            font=ctk.CTkFont(family='Segoe UI', size=14),
            anchor='center'
        )
        self.p_level.pack(fill='x', padx=2, pady=0)
        
        self.k_level = ModernLabel(
            npk_right,
            text="K: N/A",
            font=ctk.CTkFont(family='Segoe UI', size=14),
            anchor='center'
        )
        self.k_level.pack(fill='x', padx=2, pady=0)
        
        # Actuators Card with modern design
        actuators_card = ModernFrame(self.stats_frame, fg_color=COLORS['card_bg'])
        actuators_card.pack(fill='x', padx=2, pady=1)
        
        actuators_header = ModernFrame(actuators_card, fg_color="transparent")
        actuators_header.pack(fill='x', padx=2, pady=1)
        
        ModernLabel(
            actuators_header,
            text="🎛️ Actuators",
            font=ctk.CTkFont(family='Segoe UI', size=12, weight='bold')
        ).pack(side='left')
        
        actuators_content = ModernFrame(actuators_card, fg_color="transparent")
        actuators_content.pack(fill='x', padx=2, pady=0)
        
        actuators_grid = ModernFrame(actuators_content, fg_color="transparent")
        actuators_grid.pack(fill='x', expand=True)
        
        actuators_left = ModernFrame(actuators_grid, fg_color="transparent")
        actuators_left.pack(side='left', fill='x', expand=True)
        
        actuators_middle = ModernFrame(actuators_grid, fg_color="transparent")
        actuators_middle.pack(side='left', fill='x', expand=True)
        
        actuators_right = ModernFrame(actuators_grid, fg_color="transparent")
        actuators_right.pack(side='right', fill='x', expand=True)
        
        self.ventilator_status = ModernLabel(
            actuators_left,
            text="Vent: Off",
            font=ctk.CTkFont(family='Segoe UI', size=14),
            anchor='center'
        )
        self.ventilator_status.pack(fill='x', padx=2, pady=0)
        
        self.humidifier_status = ModernLabel(
            actuators_middle,
            text="Humid: Off",
            font=ctk.CTkFont(family='Segoe UI', size=14),
            anchor='center'
        )
        self.humidifier_status.pack(fill='x', padx=2, pady=0)
        
        self.curtain_status = ModernLabel(
            actuators_right,
            text="Curtain: Closed",
            font=ctk.CTkFont(family='Segoe UI', size=14),
            anchor='center'
        )
        self.curtain_status.pack(fill='x', padx=2, pady=0)
        
        # Last Update Card with modern design
        update_card = ModernFrame(self.stats_frame, fg_color=COLORS['card_bg'])
        update_card.pack(fill='x', padx=2, pady=1)
        
        self.last_update = ModernLabel(
            update_card,
            text="Last Update: N/A",
            font=ctk.CTkFont(family='Segoe UI', size=12),
            text_color=COLORS['text_secondary']
        )
        self.last_update.pack(fill='x', padx=2, pady=1)

    def create_status_bar(self):
        self.status_var = tk.StringVar()
        status_bar = ModernFrame(self, height=28)
        status_bar.pack(side='bottom', fill='x', padx=4, pady=2)
        
        self.status_label = ModernLabel(
            status_bar,
                               textvariable=self.status_var,
            font=ctk.CTkFont(family='Segoe UI', size=11),
            anchor='w',
            padx=8
        )
        self.status_label.pack(side='left', padx=8, pady=2)
        self.status_var.set("System ready for operation")

    def update_status(self, message, error=False):
        status_color = COLORS['error'] if error else COLORS['success']
        self.status_var.set(message)
        if hasattr(self, 'status_indicator'):
            self.status_indicator.configure(
                text=f"{'🔴' if error else '🟢'} {message}",
                text_color=status_color
            )
        self.update_idletasks()

    def load_csv(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Select CSV File"
        )
        if file_path:
            try:
                # Read CSV with error handling
                self.df = pd.read_csv(file_path)
                
                # Check for required columns
                required_columns = ['temperature', 'humidity']
                missing_columns = [col for col in required_columns if col not in self.df.columns]
                if missing_columns:
                    self.update_status(f"Missing required columns: {', '.join(missing_columns)}", error=True)
                    return
                
                # Check for timestamp column with different possible names
                timestamp_columns = ['timestamp', 'date', 'time', 'datetime']
                found_timestamp = False
                
                for col in timestamp_columns:
                    if col in self.df.columns:
                        # Rename the column to 'timestamp' for consistency
                        self.df = self.df.rename(columns={col: 'timestamp'})
                        found_timestamp = True
                        break
                
                if not found_timestamp:
                    self.update_status("No timestamp column found in data. Please ensure your CSV has a timestamp column.", error=True)
                    return
                
                # Convert timestamp to datetime
                try:
                    self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
                    # Remove rows where timestamp conversion failed
                    self.df = self.df.dropna(subset=['timestamp'])
                except Exception as e:
                    self.update_status(f"Error processing timestamp: {str(e)}", error=True)
                    return
                
                # Sort data by timestamp
                self.df = self.df.sort_values('timestamp')
                
                # Validate numeric columns and their ranges
                numeric_columns = ['temperature', 'humidity']
                if 'N' in self.df.columns: numeric_columns.append('N')
                if 'P' in self.df.columns: numeric_columns.append('P')
                if 'K' in self.df.columns: numeric_columns.append('K')
                
                # Define valid ranges for each parameter
                valid_ranges = {
                    'temperature': (-10, 50),  # Celsius
                    'humidity': (0, 100),      # Percentage
                    'N': (0, 255),            # NPK values
                    'P': (0, 255),
                    'K': (0, 255)
                }
                
                for col in numeric_columns:
                    try:
                        # Convert to numeric
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                        
                        # Check if column is in valid ranges
                        if col in valid_ranges:
                            min_val, max_val = valid_ranges[col]
                            invalid_rows = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)]
                            if not invalid_rows.empty:
                                self.update_status(f"Warning: Found {len(invalid_rows)} rows with {col} outside valid range ({min_val}-{max_val})", error=True)
                                # Remove invalid rows
                                self.df = self.df[(self.df[col] >= min_val) & (self.df[col] <= max_val)]
                    except Exception as e:
                        self.update_status(f"Error converting {col} to numeric: {str(e)}", error=True)
                        return
                
                # Remove rows with invalid numeric values
                self.df = self.df.dropna(subset=numeric_columns)
                
                if len(self.df) == 0:
                    self.update_status("No valid data found in the file", error=True)
                    return
                
                # Update UI with loaded data
                self.update_quick_stats()
                self.update_status(f"Data loaded successfully: {len(self.df)} rows")
                
                # Update current values in quick stats
                if not self.df.empty:
                    # Update temperature
                    current_temp = self.df['temperature'].iloc[-1]
                    self.temp_current.configure(text=f"Now: {current_temp:.1f}°C")
                    self.temp_avg.configure(text=f"Avg: {self.df['temperature'].mean():.1f}°C")
                    
                    # Update humidity
                    current_hum = self.df['humidity'].iloc[-1]
                    self.humidity_current.configure(text=f"Now: {current_hum:.1f}%")
                    self.humidity_avg.configure(text=f"Avg: {self.df['humidity'].mean():.1f}%")
                    
                    # Update NPK if available
                    if all(col in self.df.columns for col in ['N', 'P', 'K']):
                        self.n_level.configure(text=f"N: {self.df['N'].iloc[-1]:.1f}")
                        self.p_level.configure(text=f"P: {self.df['P'].iloc[-1]:.1f}")
                        self.k_level.configure(text=f"K: {self.df['K'].iloc[-1]:.1f}")
                    
                    # Update status indicator
                    self.status_indicator.configure(text="🟢 Online", text_color=COLORS['success'])
                    
                    # Update last update time
                    self.last_update.configure(
                        text=f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
                    )
                
            except Exception as e:
                self.update_status(f"Error loading file: {str(e)}", error=True)

    def preview_data(self):
        if not hasattr(self, 'df') or self.df is None:
            self.update_status("No data loaded", error=True)
            return
            
        preview_window = ctk.CTkToplevel(self)
        preview_window.title("Data Preview")
        preview_window.geometry("1000x600")
        
        # Create main container
        main_frame = ModernFrame(preview_window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create header with statistics
        header_frame = ModernFrame(main_frame, fg_color=COLORS['header_bg'])
        header_frame.pack(fill='x', padx=0, pady=0)
        
        # Add row count
        row_count = ModernLabel(
            header_frame,
            text=f"Total Rows: {len(self.df)}",
            font=ctk.CTkFont(family='Segoe UI', size=12)
        )
        row_count.pack(side='left', padx=20, pady=10)
        
        # Add date range
        date_range = ModernLabel(
            header_frame,
            text=f"Date Range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}",
            font=ctk.CTkFont(family='Segoe UI', size=12)
        )
        date_range.pack(side='right', padx=20, pady=10)
        
        # Create filter frame
        filter_frame = ModernFrame(main_frame)
        filter_frame.pack(fill='x', padx=5, pady=5)
        
        # Add column selector
        ModernLabel(
            filter_frame,
            text="Columns:",
            font=ctk.CTkFont(family='Segoe UI', size=12)
        ).pack(side='left', padx=5)
        
        self.column_vars = {}
        for col in self.df.columns:
            var = tk.BooleanVar(value=True)
            self.column_vars[col] = var
            cb = ctk.CTkCheckBox(
                filter_frame,
                text=col,
                variable=var,
                command=self.update_preview
            )
            cb.pack(side='left', padx=5)
        
        # Create data display area
        display_frame = ModernFrame(main_frame)
        display_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create treeview
        self.tree = ttk.Treeview(display_frame)
        self.tree.pack(side='left', fill='both', expand=True)
        
        # Add scrollbars
        scroll_y = ttk.Scrollbar(display_frame, orient='vertical', command=self.tree.yview)
        scroll_y.pack(side='right', fill='y')
        scroll_x = ttk.Scrollbar(main_frame, orient='horizontal', command=self.tree.xview)
        scroll_x.pack(side='bottom', fill='x')
        
        self.tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        # Configure treeview style
        style = ttk.Style()
        style.configure("Treeview",
                      background=COLORS['card_bg'],
                      foreground=COLORS['text_primary'],
                      fieldbackground=COLORS['card_bg'],
                      borderwidth=0)
        style.configure("Treeview.Heading",
                      background=COLORS['header_bg'],
                      foreground=COLORS['text_primary'],
                      font=('Segoe UI', 10, 'bold'))
        style.map('Treeview',
                 background=[('selected', COLORS['accent'])],
                 foreground=[('selected', COLORS['text_primary'])])
        
        # Initial data display
        self.update_preview()
        
    def update_preview(self):
        """Update the data preview based on selected columns"""
        # Clear existing columns
        for col in self.tree['columns']:
            self.tree.heading(col, text='')
            self.tree.column(col, width=0)
        
        # Get selected columns
        selected_cols = [col for col, var in self.column_vars.items() if var.get()]
        
        # Configure columns
        self.tree['columns'] = selected_cols
        for col in selected_cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor='center')
        
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add data
        for idx, row in self.df[selected_cols].iterrows():
            values = [str(row[col]) for col in selected_cols]
            self.tree.insert('', 'end', values=values)

    def generate_report(self):
        if not hasattr(self, 'df') or self.df is None:
            self.update_status("No data loaded", error=True)
            return

        try:
            # Get selected date range
            start_date = f"{self.start_year.get()}-{self.start_month.get()}-{self.start_day.get()}"
            end_date = f"{self.end_year.get()}-{self.end_month.get()}-{self.end_day.get()}"
            
            # Filter data by date range
            mask = (self.df['timestamp'] >= start_date) & (self.df['timestamp'] <= end_date)
            filtered_df = self.df.loc[mask]
            
            if len(filtered_df) == 0:
                self.update_status("No data found for selected date range", error=True)
                return

            # Create PDF report
            pdf = FPDF()
            pdf.add_page()
            
            # Add title and header
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Greenhouse Monitoring Report', ln=True, align='C')
            pdf.ln(10)
            
            # Add date range
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, f'Date Range: {start_date} to {end_date}', ln=True)
            pdf.ln(10)
            
            # Add summary statistics
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Summary Statistics', ln=True)
            pdf.ln(5)
            
            # Temperature statistics
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Temperature (°C):', ln=True)
            pdf.set_font('Arial', '', 10)
            temp_stats = filtered_df['temperature'].describe()
            for stat, value in temp_stats.items():
                pdf.cell(0, 10, f'{stat.capitalize()}: {value:.2f}°C', ln=True)
            pdf.ln(5)
            
            # Humidity statistics
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Humidity (%):', ln=True)
            pdf.set_font('Arial', '', 10)
            humid_stats = filtered_df['humidity'].describe()
            for stat, value in humid_stats.items():
                pdf.cell(0, 10, f'{stat.capitalize()}: {value:.2f}%', ln=True)
            pdf.ln(5)
            
            # NPK statistics if available
            if 'N' in filtered_df.columns and 'P' in filtered_df.columns and 'K' in filtered_df.columns:
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, 'NPK Levels:', ln=True)
                pdf.set_font('Arial', '', 10)
                
                for nutrient in ['N', 'P', 'K']:
                    nutrient_stats = filtered_df[nutrient].describe()
                    pdf.cell(0, 10, f'{nutrient} - Mean: {nutrient_stats["mean"]:.2f}, Std: {nutrient_stats["std"]:.2f}', ln=True)
                pdf.ln(5)
            
            # Add trends analysis
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Trends Analysis', ln=True)
            pdf.ln(5)
            
            # Calculate trends
            for col in ['temperature', 'humidity']:
                if col in filtered_df.columns:
                    x = np.arange(len(filtered_df))
                    y = filtered_df[col].values
                    slope, intercept = np.polyfit(x, y, 1)
                    trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                    
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, f'{col.capitalize()} Trend:', ln=True)
                    pdf.set_font('Arial', '', 10)
                    pdf.cell(0, 10, f'Trend: {trend} (slope: {slope:.4f})', ln=True)
                    pdf.ln(5)
            
            # Add recommendations
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Recommendations', ln=True)
            pdf.ln(5)
            
            # Temperature recommendations
            avg_temp = filtered_df['temperature'].mean()
            if avg_temp > 25:
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, 'Temperature is above optimal range. Consider increasing ventilation.', ln=True)
            elif avg_temp < 20:
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, 'Temperature is below optimal range. Consider heating or reducing ventilation.', ln=True)
            
            # Humidity recommendations
            avg_humid = filtered_df['humidity'].mean()
            if avg_humid > 80:
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, 'Humidity is above optimal range. Consider increasing ventilation.', ln=True)
            elif avg_humid < 60:
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, 'Humidity is below optimal range. Consider using humidifier.', ln=True)
            
            # Save the PDF
            report_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Save Report As"
            )
            
            if report_path:
                pdf.output(report_path)
                self.update_status("Report generated successfully")
                
        except Exception as e:
            self.update_status(f"Error generating report: {str(e)}", error=True)

    def plot_data(self):
        if not hasattr(self, 'df') or self.df is None:
            self.update_status("No data loaded", error=True)
            return
            
        try:
            # Get selected date range
            start_date = f"{self.start_year.get()}-{self.start_month.get()}-{self.start_day.get()}"
            end_date = f"{self.end_year.get()}-{self.end_month.get()}-{self.end_day.get()}"
            
            # Check if timestamp column exists
            if 'timestamp' not in self.df.columns:
                self.update_status("Timestamp column not found in data. Please load a valid CSV file.", error=True)
                return
                
            # Try to convert timestamp to datetime
            try:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
                # Remove rows where timestamp conversion failed
                self.df = self.df.dropna(subset=['timestamp'])
            except Exception as e:
                self.update_status(f"Error processing timestamp: {str(e)}", error=True)
                return
            
            # Filter data by date range
            mask = (self.df['timestamp'] >= start_date) & (self.df['timestamp'] <= end_date)
            filtered_df = self.df.loc[mask]
            
            if len(filtered_df) == 0:
                self.update_status("No data found for selected date range", error=True)
                return

            # Create a new window for the plots with modern design
            plot_window = ctk.CTkToplevel(self)
            plot_window.title("Greenhouse Analytics Dashboard")
            plot_window.geometry("1200x800")
            
            # Create main container with modern styling
            main_container = ModernFrame(plot_window, fg_color=COLORS['background'])
            main_container.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Create header with title and date range
            header_frame = ModernFrame(main_container, fg_color=COLORS['header_bg'])
            header_frame.pack(fill='x', padx=0, pady=0)
            
            title_label = HeaderLabel(
                header_frame,
                text="📊 Greenhouse Analytics",
                font=ctk.CTkFont(family='Segoe UI', size=16, weight='bold')
            )
            title_label.pack(side='left', padx=20, pady=10)
            
            date_range_label = HeaderLabel(
                header_frame,
                text=f"Date Range: {start_date} to {end_date}",
                font=ctk.CTkFont(family='Segoe UI', size=12)
            )
            date_range_label.pack(side='right', padx=20, pady=10)
            
            # Create notebook for tabs with modern styling
            notebook = ttk.Notebook(main_container)
            notebook.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Environment Tab
            env_tab = ModernFrame(notebook, fg_color=COLORS['background'])
            notebook.add(env_tab, text="🌡️ Environment")
            
            # Create figure for environment plots
            fig_env, axes_env = plt.subplots(2, 2, figsize=(10, 6))
            fig_env.patch.set_facecolor(COLORS['background'])
            
            # Temperature over time
            axes_env[0, 0].plot(filtered_df['timestamp'], filtered_df['temperature'], 
                              color='#FF6B6B', linewidth=1.5)
            axes_env[0, 0].set_title('Temperature Trend', color='white', pad=10, fontsize=10)
            axes_env[0, 0].set_xlabel('Time', color='white', fontsize=8)
            axes_env[0, 0].set_ylabel('Temperature (°C)', color='white', fontsize=8)
            axes_env[0, 0].grid(True, alpha=0.2, linestyle='--')
            axes_env[0, 0].set_facecolor(COLORS['card_bg'])
            axes_env[0, 0].tick_params(colors='white', labelsize=8)
            
            # Humidity over time
            axes_env[0, 1].plot(filtered_df['timestamp'], filtered_df['humidity'], 
                              color='#4ECDC4', linewidth=1.5)
            axes_env[0, 1].set_title('Humidity Trend', color='white', pad=10, fontsize=10)
            axes_env[0, 1].set_xlabel('Time', color='white', fontsize=8)
            axes_env[0, 1].set_ylabel('Humidity (%)', color='white', fontsize=8)
            axes_env[0, 1].grid(True, alpha=0.2, linestyle='--')
            axes_env[0, 1].set_facecolor(COLORS['card_bg'])
            axes_env[0, 1].tick_params(colors='white', labelsize=8)
            
            # Temperature distribution
            sns.histplot(data=filtered_df, x='temperature', kde=True, 
                        color='#FF6B6B', ax=axes_env[1, 0], bins=20)
            axes_env[1, 0].set_title('Temperature Distribution', color='white', pad=10, fontsize=10)
            axes_env[1, 0].set_xlabel('Temperature (°C)', color='white', fontsize=8)
            axes_env[1, 0].set_ylabel('Frequency', color='white', fontsize=8)
            axes_env[1, 0].grid(True, alpha=0.2, linestyle='--')
            axes_env[1, 0].set_facecolor(COLORS['card_bg'])
            axes_env[1, 0].tick_params(colors='white', labelsize=8)
            
            # Humidity distribution
            sns.histplot(data=filtered_df, x='humidity', kde=True, 
                        color='#4ECDC4', ax=axes_env[1, 1], bins=20)
            axes_env[1, 1].set_title('Humidity Distribution', color='white', pad=10, fontsize=10)
            axes_env[1, 1].set_xlabel('Humidity (%)', color='white', fontsize=8)
            axes_env[1, 1].set_ylabel('Frequency', color='white', fontsize=8)
            axes_env[1, 1].grid(True, alpha=0.2, linestyle='--')
            axes_env[1, 1].set_facecolor(COLORS['card_bg'])
            axes_env[1, 1].tick_params(colors='white', labelsize=8)
            
            # Format x-axis for time series plots
            for ax in [axes_env[0, 0], axes_env[0, 1]]:
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
            
            plt.tight_layout(pad=2.0)
            
            # Create canvas for environment plots
            canvas_env = FigureCanvasTkAgg(fig_env, master=env_tab)
            canvas_env.draw()
            
            # Add navigation toolbar
            toolbar_env = NavigationToolbar2Tk(canvas_env, env_tab)
            toolbar_env.update()
            
            # Pack the canvas
            canvas_env.get_tk_widget().pack(fill='both', expand=True)
            
            # NPK Levels Tab
            npk_tab = ModernFrame(notebook, fg_color=COLORS['background'])
            notebook.add(npk_tab, text="🌱 NPK Levels")
            
            # Create figure for NPK plots
            fig_npk, axes_npk = plt.subplots(2, 2, figsize=(10, 6))
            fig_npk.patch.set_facecolor(COLORS['background'])
            
            # NPK trends
            if 'N' in filtered_df.columns:
                axes_npk[0, 0].plot(filtered_df['timestamp'], filtered_df['N'], 
                                  color='#45B7D1', linewidth=1.5, label='N')
            if 'P' in filtered_df.columns:
                axes_npk[0, 0].plot(filtered_df['timestamp'], filtered_df['P'], 
                                  color='#96CEB4', linewidth=1.5, label='P')
            if 'K' in filtered_df.columns:
                axes_npk[0, 0].plot(filtered_df['timestamp'], filtered_df['K'], 
                                  color='#FFEEAD', linewidth=1.5, label='K')
            
            axes_npk[0, 0].set_title('NPK Levels Trend', color='white', pad=10, fontsize=10)
            axes_npk[0, 0].set_xlabel('Time', color='white', fontsize=8)
            axes_npk[0, 0].set_ylabel('Level', color='white', fontsize=8)
            axes_npk[0, 0].grid(True, alpha=0.2, linestyle='--')
            axes_npk[0, 0].set_facecolor(COLORS['card_bg'])
            axes_npk[0, 0].tick_params(colors='white', labelsize=8)
            axes_npk[0, 0].legend(facecolor=COLORS['card_bg'], edgecolor='none', 
                                labelcolor='white', fontsize=8)
            
            # NPK correlation
            if 'N' in filtered_df.columns and 'P' in filtered_df.columns:
                sns.scatterplot(data=filtered_df, x='N', y='P', 
                              color='#45B7D1', ax=axes_npk[0, 1], s=20)
                axes_npk[0, 1].set_title('N vs P Correlation', color='white', pad=10, fontsize=10)
                axes_npk[0, 1].set_xlabel('Nitrogen (N)', color='white', fontsize=8)
                axes_npk[0, 1].set_ylabel('Phosphorus (P)', color='white', fontsize=8)
                axes_npk[0, 1].grid(True, alpha=0.2, linestyle='--')
                axes_npk[0, 1].set_facecolor(COLORS['card_bg'])
                axes_npk[0, 1].tick_params(colors='white', labelsize=8)
            
            # NPK distributions
            if 'N' in filtered_df.columns:
                sns.histplot(data=filtered_df, x='N', kde=True, 
                            color='#45B7D1', ax=axes_npk[1, 0], bins=20)
                axes_npk[1, 0].set_title('Nitrogen Distribution', color='white', pad=10, fontsize=10)
                axes_npk[1, 0].set_xlabel('Nitrogen (N)', color='white', fontsize=8)
                axes_npk[1, 0].set_ylabel('Frequency', color='white', fontsize=8)
                axes_npk[1, 0].grid(True, alpha=0.2, linestyle='--')
                axes_npk[1, 0].set_facecolor(COLORS['card_bg'])
                axes_npk[1, 0].tick_params(colors='white', labelsize=8)
            
            if 'P' in filtered_df.columns:
                sns.histplot(data=filtered_df, x='P', kde=True, 
                            color='#96CEB4', ax=axes_npk[1, 1], bins=20)
                axes_npk[1, 1].set_title('Phosphorus Distribution', color='white', pad=10, fontsize=10)
                axes_npk[1, 1].set_xlabel('Phosphorus (P)', color='white', fontsize=8)
                axes_npk[1, 1].set_ylabel('Frequency', color='white', fontsize=8)
                axes_npk[1, 1].grid(True, alpha=0.2, linestyle='--')
                axes_npk[1, 1].set_facecolor(COLORS['card_bg'])
                axes_npk[1, 1].tick_params(colors='white', labelsize=8)
            
            # Format x-axis for time series plots
            for ax in [axes_npk[0, 0]]:
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
            
            plt.tight_layout(pad=2.0)
            
            # Create canvas for NPK plots
            canvas_npk = FigureCanvasTkAgg(fig_npk, master=npk_tab)
            canvas_npk.draw()
            
            # Add navigation toolbar
            toolbar_npk = NavigationToolbar2Tk(canvas_npk, npk_tab)
            toolbar_npk.update()
            
            # Pack the canvas
            canvas_npk.get_tk_widget().pack(fill='both', expand=True)
            
            # Statistics Tab
            stats_tab = ModernFrame(notebook, fg_color=COLORS['background'])
            notebook.add(stats_tab, text="📊 Statistics")
            
            # Create statistics cards
            stats_frame = ModernFrame(stats_tab, fg_color=COLORS['background'])
            stats_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Temperature statistics
            temp_stats = ModernFrame(stats_frame, fg_color=COLORS['card_bg'])
            temp_stats.pack(fill='x', padx=5, pady=5)
            
            ModernLabel(
                temp_stats,
                text="🌡️ Temperature Statistics",
                font=ctk.CTkFont(family='Segoe UI', size=12, weight='bold')
            ).pack(padx=10, pady=5)
            
            temp_data = filtered_df['temperature'].describe()
            for stat, value in temp_data.items():
                ModernLabel(
                    temp_stats,
                    text=f"{stat.capitalize()}: {value:.2f}°C",
                    font=ctk.CTkFont(family='Segoe UI', size=10)
                ).pack(padx=10, pady=2)
            
            # Humidity statistics
            humid_stats = ModernFrame(stats_frame, fg_color=COLORS['card_bg'])
            humid_stats.pack(fill='x', padx=5, pady=5)
            
            ModernLabel(
                humid_stats,
                text="💧 Humidity Statistics",
                font=ctk.CTkFont(family='Segoe UI', size=12, weight='bold')
            ).pack(padx=10, pady=5)
            
            humid_data = filtered_df['humidity'].describe()
            for stat, value in humid_data.items():
                ModernLabel(
                    humid_stats,
                    text=f"{stat.capitalize()}: {value:.2f}%",
                    font=ctk.CTkFont(family='Segoe UI', size=10)
                ).pack(padx=10, pady=2)
            
            # NPK statistics if available
            if 'N' in filtered_df.columns and 'P' in filtered_df.columns and 'K' in filtered_df.columns:
                npk_stats = ModernFrame(stats_frame, fg_color=COLORS['card_bg'])
                npk_stats.pack(fill='x', padx=5, pady=5)
                
                ModernLabel(
                    npk_stats,
                    text="🌱 NPK Statistics",
                    font=ctk.CTkFont(family='Segoe UI', size=12, weight='bold')
                ).pack(padx=10, pady=5)
                
                for nutrient in ['N', 'P', 'K']:
                    nutrient_data = filtered_df[nutrient].describe()
                    ModernLabel(
                        npk_stats,
                        text=f"{nutrient} - Mean: {nutrient_data['mean']:.2f}, Std: {nutrient_data['std']:.2f}",
                        font=ctk.CTkFont(family='Segoe UI', size=10)
                    ).pack(padx=10, pady=2)
            
            self.update_status("Graphs generated successfully")
            
        except Exception as e:
            self.update_status(f"Error generating graphs: {str(e)}", error=True)

    def save_file(self):
        if not hasattr(self, 'df') or self.df is None:
            self.update_status("No data loaded", error=True)
            return
            
        try:
            # Get selected date range
            start_date = f"{self.start_year.get()}-{self.start_month.get()}-{self.start_day.get()}"
            end_date = f"{self.end_year.get()}-{self.end_month.get()}-{self.end_day.get()}"
            
            # Filter data by date range
            mask = (self.df['timestamp'] >= start_date) & (self.df['timestamp'] <= end_date)
            filtered_df = self.df.loc[mask]
            
            if len(filtered_df) == 0:
                self.update_status("No data found for selected date range", error=True)
                return
                
            # Ask for file type and location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Excel files", "*.xlsx"),
                    ("All files", "*.*")
                ],
                title="Save Data As"
            )
            
            if file_path:
                if file_path.endswith('.csv'):
                    filtered_df.to_csv(file_path, index=False)
                elif file_path.endswith('.xlsx'):
                    filtered_df.to_excel(file_path, index=False)
                    
                self.update_status("File saved successfully")
                
        except Exception as e:
            self.update_status(f"Error saving file: {str(e)}", error=True)

    def actualizeaza_control(self):
        if not hasattr(self, 'df') or self.df is None:
            self.update_status("No data loaded", error=True)
            return

        try:
            # Get latest measurements
            latest = self.df.iloc[-1]
            
            # Validate measurements
            if not all(key in latest for key in ['temperature', 'humidity']):
                self.update_status("Missing required measurements", error=True)
                return
                
            temperature = float(latest['temperature'])
            humidity = float(latest['humidity'])
            
            # Safety checks
            if temperature > 50 or temperature < -10:
                self.update_status("Temperature out of safe range", error=True)
                return
                
            if humidity > 100 or humidity < 0:
                self.update_status("Humidity out of safe range", error=True)
                return
            
            # Update actuator status based on measurements with hysteresis
            temp_threshold = 25
            temp_hysteresis = 2
            humid_threshold = 60
            humid_hysteresis = 5
            
            # Get current states
            current_vent = "On" in self.ventilator_status.cget("text")
            current_humid = "On" in self.humidifier_status.cget("text")
            current_curtain = "Open" in self.curtain_status.cget("text")
            
            # Update ventilator with hysteresis
            if temperature > temp_threshold + temp_hysteresis:
                new_vent_state = "On"
            elif temperature < temp_threshold - temp_hysteresis:
                new_vent_state = "Off"
            else:
                new_vent_state = "On" if current_vent else "Off"
            
            # Update humidifier with hysteresis
            if humidity < humid_threshold - humid_hysteresis:
                new_humid_state = "On"
            elif humidity > humid_threshold + humid_hysteresis:
                new_humid_state = "Off"
            else:
                new_humid_state = "On" if current_humid else "Off"
            
            # Update curtain based on temperature
            if temperature > 30:
                new_curtain_state = "Open"
            else:
                new_curtain_state = "Closed"
            
            # Update UI
            self.ventilator_status.configure(text=f"Vent: {new_vent_state}")
            self.humidifier_status.configure(text=f"Humid: {new_humid_state}")
            self.curtain_status.configure(text=f"Curtain: {new_curtain_state}")
            
            # Log the control action
            self.log_control_action({
                'temperature': temperature,
                'humidity': humidity,
                'ventilator': new_vent_state,
                'humidifier': new_humid_state,
                'curtain': new_curtain_state
            })
            
            self.update_status("Control system updated successfully")
            
        except Exception as e:
            self.update_status(f"Error updating control: {str(e)}", error=True)
            
    def log_control_action(self, action_data):
        """Log control actions to a file"""
        try:
            log_file = "control_log.csv"
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                **action_data
            }
            
            # Create log file if it doesn't exist
            if not os.path.exists(log_file):
                pd.DataFrame([log_entry]).to_csv(log_file, index=False)
            else:
                # Append to existing log
                pd.DataFrame([log_entry]).to_csv(log_file, mode='a', header=False, index=False)
                
        except Exception as e:
            self.update_status(f"Error logging control action: {str(e)}", error=True)

    def show_help(self):
        """Show help dialog"""
        help_text = """
        Greenhouse Control System Help
        
        1. Data Operations:
           - Load CSV: Import sensor data from a CSV file
           - View Data: Preview the loaded data
           - Export: Save filtered data to CSV or Excel
        
        2. Date Range:
           - Select start and end dates for data filtering
           - Use the dropdown menus to choose day, month, and year
        
        3. Actions:
           - Generate Report: Create a PDF report with statistics
           - Graphs: View temperature and humidity plots
           - Export: Save filtered data
        
        4. Automatic Control:
           - System automatically controls actuators based on sensor readings
           - Manual update available via the Update Control button
        
        For more information, please contact the system administrator.
        """
        
        help_window = ctk.CTkToplevel(self)
        help_window.title("Help")
        help_window.geometry("600x400")
        
        help_frame = ModernFrame(help_window)
        help_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        help_textbox = ctk.CTkTextbox(help_frame, wrap='word')
        help_textbox.pack(fill='both', expand=True, padx=5, pady=5)
        help_textbox.insert('1.0', help_text)
        help_textbox.configure(state='disabled')

    def update_quick_stats(self):
        if self.df is None or self.df.empty:
            return
            
        try:
            # Get selected date range and convert to datetime
            start_date = datetime(
                int(self.start_year.get()),
                int(self.start_month.get()),
                int(self.start_day.get())
            )
            end_date = datetime(
                int(self.end_year.get()),
                int(self.end_month.get()),
                int(self.end_day.get())
            )
            
            # Validate date range
            if start_date > end_date:
                self.update_status("Invalid date range: Start date is after end date", error=True)
                return
                
            if start_date > self.df['timestamp'].max() or end_date < self.df['timestamp'].min():
                self.update_status("Selected date range is outside available data range", error=True)
                return
            
            # Filter data by date range
            mask = (self.df['timestamp'] >= start_date) & (self.df['timestamp'] <= end_date)
            filtered_df = self.df.loc[mask]
            
            if not filtered_df.empty:
                # Temperature statistics
                temp_stats = filtered_df['temperature'].agg(['mean', 'min', 'max', 'std'])
                self.temp_label.config(text=f"🌡️ {temp_stats['mean']:.1f}°C\n"
                                          f"Min: {temp_stats['min']:.1f}°C\n"
                                          f"Max: {temp_stats['max']:.1f}°C\n"
                                          f"Std: {temp_stats['std']:.1f}°C")
                
                # Humidity statistics
                hum_stats = filtered_df['humidity'].agg(['mean', 'min', 'max', 'std'])
                self.hum_label.config(text=f"💧 {hum_stats['mean']:.1f}%\n"
                                         f"Min: {hum_stats['min']:.1f}%\n"
                                         f"Max: {hum_stats['max']:.1f}%\n"
                                         f"Std: {hum_stats['std']:.1f}%")
                
                # NPK statistics if available
                if all(col in self.df.columns for col in ['N', 'P', 'K']):
                    npk_stats = filtered_df[['N', 'P', 'K']].agg(['mean', 'min', 'max'])
                    self.npk_label.config(text=f"N: {npk_stats['N']['mean']:.1f}\n"
                                             f"P: {npk_stats['P']['mean']:.1f}\n"
                                             f"K: {npk_stats['K']['mean']:.1f}")
                else:
                    self.npk_label.config(text="NPK data not available")
                
                # Update status based on current values
                current_temp = filtered_df['temperature'].iloc[-1]
                current_hum = filtered_df['humidity'].iloc[-1]
                
                # Temperature status
                if current_temp < 15:
                    temp_status = "❄️ Too Cold"
                elif current_temp > 30:
                    temp_status = "🔥 Too Hot"
                else:
                    temp_status = "✅ Optimal"
                
                # Humidity status
                if current_hum < 40:
                    hum_status = "🌵 Too Dry"
                elif current_hum > 80:
                    hum_status = "💦 Too Humid"
                else:
                    hum_status = "✅ Optimal"
                
                self.status_label.config(text=f"Status:\n"
                                            f"Temperature: {temp_status}\n"
                                            f"Humidity: {hum_status}")
                
                # Update last update timestamp
                self.last_update.configure(
                    text=f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
                )
                
            else:
                self.update_status("No data available for selected date range", error=True)
                
        except Exception as e:
            self.update_status(f"Error updating quick stats: {str(e)}", error=True)

if __name__ == "__main__":
    # Create and run the application
    app = GreenhouseApp()
    app.mainloop()