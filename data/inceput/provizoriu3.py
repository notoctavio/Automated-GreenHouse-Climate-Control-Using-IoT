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
from fpdf import FPDF  # Pentru generarea raportelor PDF
import tempfile

# Setări vizuale globale
plt.style.use('dark_background')
sns.set_theme(style="whitegrid", palette="muted")


class GreenhouseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 Smart Greenhouse Controller")
        self.root.geometry("1000x800")
        self.style = ttk.Style()

        # Configurare temă personalizată
        self.root.configure(bg='#2E3440')
        self.style.theme_use('clam')

        # Definire stiluri
        self.style.configure('TButton',
                             font=('Segoe UI', 10, 'bold'),
                             borderwidth=2,
                             relief='raised',
                             foreground='#ECEFF4',
                             background='#5E81AC',
                             padding=10)

        self.style.map('TButton',
                       foreground=[('active', '#ECEFF4'), ('disabled', '#D8DEE9')],
                       background=[('active', '#81A1C1'), ('disabled', '#4C566A')])

        self.style.configure('TLabel',
                             background='#2E3440',
                             foreground='#ECEFF4',
                             font=('Segoe UI', 9))

        self.style.configure('Header.TLabel',
                             font=('Segoe UI', 14, 'bold'),
                             foreground='#88C0D0')

        self.style.configure('TEntry',
                             fieldbackground='#4C566A',
                             foreground='#ECEFF4')

        # Inițializare date
        self.df = None
        self.df_week = None
        self.current_figure = None

        self.create_widgets()
        self.create_status_bar()
        self.create_side_panel()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(pady=20, padx=20, fill='both', expand=True)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill='x')

        ttk.Label(header_frame,
                  text="🌡️ Greenhouse Monitoring System",
                  style='Header.TLabel').pack(side='left')

        # Buton rapid pentru help
        ttk.Button(header_frame,
                   text="❔ Help",
                   command=self.show_help,
                   style='TButton').pack(side='right')

        # Panou principal
        self.main_notebook = ttk.Notebook(main_frame)
        self.main_notebook.pack(fill='both', expand=True, pady=10)

        # Tab Date
        data_tab = ttk.Frame(self.main_notebook)
        self.main_notebook.add(data_tab, text='📁 Date Senzori')

        # Sectiune încărcare date
        load_frame = ttk.LabelFrame(data_tab, text=' Operațiuni Date ', padding=15)
        load_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(load_frame,
                   text="📤 Încarcă Date CSV",
                   command=self.load_csv,
                   style='TButton').pack(side='left', padx=5)

        ttk.Button(load_frame,
                   text="🔍 Vezi Date",
                   command=self.preview_data,
                   style='TButton').pack(side='left', padx=5)

        # Sectiune selectare interval
        date_frame = ttk.LabelFrame(data_tab, text=' Selectare Interval ', padding=15)
        date_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(date_frame, text="Data început:").grid(row=0, column=0, padx=5, pady=5)
        self.start_date_entry = ttk.Entry(date_frame, width=20)
        self.start_date_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(date_frame, text="Data sfârșit:").grid(row=1, column=0, padx=5, pady=5)
        self.end_date_entry = ttk.Entry(date_frame, width=20)
        self.end_date_entry.grid(row=1, column=1, padx=5, pady=5)

        # Butoane acțiune
        action_frame = ttk.Frame(data_tab)
        action_frame.pack(pady=15)

        ttk.Button(action_frame,
                   text="📈 Generează Raport",
                   command=self.generate_report,
                   style='TButton').grid(row=0, column=0, padx=10)

        ttk.Button(action_frame,
                   text="🖼️ Vizualizare Grafice",
                   command=self.plot_data,
                   style='TButton').grid(row=0, column=1, padx=10)

        ttk.Button(action_frame,
                   text="💾 Export Date",
                   command=self.save_file,
                   style='TButton').grid(row=0, column=2, padx=10)

    def create_side_panel(self):
        # Panou lateral cu statistici
        self.side_panel = ttk.Frame(self.root, width=250)
        self.side_panel.pack(side='right', fill='y', padx=10, pady=20)

        stats_frame = ttk.LabelFrame(self.side_panel, text=' 📊 Statistici Rapide ', padding=15)
        stats_frame.pack(fill='x')

        self.temp_label = ttk.Label(stats_frame, text="🌡️ Temp: N/A")
        self.temp_label.pack(anchor='w')

        self.humidity_label = ttk.Label(stats_frame, text="💧 Umiditate: N/A")
        self.humidity_label.pack(anchor='w')

        self.npk_label = ttk.Label(stats_frame, text="🌱 NPK: N/A")
        self.npk_label.pack(anchor='w')

        ttk.Separator(stats_frame).pack(fill='x', pady=10)

        self.status_indicator = ttk.Label(stats_frame, text="🔴 Sistem neinițializat")
        self.status_indicator.pack(anchor='w')

    def create_status_bar(self):
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root,
                               textvariable=self.status_var,
                               relief='sunken',
                               anchor='center',
                               style='TLabel')
        status_bar.pack(side='bottom', fill='x')
        self.status_var.set("Sistem gata pentru operare")

    def update_status(self, message, error=False):
        status_color = "#BF616A" if error else "#A3BE8C"
        self.status_var.set(message)
        self.status_indicator.config(foreground=status_color)
        self.root.update_idletasks()

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
                self.df = self.df.dropna().drop_duplicates().reset_index(drop=True)
                self.df = self.df.sort_values('date').reset_index(drop=True)

                # Validare interval senzori
                valid_range = (0, 255)
                for col in ['N', 'P', 'K']:
                    if self.df[col].min() < valid_range[0] or self.df[col].max() > valid_range[1]:
                        self.df = self.df[(self.df[col] >= valid_range[0]) & (self.df[col] <= valid_range[1])]

                # Eliminare outliers
                for col in ['temperature', 'humidity', 'N', 'P', 'K']:
                    self.remove_outliers(col)

                # Calcul condiții optime
                self.df['optimal_condition'] = self.df.apply(self.is_optimal, axis=1)

                self.update_status("✅ Date încărcate cu succes")
                self.update_side_panel()

            except Exception as e:
                self.update_status(f"❌ Eroare la încărcare: {str(e)}", error=True)
                messagebox.showerror("Eroare", f"Eroare la procesarea fișierului:\n{str(e)}")
        else:
            self.update_status("⏹️ Încărcare anulată", error=True)

    def update_side_panel(self):
        if self.df is not None:
            self.temp_label.config(text=f"🌡️ Temp: {self.df['temperature'].mean():.1f}°C")
            self.humidity_label.config(text=f"💧 Umiditate: {self.df['humidity'].mean():.1f}%")
            npk_text = f"N: {self.df['N'].mean():.0f} | P: {self.df['P'].mean():.0f} | K: {self.df['K'].mean():.0f}"
            self.npk_label.config(text=f"🌱 NPK: {npk_text}")
            self.status_indicator.config(text="🟢 Sistem operațional")

    def remove_outliers(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]

    def is_optimal(self, row):
        conditions = [
            20 <= row['temperature'] <= 25,
            60 <= row['humidity'] <= 80,
            50 <= row['N'] <= 150,
            30 <= row['P'] <= 100,
            100 <= row['K'] <= 200
        ]
        return 1 if all(conditions) else 0

    def plot_data(self):
        if self.df is None:
            self.update_status("❌ Încărcați mai întâi datele", error=True)
            return

        try:
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()
            mask = (self.df['date'] >= start_date) & (self.df['date'] <= end_date)
            self.df_week = self.df.loc[mask].reset_index(drop=True)

            if self.df_week.empty:
                messagebox.showwarning("Avertisment", "Nu există date pentru intervalul selectat")
                return

            self.create_plots_window()

        except Exception as e:
            self.update_status(f"❌ Eroare la generarea graficelor: {str(e)}", error=True)

    def create_plots_window(self):
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Analiză Date")
        plot_window.geometry("1200x800")

        notebook = ttk.Notebook(plot_window)
        notebook.pack(fill='both', expand=True)

        # Tab Series temporale
        ts_frame = ttk.Frame(notebook)
        notebook.add(ts_frame, text='📈 Serii Temporale')
        self.create_time_series_plot(ts_frame)

        # Tab Distribuții
        dist_frame = ttk.Frame(notebook)
        notebook.add(dist_frame, text='📊 Distribuții')
        self.create_distribution_plots(dist_frame)

    def create_time_series_plot(self, parent):
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)

        variables = ['temperature', 'humidity', 'N', 'P', 'K']
        colors = ['#BF616A', '#5E81AC', '#A3BE8C', '#EBCB8B', '#B48EAD']

        for var, color in zip(variables, colors):
            ax.plot(self.df_week['date'], self.df_week[var],
                    marker='o', linestyle='-',
                    color=color, label=var.capitalize())

        ax.set_title("Serii Temporale Parametri", pad=20)
        ax.set_xlabel("Dată")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def create_distribution_plots(self, parent):
        fig = Figure(figsize=(10, 8), dpi=100)
        axes = fig.subplots(2, 3)
        axes = axes.flatten()

        variables = ['temperature', 'humidity', 'N', 'P', 'K']
        colors = ['#BF616A', '#5E81AC', '#A3BE8C', '#EBCB8B', '#B48EAD']

        for i, (var, color) in enumerate(zip(variables, colors)):
            axes[i].hist(self.df_week[var], bins=15, color=color, alpha=0.7)
            axes[i].set_title(f'Distribuție {var.capitalize()}')
            axes[i].grid(True, alpha=0.3)

        fig.delaxes(axes[5])
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def save_file(self):
        if self.df_week is not None:
            try:
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
                )
                if file_path:
                    self.df_week.to_csv(file_path, index=False)
                    self.update_status(f"✅ Date salvate în {os.path.basename(file_path)}")
            except Exception as e:
                self.update_status(f"❌ Eroare la salvare: {str(e)}", error=True)
        else:
            self.update_status("❌ Nu există date de salvat", error=True)

    def preview_data(self):
        if self.df is not None:
            preview_window = tk.Toplevel(self.root)
            preview_window.title("Previzualizare Date")

            text = tk.Text(preview_window, wrap='none')
            text.pack(fill='both', expand=True)

            text.insert('end', self.df.head(20).to_string(index=False))
            text.config(state='disabled')
        else:
            self.update_status("❌ Încărcați mai întâi datele", error=True)

    def generate_report(self):
        if self.df is None:
            self.update_status("❌ Încărcați mai întâi datele", error=True)
            return

        try:
            # Verifică dacă intervalul de date este introdus
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()

            if not start_date or not end_date:
                self.update_status("❌ Introduceți intervalul de date!", error=True)
                return

            # Filtrează datele pe intervalul selectat
            mask = (self.df['date'] >= start_date) & (self.df['date'] <= end_date)
            df_filtered = self.df.loc[mask]

            if df_filtered.empty:
                self.update_status("❌ Nu există date pentru intervalul selectat", error=True)
                return

            # Inițializează PDF cu suport pentru UTF-8
            pdf = FPDF()
            pdf.add_page()

            # Adaugă fonturile Noto Sans (salvează fișierele .ttf în același folder)
            pdf.add_font("NotoSans", "", "NotoSans-Regular.ttf", uni=True)
            pdf.add_font("NotoSans", "B", "NotoSans-Bold.ttf", uni=True)
            pdf.set_font("NotoSans", size=12)

            # Titlu raport
            pdf.set_font("NotoSans", "B", 14)
            pdf.cell(200, 10, txt="Raport Analiză Date Sere", ln=True, align='C')
            pdf.ln(10)

            # Sectiune 1: Statistici generale
            pdf.set_font("NotoSans", "B", 12)
            pdf.cell(200, 10, txt="1. Statistici Generale", ln=True)
            pdf.set_font("NotoSans", "", 10)

            stats = df_filtered[['temperature', 'humidity', 'N', 'P', 'K']].describe().to_string()
            pdf.multi_cell(0, 10, txt=stats)
            pdf.ln(10)

            # Sectiune 2: Condiții optime
            pdf.set_font("NotoSans", "B", 12)
            pdf.cell(200, 10, txt="2. Evaluare Condiții Optime", ln=True)
            pdf.set_font("NotoSans", "", 10)

            optimal_percentage = df_filtered['optimal_condition'].mean() * 100
            pdf.multi_cell(0, 10, txt=f"Procentaj timp condiții optime: {optimal_percentage:.2f}%")
            pdf.ln(10)

            # Sectiune 3: Probleme identificate
            pdf.set_font("NotoSans", "B", 12)
            pdf.cell(200, 10, txt="3. Probleme Identificate", ln=True)
            pdf.set_font("NotoSans", "", 10)

            issues = []
            if (df_filtered['temperature'] < 20).any():
                issues.append("Temperatură prea scăzută")
            if (df_filtered['temperature'] > 25).any():
                issues.append("Temperatură prea ridicată")
            if (df_filtered['humidity'] < 60).any():
                issues.append("Umiditate prea scăzută")
            if (df_filtered['humidity'] > 80).any():
                issues.append("Umiditate prea ridicată")

            if issues:
                issues_text = "Probleme identificate:\n- " + "\n- ".join(issues)
                pdf.multi_cell(0, 10, txt=issues_text)
            else:
                pdf.multi_cell(0, 10, txt="Nu s-au identificat probleme majore.")
            pdf.ln(10)

            # Sectiune 4: Grafice
            pdf.set_font("NotoSans", "B", 12)
            pdf.cell(200, 100, txt="4. Grafice", ln=True)

            # Configură matplotlib pentru diacritice
            plt.rcParams['font.family'] = 'Noto Sans'

            with tempfile.TemporaryDirectory() as temp_dir:
                # Grafic 1: Serii temporale
                plt.figure(figsize=(10, 10))
                for col in ['temperature', 'humidity']:
                    plt.plot(df_filtered['date'], df_filtered[col], label=col.capitalize())
                plt.title(f"Serii Temporale Temperatură și Umiditate ({start_date} - {end_date})")
                plt.xlabel("Dată")
                plt.ylabel("Valoare")
                plt.legend()
                plt.tight_layout()
                temp_plot_path = os.path.join(temp_dir, "plot1.png")
                plt.savefig(temp_plot_path, bbox_inches='tight')
                plt.close()

                # Adăugare grafic în PDF
                pdf.image(temp_plot_path, x=20, y=pdf.get_y(), w=150)
                pdf.ln(90)

            # Salvăm raportul
            report_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
            )
            if report_path:
                pdf.output(report_path)
                self.update_status(f"✅ Raport generat și salvat în {report_path}")

        except Exception as e:
            self.update_status(f"❌ Eroare la generarea raportului: {str(e)}", error=True)
        except Exception as e:
            self.update_status(f"❌ Eroare la generarea raportului: {str(e)}", error=True)

    def show_help(self):
        help_text = """🌿 Greenhouse Control System - Ajutor
1. Încărcați un fișier CSV cu date senzori
2. Selectați intervalul de date dorit
3. Generați grafice și rapoarte
4. Exportați date procesate"""
        messagebox.showinfo("Ajutor", help_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = GreenhouseApp(root)
    root.mainloop()