
🌿 Greenhouse Control System
-

- 📝 Descriere Proiect

Această aplicație GUI (interfață grafică) gestionează și monitorizează parametrii unei sere inteligente, analizând date de senzori (temperatură, umiditate, NPK) și oferind instrumente pentru:

 - Încărcarea și preprocesarea datelor din fișiere CSV. 
 - Vizualizarea grafică a seriilor temporale și distribuțiilor parametrilor.
 - Generarea de rapoarte PDF cu statistici, probleme identificate și grafice.
 - Controlul automat al actuatoarelor (ventilatoare, pompe) pe baza condițiilor optimale.

 Funcționalități
-
1. Încărcare de Date:

- Validare și curățare automată a datelor (eliminare outliers, valori invalide).

- Detectare condiții optime (20-25°C, 60-80% umiditate, intervale NPK specificate).

2. Vizualizare Interactivă:

- Grafice temporale și histograme pentru fiecare parametru.

- Filtrare date pe interval specificat de utilizator.

3. Rapoarte Personalizate:

- Export PDF cu statistici descriptive, procentaj timp optim, probleme identificate.

- Grafice integrate în raport, cu suport pentru diacritice românești.

4. Controlul Actuatoarelor:

- Decizii automate pentru ventilatoare (temperatură), pompe (umiditate, NPK).

Tehnologii Utilizate
-

- Python + Tkinter (interfață grafică)

- Pandas (preprocesare date)

- Matplotlib/Seaborn (vizualizări)

- FPDF (generare rapoarte PDF cu fonturi Unicode)

- Scikit-learn (analiză statistică)