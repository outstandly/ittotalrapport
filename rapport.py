import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import timedelta

# Sidkonfiguration
st.set_page_config(
    page_title="IT-Total Marknadsföringsrapport",
    page_icon="📊",
    layout="wide"
)

# Huvudrubrik
st.title("IT-Total Digital Marknadsföringsrapport")

# Sidofält för navigation
menu = st.sidebar.selectbox(
    "Välj rapport",
    ["Digital Marknadsföring", "SEO Analys", "Konkurrensanalys", "Brand Awareness"]
)

# Digital Marknadsföring sidan
if menu == "Digital Marknadsföring":
    st.header("Digital Marknadsföring - Google Ads")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Klick", "868", "+24.0%")
        st.metric("CTR", "13.5%", "+20.7%")
        st.metric("Visningar", "6.5K", "+2.7%")
    
    with col2:
        st.metric("Konverteringar", "21", "-24.1%")
        st.metric("Konverteringsgrad", "2.4%", "-39.0%")
        st.metric("Kostnad/konvertering", "242.58 kr", "+38.9%")
    
    with col3:
        st.metric("Total kostnad", "5,090 kr", "+5.4%")
        st.metric("CPC (Kostnad per klick)", "5.87 kr", "-15.0%")
        st.metric("CPM", "789.56 kr", "+2.6%")

    # Lägg till grafer för trendanalys
    st.subheader("Trender över tid")
    
    # Skapa datumperiod
    dates = pd.date_range(start='2023-08-01', end='2023-11-29')
    n_days = len(dates)  # Antal dagar i perioden

    # Generera data med korrekt längd
    data = {
        'Datum': dates,
        'Klick': np.random.randint(10, 60, n_days),  # Slumpmässiga värden mellan 10-60
        'CTR': np.random.uniform(0.05, 0.35, n_days),  # Slumpmässiga värden mellan 5-35%
        'Konverteringar': np.random.randint(0, 8, n_days),  # Slumpmässiga värden mellan 0-8
        'Kostnad_per_konv': np.random.uniform(50, 300, n_days),  # Slumpmässiga värden mellan 50-300
        'Total_kostnad': np.linspace(0, 21000, n_days),  # Linjärt ökande från 0 till 21000
        'CPC': np.random.uniform(6, 12, n_days)  # Slumpmässiga värden mellan 6-12
    }
    
    # Skapa dataframe
    df = pd.DataFrame(data)

    # Lägg till smoothing för att efterlikna trenderna i bilderna
    df['Klick'] = df['Klick'].rolling(window=5, center=True).mean()
    df['CTR'] = df['CTR'].rolling(window=5, center=True).mean()
    df['Konverteringar'] = df['Konverteringar'].rolling(window=5, center=True).mean()
    df['CPC'] = df['CPC'].rolling(window=5, center=True).mean()

    # Fyll i NaN-värden med närmaste giltiga värde
    df = df.fillna(method='bfill').fillna(method='ffill')

    # Exempel på hur man kan visa trendgraferna
    tab1, tab2, tab3 = st.tabs(["Klick & CTR", "Konverteringar", "Kostnader"])
    
    with tab1:
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig1.add_trace(
            go.Scatter(
                x=df['Datum'],
                y=df['Klick'],
                name="Klick",
                line=dict(color="#34A853"),
                mode="lines+markers"
            ),
            secondary_y=False
        )
        
        fig1.add_trace(
            go.Scatter(
                x=df['Datum'],
                y=df['CTR'],
                name="CTR",
                line=dict(color="#4285F4"),
                mode="lines"
            ),
            secondary_y=True
        )
        
        fig1.update_layout(
            title="Klick och CTR utveckling",
            height=400,
            hovermode="x unified",
            yaxis=dict(title="Antal klick", range=[0, 60]),
            yaxis2=dict(title="CTR %", range=[0, 40], tickformat='.0%')
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig2.add_trace(
            go.Scatter(
                x=df['Datum'],
                y=df['Konverteringar'],
                name="Konverteringar",
                line=dict(color="#34A853"),
                mode="lines+markers"
            ),
            secondary_y=False
        )
        
        fig2.add_trace(
            go.Scatter(
                x=df['Datum'],
                y=df['Kostnad_per_konv'],
                name="Kostnad/konv.",
                line=dict(color="#4285F4"),
                mode="lines"
            ),
            secondary_y=True
        )
        
        fig2.update_layout(
            title="Konverteringar och kostnad per konvertering",
            height=400,
            hovermode="x unified",
            yaxis=dict(title="Antal konverteringar", range=[0, 8]),
            yaxis2=dict(title="Kostnad/konv. (kr)", range=[0, 400])
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig3.add_trace(
            go.Scatter(
                x=df['Datum'],
                y=df['Total_kostnad'],
                name="Total kostnad",
                line=dict(color="#34A853"),
                mode="lines"
            ),
            secondary_y=False
        )
        
        fig3.add_trace(
            go.Scatter(
                x=df['Datum'],
                y=df['CPC'],
                name="Genomsnittlig CPC",
                line=dict(color="#4285F4"),
                mode="lines"
            ),
            secondary_y=True
        )
        
        fig3.update_layout(
            title="Kostnadsutveckling och CPC",
            height=400,
            hovermode="x unified",
            yaxis=dict(title="Total kostnad (kr)", range=[0, 30000]),
            yaxis2=dict(title="Genomsnittlig CPC (kr)", range=[0, 40])
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Lägg till prediktiv analys
    st.subheader("Prediktiv Analys")
    
    # Förbereder data för prediktion
    X = np.array(range(len(df))).reshape(-1, 1)  # Dagar som feature
    future_days = 30  # Antal dagar att predicera framåt
    future_dates = pd.date_range(start=dates[-1], periods=future_days + 1)[1:]
    future_X = np.array(range(len(df), len(df) + future_days)).reshape(-1, 1)

    col1, col2 = st.columns(2)
    
    with col1:
        # Prediktion för konverteringar
        model_conv = LinearRegression()
        model_conv.fit(X, df['Konverteringar'])
        
        future_conv = model_conv.predict(future_X)
        
        fig_pred_conv = go.Figure()
        
        # Historisk data
        fig_pred_conv.add_trace(
            go.Scatter(
                x=dates,
                y=df['Konverteringar'],
                name="Historiska konverteringar",
                line=dict(color="#34A853")
            )
        )
        
        # Predikterad data
        fig_pred_conv.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_conv,
                name="Predikterade konverteringar",
                line=dict(color="#34A853", dash='dash')
            )
        )
        
        fig_pred_conv.update_layout(
            title="Konverteringsprognos (30 dagar)",
            height=400,
            hovermode="x unified",
            yaxis=dict(title="Antal konverteringar")
        )
        st.plotly_chart(fig_pred_conv, use_container_width=True)

    with col2:
        # Prediktion för kostnad
        model_cost = LinearRegression()
        model_cost.fit(X, df['Total_kostnad'])
        
        future_cost = model_cost.predict(future_X)
        
        fig_pred_cost = go.Figure()
        
        # Historisk data
        fig_pred_cost.add_trace(
            go.Scatter(
                x=dates,
                y=df['Total_kostnad'],
                name="Historisk kostnad",
                line=dict(color="#4285F4")
            )
        )
        
        # Predikterad data
        fig_pred_cost.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_cost,
                name="Predikterad kostnad",
                line=dict(color="#4285F4", dash='dash')
            )
        )
        
        fig_pred_cost.update_layout(
            title="Kostnadsprognos (30 dagar)",
            height=400,
            hovermode="x unified",
            yaxis=dict(title="Total kostnad (kr)")
        )
        st.plotly_chart(fig_pred_cost, use_container_width=True)

    # Lägg till prediktionsinsikter
    st.subheader("Prediktionsinsikter")
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        avg_current_conv = df['Konverteringar'].mean()
        avg_predicted_conv = future_conv.mean()
        conv_change = ((avg_predicted_conv - avg_current_conv) / avg_current_conv) * 100
        
        st.metric(
            "Förväntade konverteringar (medel/dag)", 
            f"{avg_predicted_conv:.1f}",
            f"{conv_change:+.1f}%"
        )
    
    with insight_col2:
        current_cost_per_conv = df['Kostnad_per_konv'].mean()
        predicted_cost_per_conv = future_cost.mean() / future_conv.mean()
        cost_change = ((predicted_cost_per_conv - current_cost_per_conv) / current_cost_per_conv) * 100
        
        st.metric(
            "Förväntad kostnad/konvertering", 
            f"{predicted_cost_per_conv:.2f} kr",
            f"{cost_change:+.1f}%"
        )
    
    with insight_col3:
        roi_current = (df['Konverteringar'].sum() * 1000 - df['Total_kostnad'].iloc[-1]) / df['Total_kostnad'].iloc[-1] * 100
        roi_predicted = (future_conv.sum() * 1000 - future_cost[-1]) / future_cost[-1] * 100
        roi_change = roi_predicted - roi_current
        
        st.metric(
            "Förväntad ROI", 
            f"{roi_predicted:.1f}%",
            f"{roi_change:+.1f}%"
        )

    # Konkurrentanalys inom Digital Marknadsföring
    st.subheader("Konkurrentanalys - Digital Marknadsföring")
    
    # Data för konkurrentanalys
    competitor_data = {
        'Företag': ['IT-Total', 'Iver', 'Advania', 'Dustin', 'Atea', 'Nordlo', 'Orange Cyberdefence', 'Proact', 'Softronic'],
        'Digital_Annonsering': [85, 92, 88, 95, 90, 87, 94, 89, 86],
        'Sökords_Position': [3.2, 2.8, 3.5, 2.5, 2.1, 2.9, 2.3, 3.1, 3.4],
        'Annonsinvestering': [50000, 75000, 65000, 85000, 80000, 70000, 82000, 68000, 55000],
        'Digital_Leads': [180, 250, 220, 280, 260, 230, 270, 210, 190]
    }
    comp_df = pd.DataFrame(competitor_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Digital marknadsföringsbudget
        fig_budget = px.bar(
            comp_df,
            x='Företag',
            y='Digital_Annonsering',
            title='Digital Marknadsföringsandel av Total Budget',
            labels={'Digital_Annonsering': 'Andel av Budget (%)'},
            color='Företag',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_budget.update_layout(height=400)
        st.plotly_chart(fig_budget, use_container_width=True)
        
        # Insikt om digital närvaro
        st.info("""
        **Digital Närvaro Insikt:**
        • Branschen investerar i genomsnitt 90% av marknadsföringsbudgeten i digitala kanaler
        • IT-Total har möjlighet att öka digital närvaro för att matcha konkurrenterna
        """)
    
    with col2:
        # Lead generation jämförelse
        fig_leads = px.scatter(
            comp_df,
            x='Annonsinvestering',
            y='Digital_Leads',
            size='Digital_Annonsering',
            color='Företag',
            title='Digital Leads vs Annonsinvestering',
            labels={
                'Annonsinvestering': 'Månatlig Annonsinvestering (SEK)',
                'Digital_Leads': 'Digitala Leads per Månad'
            }
        )
        fig_leads.update_layout(height=400)
        st.plotly_chart(fig_leads, use_container_width=True)
    
    # Sökordsstrategi tabell
    st.subheader("Konkurrenternas Sökordsstrategi")
    
    keyword_data = {
        'Sökord': ['IT konsult', 'Molntjänster', 'IT drift', 'Cybersäkerhet', 'IT support'],
        'IT-Total Position': [3, 5, 4, 6, 2],
        'Topp Konkurrent': ['Atea', 'Dustin', 'Iver', 'Advania', 'Atea'],
        'Konkurrent Position': [1, 2, 1, 2, 1],
        'Månadsvolym': [2200, 1800, 1500, 2500, 3000]
    }
    
    st.dataframe(
        pd.DataFrame(keyword_data)
        .style.highlight_min(subset=['IT-Total Position'], color='#34A853')
        .highlight_max(subset=['Månadsvolym'], color='#4285F4'),
        hide_index=True
    )
    
    # Strategiska insikter
    st.subheader("Strategiska Insikter")
    insikt_col1, insikt_col2 = st.columns(2)
    
    with insikt_col1:
        st.markdown("""
        **Möjligheter:**
        * Öka digital annonsering för att matcha branschgenomsnittet
        * Fokusera på högvolym-sökord inom cybersäkerhet
        * Optimera konverteringsgrad för att maximera ROI
        """)
    
    with insikt_col2:
        st.markdown("""
        **Konkurrenternas Framgångsfaktorer:**
        * Högre investering i sökordsannonsering
        * Fokus på branschspecifika långsvans-sökord
        * Stark närvaro i digitala B2B-kanaler
        """)

# SEO Analys
elif menu == "SEO Analys":
    st.header("SEO Prestanda")
    
    # Översiktsmått
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Domain Rating (DR)", "39/100", "+2")
        st.metric("URL Rating (UR)", "4.6", "-0.2")
    with col2:
        st.metric("Länkande domäner", "267", "+9")
        st.metric("Totala länkar", "14.9K", "-156")
    with col3:
        st.metric("Organiska sökord", "1.3K", "-7")
        st.metric("Organisk trafik", "511", "+1")

    # Geografisk fördelning
    st.subheader("Geografisk SEO-prestanda")
    
    geo_data = {
        'Land': ['Sweden', 'United States', 'United Kingdom', 'Russian Federation', 'Germany'],
        'Trafik': [507, 3, 0, 0, 0],
        'Andel': [99.4, 0.6, 0, 0, 0],
        'Sökord': [1200, 140, 11, 8, 8]
    }
    
    st.dataframe(
        pd.DataFrame(geo_data)
        .style.highlight_max(subset=['Trafik', 'Andel', 'Sökord'], color='#34A853'),
        hide_index=True
    )

    # SEO Aktivitetsanalys
    st.subheader("SEO Aktivitetsjämförelse med Konkurrenter")
    
    seo_activity_data = {
        'Företag': ['IT-Total', 'Iver', 'Advania', 'Dustin', 'Atea', 'Nordlo', 'Orange Cyberdefence', 'Proact', 'Softronic'],
        'Innehållsuppdateringar/mån': [4, 12, 8, 15, 10, 9, 14, 7, 6],
        'Tekniska optimeringar/kvartal': [2, 6, 5, 8, 7, 5, 7, 4, 3],
        'Nya sidor/mån': [3, 8, 6, 12, 9, 7, 10, 5, 4],
        'Backlink-tillväxt/mån': [9, 25, 18, 30, 22, 20, 28, 15, 12]
    }
    
    df_seo_activity = pd.DataFrame(seo_activity_data)
    
    # Visualisering av SEO-aktiviteter
    fig_seo_activity = px.bar(
        df_seo_activity,
        x='Företag',
        y=['Innehållsuppdateringar/mån', 'Tekniska optimeringar/kvartal', 
           'Nya sidor/mån', 'Backlink-tillväxt/mån'],
        title='SEO-aktiviteter: IT-Total vs Konkurrenter',
        barmode='group',
        labels={
            'value': 'Antal aktiviteter',
            'variable': 'Aktivitetstyp'
        }
    )
    fig_seo_activity.update_layout(height=400)
    st.plotly_chart(fig_seo_activity, use_container_width=True)
    
    # SEO Investeringsanalys
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **SEO Aktivitetsanalys:**
        • Konkurrenter publicerar i genomsnitt 3x mer innehåll
        • Tekniska optimeringar görs 2-4x oftare av konkurrenter
        • Backlink-tillväxten är 2.5x högre hos konkurrenterna
        • Dustin och Atea visar högst aktivitetsnivå
        """)
        
    with col2:
        st.warning("""
        **Rekommenderade Åtgärder:**
        • Öka frekvensen av innehållspublicering till minst 8-10 artiklar/mån
        • Implementera månatlig teknisk SEO-genomgång
        • Utveckla aktiv strategi för länkbyggande
        • Skapa dedikerat team för innehållsproduktion
        """)
    
    # SEO Mognadsgrad
    st.subheader("SEO Mognadsgrad")
    
    # Uppdatera SEO mognadsdata
    maturity_data = {
        'Företag': ['IT-Total', 'Iver', 'Advania', 'Dustin', 'Atea', 'Nordlo', 'Orange Cyberdefence', 'Proact', 'Softronic'],
        'Innehållsstrategi': [2, 4, 3, 5, 4, 3, 4, 3, 2],  # Skala 1-5
        'Teknisk SEO': [3, 4, 4, 5, 4, 3, 5, 4, 3],
        'Länkprofil': [2, 4, 3, 4, 5, 3, 4, 3, 2],
        'Innehållskvalitet': [4, 4, 4, 5, 4, 3, 5, 4, 3]
    }
    
    # Konvertera till DataFrame
    df_maturity = pd.DataFrame(maturity_data)
    
    # Radar chart för mognadsjämförelse
    fig_maturity = go.Figure()
    
    for company in df_maturity['Företag']:
        fig_maturity.add_trace(go.Scatterpolar(
            r=[df_maturity.loc[df_maturity['Företag'] == company, col].iloc[0] 
               for col in ['Innehållsstrategi', 'Teknisk SEO', 'Länkprofil', 'Innehållskvalitet']],
            theta=['Innehållsstrategi', 'Teknisk SEO', 'Länkprofil', 'Innehållskvalitet'],
            name=company,
            fill='toself'
        ))
    
    fig_maturity.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True,
        title="SEO Mognadsgrad Jämförelse"
    )
    
    st.plotly_chart(fig_maturity, use_container_width=True)

    # Sökordskategorier
    st.subheader("Sökordskategorier")
    
    keyword_categories = {
        'Kategori': ['Branded', 'Non-branded'],
        'Sökord': [737, 662],
        'Förändring': [-10, +3],
        'Trafik': [382, 128],
        'Trafik_förändring': [+1, 0]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Branded sökord", "737", "-10")
        st.metric("Branded trafik", "382", "+1")
    
    with col2:
        st.metric("Non-branded sökord", "662", "+3")
        st.metric("Non-branded trafik", "128", "0")

    # Sökordsprestanda
    st.subheader("Sökordsprestanda")
    
    keyword_performance = {
        'Sökord': ['Cybersäkerhet företag', 'Molntjänster Stockholm', 'IT-leverantör', 
                  'Managed IT services', 'IT drift outsourcing', 'IT säkerhet företag',
                  'Molnmigrering', 'IT konsult Stockholm', 'Microsoft 365 partner'],
        'Position': [5, 3, 4, 6, 2, 4, 8, 3, 2],
        'Sökvolym': [1200, 880, 720, 1500, 900, 2200, 600, 1800, 1100],
        'CTR': [3.2, 4.5, 3.8, 2.9, 5.1, 3.6, 2.2, 4.8, 5.2],
        'Konkurrent_Position': [2, 1, 3, 2, 1, 2, 3, 1, 1],
        'Svårighet': [68, 72, 65, 75, 70, 82, 58, 85, 78]
    }
    
    df_keywords = pd.DataFrame(keyword_performance)
    
    # Visualisering av sökordsprestanda
    fig_keywords = px.scatter(
        df_keywords,
        x='Sökvolym',
        y='Position',
        size='CTR',
        color='Svårighet',
        hover_name='Sökord',
        title='Sökordsprestanda och Potential',
        labels={
            'Position': 'Nuvarande Position',
            'Sökvolym': 'Månatlig Sökvolym',
            'CTR': 'Klickfrekvens (%)',
            'Svårighet': 'SEO Svårighetsgrad'
        }
    )
    fig_keywords.update_yaxes(autorange="reversed")  # Inverterar y-axeln så position 1 är högst upp
    st.plotly_chart(fig_keywords, use_container_width=True)

    # EEAT Analys
    st.subheader("EEAT Analys och Strategi")
    
    # EEAT jämförelse
    eeat_data = {
        'Företag': ['IT-Total', 'Iver', 'Advania', 'Dustin', 'Atea', 'Nordlo', 'Orange Cyberdefence', 'Proact', 'Softronic'],
        'Expertis': [75, 85, 82, 80, 88, 78, 90, 80, 76],
        'Auktoritet': [70, 85, 80, 82, 86, 75, 88, 78, 72],
        'Trovärdighet': [78, 82, 80, 85, 84, 80, 86, 79, 75],
        'Erfarenhet': [72, 80, 78, 82, 85, 76, 84, 77, 74]
    }
    
    df_eeat = pd.DataFrame(eeat_data)
    
    # Radar chart för EEAT jämförelse
    fig_eeat = go.Figure()
    
    for company in df_eeat['Företag']:
        fig_eeat.add_trace(go.Scatterpolar(
            r=[df_eeat.loc[df_eeat['Företag'] == company, col].iloc[0] 
               for col in ['Expertis', 'Auktoritet', 'Trovärdighet', 'Erfarenhet']],
            theta=['Expertis', 'Auktoritet', 'Trovärdighet', 'Erfarenhet'],
            name=company,
            fill='toself'
        ))
    
    fig_eeat.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="EEAT Jämförelse med Konkurrenter"
    )
    
    st.plotly_chart(fig_eeat, use_container_width=True)

    # EEAT Strategi
    st.subheader("EEAT Förbättringsstrategi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Expertis (E)**
        * Utveckla detaljerade tekniska guider och whitepapers
        * Publicera expertartiklar från IT-säkerhetsspecialister
        * Skapa fallstudier med teknisk djupdykning
        * Certifieringar och partnerskapsnivåer tydligt presenterade
        
        **Erfarenhet (E)**
        * Utöka kundrecensioner och testimonials
        * Dokumentera långvariga kundrelationer
        * Visa upp team-medlemmars erfarenhet
        * Dela projekthistorik och resultat
        """)
    
    with col2:
        st.markdown("""
        **Auktoritet (A)**
        * Öka närvaro i branschmedier
        * Utveckla samarbeten med erkända partners
        * Stärk länkprofilen från auktoritativa källor
        * Delta i branschkonferenser och events
        
        **Trovärdighet (T)**
        * Transparent prissättning och villkor
        * Regelbunden uppdatering av innehåll
        * Tydlig kontaktinformation och support
        * Säkerhetscertifieringar och compliance
        """)

    # Prioriterade åtgärder
    st.subheader("Prioriterade SEO-åtgärder")
    
    priority_data = {
        'Åtgärd': [
            'Expertinnehåll Cybersäkerhet',
            'Tekniska guider Molntjänster',
            'Kundcase IT-drift',
            'Partner-certifieringar',
            'Branschexpert-artiklar'
        ],
        'Prioritet': ['Hög', 'Hög', 'Medium', 'Medium', 'Hög'],
        'Tidsram': ['Q1', 'Q1', 'Q2', 'Q2', 'Q1'],
        'Förväntad Effekt': ['Stor', 'Stor', 'Medium', 'Medium', 'Stor']
    }
    
    st.dataframe(
        pd.DataFrame(priority_data)
        .style.highlight_min(subset=['Prioritet'], color='#34A853'),
        hide_index=True
    )

# Konkurrensanalys
elif menu == "Konkurrensanalys":
    st.header("Konkurrensanalys")
    
    # USP Jämförelse
    st.subheader("Unique Selling Propositions (USP)")
    
    usp_data = {
        'Företag': ['IT-Total', 'Iver', 'Advania', 'Dustin', 'Atea'],
        'Huvudfokus': ['Personlig IT-partner', 'Enterprise Solutions', 'Digital Transformation', 'Produktbredd', 'Helhetsleverantör'],
        'Teknisk_Expertis': [4, 5, 5, 3, 4],
        'Kundservice': [5, 3, 4, 4, 4],
        'Pris': [4, 3, 3, 5, 3],
        'Leveranstid': [5, 4, 4, 5, 4],
        'Innovation': [3, 5, 5, 4, 4]
    }
    
    df_usp = pd.DataFrame(usp_data)
    
    # Radar chart för USP jämförelse
    fig_usp = go.Figure()
    
    for company in df_usp['Företag']:
        fig_usp.add_trace(go.Scatterpolar(
            r=[df_usp.loc[df_usp['Företag'] == company, col].iloc[0] 
               for col in ['Teknisk_Expertis', 'Kundservice', 'Pris', 'Leveranstid', 'Innovation']],
            theta=['Teknisk Expertis', 'Kundservice', 'Pris', 'Leveranstid', 'Innovation'],
            name=company,
            fill='toself'
        ))
    
    fig_usp.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True,
        title="USP Jämförelse"
    )
    
    st.plotly_chart(fig_usp, use_container_width=True)
    
    # Trafikfunnel Analys
    st.subheader("Trafikfördelning per Kanal")
    
    traffic_data = {
        'Företag': ['IT-Total', 'Iver', 'Advania', 'Dustin', 'Atea', 'Nordlo', 'Orange Cyberdefence', 'Proact', 'Softronic'],
        'Organisk_Sök': [35, 55, 52, 48, 50, 45, 58, 42, 40],
        'Direkt': [30, 15, 18, 20, 17, 22, 16, 25, 28],
        'Betald_Sök': [8, 15, 14, 18, 16, 12, 15, 10, 9],
        'Social': [5, 8, 9, 7, 10, 8, 6, 7, 6],
        'Referral': [20, 4, 5, 5, 5, 10, 3, 13, 15],
        'Email': [2, 3, 2, 2, 2, 3, 2, 3, 2]
    }
    
    df_traffic = pd.DataFrame(traffic_data)
    
    # Stacked bar chart för trafikfördelning
    fig_traffic = px.bar(
        df_traffic,
        x='Företag',
        y=['Organisk_Sök', 'Direkt', 'Betald_Sök', 'Social', 'Referral', 'Email'],
        title='Trafikfördelning per Kanal (%)',
        labels={
            'value': 'Andel av total trafik (%)',
            'variable': 'Trafikkanal'
        },
        barmode='stack',
        color_discrete_sequence=['#34A853', '#4285F4', '#FBBC05', '#EA4335', '#5F6368', '#185ABC']
    )
    
    st.plotly_chart(fig_traffic, use_container_width=True)
    
    # Lägg till insiktsruta
    st.info("""
    **Trafikfördelningsanalys:**
    • IT-Total har lägre andel organisk söktrafik (35% vs. branschsnitt ~51%)
    • Betald söktrafik är betydligt lägre än konkurrenter (8% vs. branschsnitt ~16%)
    • Social media trafik är under branschgenomsnittet (5% vs. ~8.5%)
    • Högre andel direkttrafik och referrals indikerar stark varumärkeskännedom
    • Potential att öka digital närvaro genom förbättrad SEO och SEM-strategi
    """)
    
    # Konverteringsfunnel
    st.subheader("Konverteringsfunnel per Kanal")
    
    # Utökad funnel-data med alla företag
    funnel_data = {
        'IT-Total': {
            'Kanal': ['Organisk Sök', 'Betald Sök', 'Direkt', 'Social', 'Referral', 'Email'],
            'Besökare': [15000, 5000, 8000, 3000, 2500, 1000],
            'Leads': [450, 200, 320, 90, 100, 50],
            'Konverteringar': [45, 30, 48, 9, 15, 10]
        },
        'Iver': {
            'Kanal': ['Organisk Sök', 'Betald Sök', 'Direkt', 'Social', 'Referral', 'Email'],
            'Besökare': [25000, 8000, 6000, 4500, 2000, 1500],
            'Leads': [875, 400, 300, 180, 100, 90],
            'Konverteringar': [105, 60, 45, 27, 15, 18]
        },
        'Advania': {
            'Kanal': ['Organisk Sök', 'Betald Sök', 'Direkt', 'Social', 'Referral', 'Email'],
            'Besökare': [22000, 7500, 7000, 4000, 2200, 1300],
            'Leads': [770, 375, 350, 160, 110, 78],
            'Konverteringar': [92, 56, 53, 24, 17, 16]
        },
        'Dustin': {
            'Kanal': ['Organisk Sök', 'Betald Sök', 'Direkt', 'Social', 'Referral', 'Email'],
            'Besökare': [30000, 10000, 8000, 5000, 2800, 1800],
            'Leads': [1050, 500, 400, 200, 140, 108],
            'Konverteringar': [126, 75, 60, 30, 21, 22]
        },
        'Atea': {
            'Kanal': ['Organisk Sök', 'Betald Sök', 'Direkt', 'Social', 'Referral', 'Email'],
            'Besökare': [28000, 9000, 7500, 4800, 2600, 1600],
            'Leads': [980, 450, 375, 192, 130, 96],
            'Konverteringar': [118, 68, 56, 29, 20, 19]
        }
    }
    
    # Företagsväljare
    selected_company = st.selectbox(
        "Välj företag för detaljerad analys",
        ['IT-Total', 'Iver', 'Advania', 'Dustin', 'Atea']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Funnel chart för valt företag
        fig_funnel = go.Figure(go.Funnel(
            y=['Besökare', 'Leads', 'Konverteringar'],
            x=[sum(funnel_data[selected_company]['Besökare']), 
               sum(funnel_data[selected_company]['Leads']), 
               sum(funnel_data[selected_company]['Konverteringar'])],
            textinfo="value+percent initial",
            name=selected_company
        ))
        
        fig_funnel.update_layout(
            title=f"Konverteringsfunnel - {selected_company}",
            height=400
        )
        st.plotly_chart(fig_funnel, use_container_width=True)
    
    with col2:
        # Konverteringsgrad per kanal för valt företag
        df_selected = pd.DataFrame(funnel_data[selected_company])
        conv_rates = (df_selected['Konverteringar'] / df_selected['Besökare'] * 100).round(2)
        
        fig_conv = px.bar(
            x=df_selected['Kanal'],
            y=conv_rates,
            title=f"Konverteringsgrad per Kanal (%) - {selected_company}",
            labels={'x': 'Kanal', 'y': 'Konverteringsgrad (%)'},
            color=conv_rates,
            color_continuous_scale='Viridis'
        )
        fig_conv.update_layout(height=400)
        st.plotly_chart(fig_conv, use_container_width=True)
    
    # Jämförande statistik
    st.subheader("Jämförande Konverteringsstatistik")
    
    # Beräkna genomsnittlig konverteringsgrad för alla företag
    avg_conv_rates = {}
    for company in funnel_data:
        df_comp = pd.DataFrame(funnel_data[company])
        total_conv_rate = round((sum(df_comp['Konverteringar']) / sum(df_comp['Besökare']) * 100), 2)
        avg_conv_rates[company] = total_conv_rate
    
    # Visa jämförande metrics
    metric_cols = st.columns(len(funnel_data))
    for i, (company, rate) in enumerate(avg_conv_rates.items()):
        diff = rate - avg_conv_rates['IT-Total'] if company != 'IT-Total' else 0
        metric_cols[i].metric(
            company,
            f"{rate}%",
            f"{diff:+.2f}%" if diff != 0 else None,
            delta_color="normal" if company == 'IT-Total' else "off"
        )

# Brand Awareness
else:
    st.header("Brand Awareness")
    
    # Översiktsmått
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Varumärkeskännedom", "68%", "+5%")
        st.metric("Sociala medier följare", "4,029", "+15%")
    with col2:
        st.metric("Kundnöjdhet", "4.8/5", "+0.3")
        st.metric("Rekommendationsgrad", "92%", "+7%")

    # Brand Awareness Trender
    st.subheader("Varumärkeskännedom per Målgrupp")
    
    awareness_data = {
        'Målgrupp': ['Små företag', 'Medelstora företag', 'Stora företag', 'IT-beslutsfattare', 'C-level'],
        'Kännedom': [72, 65, 58, 75, 62],
        'Förändring': [+8, +5, +3, +7, +4],
        'Konkurrent_Snitt': [80, 78, 75, 82, 77]
    }
    
    df_awareness = pd.DataFrame(awareness_data)
    
    # Visualisering av varumärkeskännedom
    fig_awareness = go.Figure()
    
    fig_awareness.add_trace(go.Bar(
        name='IT-Total',
        x=df_awareness['Målgrupp'],
        y=df_awareness['Kännedom'],
        marker_color='#34A853'
    ))
    
    fig_awareness.add_trace(go.Bar(
        name='Branschsnitt',
        x=df_awareness['Målgrupp'],
        y=df_awareness['Konkurrent_Snitt'],
        marker_color='#4285F4'
    ))
    
    fig_awareness.update_layout(
        title="Varumärkeskännedom: IT-Total vs Branschsnitt",
        barmode='group'
    )
    
    st.plotly_chart(fig_awareness, use_container_width=True)
    
    # Kanaleffektivitet
    st.subheader("Kanaleffektivitet för Varumärkesbyggande")
    
    channel_data = {
        'Kanal': ['LinkedIn', 'Branschevent', 'PR/Media', 'Webbplats', 'Kundcase', 'Nyhetsbrev'],
        'Effektivitet': [85, 78, 65, 72, 80, 58],
        'Räckvidd': [15000, 8500, 25000, 35000, 12000, 5000],
        'Engagemang': [4.2, 4.8, 3.5, 3.8, 4.5, 3.2]
    }
    
    df_channels = pd.DataFrame(channel_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Effektivitet per kanal
        fig_effectiveness = px.bar(
            df_channels,
            x='Kanal',
            y='Effektivitet',
            color='Effektivitet',
            title='Kanaleffektivitet (%)',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_effectiveness, use_container_width=True)
    
    with col2:
        # Räckvidd vs Engagemang
        fig_reach = px.scatter(
            df_channels,
            x='Räckvidd',
            y='Engagemang',
            size='Effektivitet',
            color='Kanal',
            title='Räckvidd vs Engagemang',
            labels={'Räckvidd': 'Total Räckvidd', 'Engagemang': 'Engagemangsnivå (1-5)'}
        )
        st.plotly_chart(fig_reach, use_container_width=True)

    # Rekommendationer
    st.subheader("Rekommendationer för Ökad Varumärkeskännedom")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Kortsiktiga Åtgärder (0-6 månader):**
        1. **LinkedIn-strategi**
           * Öka publiceringsfrekvens till 4-5 ggr/vecka
           * Fokusera på thought leadership innehåll
           * Aktivera medarbetare som varumärkesambassadörer
        
        2. **Event-närvaro**
           * Delta i minst 2 större branschevent per kvartal
           * Arrangera egna kundevent och webinarier
           * Utveckla partnerskap med branschorganisationer
        
        3. **Digital Närvaro**
           * Optimera webbplats för varumärkessökord
           * Producera mer videoinnehåll
           * Förbättra kundcase-presentationer
        """)
        
    with col2:
        st.warning("""
        **Långsiktiga Strategier (6-18 månader):**
        1. **Innehållsstrategi**
           * Utveckla omfattande resursbibliotek
           * Starta podcast om IT-trender
           * Skapa branschrapporter och whitepapers
        
        2. **PR och Media**
           * Bygga relationer med branschmedia
           * Utveckla pressrum på webbplatsen
           * Öka synlighet i facktidningar
        
        3. **Kundengagemang**
           * Starta kundråd för feedback
           * Utveckla lojalitetsprogram
           * Skapa referensprogram
        """)
    
    # Målsättningar
    st.subheader("Målsättningar för Varumärkeskännedom")
    
    goals_data = {
        'Mätpunkt': ['Varumärkeskännedom', 'LinkedIn-följare', 'PR-omnämnanden/kvartal', 'Event-deltagare/år', 'Kundcase'],
        'Nuläge': [68, 4029, 15, 850, 24],
        'Mål_6_mån': [75, 6000, 25, 1200, 36],
        'Mål_12_mån': [82, 8000, 40, 2000, 48]
    }
    
    df_goals = pd.DataFrame(goals_data)
    st.dataframe(
        df_goals.style.highlight_max(subset=['Mål_12_mån'], color='#34A853'),
        hide_index=True
    )

# Footer
st.markdown("---")
st.markdown("*Rapport genererad för IT-Total AB*") 