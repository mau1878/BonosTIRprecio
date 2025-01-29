import streamlit as st
import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
from scipy.optimize import newton
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup


# Set page language and title
st.set_page_config(page_title="Calculadora de TIR de Bonos")


def get_argentine_holidays(year):
    """
    Generate Argentine holidays for a specific year
    """
    def easter(year):
        """Calculate Easter Sunday date for a given year"""
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return pd.Timestamp(year, month, day)

    # Calculate Easter-dependent dates
    easter_sunday = easter(year)
    good_friday = easter_sunday - pd.Timedelta(days=2)
    carnival_monday = easter_sunday - pd.Timedelta(days=48)
    carnival_tuesday = easter_sunday - pd.Timedelta(days=47)

    # Fixed holidays
    fixed_holidays = [
        (1, 1),   # New Year's Day
        (3, 24),  # Day of Remembrance
        (4, 2),   # Malvinas Day
        (5, 1),   # Labor Day
        (5, 25),  # May Revolution Day
        (6, 17),  # Martin Miguel de Güemes Day
        (6, 20),  # Flag Day
        (7, 9),   # Independence Day
        (8, 17),  # San Martín Day
        (10, 12), # Day of Respect for Cultural Diversity
        (11, 20), # National Sovereignty Day
        (12, 8),  # Immaculate Conception
        (12, 25), # Christmas Day
    ]

    # Convert fixed holidays to timestamps
    holidays = [pd.Timestamp(year, month, day) for month, day in fixed_holidays]

    # Add movable holidays
    holidays.extend([
        carnival_monday,
        carnival_tuesday,
        good_friday
    ])

    # Bridge holidays (if holiday falls on Tuesday or Thursday, Monday or Friday becomes holiday)
    bridge_holidays = []
    for holiday in holidays:
        if holiday.dayofweek == 1:  # Tuesday
            bridge_holidays.append(holiday - pd.Timedelta(days=1))  # Monday
        elif holiday.dayofweek == 3:  # Thursday
            bridge_holidays.append(holiday + pd.Timedelta(days=1))  # Friday

    # Add bridge holidays to the main list
    holidays.extend(bridge_holidays)

    # Sort all holidays
    return pd.DatetimeIndex(sorted(set(holidays)))

def get_previous_business_day_argentina(date, days_back=10):
    """
    Get the Argentine business day that is 'days_back' business days before the given date
    """
    # Convert date to pandas timestamp if it's not already
    date = pd.Timestamp(date)

    # Get holidays for the relevant years
    current_year = date.year
    prev_year = current_year - 1

    holidays = pd.DatetimeIndex([])
    for year in [prev_year, current_year]:
        holidays = holidays.append(get_argentine_holidays(year))

    # Create business day offset excluding holidays
    bd = pd.offsets.CustomBusinessDay(holidays=holidays)

    # Get the date that is 'days_back' business days before the reference date
    reference_date = date - days_back * bd

    return reference_date.to_pydatetime()
def fetch_cer_for_date(date):
    """
    Fetch CER value for a specific date from BCRA API

    Args:
        date (datetime): Date to fetch CER for

    Returns:
        float: CER value for the specified date, or None if not found
    """
    # Get the Argentine business day 10 days before settlement
    reference_date = get_previous_business_day_argentina(date)

    # Format date as required by the API
    date_str = reference_date.strftime('%Y-%m-%d')

    # API endpoint for CER (ID 30)
    url = "https://api.bcra.gob.ar/estadisticas/v3.0/Monetarias/30"

    # Parameters for the request
    params = {
        'desde': date_str,
        'hasta': date_str,
        'limit': 1
    }

    try:
        # Make the GET request with SSL verification disabled
        response = requests.get(url, params=params, verify=False)

        if response.status_code == 200:
            data = response.json()

            # Check if we got any results
            if data['results'] and len(data['results']) > 0:
                cer_value = data['results'][0]['valor']
                st.info(f"Usando CER del {reference_date.strftime('%d/%m/%Y')}: {cer_value:.4f}")
                return cer_value
            else:
                st.warning(f"No se encontró valor de CER para {reference_date.strftime('%d/%m/%Y')}")
                return None
        else:
            st.error(f"Error al obtener CER: Status code {response.status_code}")
            return None

    except Exception as e:
        st.error(f"Error al obtener CER: {str(e)}")
        return None
def fetch_current_prices():
    """Fetch current prices from Google Sheets CSV"""
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSh_h0veiwhaDs-8u0W75LcPc7DoKQ_zWjsMN6EHzfMlgWvGJMC_AFe319FTQXps3ACnkgEZaNPJopz/pub?gid=1705467858&single=true&output=csv"

    try:
        # Read CSV directly into DataFrame
        df = pd.read_csv(url)

        # Create a dictionary of ticker:price pairs
        price_dict = {}
        for _, row in df.iterrows():
            ticker = str(row.iloc[0])
            price_str = str(row.iloc[2])

            # Skip rows with #N/A or invalid prices
            if price_str != '#N/A' and not pd.isna(price_str):
                try:
                    # Convert price string to float, handling different number formats
                    price_str = price_str.replace('.', '').replace(',', '.')
                    price = float(price_str)
                    price_dict[ticker] = price
                except ValueError:
                    continue

        return price_dict

    except Exception as e:
        st.error(f"Error al obtener los precios: {str(e)}")
        return {}

def xnpv(rate, values, dates):
    """Calculate the Net Present Value with irregular time periods"""
    if rate <= -1.0:
        return float('inf')
    d0 = dates[0]
    return sum([
        vi / (1.0 + rate) ** ((di - d0).days / 365.0)
        for vi, di in zip(values, dates)
    ])

def xirr(values, dates):
    """Calculate IRR for irregular time periods"""
    try:
        return newton(
            lambda r: xnpv(r, values, dates),
            x0=0.1,  # Initial guess
            tol=1e-5,
            maxiter=1000
        )
    except (RuntimeError, ValueError):
        return float('nan')

def calculate_irr_with_timing(cashflows, dates, price, start_date):
    """Calculate IRR considering the actual timing of cash flows"""
    values = [-price] + cashflows
    dates = [start_date] + dates
    return xirr(values, dates)

def parse_number(number_str, format_type):
    """Parse a number string based on the selected format"""
    try:
        if format_type == "Punto decimal y coma para miles (1,234.56)":
            # Remove thousand separators (commas) and keep decimal point
            cleaned_str = number_str.replace(',', '')
            return float(cleaned_str)
        else:
            # Format with comma as decimal and dots as thousand separators
            cleaned_str = number_str.replace('.', '')
            cleaned_str = cleaned_str.replace(',', '.')
            return float(cleaned_str)
    except ValueError as e:
        raise ValueError(f"No se pudo procesar el número: {number_str}") from e

def parse_cashflows(text, number_format, settlement_date):
    """Parse cash flows from text input with 'DD/MM/YYYY Coupon' and filter by settlement date"""
    cashflows = []
    dates = []

    for line in text.strip().split('\n'):
        if line.strip():  # Skip empty lines
            try:
                parts = re.split(r'\s+', line.strip())
                date_str = parts[0]
                amount_str = ''.join(parts[1:])

                try:
                    date = datetime.strptime(date_str, '%d/%m/%Y')
                    # Skip if date is equal to or before settlement date
                    if date <= settlement_date:
                        continue
                except ValueError:
                    st.error(f"Formato de fecha inválido en la línea: {line}. Use DD/MM/YYYY")
                    continue

                try:
                    amount = parse_number(amount_str.strip(), number_format)
                except ValueError:
                    format_example = "1,234.56" if number_format == "Punto decimal y coma para miles (1,234.56)" else "1.234,56"
                    st.error(f"Formato de monto inválido en la línea: {line}. Use el formato {format_example}")
                    continue

                cashflows.append(amount)
                dates.append(date)
            except Exception as e:
                st.error(f"Error al procesar la línea: {line}. Error: {str(e)}")
                continue

    if not cashflows:
        return None, None

    return cashflows, dates

def load_bonds_from_csv():
    try:
        df = pd.read_csv('Cashflowbonos.csv')
        available_bonds = df['Bono'].unique()
        return df, available_bonds
    except FileNotFoundError:
        st.error("No se encontró el archivo Cashflowbonos.csv en el directorio")
        return None, None
def calculate_modified_duration(cashflows, dates, price, settlement_date, irr):
    """Calculate Modified Duration for a bond"""
    if not cashflows or not dates:
        return 0

    # Convert IRR from percentage to decimal
    irr = irr / 100

    # Calculate present value of each cash flow and its contribution to duration
    total_pv = 0
    weighted_time = 0

    for cf, date in zip(cashflows, dates):
        time_to_cf = (date - settlement_date).days / 365.0
        pv_factor = 1 / ((1 + irr) ** time_to_cf)
        pv = cf * pv_factor
        total_pv += pv
        weighted_time += pv * time_to_cf

    # Calculate Macaulay Duration
    macaulay_duration = weighted_time / price

    # Calculate Modified Duration
    modified_duration = macaulay_duration / (1 + irr)

    return modified_duration

# Set up the Streamlit page
st.title('Calculadora de TIR de Bonos con Análisis de Sensibilidad')

# Input method selection
# Add this near the top of the script, after the imports and before the input method selection
default_number_format = "Punto decimal y coma para miles (1,234.56)"  # default format

# Input method selection
input_method = st.radio(
    "Seleccione el método de entrada:",
    ["Seleccionar bonos predefinidos", "Ingresar flujos manualmente"],
    horizontal=True
)

# Initialize number_format with default value
number_format = default_number_format



# Get today's date
today = datetime.now()
# Add this near the settlement date input
settlement_date = st.date_input("Fecha de Liquidación", today)
settlement_date = datetime.combine(settlement_date, datetime.min.time())

# Fetch CER value for settlement date
cer_value = fetch_cer_for_date(settlement_date)
if cer_value is not None:
    st.info(f"CER para la fecha de liquidación: {cer_value:.4f}")
else:
    st.warning("No se pudo obtener el valor del CER para la fecha especificada")

if input_method == "Seleccionar bonos predefinidos":
    # Load bonds from CSV
    bonds_df, available_bonds = load_bonds_from_csv()

    if bonds_df is not None:
        selected_bonds = st.multiselect(
            "Seleccione los bonos a analizar:",
            available_bonds
        )

        if selected_bonds:
            # Fetch current prices
            current_prices = fetch_current_prices()

            # Create a dictionary to store prices for each bond
            bond_prices = {}

            # Create columns for price inputs
            cols = st.columns(len(selected_bonds))
            for idx, bond in enumerate(selected_bonds):
                with cols[idx]:
                    # Use fetched price as default if available
                    default_price = current_prices.get(bond, 1000.0)
                    bond_prices[bond] = st.number_input(
                        f'Precio de {bond}',
                        min_value=0.01,
                        value=default_price,
                        key=f'price_{bond}'
                    )

            # Get cashflows for each bond separately
            all_cashflows = []
            all_dates = []
            all_data = []  # Initialize all_data list here

            # Inside the "Seleccionar bonos predefinidos" section:
            # Inside the "Seleccionar bonos predefinidos" section, modify the cashflow processing part:
            for bond in selected_bonds:
                bond_data = bonds_df[bonds_df['Bono'] == bond]
                bond_cashflows = []
                bond_dates = []

                # Check if this bond has CER adjustment
                has_cer = 'CERinicial' in bond_data.columns and not pd.isna(bond_data['CERinicial'].iloc[0])

                for _, row in bond_data.iterrows():
                    date = datetime.strptime(row['Fecha'], '%d/%m/%Y')
                    # Skip if date is equal to or before settlement date
                    if date <= settlement_date:
                        continue

                    cashflow = float(row['Cashflow'])

                    # Apply CER adjustment if applicable
                    # Apply CER adjustment if applicable
                    if has_cer and cer_value is not None:
                        initial_cer = float(row['CERinicial'])
                        cashflow = cashflow * cer_value / initial_cer
                        # Only show one CER adjustment message per bond
                        if not hasattr(st.session_state, f'cer_message_{bond}'):
                            st.info(f"Flujos de {bond} ajustados por CER (CER actual: {cer_value:.4f})")
                            setattr(st.session_state, f'cer_message_{bond}', True)

                    bond_cashflows.append(cashflow)
                    bond_dates.append(date)

                    # Add data to all_data list
                    all_data.append({
                        'Bono': bond,
                        'Fecha': date,
                        'Días desde Liquidación': (date - settlement_date).days,
                        'Años desde Liquidación': (date - settlement_date).days / 365,
                        'Flujo de Caja Original': float(row['Cashflow']),
                        'Flujo de Caja Ajustado': cashflow if has_cer else float(row['Cashflow']),
                        'CER Inicial': row['CERinicial'] if has_cer else None,
                        'CER Actual': cer_value if has_cer else None
                    })

                all_cashflows.append(bond_cashflows)
                all_dates.append(bond_dates)

else:
    # [Rest of the code for manual input remains the same]
    # Add number format selector
    number_format = st.radio(
        "Seleccione el formato de números:",
        ["Punto decimal y coma para miles (1,234.56)",
         "Coma decimal y punto para miles (1.234,56)"],
        horizontal=True
    )

    st.subheader('Ingrese los Flujos de Caja (Formato: DD/MM/YYYY Cupón)')

    # Show example based on selected format
    if number_format == "Punto decimal y coma para miles (1,234.56)":
        st.text('Ejemplos:\n01/01/2024    50.5\n01/07/2024\t1,050.5\n01/01/2025 1,000,050.5')
    else:
        st.text('Ejemplos:\n01/01/2024    50,5\n01/07/2024\t1.050,5\n01/01/2025 1.000.050,5')

    cashflow_text = st.text_area(
        'Pegue sus flujos de caja aquí (uno por línea):',
        height=200
    )

    if cashflow_text:
        cashflows, dates = parse_cashflows(cashflow_text, number_format, settlement_date)
        current_price = st.number_input('Precio Actual del Bono', min_value=0.01, value=1000.0)

# [Rest of your original script continues here...]
# Display processed cash flows and sensitivity analysis
if (input_method == "Seleccionar bonos predefinidos" and selected_bonds) or \
        (input_method == "Ingresar flujos manualmente" and cashflow_text):

    # Modified check to handle both cases
    if input_method == "Seleccionar bonos predefinidos":
        if all(len(cf) == 0 for cf in all_cashflows):
            st.warning("No hay flujos de caja futuros disponibles después de la fecha de liquidación.")
            st.stop()
    else:  # manual input
        if not cashflows or len(cashflows) == 0:
            st.warning("No hay flujos de caja futuros disponibles después de la fecha de liquidación.")
            st.stop()

    # Display processed cash flows
    st.subheader('Flujos de Caja Procesados')
    if input_method == "Seleccionar bonos predefinidos":
        df_input = pd.DataFrame(all_data)
        df_input = df_input.sort_values('Fecha')

        # Format the cashflow columns
        for col in ['Flujo de Caja Original', 'Flujo de Caja Ajustado']:
            df_input[col] = df_input[col].apply(
                lambda x: f"{x:,.2f}".replace(',', '@').replace('.', ',').replace('@', '.')
            )

        # Display the DataFrame with all columns
        st.dataframe(df_input)
    else:
        df_manual = pd.DataFrame({
            'Fecha': dates,
            'Días desde Liquidación': [(date - settlement_date).days for date in dates],
            'Años desde Liquidación': [(date - settlement_date).days / 365 for date in dates],
            'Flujo de Caja': cashflows
        })

        # Format numbers according to selected format
        if number_format == "Punto decimal y coma para miles (1,234.56)":
            formatted_cashflows = df_manual['Flujo de Caja'].apply(lambda x: f"{x:,.2f}")
        else:
            formatted_cashflows = df_manual['Flujo de Caja'].apply(
                lambda x: f"{x:,.2f}".replace(',', '@').replace('.', ',').replace('@', '.')
            )

        df_manual['Flujo de Caja'] = formatted_cashflows
        st.dataframe(df_manual)

    # Price sensitivity range
    price_change_percent = st.slider('Rango de Variación de Precio (%)', -50, 50, (-20, 20))
    plot_type = st.radio(
        "Seleccione el tipo de gráfico:",
        ["Precio Absoluto", "Variación Porcentual del Precio", "Variación Porcentual Precio vs Variación Porcentual TIR"],
        horizontal=True
    )

    # Create sensitivity analysis chart
    # Create IRR vs Modified Duration scatter plot
    # Create IRR vs Modified Duration scatter plot
    # Create IRR vs Modified Duration scatter plot
    # Create IRR vs Modified Duration scatter plot
    if input_method == "Seleccionar bonos predefinidos" and selected_bonds:
        fig_md = go.Figure()

        # Define an extended color palette that supports up to 20 items
        colors = (px.colors.qualitative.Set3 +
                  px.colors.qualitative.Set1 +
                  px.colors.qualitative.Pastel1 +
                  px.colors.qualitative.Safe)[:20]  # Combine multiple palettes

        # Calculate IRR and MD for each bond
        irr_md_data = []
        for bond, cashflows, dates, price, color in zip(selected_bonds, all_cashflows, all_dates, bond_prices.values(),
                                                        colors):
            current_irr = calculate_irr_with_timing(cashflows, dates, price, settlement_date) * 100
            modified_duration = calculate_modified_duration(cashflows, dates, price, settlement_date, current_irr)
            irr_md_data.append({
                'Bond': bond,
                'IRR': current_irr,
                'MD': modified_duration,
                'Color': color
            })

        # Create DataFrame
        df_irr_md = pd.DataFrame(irr_md_data)

        # Add scatter plot for each bond
        for bond, color in zip(selected_bonds, colors):
            bond_data = df_irr_md[df_irr_md['Bond'] == bond]
            fig_md.add_trace(go.Scatter(
                x=bond_data['MD'],
                y=bond_data['IRR'],
                mode='markers+text',
                text=bond_data['Bond'],
                textposition='top center',
                name=bond,
                marker=dict(size=10, color=color),
                showlegend=True
            ))

        # Add polynomial trendline
        z = np.polyfit(df_irr_md['MD'], df_irr_md['IRR'], 2)
        p = np.poly1d(z)

        x_range = np.linspace(df_irr_md['MD'].min(), df_irr_md['MD'].max(), 100)
        y_trend = p(x_range)

        # Add trendline to plot
        fig_md.add_trace(go.Scatter(
            x=x_range,
            y=y_trend,
            mode='lines',
            name='Tendencia Polinómica',
            line=dict(color='rgba(255, 0, 0, 0.5)', dash='dash'),
            hovertemplate='MD: %{x:.2f}<br>TIR: %{y:.2f}%<extra></extra>'
        ))

        # Update layout
        fig_md.update_layout(
            title='TIR vs Duración Modificada',
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            title_font_size=24,
            xaxis=dict(
                title='Duración Modificada (años)',
                title_font=dict(size=18),
                tickfont=dict(size=14),
                gridcolor='dimgray',
                showgrid=True,
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1
            ),
            yaxis=dict(
                title='TIR (%)',
                title_font=dict(size=18),
                tickfont=dict(size=14),
                gridcolor='dimgray',
                showgrid=True,
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1
            ),
            annotations=[
                dict(
                    text="MTaurus - X: @mtaurus_ok",
                    x=0.5,
                    y=0.5,  # Adjusted to be above the legend box
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(
                        size=30,
                        color="rgba(100,100, 100, 0.1)"
                    ),
                    textangle=0,
                    align="right",
                )
            ],
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(0, 0, 0, 0.7)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                itemsizing='constant',  # Makes legend items constant size
                font=dict(size=10),  # Smaller font size for legend
            ),
            margin=dict(r=200)  # Increased right margin for larger legend
        )

        # Display the chart
        st.plotly_chart(fig_md)
    fig = go.Figure()

    # Modify the plotting section:
    if input_method == "Seleccionar bonos predefinidos":
        colors = px.colors.qualitative.Set3[:len(selected_bonds)]
        for bond, cashflows, dates, price, color in zip(selected_bonds, all_cashflows, all_dates, bond_prices.values(),
                                                        colors):
            # Calculate current IRR as reference point
            current_irr = calculate_irr_with_timing(cashflows, dates, price, settlement_date) * 100

            # Calculate price range
            price_range = np.linspace(
                price * (1 + price_change_percent[0] / 100),
                price * (1 + price_change_percent[1] / 100),
                100
            )

            # Calculate IRRs
            irrs = [calculate_irr_with_timing(cashflows, dates, p, settlement_date) * 100 for p in price_range]

            # Calculate percentage changes
            price_changes_range = [(p / price - 1) * 100 for p in price_range]
            irr_changes_range = [(irr / current_irr - 1) * 100 for irr in irrs]

            # Select y-values based on plot type
            if plot_type == "Precio Absoluto":
                y_values = price_range
                y_axis_title = "Precio del Bono"
                current_y = price
            elif plot_type == "Variación Porcentual del Precio":
                y_values = price_changes_range
                y_axis_title = "Variación del Precio (%)"
                current_y = 0
            else:  # Variación Porcentual Precio vs TIR
                y_values = price_changes_range
                x_values = irr_changes_range
                y_axis_title = "Variación del Precio (%)"
                current_y = 0
                current_x = 0

            # Add trace with appropriate x and y values
            if plot_type == "Variación Porcentual Precio vs Variación Porcentual TIR":
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines',
                    name=bond,
                    line=dict(color=color)
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=irrs,
                    y=y_values,
                    mode='lines',
                    name=bond,
                    line=dict(color=color)
                ))

            # Add current point
            if plot_type == "Variación Porcentual Precio vs Variación Porcentual TIR":
                fig.add_trace(go.Scatter(
                    x=[current_x],
                    y=[current_y],
                    mode='markers',
                    name=f'{bond} (Actual)',
                    marker=dict(color=color, size=10)
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[current_irr],
                    y=[current_y],
                    mode='markers',
                    name=f'{bond} (Actual)',
                    marker=dict(color=color, size=10)
                ))

    else:
        # Original single bond analysis
        price_range = np.linspace(
            current_price * (1 + price_change_percent[0] / 100),
            current_price * (1 + price_change_percent[1] / 100),
            100
        )
        irrs = [calculate_irr_with_timing(cashflows, dates, price, settlement_date) * 100 for price in price_range]
        price_changes_range = [(p / current_price - 1) * 100 for p in price_range]

        y_values = price_range if x_axis_option == "Precio" else price_changes_range
        fig.add_trace(go.Scatter(
            x=irrs,
            y=y_values,
            mode='lines',
            name='TIR',
            line=dict(color='lightblue')
        ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Sensibilidad de la TIR a Cambios en el Precio',
            'font': {'size': 24, 'color': 'lightgrey'}
        },
        xaxis=dict(
            title=dict(
                text="Variación TIR (%)" if plot_type == "Variación Porcentual Precio vs Variación Porcentual TIR" else "TIR (%)",
                font=dict(size=18, color='lightgrey')
            ),
            gridcolor='dimgray',
            gridwidth=0.5,
            tickfont=dict(size=14, color='lightgrey'),
            showgrid=True,
            zeroline=True,
            zerolinecolor='darkmagenta',
            zerolinewidth=2
        ),
        yaxis=dict(
            title=dict(
                text=y_axis_title,
                font=dict(size=18, color='lightgrey')
            ),
            gridcolor='dimgray',
            gridwidth=0.5,
            tickfont=dict(size=14, color='lightgrey'),
            showgrid=True,
            zeroline=True,
            zerolinecolor='darkmagenta',
            zerolinewidth=2
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=12, color='black'),
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(t=50, b=50, l=50, r=50),
        hovermode='x unified',
        plot_bgcolor='dark gray',
        paper_bgcolor='dark gray',
        annotations=[
            dict(
                text="MTaurus - X: @mtaurus_ok",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(
                    size=30,
                    color="rgba(0, 0, 0, 0.5)"
                ),
                textangle=0,
                align="center",
            )
        ]
    )

    # Update grid and tick marks
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)


    # Display the chart
    st.plotly_chart(fig)

    # Create sensitivity tables
    st.subheader('Tablas de Sensibilidad de la TIR')

    if input_method == "Seleccionar bonos predefinidos":
        for bond, cashflows, dates, price in zip(selected_bonds, all_cashflows, all_dates, bond_prices.values()):
            price_changes = [-20, -10, -5, 0, 5, 10, 20]
            sensitivity_data = {
                'Variación de Precio (%)': price_changes,
                'Precio': [price * (1 + pc / 100) for pc in price_changes],
                'TIR (%)': [calculate_irr_with_timing(cashflows, dates, price * (1 + pc / 100), settlement_date) * 100
                           for pc in price_changes]
            }

            sensitivity_df = pd.DataFrame(sensitivity_data)
            st.subheader(f'Sensibilidad para {bond}')

            # Format numbers according to selected format
            formatted_df = sensitivity_df.round(2)
            if input_method == "Ingresar flujos manualmente" and number_format == "Punto decimal y coma para miles (1,234.56)":
                formatted_df = formatted_df.apply(lambda x: x.apply(lambda y: f"{y:,.2f}")
                                               if x.name != 'Variación de Precio (%)' else x)
            else:
                formatted_df = formatted_df.apply(
                    lambda x: x.apply(lambda y: f"{y:,.2f}".replace(',', '@').replace('.', ',').replace('@', '.'))
                    if x.name != 'Variación de Precio (%)' else x
                )

            st.dataframe(formatted_df)

            # Download button for each bond
            csv = sensitivity_df.to_csv(index=False, decimal=',' if number_format == "Coma decimal y punto para miles (1.234,56)" else '.')
            st.download_button(
                label=f"Descargar Análisis de Sensibilidad de {bond} como CSV",
                data=csv,
                file_name=f"analisis_sensibilidad_tir_{bond}.csv",
                mime="text/csv"
            )
    else:
        # Original single bond sensitivity table
        price_changes = [-20, -10, -5, 0, 5, 10, 20]
        sensitivity_data = {
            'Variación de Precio (%)': price_changes,
            'Precio': [current_price * (1 + pc / 100) for pc in price_changes],
            'TIR (%)': [
                calculate_irr_with_timing(cashflows, dates, current_price * (1 + pc / 100), settlement_date) * 100
                for pc in price_changes]
        }

        sensitivity_df = pd.DataFrame(sensitivity_data)

        # Format numbers according to selected format
        if number_format == "Punto decimal y coma para miles (1,234.56)":
            formatted_df = sensitivity_df.round(2).apply(
                lambda x: x.apply(lambda y: f"{y:,.2f}")
                if x.name != 'Variación de Precio (%)' else x
            )
        else:
            formatted_df = sensitivity_df.round(2).apply(
                lambda x: x.apply(lambda y: f"{y:,.2f}".replace(',', '@').replace('.', ',').replace('@', '.'))
                if x.name != 'Variación de Precio (%)' else x
            )

        st.dataframe(formatted_df)

        # Download button for single bond
        csv = sensitivity_df.to_csv(index=False, decimal=',' if number_format == "Coma decimal y punto para miles (1.234,56)" else '.')
        st.download_button(
            label="Descargar Análisis de Sensibilidad como CSV",
            data=csv,
            file_name="analisis_sensibilidad_tir.csv",
            mime="text/csv"
        )
