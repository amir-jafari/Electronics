import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# Set page configuration
st.set_page_config(
    page_title="RF Attenuator Calculator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .circuit-diagram {
        font-family: monospace;
        font-size: 1.2rem;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def calculate_t_attenuator(attenuation_db, z0=50):
    """Calculate T-section attenuator resistor values."""
    if attenuation_db <= 0:
        raise ValueError("Attenuation must be positive (> 0 dB)")

    k = 10 ** (attenuation_db / 20)
    r1 = z0 * (k - 1) / (k + 1)
    r2 = 2 * z0 * k / (k ** 2 - 1)
    r3 = r1

    return {
        'type': 'T-section',
        'attenuation_db': attenuation_db,
        'characteristic_impedance': z0,
        'R1_ohms': r1,
        'R2_ohms': r2,
        'R3_ohms': r3,
        'R1_description': 'Series (Input)',
        'R2_description': 'Shunt (Middle)',
        'R3_description': 'Series (Output)',
        'voltage_ratio': k,
        'power_ratio': k ** 2,
        'input_impedance': z0,
        'output_impedance': z0
    }


def calculate_pi_attenuator(attenuation_db, z0=50):
    """Calculate PI-section attenuator resistor values."""
    if attenuation_db <= 0:
        raise ValueError("Attenuation must be positive (> 0 dB)")

    k = 10 ** (attenuation_db / 20)
    r1 = z0 * (k + 1) / (k - 1)
    r2 = z0 * (k ** 2 - 1) / (2 * k)
    r3 = r1

    return {
        'type': 'PI-section',
        'attenuation_db': attenuation_db,
        'characteristic_impedance': z0,
        'R1_ohms': r1,
        'R2_ohms': r2,
        'R3_ohms': r3,
        'R1_description': 'Shunt (Input)',
        'R2_description': 'Series (Middle)',
        'R3_description': 'Shunt (Output)',
        'voltage_ratio': k,
        'power_ratio': k ** 2,
        'input_impedance': z0,
        'output_impedance': z0
    }


def create_circuit_diagram(attenuator_type):
    """Create ASCII circuit diagrams."""
    if attenuator_type == "T-section":
        return """
    Input ──[R1]──┬──[R3]── Output
                  │
                 [R2]
                  │
                 GND
        """
    else:  # PI-section
        return """
    Input ──┬──[R2]──┬── Output
            │        │
           [R1]     [R3]
            │        │
           GND      GND
        """


def generate_comparison_table(attenuation_range, z0):
    """Generate comparison table for different attenuation values."""
    data = []
    for db in attenuation_range:
        try:
            t_results = calculate_t_attenuator(db, z0)
            pi_results = calculate_pi_attenuator(db, z0)

            data.append({
                'Attenuation (dB)': db,
                'T-R1 (Ω)': round(t_results['R1_ohms'], 2),
                'T-R2 (Ω)': round(t_results['R2_ohms'], 2),
                'T-R3 (Ω)': round(t_results['R3_ohms'], 2),
                'PI-R1 (Ω)': round(pi_results['R1_ohms'], 2),
                'PI-R2 (Ω)': round(pi_results['R2_ohms'], 2),
                'PI-R3 (Ω)': round(pi_results['R3_ohms'], 2),
                'Voltage Ratio': f"1:{round(t_results['voltage_ratio'], 3)}",
                'Power Ratio': f"1:{round(t_results['power_ratio'], 3)}"
            })
        except ValueError:
            continue

    return pd.DataFrame(data)


def create_resistance_plot(attenuation_range, z0):
    """Create interactive plot showing resistance values vs attenuation."""
    t_r1_values = []
    t_r2_values = []
    pi_r1_values = []
    pi_r2_values = []

    for db in attenuation_range:
        try:
            t_results = calculate_t_attenuator(db, z0)
            pi_results = calculate_pi_attenuator(db, z0)

            t_r1_values.append(t_results['R1_ohms'])
            t_r2_values.append(t_results['R2_ohms'])
            pi_r1_values.append(pi_results['R1_ohms'])
            pi_r2_values.append(pi_results['R2_ohms'])
        except ValueError:
            t_r1_values.append(None)
            t_r2_values.append(None)
            pi_r1_values.append(None)
            pi_r2_values.append(None)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('T-Section Attenuator', 'PI-Section Attenuator'),
        shared_xaxes=True
    )

    # T-section plot
    fig.add_trace(
        go.Scatter(x=attenuation_range, y=t_r1_values, name='T-R1 (Series)',
                   line=dict(color='blue', width=2), mode='lines+markers'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=attenuation_range, y=t_r2_values, name='T-R2 (Shunt)',
                   line=dict(color='red', width=2), mode='lines+markers'),
        row=1, col=1
    )

    # PI-section plot
    fig.add_trace(
        go.Scatter(x=attenuation_range, y=pi_r1_values, name='PI-R1 (Shunt)',
                   line=dict(color='green', width=2), mode='lines+markers'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=attenuation_range, y=pi_r2_values, name='PI-R2 (Series)',
                   line=dict(color='orange', width=2), mode='lines+markers'),
        row=2, col=1
    )

    fig.update_layout(
        title=f'Resistor Values vs Attenuation (Z₀ = {z0}Ω)',
        height=600,
        showlegend=True
    )
    fig.update_xaxes(title_text="Attenuation (dB)", row=2, col=1)
    fig.update_yaxes(title_text="Resistance (Ω)")

    return fig


def main():
    # Title
    st.markdown('<div class="main-header">📡 RF Attenuator Calculator</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")

    # Impedance selection
    impedance_options = [50, 75, 100, 600]
    z0 = st.sidebar.selectbox(
        "Characteristic Impedance (Ω)",
        options=impedance_options,
        index=0,
        help="Select the system impedance"
    )

    # Custom impedance option
    if st.sidebar.checkbox("Use custom impedance"):
        z0 = st.sidebar.number_input(
            "Custom impedance (Ω)",
            min_value=1.0,
            max_value=1000.0,
            value=50.0,
            step=1.0
        )

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Calculator", "📊 Comparison Table", "📈 Analysis Charts", "📖 Reference"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">T-Section Attenuator</div>', unsafe_allow_html=True)

            # Attenuation input for T-section
            t_attenuation = st.number_input(
                "Attenuation (dB) - T-Section",
                min_value=0.1,
                max_value=60.0,
                value=10.0,
                step=0.1,
                key="t_atten"
            )

            try:
                t_results = calculate_t_attenuator(t_attenuation, z0)

                # Circuit diagram
                st.markdown('<div class="circuit-diagram">', unsafe_allow_html=True)
                st.code(create_circuit_diagram("T-section"), language=None)
                st.markdown('</div>', unsafe_allow_html=True)

                # Results
                st.subheader("Resistor Values")
                col_t1, col_t2, col_t3 = st.columns(3)

                with col_t1:
                    st.metric(
                        "R1 (Series Input)",
                        f"{t_results['R1_ohms']:.2f} Ω",
                        help="Series resistor at input"
                    )

                with col_t2:
                    st.metric(
                        "R2 (Shunt Middle)",
                        f"{t_results['R2_ohms']:.2f} Ω",
                        help="Shunt resistor in middle"
                    )

                with col_t3:
                    st.metric(
                        "R3 (Series Output)",
                        f"{t_results['R3_ohms']:.2f} Ω",
                        help="Series resistor at output"
                    )

                # Performance metrics
                st.subheader("Performance")
                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    st.metric("Voltage Ratio", f"1:{t_results['voltage_ratio']:.3f}")

                with col_p2:
                    st.metric("Power Ratio", f"1:{t_results['power_ratio']:.3f}")

            except ValueError as e:
                st.error(f"Error: {e}")

        with col2:
            st.markdown('<div class="section-header">PI-Section Attenuator</div>', unsafe_allow_html=True)

            # Attenuation input for PI-section
            pi_attenuation = st.number_input(
                "Attenuation (dB) - PI-Section",
                min_value=0.1,
                max_value=60.0,
                value=10.0,
                step=0.1,
                key="pi_atten"
            )

            try:
                pi_results = calculate_pi_attenuator(pi_attenuation, z0)

                # Circuit diagram
                st.markdown('<div class="circuit-diagram">', unsafe_allow_html=True)
                st.code(create_circuit_diagram("PI-section"), language=None)
                st.markdown('</div>', unsafe_allow_html=True)

                # Results
                st.subheader("Resistor Values")
                col_pi1, col_pi2, col_pi3 = st.columns(3)

                with col_pi1:
                    st.metric(
                        "R1 (Shunt Input)",
                        f"{pi_results['R1_ohms']:.2f} Ω",
                        help="Shunt resistor at input"
                    )

                with col_pi2:
                    st.metric(
                        "R2 (Series Middle)",
                        f"{pi_results['R2_ohms']:.2f} Ω",
                        help="Series resistor in middle"
                    )

                with col_pi3:
                    st.metric(
                        "R3 (Shunt Output)",
                        f"{pi_results['R3_ohms']:.2f} Ω",
                        help="Shunt resistor at output"
                    )

                # Performance metrics
                st.subheader("Performance")
                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    st.metric("Voltage Ratio", f"1:{pi_results['voltage_ratio']:.3f}")

                with col_p2:
                    st.metric("Power Ratio", f"1:{pi_results['power_ratio']:.3f}")

            except ValueError as e:
                st.error(f"Error: {e}")

    with tab2:
        st.markdown('<div class="section-header">📊 Attenuator Comparison Table</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            min_atten = st.number_input("Min Attenuation (dB)", min_value=0.1, max_value=50.0, value=1.0, step=0.1)

        with col2:
            max_atten = st.number_input("Max Attenuation (dB)", min_value=0.2, max_value=60.0, value=20.0, step=0.1)

        with col3:
            step_atten = st.number_input("Step Size (dB)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

        if st.button("Generate Table", type="primary"):
            attenuation_range = np.arange(min_atten, max_atten + step_atten, step_atten)
            df = generate_comparison_table(attenuation_range, z0)

            if not df.empty:
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=400
                )

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"attenuator_values_Z{z0}ohm.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No valid data in the specified range.")

    with tab3:
        st.markdown('<div class="section-header">📈 Resistance Analysis Charts</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            chart_min = st.number_input("Chart Min (dB)", min_value=0.1, max_value=50.0, value=1.0, step=0.1,
                                        key="chart_min")
            chart_max = st.number_input("Chart Max (dB)", min_value=0.2, max_value=60.0, value=30.0, step=0.1,
                                        key="chart_max")

        with col2:
            chart_step = st.number_input("Chart Step (dB)", min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                                         key="chart_step")

        if st.button("Generate Chart", type="primary"):
            attenuation_range = np.arange(chart_min, chart_max + chart_step, chart_step)
            fig = create_resistance_plot(attenuation_range, z0)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown('<div class="section-header">📖 Design Reference</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("T-Section Formulas")
            st.latex(r"K = 10^{A_{dB}/20}")
            st.latex(r"R_1 = R_3 = Z_0 \cdot \frac{K-1}{K+1}")
            st.latex(r"R_2 = \frac{2Z_0 K}{K^2-1}")

            st.subheader("When to Use T-Section")
            st.write("✅ Low to moderate attenuation (< 10 dB)")
            st.write("✅ When you need lower resistor value spread")
            st.write("✅ Easier to implement with precision resistors")

        with col2:
            st.subheader("PI-Section Formulas")
            st.latex(r"K = 10^{A_{dB}/20}")
            st.latex(r"R_1 = R_3 = Z_0 \cdot \frac{K+1}{K-1}")
            st.latex(r"R_2 = Z_0 \cdot \frac{K^2-1}{2K}")

            st.subheader("When to Use PI-Section")
            st.write("✅ High attenuation (> 10 dB)")
            st.write("✅ Better power handling")
            st.write("✅ More reasonable resistor values at high attenuation")

        st.subheader("General Design Notes")
        st.info("""
        🔧 **Impedance Matching**: Both designs maintain input/output impedance equal to Z₀

        ⚡ **Power Rating**: Ensure resistors can handle the power dissipation

        🎯 **Tolerance**: Use precision resistors (1% or better) for accurate attenuation

        📏 **Frequency Response**: Consider parasitic effects at high frequencies
        """)


if __name__ == "__main__":
    main()