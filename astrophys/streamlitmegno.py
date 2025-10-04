import streamlit as st
import rebound
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import pandas as pd

st.set_page_config(layout="wide")
st.title("MEGNO Simulation")

st.sidebar.header("Modes")
feature = st.sidebar.radio(
    "Pick one of these, thank you",
    ["MEGNO Stability Map", "Planet Trajectories", "MEGNO Between Planets"]
)


if feature == "MEGNO Stability Map":
    st.header("MEGNO Stability Map Settings")
    M_star = st.slider("Stellar Mass [Msun]", 0.1, 5.0, 1.0)
    M_planet_earth = st.slider("Planet Mass [Earth masses]", 1, 1000, 318)
    M_planet = M_planet_earth / 332946.0

    a_min, a_max = st.slider("Semi-major axis range [AU]", 0.1, 20.0, (6.5, 10.5))
    e_min, e_max = st.slider("Eccentricity range", 0.0, 1.0, (0.0, 0.5))
    Ngrid = st.slider("Grid points per axis", 10, 50, 20)

    if st.button("Generate MEGNO Map"):
        st.write("Running MEGNO simulations... (may take some time)")

        def simulation(par, M_star, M_planet):
            a, e = par
            sim = rebound.Simulation()
            sim.integrator = "whfast"
            sim.dt = 5.0
            sim.add(m=M_star)
            sim.add(m=M_planet, a=5.204, M=0.600, omega=0.257, e=0.048)
            sim.add(m=0.000285, a=a, M=0.871, omega=1.616, e=e)
            sim.move_to_com()
            sim.init_megno()
            sim.exit_max_distance = 20.0
            try:
                sim.integrate(5e2*2*np.pi, exact_finish_time=0)
                return sim.megno()
            except rebound.Escape:
                return 10.0

        par_a = np.linspace(a_min, a_max, Ngrid)
        par_e = np.linspace(e_min, e_max, Ngrid)
        parameters = [(a, e) for e in par_e for a in par_a]

        results = [simulation(par, M_star, M_planet) for par in parameters]
        results_grid = np.array(results).reshape((Ngrid, Ngrid))
        results_grid_plot = np.clip(results_grid, 0, 5)

        fig, ax = plt.subplots(figsize=(6,5))
        cax = ax.imshow(results_grid_plot, origin='lower', extent=[a_min, a_max, e_min, e_max],
                        aspect='auto', cmap='plasma')
        fig.colorbar(cax, label='MEGNO')
        ax.set_xlabel('Semi-major axis [AU]')
        ax.set_ylabel('Eccentricity')
        ax.set_title(f'MEGNO Map: Star={M_star} Msun, Planet={M_planet_earth} Earth masses')
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.download_button("Download MEGNO Map", buf, "megno_map.png", "image/png")

# Hypothetical(very important) Planet Trajectories
elif feature == "Planet Trajectories":
    st.header("Planet Trajectory Settings")
    M_star = st.slider("Star mass [Msun]", 0.1, 5.0, 1.0, key="traj_star")
    N_planets = st.slider("Number of planets", 1, 5, 3, key="traj_num")

    planet_masses = [st.slider(f"Planet {i+1} mass [M_sun]", 1e-6, 0.05, 0.001*(i+1), key=f"traj_m{i}") 
                     for i in range(N_planets)]
    planet_a = [st.slider(f"Planet {i+1} SMA [AU]", 0.1, 10.0, 1.0 + i*0.5, key=f"traj_a{i}") 
                for i in range(N_planets)]
    planet_e = [st.slider(f"Planet {i+1} eccentricity", 0.0, 0.5, 0.05, key=f"traj_e{i}") 
                for i in range(N_planets)]
    planet_f = [st.slider(f"Planet {i+1} true anomaly [rad]", 0.0, 6.28, np.random.uniform(0,2*np.pi), key=f"traj_f{i}") 
                for i in range(N_planets)]

    n_steps = st.slider("Number of steps", 100, 5000, 1000, key="traj_nsteps")
    dt = st.slider("Timestep", 0.001, 0.1, 0.01, key="traj_dt")

    if st.button("Compute Trajectories"):
        sim = rebound.Simulation()
        sim.integrator = "ias15"
        sim.add(m=M_star)
        planet_indices = []

        for i in range(N_planets):
            sim.add(m=planet_masses[i], a=planet_a[i], e=planet_e[i], f=planet_f[i])
            planet_indices.append(len(sim.particles)-1)

        sim.move_to_com()

        positions = {idx: {"x": [], "y": []} for idx in planet_indices}
        for _ in range(n_steps):
            sim.integrate(sim.t + dt)
            for idx in planet_indices:
                positions[idx]["x"].append(sim.particles[idx].x)
                positions[idx]["y"].append(sim.particles[idx].y)

        fig, ax = plt.subplots(figsize=(6,6))
        cmap = cm.get_cmap("tab10", N_planets)
        for i, idx in enumerate(planet_indices):
            ax.plot(positions[idx]["x"], positions[idx]["y"], color=cmap(i), label=f"Planet {i+1}")
            ax.plot(positions[idx]["x"][-1], positions[idx]["y"][-1], "o", color=cmap(i))
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_title("Planetary Trajectories")
        ax.set_aspect("equal")
        ax.legend()
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.download_button("Download Trajectory Plot", buf, "trajectories.png", "image/png")



# MEGNO Hypothetical Planet
elif feature == "MEGNO Between Planets":
    st.header("Hypothetical Exoplanet MEGNO Between Two Planets")

    uploaded_file = st.file_uploader("Upload CSV with columns: semi_major_axis, planet_mass (eccentricity optional)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().lower() for c in df.columns]

        if "semi_major_axis" not in df.columns or "planet_mass" not in df.columns:
            st.error("CSV must contain at least 'semi_major_axis' and 'planet_mass' columns")
        elif len(df) < 2:
            st.warning("Please upload at least two planets.")
        else:
            
            if "eccentricity" not in df.columns:
                df["eccentricity"] = 0.0

            st.dataframe(df)
            planet_options = list(df.index)
            col1, col2 = st.columns(2)
            with col1:
                p1_idx = st.selectbox("Select inner planet", planet_options)
            with col2:
                p2_idx = st.selectbox("Select outer planet", planet_options, index=1)

            a1 = float(df.loc[p1_idx, "semi_major_axis"])
            m1 = float(df.loc[p1_idx, "planet_mass"])
            e1 = float(df.loc[p1_idx, "eccentricity"])

            a2 = float(df.loc[p2_idx, "semi_major_axis"])
            m2 = float(df.loc[p2_idx, "planet_mass"])
            e2 = float(df.loc[p2_idx, "eccentricity"])

            if a1 > a2:
                a1, a2 = a2, a1
                m1, m2 = m2, m1
                e1, e2 = e2, e1

            M_star = st.slider("Star mass [Msun]", 0.1, 5.0, 1.0)

            st.subheader("Hypothetical Planet Grid Parameters")
            e_min, e_max = st.slider("Eccentricity range", 0.0, 0.5, (0.0, 0.2))
            N_sma = st.slider("SMA grid points", 5, 50, 20)
            N_e = st.slider("Eccentricity grid points", 5, 20, 10)
            m_test_earth = st.slider("Planet mass [Earth masses]", 1, 500, 10)
            m_test = m_test_earth / 332946.0

            if st.button("Generate MEGNO Grid"):
                st.write("Running MEGNO simulations for grid...")

                sma_vals = np.linspace(a1, a2, N_sma)
                e_vals = np.linspace(e_min, e_max, N_e)
                parameters = [(a, e) for e in e_vals for a in sma_vals]

                def simulation(par):
                    a, e = par
                    sim = rebound.Simulation()
                    sim.integrator = "whfast"
                    sim.dt = 5.0
                    sim.add(m=M_star)
                    sim.add(m=m1, a=a1, e=e1)
                    sim.add(m=m2, a=a2, e=e2)
                    sim.add(m=m_test, a=a, e=e)
                    sim.move_to_com()
                    sim.init_megno()
                    sim.exit_max_distance = 20.0
                    try:
                        sim.integrate(500*2*np.pi, exact_finish_time=0)
                        return sim.megno()
                    except rebound.Escape:
                        return 10.0

                results = [simulation(par) for par in parameters]
                results_grid = np.array(results).reshape((N_e, N_sma))
                results_grid_plot = np.clip(results_grid, 0, 5)

                fig, ax = plt.subplots(figsize=(6,5))
                cax = ax.imshow(results_grid_plot, origin='lower', extent=[a1, a2, e_min, e_max],
                                aspect='auto', cmap='plasma')
                fig.colorbar(cax, label='MEGNO')
                ax.set_xlabel("Semi-major axis [AU]")
                ax.set_ylabel("Eccentricity")
                ax.set_title(f"Hypothetical Planet MEGNO Grid: Mass={m_test_earth} Earth masses")
                st.pyplot(fig)

                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                st.download_button("Download MEGNO Grid", buf, "megno_grid.png", "image/png")


